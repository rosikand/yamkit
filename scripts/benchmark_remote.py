"""Measure the final LeRobot worker/strategy with fake hardware and delayed fake RPC.

Run from the repository: .venv/bin/python -m scripts.benchmark_remote --output PATH
No Modal import, credentials, model download, camera capture or motor activation.
The gate is patched only inside this explicit fake-hardware diagnostic process.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np


def run_scenario(name: str, delays: list[float], *, duration: float, image_hw=(480, 640)) -> dict:
    from lerobot.rollout.configs import RolloutConfig
    from lerobot_robot_yamkit import BiYamFollowerConfig

    from tests.conftest import FakeRobot
    from yamkit.arm import YamArm
    from yamkit.camera_ownership import CameraLease
    from yamkit.config import ArmSpec, ControlSpec, PairSpec, RigConfig
    from yamkit.inference.client import InvalidatedRequest, RemoteFault
    from yamkit.inference.profiles import get_profile
    from yamkit.paths import ROOT
    from yamkit.preview import NullPreview
    from yamkit.remote_policy import YamkitRemoteConfig
    from yamkit.remote_rollout import run_remote_rollout

    profile = get_profile("molmoact2")
    stop = threading.Event()
    robots = []
    execution_times = []
    transport = None

    class Camera:
        is_connected = False

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def read_latest(self):
            return np.zeros((*image_hw, 3), dtype=np.uint8)

    class Transport:
        def __init__(self):
            self.requests = []
            self.last_timing = {"network_only_s": None, "modal_queue_s": None,
                                "source": "injected fake RPC sleep; no real SDK dispatch"}

        def ready(self, timeout_s):
            return {**profile.metadata(), "ready": True, "instance_id": "synthetic-benchmark",
                    "fresh_chunk": True, "saved_processors": True, "image_encoding": "rgb8"}

        def cancel(self):
            pass

        def predict_chunk(self, request, timeout_s):
            delay = delays[len(self.requests) % len(delays)]
            event = {"started": time.monotonic(), "delay_s": delay}
            self.requests.append(event)
            if stop.wait(delay):
                raise InvalidatedRequest("fake diagnostic stopped")
            event["returned"] = time.monotonic()
            return {**{key: request[key] for key in (
                "protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time")},
                "action_units": "robot", "instance_id": "synthetic-benchmark",
                "action_names": list(profile.action_names), "chunk": [[0.2] * 14 for _ in range(30)],
                "timing": dict.fromkeys(("preprocess_s", "inference_s", "postprocess_s", "total_s"), 0.0)}

    def connect(spec, channel, **kwargs):
        robot = FakeRobot()
        original = robot.command_joint_pos

        def command(value):
            original(value)
            execution_times.append(time.monotonic())

        robot.command_joint_pos = command
        robots.append(robot)
        return YamArm(spec, channel, robot, max_joint_speed=kwargs["max_joint_speed"],
                      max_gripper_speed=kwargs["max_gripper_speed"])

    context_dir = ROOT / ".context"
    context_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="remote-benchmark-", dir=context_dir) as temporary, ExitStack() as stack:
        arms = {f"{side}_follower": ArmSpec(name=f"{side}_follower", role="follower", side=side,
                    gripper="linear_4310", gripper_limits=[0.0, 6.5], can_serial=f"FAKE-{side}")
                for side in ("left", "right")}
        # Include leaders for the existing rig validator; they are never connected.
        arms.update({f"{side}_leader": ArmSpec(name=f"{side}_leader", role="leader", side=side,
                      gripper="yam_teaching_handle", can_serial=f"FAKE-LEADER-{side}")
                     for side in ("left", "right")})
        rig = RigConfig(arms=arms, pairs=[PairSpec(f"{side}_leader", f"{side}_follower")
                                         for side in ("left", "right")], control=ControlSpec(home_speed=0))
        rig.cameras = {key: {"type": "opencv", "index_or_path": i, "width": image_hw[1],
                            "height": image_hw[0], "fps": 30} for i, key in enumerate(profile.image_keys)}
        rig.save(Path(temporary) / "rig.yaml")
        transport = Transport()
        stack.enter_context(patch("yamkit.inference.performance.require_physical_modal_rollout", lambda: None))
        stack.enter_context(patch("yamkit.remote_policy.modeling_yamkit_remote.make_transport", lambda cfg: transport))
        stack.enter_context(patch("yamkit.arm.YamArm.connect", connect))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.resolve_channel", lambda spec: "FAKE-" + spec.name))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.make_cameras_from_configs",
                                  lambda configs: {key: Camera() for key in configs}))
        # Diagnostic subprocess must never inherit a live UI ownership session.
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.claim_from_env",
                                  lambda names: CameraLease()))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.start_from_env", lambda *a, **k: NullPreview()))
        cfg = RolloutConfig(robot=BiYamFollowerConfig(rig=str(rig.path)),
                            policy=YamkitRemoteConfig(modal_app="fake-benchmark"), device="cpu",
                            task="synthetic timing diagnostic", duration=duration, fps=30,
                            return_to_initial_position=False)
        started = time.monotonic()
        error = None
        try:
            metrics = run_remote_rollout(cfg, shutdown_event=stop)
        except RemoteFault as exc:
            metrics = exc.metrics
            error = str(exc)
        wall_s = time.monotonic() - started
    # Independent check against successful FakeRobot SDK commands, not queue pops.
    overlap = [sum(event["started"] < at < event["returned"] for at in execution_times) // 2
               for event in transport.requests if "returned" in event]
    return {"name": name, "source": "final LeRobot worker/strategy; fake RGB cameras, fake YAM, fake RPC",
            "injected_delay_cycle_s": delays, "fps": 30, "chunk_steps": 30,
            "image_hw": list(image_hw), "elapsed_s": wall_s, "error": error,
            "all_fake_robots_released": all(robot.closed for robot in robots),
            "sdk_commands_during_completed_rpc": overlap, **metrics}


def compact_report(report: dict) -> dict:
    """Small reviewable artifact; raw per-request timings remain in the full report."""
    result = {"measured_on": datetime.now(UTC).date().isoformat(), "source": report["measurement"],
              "hardware": "FakeRobot and generated zero RGB frames only",
              "model": "fake finite 30x14 chunk; no model inference or Modal SDK calls",
              "fps": 30, "chunk_steps": 30, "image_hw": [480, 640],
              "historical_real_service_warm_sample_count": 2,
              "physical_modal_rollout_allowed": False, "scenarios": []}

    def bounds(key, events):
        values = [event[key] for event in events if key in event]
        return [min(values), max(values)] if values else None

    for scenario in report["scenarios"]:
        events = scenario["prediction_samples"]
        summary = {key: scenario[key] for key in (
            "name", "injected_delay_cycle_s", "elapsed_s", "sample_count", "warm_sample_count",
            "warm_round_trip_s", "failed", "underruns", "executed_actions", "expired_prefix_dropped",
            "overlap_prefix_dropped", "expired_chunks", "peak_queue_depth", "all_fake_robots_released", "error")}
        summary.update(warm_start_depth_range=bounds("queue_depth_at_start", events[1:]),
                       warm_return_depth_range=bounds("queue_depth_at_return", events[1:]),
                       actions_executed_during_prediction_range=bounds("actions_executed_during_prediction", events),
                       valid_horizon_s_range=bounds("remaining_valid_action_horizon_s", events),
                       raw_rgb_payload_bytes=3 * scenario["image_hw"][0] * scenario["image_hw"][1] * 3,
                       overlapping_completed_predictions=sum(
                           count > 0 for count in scenario["sdk_commands_during_completed_rpc"]))
        result["scenarios"].append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, help="optional compact review artifact")
    parser.add_argument("--duration", type=float, default=32,
                        help="seconds per healthy scenario (default gives roughly 60 warm requests)")
    args = parser.parse_args()
    results = []
    for name, delays in (("healthy_50ms", [0.05]),
                         ("healthy_jitter", [0.05, 0.12, 0.08, 0.20, 0.06, 0.15, 0.09, 0.25])):
        result = run_scenario(name, delays, duration=args.duration)
        results.append(result)
        print(f"{name}: {result['sample_count']} successful requests, {result['underruns']} underruns", flush=True)
    # Each historical delay is a separate run: fail-closed execution cannot keep
    # generating consecutive requests once its first chunk has already expired.
    result = run_scenario("late_jitter_after_overlap", [0.05, 0.05, 0.05, 0.7], duration=6)
    results.append(result)
    print(f"{result['name']}: {result['error']}", flush=True)
    for delay in (1.48, 2.378, 3.166):
        result = run_scenario(f"historical_molmo_delay_{delay:g}", [delay], duration=6)
        results.append(result)
        print(f"{result['name']}: {result['error']}", flush=True)
    report = {"measurement": "synthetic delay reproduction; NOT new Modal/network/model measurements",
              "historical_source": "docs/MODAL_VALIDATION.md; original warm n=2 remains n=2",
              "physical_modal_rollout_allowed": False, "scenarios": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(compact_report(report), indent=2) + "\n")


if __name__ == "__main__":
    main()
