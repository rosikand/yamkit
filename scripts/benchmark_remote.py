"""Measure the final LeRobot worker/strategy with fake hardware, or profile Modal.

Run from the repository: .venv/bin/python -m scripts.benchmark_remote --output PATH
Default runs use delayed fake RPC with no cloud access. --modal-app explicitly
profiles an existing service with non-executable generated RGB fixtures; it never
deploys a service. --integrated-modal tests that service through the final LeRobot
path with fake hardware. No mode captures cameras or activates motors. The gate is
patched only inside the explicit fake-hardware diagnostic context.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import uuid
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np


@contextmanager
def modal_sdk_measurements():
    """Measure actual pinned-SDK work once; retain sizes/times, never payload data.

    Private SDK hooks are scoped to this explicit single-request diagnostic. No
    production transport behavior or serialization is replaced.
    """
    import modal._utils.function_utils as sdk

    metrics = {}

    def record(key, value):
        metrics[key] = metrics.get(key, 0) + value

    serialize = sdk._serialize_data_format
    deserialize = sdk.deserialize_data_format
    upload = sdk.blob_upload_with_r2_failure_info
    download = sdk.blob_download

    def measured_serialize(*args, **kwargs):
        started = time.monotonic()
        result = serialize(*args, **kwargs)
        record("request_serialization_s", time.monotonic() - started)
        record("serialized_request_bytes", len(result))
        record("request_serialization_calls", 1)
        return result

    def measured_deserialize(data, *args, **kwargs):
        started = time.monotonic()
        result = deserialize(data, *args, **kwargs)
        record("response_deserialization_s", time.monotonic() - started)
        record("serialized_response_bytes", len(data))
        return result

    async def measured_upload(data, *args, **kwargs):
        started = time.monotonic()
        try:
            return await upload(data, *args, **kwargs)
        finally:
            record("input_blob_upload_s", time.monotonic() - started)
            record("input_blob_bytes", len(data))

    async def measured_download(*args, **kwargs):
        started = time.monotonic()
        result = await download(*args, **kwargs)
        record("output_blob_download_s", time.monotonic() - started)
        record("output_blob_bytes", len(result))
        return result

    with ExitStack() as stack:
        for name, function in (("_serialize_data_format", measured_serialize),
                               ("deserialize_data_format", measured_deserialize),
                               ("blob_upload_with_r2_failure_info", measured_upload),
                               ("blob_download", measured_download)):
            stack.enter_context(patch.object(sdk, name, function))
        yield metrics


def make_benchmark_transport(app_name, profile_name="molmoact2", *, shutdown_event=None,
                             uncached_handles=False, sdk_metrics=None, call_mode="remote"):
    from yamkit.inference.client import ModalTransport

    class MeasuredModalTransport(ModalTransport):
        def _invoke(self, method, payload, timeout_s):
            if uncached_handles:
                self._service = None
            if sdk_metrics is not None:
                sdk_metrics.clear()
            try:
                return super()._invoke(method, payload, timeout_s)
            finally:
                if sdk_metrics is not None:
                    self.last_timing["sdk"] = dict(sdk_metrics)

    return MeasuredModalTransport(app_name, profile_name, shutdown_event=shutdown_event, call_mode=call_mode)


def run_scenario(name: str, delays: list[float], *, duration: float, image_hw=(480, 640),
                 transport_factory=None, target_warm_samples: int | None = None, policy_options=None) -> dict:
    from lerobot.rollout.configs import RolloutConfig
    from lerobot_robot_yamkit import BiYamFollowerConfig

    from tests.conftest import FakeRobot
    from yamkit.arm import YamArm
    from yamkit.camera_ownership import CameraLease
    from yamkit.config import ArmSpec, ControlSpec, PairSpec, RigConfig
    from yamkit.inference.client import InvalidatedRequest, RemoteFault
    from yamkit.inference.profiles import get_profile
    from yamkit.inference.qualification import host_identity
    from yamkit.paths import ROOT
    from yamkit.preview import NullPreview
    from yamkit.remote_policy import YamkitRemoteConfig
    from yamkit.remote_rollout import run_remote_rollout

    profile = get_profile("molmoact2")
    class ObservedStop(threading.Event):
        requested_at = None

        def set(self):
            if self.requested_at is None:
                self.requested_at = time.monotonic()
            super().set()

    stop = ObservedStop()
    robots = []
    execution_times = []
    transport = None

    if duration <= 0 or not delays or any(delay < 0 for delay in delays):
        raise ValueError("Scenario requires a positive duration and nonnegative RPC delays")
    if target_warm_samples is not None and target_warm_samples < 1:
        raise ValueError("target_warm_samples must be positive")

    class Camera:
        is_connected = False

        def __init__(self, seed):
            # Fixed generated textures keep JPEG payload entropy comparable to
            # the direct fixture profile; zero frames would unfairly shrink RPCs.
            self.frame = np.random.default_rng(seed).integers(0, 256, (*image_hw, 3), dtype=np.uint8)

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def read_latest(self):
            return self.frame

    class Transport:
        def __init__(self):
            self.requests = []
            self.last_timing = {"network_only_s": None, "modal_queue_s": None,
                                "source": "injected fake RPC sleep; no real SDK dispatch"}

        def ready(self, timeout_s):
            return {**profile.metadata(), "ready": True, "instance_id": "synthetic-benchmark",
                    "fresh_chunk": True, "saved_processors": True, "image_encoding": "rgb8",
                    "supported_image_encodings": ["jpeg", "rgb8"], "preferred_image_encoding": "rgb8",
                    "jpeg_quality": 85, "prediction_count": 1}

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

    class ObservedTransport:
        """Observe the same transport without touching its dispatch or freshness rules."""

        def __init__(self, inner):
            self.inner = inner
            self.requests = []
            self.readiness = None
            self.readiness_s = None

        @property
        def last_timing(self):
            return getattr(self.inner, "last_timing", {})

        def ready(self, timeout_s):
            started = time.monotonic()
            self.readiness = self.inner.ready(timeout_s)
            self.readiness_s = time.monotonic() - started
            return self.readiness

        def cancel(self):
            self.inner.cancel()

        def predict_chunk(self, request, timeout_s):
            event = {"started": time.monotonic()}
            self.requests.append(event)
            response = self.inner.predict_chunk(request, timeout_s)
            event["returned"] = time.monotonic()
            return response

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
        transport = ObservedTransport(transport_factory(stop) if transport_factory else Transport())
        stack.enter_context(patch("yamkit.inference.performance.require_physical_modal_rollout", lambda *a, **k: None))
        stack.enter_context(patch("yamkit.remote_policy.modeling_yamkit_remote.make_transport", lambda cfg: transport))
        stack.enter_context(patch("yamkit.arm.YamArm.connect", connect))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.resolve_channel", lambda spec: "FAKE-" + spec.name))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.make_cameras_from_configs",
                                  lambda configs: {key: Camera(index) for index, key in enumerate(configs)}))
        # Diagnostic subprocess must never inherit a live UI ownership session.
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.claim_from_env",
                                  lambda names: CameraLease()))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.start_from_env", lambda *a, **k: NullPreview()))
        cfg = RolloutConfig(robot=BiYamFollowerConfig(rig=str(rig.path)),
                            policy=YamkitRemoteConfig(modal_app="fake-benchmark", **(policy_options or {})), device="cpu",
                            task="synthetic timing diagnostic", duration=duration, fps=30,
                            return_to_initial_position=False)
        started = time.monotonic()
        error = None
        monitor_done = threading.Event()

        def stop_at_target():
            # Observe completed responses only. The worker still validates/merges
            # normally; Stop invalidates any final unconsumed response as usual.
            while not monitor_done.wait(0.05):
                completed = sum("returned" in event for event in transport.requests)
                if (target_warm_samples is not None and completed >= target_warm_samples + 1
                        and len(transport.requests) > completed):
                    stop.set()
                    return

        monitor = threading.Thread(target=stop_at_target, daemon=True, name="benchmark-request-limit")
        if target_warm_samples is not None:
            monitor.start()
        try:
            metrics = run_remote_rollout(cfg, shutdown_event=stop)
        except RemoteFault as exc:
            metrics = exc.metrics
            error = str(exc)
        finally:
            monitor_done.set()
            if monitor.is_alive():
                monitor.join(timeout=1)
        wall_s = time.monotonic() - started
    # Independent check against successful FakeRobot SDK commands, not queue pops.
    overlap = [sum(event["started"] < at < event["returned"] for at in execution_times) // 2
               for event in transport.requests if "returned" in event]
    return {"name": name, "source": "final LeRobot worker/strategy; fake RGB cameras and fake YAM; "
            + ("real Modal RPC" if transport_factory else "fake RPC"),
            "measurement_host": host_identity(),
            "injected_delay_cycle_s": None if transport_factory else delays, "fps": 30, "chunk_steps": 30,
            "image_hw": list(image_hw), "elapsed_s": wall_s, "error": error,
            "fixture_pattern": "fixed seeded random RGB per camera; fake measured state follows commanded actions",
            "policy_options": {"image_encoding": cfg.policy.image_encoding, "jpeg_quality": cfg.policy.jpeg_quality,
                               "call_mode": cfg.policy.call_mode, "center_crop": cfg.policy.center_crop,
                               "prediction_queue_threshold": cfg.policy.prediction_queue_threshold
                               if cfg.policy.prediction_queue_threshold is not None else profile.chunk_size},
            "readiness_s": transport.readiness_s, "readiness": transport.readiness,
            "stop_requested_monotonic_s": stop.requested_at,
            "commands_after_stop": sum(at >= stop.requested_at for at in execution_times)
            if stop.requested_at is not None else None,
            "stop_requested_during_inflight_rpc": stop.requested_at is not None and any(
                event["started"] < stop.requested_at < event.get("returned", float("inf"))
                for event in transport.requests),
            "all_fake_robots_released": all(robot.closed for robot in robots),
            "sdk_commands_during_completed_rpc": overlap, **metrics}


def profile_modal(transport, *, profile_name="molmoact2", warm_samples=100,
                  max_wall_s=600.0, image_hw=(480, 640), on_sample=None,
                  image_encoding="rgb8", jpeg_quality=85, center_crop=False,
                  diagnostic_num_inference_steps=None, diagnostic_cuda_graph=None) -> dict:
    """Bounded direct protocol profiling; generated fixture results never enter a queue.

    In contrast to rollout, non-executable native fixtures can be measured after
    the physical action horizon has elapsed. No rollout freshness check is changed.
    """
    from yamkit.inference.performance import summarize_measurements
    from yamkit.inference.profiles import get_profile
    from yamkit.inference.protocol import (
        MAX_IMAGE_HEIGHT,
        MAX_IMAGE_WIDTH,
        PROTOCOL_VERSION,
        encode_image,
        validate_request,
        validate_response,
    )
    from yamkit.inference.qualification import host_identity
    from yamkit.modal_ops import _validate_ready

    if not 1 <= warm_samples <= 500 or not 0 < max_wall_s <= 1800:
        raise ValueError("Use 1–500 warm requests and a wall budget of at most 1800 seconds")
    if image_encoding not in ("jpeg", "rgb8") or type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("Use jpeg or rgb8 and an integer JPEG quality in 1–100")
    if diagnostic_num_inference_steps is not None and (
            type(diagnostic_num_inference_steps) is not int or diagnostic_num_inference_steps not in (5, 10)):
        raise ValueError("Molmo diagnostic inference steps must be 5 or 10; production defaults are unchanged")
    if diagnostic_cuda_graph is not None and type(diagnostic_cuda_graph) is not bool:
        raise ValueError("diagnostic_cuda_graph must be a boolean")
    height, width = image_hw
    if not 1 <= height <= MAX_IMAGE_HEIGHT or not 1 <= width <= MAX_IMAGE_WIDTH:
        raise ValueError("Fixture dimensions exceed the unchanged protocol bounds")
    profile = get_profile(profile_name)
    started = time.monotonic()
    deadline = started + max_wall_s
    samples = []
    failures = []
    metadata = None
    readiness_s = None
    readiness_transport_timing = {}
    session_id = uuid.uuid4().hex
    terminated = "request_limit"
    try:
        ready_started = time.monotonic()
        metadata = transport.ready(min(120.0, max_wall_s))
        readiness_s = time.monotonic() - ready_started
        readiness_transport_timing = dict(getattr(transport, "last_timing", {}))
        _validate_ready(metadata, profile)
        if not isinstance(metadata.get("instance_id"), str) or not metadata["instance_id"]:
            raise ValueError("Benchmark readiness requires a container instance identity")
        for sequence in range(warm_samples + 1):
            remaining = deadline - time.monotonic()
            if remaining < 0.01:
                terminated = "wall_budget"
                break
            generated_at = time.monotonic()
            rng = np.random.default_rng(sequence)
            frames = {key: rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
                      for key in profile.native_image_keys}
            fixture_generation_s = time.monotonic() - generated_at
            # This timestamp describes fixture availability, never camera exposure.
            observation_time = time.monotonic()
            images = {}
            camera_timings = {}
            for key, frame in frames.items():
                encode_started = time.monotonic()
                images[key] = encode_image(frame, encoding=image_encoding, quality=jpeg_quality)
                encode_s = time.monotonic() - encode_started
                camera_timings[key] = {"encode_s": encode_s,
                                       "jpeg_encode_s": encode_s if image_encoding == "jpeg" else 0.0,
                                       "payload_bytes": len(images[key]["data"])}
            request = {"protocol_version": PROTOCOL_VERSION, "profile": profile.id,
                       "model_revision": profile.revision, "session_id": session_id,
                       "sequence_id": sequence, "observation_time": observation_time,
                       "observation_age_s": time.monotonic() - observation_time,
                       "timeout_s": min(120.0, remaining), "task": "pick up the red cube",
                       "state": [0.0] * len(profile.state_names), "state_names": list(profile.state_names),
                       "images": images, "mode": "native_fixture",
                       "crop": "center_16_9" if center_crop else "none", "continuation": None}
            if diagnostic_num_inference_steps is not None:
                request["diagnostic_num_inference_steps"] = diagnostic_num_inference_steps
            if diagnostic_cuda_graph is not None:
                request["diagnostic_cuda_graph"] = diagnostic_cuda_graph
            validation_started = time.monotonic()
            validate_request(request, profile)
            request_validation_s = time.monotonic() - validation_started
            dispatched_at = time.monotonic()
            if deadline - dispatched_at < 0.01:
                terminated = "wall_budget"
                break
            response = transport.predict_chunk(request, min(request["timeout_s"], deadline - dispatched_at))
            returned_at = time.monotonic()
            validation_started = time.monotonic()
            validate_response(response, request, profile)
            response_validation_s = time.monotonic() - validation_started
            if response.get("instance_id") != metadata.get("instance_id"):
                raise ValueError("Container changed during benchmark; a mixed-container warm distribution is invalid")
            server_timing = response.get("timing", {})
            elapsed = returned_at - dispatched_at
            sample = {"sequence_id": sequence, "instance_id": response.get("instance_id"),
                      "round_trip_s": elapsed, "total_request_s": time.monotonic() - observation_time,
                      "observation_age_at_dispatch_s": dispatched_at - observation_time,
                      "fixture_generation_s": fixture_generation_s, "camera_exposure_timestamp_s": None,
                      "timestamp_basis": "local generated-fixture availability, not camera capture",
                      "image_encoding": image_encoding, "jpeg_quality": jpeg_quality if image_encoding == "jpeg" else None,
                      "image_hw": [height, width], "per_camera": camera_timings,
                      "payload_bytes": sum(len(image["data"]) for image in images.values()),
                      "wire_payload_bytes": None,
                      "payload_size_basis": "encoded image bytes; SDK request framing size is unobservable",
                      "request_validation_s": request_validation_s,
                      "response_validation_s": response_validation_s,
                      "transport_timing": dict(getattr(transport, "last_timing", {})),
                      "server_timing": server_timing, "lifecycle": response.get("lifecycle"),
                      "model_execution": response.get("model_execution"),
                      "diagnostic_num_inference_steps": diagnostic_num_inference_steps,
                      "diagnostic_cuda_graph": diagnostic_cuda_graph,
                      "rpc_minus_server_s": max(0.0, elapsed - server_timing["total_s"]),
                      "rpc_residual_note": "Includes SDK serialization, dispatch, routing and network; not network-only",
                      "shape": [len(response["chunk"]), len(response["chunk"][0])]}
            samples.append(sample)
            if on_sample is not None:
                on_sample(sample)
    except Exception as exc:  # noqa: BLE001 — keep SDK exceptions and credentials out of artifacts
        # SDK exceptions can contain request data/credentials. Retain the type only.
        failures.append({"reason": type(exc).__name__, "elapsed_s": time.monotonic() - started,
                         "transport_timing": dict(getattr(transport, "last_timing", {}))})
        terminated = "failure"
    except KeyboardInterrupt:
        terminated = "interrupted"
    finally:
        transport.cancel()
    return {"measurement": "real Modal RPC; generated checkpoint-native fixtures; no action execution",
            "measurement_host": host_identity(),
            "profile": profile.id, "revision": profile.revision, "image_hw": [height, width],
            "fixture_pattern": "seeded random RGB generated per request; zero native state",
            "call_mode": getattr(transport, "call_mode", None),
            "image_encoding": image_encoding, "jpeg_quality": jpeg_quality if image_encoding == "jpeg" else None,
            "crop": "center_16_9" if center_crop else "none",
            "diagnostic_num_inference_steps": diagnostic_num_inference_steps,
            "diagnostic_cuda_graph": diagnostic_cuda_graph,
            "experiment_only": diagnostic_num_inference_steps is not None or diagnostic_cuda_graph is True,
            "requested_warm_samples": warm_samples, "max_wall_s": max_wall_s,
            "readiness_s": readiness_s, "readiness": metadata,
            "readiness_transport_timing": readiness_transport_timing,
            "readiness_cold_start_verified": False,
            "cold_start_note": "Readiness can reuse a warm pool. Model load telemetry is server-local, not measured cold RPC.",
            "elapsed_s": time.monotonic() - started, "terminated": terminated,
            "failures": failures, "samples": samples,
            **summarize_measurements(samples, minimum_warm_samples=max(50, min(100, warm_samples))),
            "queue_depth": None, "underruns": None, "physical_modal_rollout_allowed": False}


def compare_encodings(transport, *, pairs=4, image_hw=(480, 640), jpeg_quality=85, center_crop=False) -> dict:
    """Paired fixed images/state/task and seeded model noise; predictions never execute."""
    from yamkit.inference.profiles import get_profile
    from yamkit.inference.protocol import (
        MAX_IMAGE_HEIGHT,
        MAX_IMAGE_WIDTH,
        PROTOCOL_VERSION,
        encode_image,
        validate_request,
        validate_response,
    )

    if type(pairs) is not int or not 1 <= pairs <= 6:
        raise ValueError("Encoding comparison is bounded to 1–6 pairs")
    profile = get_profile("molmoact2")
    height, width = image_hw
    if (type(height) is not int or type(width) is not int
            or not 1 <= height <= MAX_IMAGE_HEIGHT or not 1 <= width <= MAX_IMAGE_WIDTH):
        raise ValueError("Fixture dimensions exceed the unchanged protocol bounds")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("JPEG quality must be an integer in 1–100")
    frames = {key: np.random.default_rng(index).integers(0, 256, (height, width, 3), dtype=np.uint8)
              for index, key in enumerate(profile.native_image_keys)}
    results = []
    instance_id = None
    try:
        for pair in range(pairs):
            responses = {}
            samples = {}
            for encoding in ("rgb8", "jpeg"):
                observation_time = time.monotonic()
                images = {key: encode_image(frame, encoding=encoding, quality=jpeg_quality)
                          for key, frame in frames.items()}
                request = {"protocol_version": PROTOCOL_VERSION, "profile": profile.id,
                           "model_revision": profile.revision, "session_id": uuid.uuid4().hex,
                           "sequence_id": pair * 2 + (encoding == "jpeg"),
                           "observation_time": observation_time,
                           "observation_age_s": time.monotonic() - observation_time,
                           "timeout_s": 30.0, "task": "pick up the red cube",
                           "state": [0.0] * len(profile.state_names), "state_names": list(profile.state_names),
                           "images": images, "mode": "native_fixture", "diagnostic_seed": pair,
                           "crop": "center_16_9" if center_crop else "none", "continuation": None}
                validate_request(request, profile)
                started = time.monotonic()
                response = transport.predict_chunk(request, 30.0)
                elapsed = time.monotonic() - started
                validate_response(response, request, profile)
                if response.get("diagnostic_seed") != pair:
                    raise ValueError("Service did not confirm the paired diagnostic seed")
                if instance_id is not None and response.get("instance_id") != instance_id:
                    raise ValueError("Container changed during paired encoding comparison")
                instance_id = response.get("instance_id")
                responses[encoding] = np.asarray(response["chunk"], dtype=np.float64)
                samples[encoding] = {"round_trip_s": elapsed, "server_timing": response["timing"],
                                     "transport_timing": dict(getattr(transport, "last_timing", {})),
                                     "payload_bytes": sum(len(image["data"]) for image in request["images"].values())}
            if responses["rgb8"].shape != responses["jpeg"].shape:
                raise ValueError("Paired encodings returned different action chunk shapes")
            delta = np.abs(responses["rgb8"] - responses["jpeg"])
            results.append({"diagnostic_seed": pair, "samples": samples, "shape": list(delta.shape),
                            "action_max_absolute_delta": float(delta.max()),
                            "action_mean_absolute_delta": float(delta.mean()),
                            "action_p95_absolute_delta": float(np.percentile(delta, 95)),
                            "per_action_max_absolute_delta": dict(zip(profile.action_names, delta.max(axis=0).tolist()))})
    finally:
        transport.cancel()
    return {"measurement": "paired generated RGB fixtures, identical state/task and seeded Molmo noise; no execution",
            "image_hw": [height, width], "jpeg_quality": jpeg_quality, "pairs": results,
            "instance_id": instance_id, "note": "Numerical sensitivity diagnostic, not physical policy fidelity validation."}


def _compare_variants(transport, variants, *, pairs, image_hw, center_crop=False, max_wall_s=600) -> dict:
    """Bounded native-fixture experiments with common images, state, task and model noise."""
    from yamkit.inference.performance import percentile_summary
    from yamkit.inference.profiles import get_profile
    from yamkit.inference.protocol import (
        MAX_IMAGE_HEIGHT,
        MAX_IMAGE_WIDTH,
        PROTOCOL_VERSION,
        decode_image,
        encode_image,
        validate_request,
        validate_response,
    )

    if type(pairs) is not int or not 1 <= pairs <= 25 or not 0 < max_wall_s <= 1800:
        raise ValueError("Use 1–25 paired seeds and at most 1800 seconds")
    height, width = image_hw
    if (type(height) is not int or type(width) is not int
            or not 1 <= height <= MAX_IMAGE_HEIGHT or not 1 <= width <= MAX_IMAGE_WIDTH):
        raise ValueError("Fixture dimensions exceed the unchanged protocol bounds")
    profile = get_profile("molmoact2")
    frames = {key: np.random.default_rng(index).integers(0, 256, (height, width, 3), dtype=np.uint8)
              for index, key in enumerate(profile.native_image_keys)}
    gripper_indices = [index for index, name in enumerate(profile.action_names) if "gripper" in name]
    joint_indices = [index for index in range(len(profile.action_names)) if index not in gripper_indices]
    started = time.monotonic()
    deadline = started + max_wall_s
    results = []
    instance_id = None
    failures = []
    terminated = "request_limit"
    session_id = uuid.uuid4().hex
    try:
        for pair in range(pairs):
            reference = None
            paired = {"diagnostic_seed": pair, "variants": {}}
            results.append(paired)
            for variant_index, variant in enumerate(variants):
                observation_time = time.monotonic()
                images = {}
                cameras = {}
                for key, frame in frames.items():
                    encoded_at = time.monotonic()
                    images[key] = encode_image(frame, encoding=variant["image_encoding"],
                                               quality=variant.get("jpeg_quality", 85))
                    encode_s = time.monotonic() - encoded_at
                    decoded_at = time.monotonic()
                    decoded = decode_image(images[key])
                    decode_s = time.monotonic() - decoded_at
                    cameras[key] = {"encode_s": encode_s, "diagnostic_decode_s": decode_s,
                                    "payload_bytes": len(images[key]["data"]),
                                    "pixel_mean_absolute_delta": float(np.abs(decoded.astype(float) - frame).mean())}
                remaining = deadline - time.monotonic()
                if remaining < 0.01:
                    raise TimeoutError("Paired diagnostic wall budget reached")
                request = {"protocol_version": PROTOCOL_VERSION, "profile": profile.id,
                           "model_revision": profile.revision, "session_id": session_id,
                           "sequence_id": pair * len(variants) + variant_index,
                           "observation_time": observation_time,
                           "observation_age_s": time.monotonic() - observation_time,
                           "timeout_s": min(30.0, remaining), "task": "pick up the red cube",
                           "state": [0.0] * len(profile.state_names), "state_names": list(profile.state_names),
                           "images": images, "mode": "native_fixture", "diagnostic_seed": pair,
                           "crop": "center_16_9" if center_crop else "none", "continuation": None}
                steps = variant.get("diagnostic_num_inference_steps")
                if steps is not None:
                    request["diagnostic_num_inference_steps"] = steps
                if "diagnostic_cuda_graph" in variant:
                    request["diagnostic_cuda_graph"] = variant["diagnostic_cuda_graph"]
                validate_request(request, profile)
                dispatched = time.monotonic()
                response = transport.predict_chunk(request, request["timeout_s"])
                elapsed = time.monotonic() - dispatched
                validate_response(response, request, profile)
                if not isinstance(response.get("instance_id"), str) or not response["instance_id"]:
                    raise ValueError("Paired diagnostics require a container identity")
                if instance_id is not None and response.get("instance_id") != instance_id:
                    raise ValueError("Container changed during paired diagnostic")
                instance_id = response.get("instance_id")
                actions = np.asarray(response["chunk"], dtype=np.float64)
                if reference is None:
                    reference = actions
                if actions.shape != reference.shape:
                    raise ValueError("Paired variants returned different action chunk shapes")
                delta = np.abs(actions - reference)
                paired["variants"][variant["name"]] = {
                    **variant, "round_trip_s": elapsed, "server_timing": response["timing"],
                    "transport_timing": dict(getattr(transport, "last_timing", {})),
                    "model_execution": response.get("model_execution"), "per_camera": cameras,
                    "payload_bytes": sum(camera["payload_bytes"] for camera in cameras.values()),
                    "image_encode_s": sum(camera["encode_s"] for camera in cameras.values()),
                    "diagnostic_image_decode_s": sum(camera["diagnostic_decode_s"] for camera in cameras.values()),
                    "shape": list(delta.shape), "action_max_absolute_delta": float(delta.max()),
                    "joint_max_absolute_delta_rad": float(delta[:, joint_indices].max()),
                    "gripper_max_absolute_delta": float(delta[:, gripper_indices].max()),
                    "per_action_max_absolute_delta": dict(zip(profile.action_names, delta.max(axis=0).tolist()))}
    except TimeoutError:
        terminated = "wall_budget"
    except Exception as exc:  # noqa: BLE001 — retain partial diagnostics without SDK data or credentials
        failures.append({"reason": type(exc).__name__, "elapsed_s": time.monotonic() - started})
        terminated = "failure"
    except KeyboardInterrupt:
        terminated = "interrupted"
    finally:
        transport.cancel()

    def _variant_report():
        summaries = {}
        for variant in variants:
            samples = [pair["variants"][variant["name"]] for pair in results if variant["name"] in pair["variants"]]
            summaries[variant["name"]] = {
                **variant, "sample_count": len(samples), "first_sample": samples[0] if samples else None,
                "warm_sample_count": max(0, len(samples) - 1),
                "warm_round_trip_s": percentile_summary(sample["round_trip_s"] for sample in samples[1:]),
                "warm_model_forward_s": percentile_summary(sample["server_timing"]["inference_s"] for sample in samples[1:]),
                "payload_bytes": percentile_summary(sample["payload_bytes"] for sample in samples),
                "image_encode_s": percentile_summary(sample["image_encode_s"] for sample in samples),
                "diagnostic_image_decode_s": percentile_summary(sample["diagnostic_image_decode_s"] for sample in samples),
                "action_max_absolute_delta": max((sample["action_max_absolute_delta"] for sample in samples), default=None),
                "joint_max_absolute_delta_rad": max((sample["joint_max_absolute_delta_rad"] for sample in samples), default=None),
                "gripper_max_absolute_delta": max((sample["gripper_max_absolute_delta"] for sample in samples), default=None)}
        return {"measurement": "native-fixture experiment; fixed images/state/task and identical paired model seeds",
                "experiment_only": True, "physical_modal_rollout_allowed": False,
                "image_hw": [height, width], "requested_pairs": pairs, "pairs": results,
                "variants": summaries, "instance_id": instance_id, "terminated": terminated,
                "elapsed_s": time.monotonic() - started, "failures": failures,
                "note": "Per-variant first requests are separated; small paired samples are not robust network tail estimates."}

    return _variant_report()


def compare_quality_sweep(transport, *, pairs=4, jpeg_qualities=(85, 90, 95), image_hw=(480, 640),
                          center_crop=False, max_wall_s=600) -> dict:
    """Compare all three cameras together and select the smallest qualifying fixture payload."""
    if (not 1 <= len(jpeg_qualities) <= 3 or len(set(jpeg_qualities)) != len(jpeg_qualities)
            or any(type(quality) is not int or quality not in (85, 90, 95) for quality in jpeg_qualities)):
        raise ValueError("Quality diagnostics support unique JPEG qualities from 85, 90 and 95")
    variants = [{"name": "raw_reference", "image_encoding": "rgb8"},
                {"name": "raw_repeat", "image_encoding": "rgb8"},
                *({"name": f"jpeg_q{quality}", "image_encoding": "jpeg", "jpeg_quality": quality}
                  for quality in jpeg_qualities)]
    report = _compare_variants(transport, variants, pairs=pairs, image_hw=image_hw,
                               center_crop=center_crop, max_wall_s=max_wall_s)
    repeat = report["variants"]["raw_repeat"]
    deterministic = (repeat["sample_count"] == pairs and repeat["action_max_absolute_delta"] is not None
                     and repeat["action_max_absolute_delta"] <= 1e-6)
    candidates = [summary for summary in report["variants"].values()
                  if summary["image_encoding"] == "jpeg" and summary["sample_count"] == pairs
                  and summary["gripper_max_absolute_delta"] is not None
                  and summary["gripper_max_absolute_delta"] < 0.02
                  and summary["joint_max_absolute_delta_rad"] <= 0.01]
    complete = report["terminated"] == "request_limit"
    chosen = (min(candidates, key=lambda sample: sample["payload_bytes"]["p50"])
              if candidates and deterministic and complete else None)
    report.update(raw_repeat_deterministic=deterministic, gripper_delta_limit=0.02,
                  joint_delta_limit_rad=0.01,
                  selected_encoding=chosen["image_encoding"] if chosen else "rgb8",
                  selected_jpeg_quality=chosen["jpeg_quality"] if chosen else None,
                  selection_basis="Complete sweep: smallest payload with max gripper delta <0.02, "
                                  "max joint delta <=0.01 rad and deterministic raw repeats; "
                                  "raw fallback otherwise. Fixture evidence only; no physical fidelity claim.")
    return report


def compare_denoising(transport, *, half_steps=5, pairs=11, image_hw=(480, 640), center_crop=False,
                      max_wall_s=600, include_graph=False) -> dict:
    """Compare default denoising and explicit half steps with raw images and the same model noise."""
    if type(half_steps) is not int or half_steps != 5:
        raise ValueError("The pinned Molmo default is 10 inference steps; this diagnostic compares 5")
    variants = [{"name": "raw_default_steps", "image_encoding": "rgb8"},
                {"name": "raw_half_steps", "image_encoding": "rgb8", "diagnostic_num_inference_steps": half_steps}]
    if include_graph:
        variants.append({"name": "raw_default_steps_cuda_graph", "image_encoding": "rgb8",
                         "diagnostic_cuda_graph": True})
    return _compare_variants(transport, variants, pairs=pairs, image_hw=image_hw,
                             center_crop=center_crop, max_wall_s=max_wall_s)


def compact_report(report: dict) -> dict:
    """Small reviewable artifact; raw per-request timings remain in the full report."""
    result = {"measured_on": datetime.now(UTC).date().isoformat(), "source": report["measurement"],
              "hardware": "FakeRobot and generated fixed random RGB textures only",
              "model": "fake finite 30x14 chunk; no model inference or Modal SDK calls",
              "fps": 30, "chunk_steps": 30, "image_hw": [480, 640],
              "historical_real_service_warm_sample_count": 2,
              "physical_modal_rollout_allowed": False, "scenarios": []}

    def bounds(key, events):
        values = [event[key] for event in events if type(event.get(key)) in (int, float)]
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
        summary["warm_sample_requirement_met"] = scenario["warm_sample_count"] >= 100
        for key in ("queue_deadline_horizon_at_start_s", "queue_deadline_horizon_at_return_s",
                    "queue_deadline_horizon_at_merge_s", "queue_deadline_horizon_after_merge_s",
                    "next_action_margin_at_start_s", "next_action_margin_at_return_s"):
            summary[f"{key}_range"] = bounds(key, events)
        for key in ("minimum_execution_queue_depth", "minimum_dispatch_margin_s", "expired_queued_actions",
                    "expired_before_dispatch", "readiness_s", "stop_to_robot_release_s"):
            if key in scenario:
                summary[key] = scenario[key]
        result["scenarios"].append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, help="optional compact review artifact")
    parser.add_argument("--duration", type=float, default=75,
                        help="maximum seconds per integrated healthy scenario")
    parser.add_argument("--modal-app", help="explicit existing Modal service; no deployment or shutdown")
    parser.add_argument("--warm-samples", type=int, default=100,
                        help="target warm requests, excluding first request (50–500)")
    parser.add_argument("--max-wall-s", type=float, default=600,
                        help="direct Modal profile wall budget including readiness (at most 1800)")
    parser.add_argument("--integrated-modal", action="store_true",
                        help="also run existing Modal service through LeRobot with fake hardware")
    parser.add_argument("--integrated-only", action="store_true",
                        help="only run the final integrated fake-hardware Modal scenario")
    parser.add_argument("--uncached-handles", action="store_true",
                        help="diagnostic baseline: clear the cached service handle before each call")
    parser.add_argument("--sdk-profile", action="store_true",
                        help="measure actual pinned SDK serialization, byte size and blob transfer")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--image-encoding", choices=("jpeg", "rgb8"), default="rgb8")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--call-mode", choices=("remote", "spawn"), default="remote")
    parser.add_argument("--center-crop", action="store_true")
    parser.add_argument("--encoding-pairs", type=int, default=0,
                        help="optional 1–6 seeded raw/JPEG pairs before warm profiling")
    parser.add_argument("--quality-pairs", type=int, default=0,
                        help="optional 1–25 paired raw repeats and JPEG q85/q90/q95; reports a fixture-only codec choice")
    parser.add_argument("--denoising-pairs", type=int, default=0,
                        help="optional 1–25 paired raw default/half-step experiments; 11 gives 10 warm each")
    parser.add_argument("--denoising-graphs", action="store_true",
                        help="include default-step CUDA graphs in the paired denoising diagnostic")
    parser.add_argument("--diagnostic-num-inference-steps", type=int, choices=(5, 10),
                        help="native-fixture profile experiment only; never changes rollout defaults")
    parser.add_argument("--diagnostic-cuda-graph", action="store_true", default=None,
                        help="native-fixture profile experiment only; enable supported CUDA graphs")
    args = parser.parse_args()
    if not 50 <= args.warm_samples <= 500:
        parser.error("--warm-samples must be 50–500; interrupted runs are marked incomplete")
    if not 0 < args.duration <= 1800 or not 0 < args.max_wall_s <= 1800:
        parser.error("duration and wall budget must be in (0, 1800] seconds")
    if not args.modal_app and any((args.integrated_modal, args.integrated_only,
                                   args.uncached_handles, args.sdk_profile, args.encoding_pairs,
                                   args.quality_pairs, args.denoising_pairs, args.denoising_graphs,
                                   args.diagnostic_num_inference_steps, args.diagnostic_cuda_graph)):
        parser.error("Modal options require an explicit --modal-app")
    if not 0 <= args.quality_pairs <= 25 or not 0 <= args.denoising_pairs <= 25:
        parser.error("--quality-pairs and --denoising-pairs must be 0–25")
    if args.denoising_graphs and not args.denoising_pairs:
        parser.error("--denoising-graphs requires --denoising-pairs")
    if (args.integrated_modal or args.integrated_only) and (
            args.diagnostic_num_inference_steps is not None or args.diagnostic_cuda_graph is not None):
        parser.error("Model overrides are restricted to native-fixture profiling; use a separate integrated run")
    if args.modal_app:
        with ExitStack() as stack:
            sdk_metrics = stack.enter_context(modal_sdk_measurements()) if args.sdk_profile else None

            def transport_factory(stop=None):
                return make_benchmark_transport(args.modal_app, shutdown_event=stop,
                                                uncached_handles=args.uncached_handles, sdk_metrics=sdk_metrics,
                                                call_mode=args.call_mode)

            def progress(sample):
                count = sample["sequence_id"]
                if count == 0 or count % 10 == 0:
                    print(f"request {count}: RPC {sample['round_trip_s']:.3f}s, "
                          f"server {sample['server_timing']['total_s']:.3f}s", flush=True)

            report = {"measurement": "real Modal through final LeRobot worker; fake hardware only"}
            comparison = None
            quality_comparison = None
            denoising_comparison = None
            if args.encoding_pairs:
                comparison = compare_encodings(transport_factory(), pairs=args.encoding_pairs,
                                               image_hw=(args.height, args.width), jpeg_quality=args.jpeg_quality,
                                               center_crop=args.center_crop)
            if args.quality_pairs:
                quality_comparison = compare_quality_sweep(
                    transport_factory(), pairs=args.quality_pairs, image_hw=(args.height, args.width),
                    center_crop=args.center_crop, max_wall_s=args.max_wall_s)
            if args.denoising_pairs:
                denoising_comparison = compare_denoising(
                    transport_factory(), pairs=args.denoising_pairs, image_hw=(args.height, args.width),
                    center_crop=args.center_crop, max_wall_s=args.max_wall_s, include_graph=args.denoising_graphs)
            if not args.integrated_only:
                report = profile_modal(transport_factory(), warm_samples=args.warm_samples,
                                       max_wall_s=args.max_wall_s, image_hw=(args.height, args.width),
                                       on_sample=progress, image_encoding=args.image_encoding,
                                       jpeg_quality=args.jpeg_quality, center_crop=args.center_crop,
                                       diagnostic_num_inference_steps=args.diagnostic_num_inference_steps,
                                       diagnostic_cuda_graph=args.diagnostic_cuda_graph)
            if comparison is not None:
                report["encoding_comparison"] = comparison
            if quality_comparison is not None:
                report["quality_comparison"] = quality_comparison
            if denoising_comparison is not None:
                report["denoising_comparison"] = denoising_comparison
            if args.integrated_modal or args.integrated_only:
                report["integrated_scenario"] = run_scenario(
                    "real_modal_fake_hardware", [0], duration=args.duration,
                    image_hw=(args.height, args.width), transport_factory=transport_factory,
                    target_warm_samples=args.warm_samples,
                    policy_options={"image_encoding": args.image_encoding, "jpeg_quality": args.jpeg_quality,
                                    "call_mode": args.call_mode, "center_crop": args.center_crop})
            report.update(measured_on=datetime.now(UTC).isoformat(), modal_app=args.modal_app,
                          uncached_handles=args.uncached_handles, sdk_profile=args.sdk_profile,
                          call_mode=args.call_mode,
                          physical_modal_rollout_allowed=False)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.summary_output is not None:
            summary = {key: value for key, value in report.items() if key not in ("samples", "first_request")}
            args.summary_output.parent.mkdir(parents=True, exist_ok=True)
            args.summary_output.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved {args.output}; warm samples {report.get('warm_sample_count', 'integrated only')}", flush=True)
        return
    results = []
    for name, delays in (("healthy_50ms", [0.05]),
                         ("healthy_jitter", [0.05, 0.12, 0.08, 0.20, 0.06, 0.15, 0.09, 0.25])):
        result = run_scenario(name, delays, duration=args.duration, image_hw=(args.height, args.width),
                              target_warm_samples=args.warm_samples)
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
