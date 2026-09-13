"""Explicitly supervised physical entrypoint for the independent PI05 executor.

Importing this module is inert. No robot module is imported until all software
admission and explicit motion-confirmation checks pass inside run_rollout.
"""

from __future__ import annotations

import json
import math
import signal
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from yamkit.inference.mapping import YAM_NAMES

from .admission import validate_qualification
from .contract import CONTRACT_ID, PROFILE, build_id
from .executor import Pi05ReferenceExecutor
from .qualification import make_request


class _BoundedStop:
    """One stop flag covers startup, inference, row dispatch and return home."""

    def __init__(self, event, transport, wall_s=150.0):
        self.event, self.transport = event, transport
        self.deadline = time.monotonic() + wall_s
        self.home_deadline = float("inf")

    def is_set(self):
        if self.event.is_set():
            return True
        if time.monotonic() >= min(self.deadline, self.home_deadline):
            self.event.set()
        try:
            self.transport.ensure_session_active()
        except Exception:  # noqa: BLE001 — a failed session can never continue motion
            self.event.set()
        return self.event.is_set()

    def set(self):
        self.event.set()

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.is_set() and time.monotonic() < deadline:
            self.event.wait(min(0.01, max(0, deadline - time.monotonic())))
        return self.is_set()


def _make_robot(rig_path, stop):
    from lerobot_robot_yamkit.config_yam_follower import BiYamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    config = BiYamFollowerConfig(rig=str(rig_path), left="left_follower", right="right_follower")
    config._session_shutdown_event = stop
    # This existing plugin flag controls configured startup home/open only;
    # the PI engine never consumes Molmo's cached-command state or interpolation.
    config._reference_startup = True
    return BiYamFollower(config)


def _home(robot, stop):
    from yamkit.arm import go_home_all

    stop.home_deadline = min(stop.deadline, time.monotonic() + 30.0)
    jobs = [handle.home_job for handle in robot._sides.values() if handle.home_job is not None]
    # Native YamArm.go_home(gripper=None) preserves each measured final opening.
    if jobs:
        go_home_all(jobs, stop=stop)
    if stop.is_set():
        raise RuntimeError("π0.5 return home stopped or expired; release without retry")


def run_rollout(transport, *, task: str, duration_s: float, rig_path: Path, qualification: dict,
                accept_mapping: bool = False, confirm_supervised: bool = False,
                artifact_dir: Path, shutdown_event=None, robot_factory=None, home=None) -> dict:
    """One explicit physical run. Call only after fresh operator approval.

    Internal factory/home seams exist solely for fake-hardware tests. Production
    uses the existing plugin's cooperative ownership, measured-state/target
    bounds, two-arm prevalidation, interruptible home and release paths.
    """
    from yamkit.paths import ROOT

    if accept_mapping is not True or confirm_supervised is not True:
        raise ValueError("π0.5 physical rollout requires mapping acceptance and fresh supervised confirmation")
    if (type(duration_s) not in (float, int) or not math.isfinite(duration_s) or not 0 < duration_s <= 90):
        raise ValueError("π0.5 rollout duration must be positive and at most 90 seconds")
    metadata = transport.ready(15.0)
    validate_qualification(qualification, metadata, task=task, rig_path=rig_path)
    destination = artifact_dir.resolve()
    destination.relative_to(Path(ROOT).resolve())
    destination.mkdir(parents=True, exist_ok=False)
    event = shutdown_event or threading.Event()
    stop = _BoundedStop(event, transport)
    if stop.is_set():
        raise ValueError("π0.5 rollout was stopped before hardware activation")
    session_id, sequence, trace = str(uuid.uuid4()), 0, []
    robot, engine, failure = None, None, None
    handlers, watch_done = {}, threading.Event()
    report = {"controller_mode": CONTRACT_ID, "model_revision": PROFILE.revision,
              "pi05_build_id": build_id(), "instance_id": metadata["instance_id"],
              "task": task, "duration_s": duration_s, "session_id": session_id,
              "hardware_tested": robot_factory is None, "released": False,
              "home_attempted": False, "home_completed": False, "status": "preparing"}

    def cancel_on_stop():
        while not watch_done.wait(0.01):
            if stop.is_set():
                transport.cancel()
                return

    watcher = threading.Thread(target=cancel_on_stop, daemon=True, name="pi05-local-stop")

    def predict(observation, timeout):
        nonlocal sequence
        request = make_request(observation, task=task, session_id=session_id,
                               sequence_id=sequence, timeout_s=timeout)
        sequence += 1
        return transport.predict_chunk(request, timeout)["chunk"]

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: event.set())
        watcher.start()
        if stop.is_set():
            raise RuntimeError("π0.5 startup was stopped")
        robot = (robot_factory or _make_robot)(rig_path, stop)
        robot.connect()
        if stop.is_set():
            raise RuntimeError("π0.5 startup was stopped before inference")

        def observe():
            values = robot.get_observation()
            return {"state": np.array([values[name] for name in YAM_NAMES]),
                    **{name: values[name] for name in PROFILE.image_keys}}

        def send(target, check):
            return robot.send_reference_action(target, dispatch_check=check)

        report["status"] = "running"
        engine = Pi05ReferenceExecutor(predict=predict, observe=observe, send=send,
                                      validate_target=robot.validate_action_target, stop=stop,
                                      session_check=transport.ensure_session_active,
                                      event=lambda kind, **record: trace.append({"kind": kind, **record}))
        engine.run(duration_s=duration_s)
        if not stop.is_set():
            report["home_attempted"] = True
            (home or _home)(robot, stop)
            report["home_completed"] = True
        report["status"] = "stopped" if stop.is_set() else "completed"
    except BaseException as exc:  # noqa: BLE001 — release hardware even on interrupt, then re-raise
        failure = exc
        report.update(status="failed", error_type=type(exc).__name__)
        stop.set()
    finally:
        # Release before any artifact serialization or upload; no home on fault.
        try:
            if robot is not None:
                robot.disconnect(home=False)
            report["released"] = True
        except BaseException as exc:  # noqa: BLE001 — preserve release failures and stop every resource
            report.update(status="release_failed", release_error_type=type(exc).__name__)
            failure = failure or exc
        finally:
            watch_done.set()
            if watcher.ident is not None:
                watcher.join(timeout=0.2)
            transport.close()
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        if engine is not None:
            report["execution"] = engine.metrics()
        report["completed_at"] = time.time()
        (destination / "trace.json").write_text(json.dumps({"events": trace}, indent=2, allow_nan=False) + "\n")
        (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if failure is not None:
        raise RuntimeError(f"π0.5 rollout {report['status']}; inspect its retained report") from failure
    return report
