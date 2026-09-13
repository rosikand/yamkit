"""Gated physical lifecycle; explicit fake mode guards all device construction."""

from __future__ import annotations

import math
import signal
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from yamkit.inference.mapping import YAM_NAMES

from .admission import adapter_build_id, rig_contract, validate_qualification
from .contract import SAVED_IMAGE_MAP
from .executor import OpenPiYamExecutor
from .interface import CONTRACT_ID
from .qualification import decoded_response, request


def make_robot(rig_path, stop):
    # This is reached only after explicit physical confirmation or inside the
    # mandatory fake SDK/CAN/camera guard context below.
    from lerobot_robot_yamkit.config_yam_follower import BiYamFollowerConfig
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    config = BiYamFollowerConfig(rig=str(rig_path), left="left_follower", right="right_follower")
    config._session_shutdown_event = stop
    config._reference_startup = True  # Existing bounded startup home/open only.
    return BiYamFollower(config)


def ordinary_send(robot, target, dispatch_check):
    """Prevalidate both arms; preserve ordinary speed clamps and per-arm Stop."""
    if type(target) is not dict or set(target) != set(YAM_NAMES):
        raise ValueError("OpenPI dispatch needs exactly fourteen named scalar targets")
    dispatch_check()
    targets = {}
    for side in ("left", "right"):
        handle = robot._sides[side]
        local = {name: target[side + "_" + name] for name in handle.features}
        q, gripper = handle.target(local)
        handle.arm.validate_command(q, gripper, limit_speed=True)
        targets[side] = (q, gripper)
    result = {}
    for side in ("left", "right"):
        dispatch_check()
        handle = robot._sides[side]
        q, gripper = targets[side]
        sent = handle.arm.command(q, gripper, limit_speed=True)
        result.update({f"{side}_{name}.pos": float(value) for name, value in zip(handle.names, sent, strict=True)})
    return result


def run_rollout(transport, *, task, duration_s, rig_path: Path, statistics, qualification,
                artifact_dir: Path, confirm_supervised=False, accept_mapping=False,
                fake_hardware=False, fake_observations=None, capture_trace=False,
                upload_repo_id=None, shutdown_event=None):
    if type(fake_hardware) is not bool:
        raise ValueError("Explicit fake mode must be a boolean")
    if fake_hardware:
        if confirm_supervised or accept_mapping or not isinstance(fake_observations, list) or not fake_observations:
            raise ValueError("Explicit fake execution requires saved observations and cannot carry physical approval")
    elif confirm_supervised is not True or accept_mapping is not True or fake_observations is not None:
        raise ValueError("OpenPI physical activation requires fresh supervised confirmation and mapping acceptance")
    if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or not 0 < duration_s <= 60:
        raise ValueError("OpenPI rollout duration must be 1–60 seconds")
    # Reuse inert, already hardened bounded lifecycle/home helpers only; no
    # PI-YAM model, mapping or executor participates in this runner.
    from yamkit.pi05.rollout import _BoundedStop, _home

    from .artifacts import OpenPiCapture, repository_path, rollout_phase

    def phase(value):
        try:
            rollout_phase(value)
        except BaseException as exc:  # noqa: BLE001 — display cannot bypass cleanup, including on interrupt
            return exc
        return None

    metadata = transport.ready()
    validate_qualification(qualification, metadata, task=task, rig_path=rig_path, statistics=statistics)
    binding = rig_contract(rig_path)
    if metadata["session_expires_at"] <= time.time() + duration_s + 60:
        raise ValueError("OpenPI service does not have time for this bounded rollout and cleanup")
    if upload_repo_id is not None:
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(upload_repo_id)
        if upload_repo_id.count("/") != 1:
            raise ValueError("Private upload needs an explicit namespace/dataset")
        capture_trace = True
    destination = repository_path(artifact_dir)
    destination.mkdir(parents=True, exist_ok=False)
    capture = OpenPiCapture(task=task, duration_s=duration_s, capture_trace=capture_trace)
    stop = _BoundedStop(shutdown_event or threading.Event(), transport)
    report = {"controller_mode": CONTRACT_ID, "policy": "pi05-base", "adapter_build_id": adapter_build_id(),
              "instance_id": metadata["instance_id"], "task": task, "duration_s": duration_s,
              "service_identity": qualification["service_identity"], "hardware_tested": False,
              "hardware_activation_attempted": False,
              "fake_hardware": fake_hardware, "released": False, "status": "preparing",
              "home_attempted": False, "home_completed": False, "started_at": time.time()}
    if fake_hardware:
        report["fake_scope"] = ("Paired saved state/RGB at each model request; independent perfect-tracking fake SDK "
                                "for initial/cross-chunk transitions. Not a closed-loop scene or dynamics simulation.")
    robot, engine, failure = None, None, None
    handlers, finished, devices = {}, threading.Event(), None
    stack = ExitStack()

    def note_cleanup_failure(exc, field):
        nonlocal failure
        report[field] = type(exc).__name__
        if failure is None:
            failure = exc
            report["failure_type"] = type(exc).__name__
            report["status"] = "stopped" if stop.is_set() else "failed"

    def watch_stop():
        while not finished.wait(.01):
            if stop.is_set():
                transport.cancel()
                return

    watcher = threading.Thread(target=watch_stop, name="openpi-stop", daemon=True)
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: stop.set())
        watcher.start()
        capture.reserve()  # Complete capture admission/prefault before any device constructor.
        if stop.is_set():
            raise RuntimeError("OpenPI startup stopped before device construction")
        if fake_hardware:
            from yamkit.fake_inference import molmo_fake_devices

            devices = stack.enter_context(molmo_fake_devices(fake_observations))
        report["hardware_activation_attempted"] = report["hardware_tested"] = not fake_hardware
        robot = make_robot(rig_path, stop)
        robot.connect()
        if stop.is_set():
            raise RuntimeError("OpenPI startup stopped before policy execution")

        replay_index = 0

        def observe(*, policy_input=False):
            nonlocal replay_index
            values = robot.get_observation()
            value = {"state": np.array([values[name] for name in YAM_NAMES]),
                     **{name: values[name] for name in SAVED_IMAGE_MAP}}
            if fake_hardware and policy_input:
                # A recorded image cannot respond to a predicted fake pose.
                # Replay its own measured state intact; keep the independent
                # fake SDK state for actual transition planning and receipts.
                # This branch is unreachable during a physical rollout.
                saved = fake_observations[replay_index % len(fake_observations)]
                capture.safely(capture.event, "fake_saved_policy_input", saved_index=replay_index % len(fake_observations),
                               actuator_state=value["state"].tolist(), replay_state=saved["state"].tolist())
                value = {"state": saved["state"].copy(), **{name: saved[name] for name in SAVED_IMAGE_MAP}}
                replay_index += 1
            capture.safely(capture.observation, value, policy_input=policy_input)
            return value

        sequence = 0

        def predict(obs, timeout):
            nonlocal sequence
            response = transport.predict_chunk(request(obs, task), timeout_s=timeout)
            value = {"raw_normalized_chunk": response["raw_normalized_chunk"]}
            try:
                value = decoded_response(response, obs, statistics)
            finally:
                capture.safely(capture.response, value, sequence_id=sequence,
                               observation_index=capture.counts["observation_frames_seen"] - 1)
            sequence += 1
            return value

        engine = OpenPiYamExecutor(predict=predict, observe=observe,
                                  observe_policy_input=lambda: observe(policy_input=True),
                                  send=lambda target, check: ordinary_send(robot, target, check),
                                  validate_target=robot.validate_action_target, stop=stop,
                                  session_check=transport.ensure_session_active,
                                  max_joint_speed=binding["max_joint_speed"], max_gripper_speed=binding["max_gripper_speed"],
                                  event=lambda kind, **data: capture.safely(capture.event, kind, **data))
        report["status"] = "running"
        display_error = phase("running")
        if display_error is not None:
            raise display_error
        capture.start()
        try:
            engine.run(duration_s=duration_s)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            try:
                capture.end()
            except BaseException as exc:  # Preserve the primary control fault on telemetry failure.
                if failure is None:
                    raise
                report["capture_cleanup_failure_type"] = type(exc).__name__
        if not stop.is_set():
            report["home_attempted"] = True
            display_error = phase("returning_home")
            if display_error is not None:
                raise display_error
            _home(robot, stop)
            report["home_completed"] = True
        report["status"] = "stopped" if stop.is_set() else "completed"
    except BaseException as exc:  # noqa: BLE001 — every control failure must pass through release
        failure = exc
        report["status"] = "stopped" if stop.is_set() else "failed"
        report["failure_type"] = type(exc).__name__
    finally:
        # Telemetry is not allowed to sit outside the release invariant. In
        # particular a low-memory event/metrics allocation must still release.
        try:
            capture.end()
        except BaseException as exc:  # noqa: BLE001 — best effort telemetry, unconditional release below
            note_cleanup_failure(exc, "capture_cleanup_failure_type")
        try:
            report["execution"] = engine.metrics() if engine is not None else {}
        except BaseException as exc:  # noqa: BLE001 — failed metrics cannot strand either arm
            report["execution"] = {}
            note_cleanup_failure(exc, "metrics_failure_type")
        display_error = phase("releasing")
        if display_error is not None:
            note_cleanup_failure(display_error, "display_failure_type")
        try:
            if robot is not None:
                robot.disconnect(home=False)
            report["released"] = True
            if devices is not None and not all(device.closed for device in devices):
                report["released"] = False
                report["release_failure_type"] = "FakeSdkNotReleased"
        except BaseException as exc:  # noqa: BLE001 — retain release failure without suppressing the control fault
            report["release_failure_type"] = type(exc).__name__
            report["released"] = False
            failure = failure or exc
        finally:
            # Even a failing fake/device-context exit must not strand the Stop
            # watcher, retain our signal handlers or hide the primary fault.
            try:
                stack.close()
            except BaseException as exc:  # noqa: BLE001 — restore all remaining lifecycle resources even on interrupt
                report.setdefault("release_failure_type", type(exc).__name__)
                report["context_cleanup_failure_type"] = type(exc).__name__
                report["released"] = False
                failure = failure or exc
            finished.set()
            try:
                if watcher.ident is not None:
                    watcher.join(timeout=1)
            except BaseException as exc:  # noqa: BLE001 — signal restoration must still run
                report["watcher_cleanup_failure_type"] = type(exc).__name__
                report["released"] = False
                failure = failure or exc
            for signum, handler in handlers.items():
                try:
                    signal.signal(signum, handler)
                except BaseException as exc:  # noqa: BLE001 — attempt every saved signal handler independently
                    report["signal_cleanup_failure_type"] = type(exc).__name__
                    report["released"] = False
                    failure = failure or exc
        if report["released"]:
            display_error = phase("released")
            if display_error is not None:
                note_cleanup_failure(display_error, "display_failure_type")
        report["completed_at"] = time.time()
    summary = capture.finalize(destination, report, upload_repo_id=upload_repo_id,
                               metadata={"qualification_path_is_host_bound": True,
                                         "fake_hardware": fake_hardware, "physical_task_success": None})
    result = {**report, "capture": summary, "artifact_directory": str(destination),
              "exit_status": summary.get("exit_status", 1 if failure else 0)}
    if failure is not None and not isinstance(failure, (Exception, KeyboardInterrupt)):
        raise failure
    return result
