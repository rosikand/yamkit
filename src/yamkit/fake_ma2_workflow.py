"""MA2 recorded CLI runs and explicit saved-observation software replay.

``run_fake_molmoact2`` has no physical fallback: its SDK/camera constructors are
replaced inside ``molmo_fake_devices`` before the normal runner is invoked.
The separate ``run_recorded_molmoact2`` physical entry requires explicit fresh
confirmation. Both retain the frozen production qualification, service identity,
reference runner and recorder. Fake receipts never authorize real hardware.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import signal
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from .backend_workflow import WorkflowError, _local_path, load_target
from .deployment import InferenceOptions
from .fake_inference import molmo_fake_devices, saved_observations
from .inference_workflow import require_prepared_current
from .paths import ROOT
from .rollout_artifacts import ARTIFACTS, _copy, _json, _safe_path, package_rollout, sanitize, upload_rollout


@contextmanager
def _fake_process_state():
    """Restore caller-owned signals/logging after an embedded software replay.

    The unchanged LeRobot runner installs process signal handlers and the trace
    CLI configures root logging. Keep those exact behaviors during replay and
    cleanup, but do not leave a consumed Stop handler or temporary stderr handler
    in the caller after fake devices have been released. Never wraps real runs.
    """
    if threading.current_thread() is not threading.main_thread():
        raise WorkflowError("Fake-device guards require their own CLI process, never a UI worker thread")
    signals = {getattr(signal, name): signal.getsignal(getattr(signal, name))
               for name in ("SIGINT", "SIGTERM", "SIGHUP", "SIGQUIT") if hasattr(signal, name)}
    root = logging.getLogger()
    handlers, level, filters = list(root.handlers), root.level, list(root.filters)
    handler_state = [(handler, handler.level, list(handler.filters), handler.formatter) for handler in handlers]
    # These are the only named levels set by the normal CLI / LeRobot logging setup.
    loggers = [(logging.getLogger(name), logging.getLogger(name).level)
               for name in ("i2rt", "can", "urllib3", "httpx")]
    try:
        yield
    finally:
        for signum, previous in signals.items():
            signal.signal(signum, previous)
        temporary = [handler for handler in root.handlers if handler not in handlers]
        root.handlers[:] = handlers
        root.filters[:] = filters
        root.setLevel(level)
        for handler, previous_level, previous_filters, formatter in handler_state:
            handler.setLevel(previous_level)
            handler.filters[:] = previous_filters
            handler.setFormatter(formatter)
        for logger, previous_level in loggers:
            logger.setLevel(previous_level)
        for handler in temporary:
            handler.close()


def _tools():
    spec = importlib.util.spec_from_file_location("_yamkit_fake_ma2_trace", Path(ROOT) / "scripts/trace_rollout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(path, value):
    _safe_path(path).write_text(json.dumps(sanitize(value), indent=2, allow_nan=False) + "\n")


def _trace_arguments(selection, directory):
    return SimpleNamespace(output_dir=directory, rig=Path(selection.rig_path),
                           duration=int(selection.duration), controller_mode="reference",
                           task=selection.task, backend="external",
                           external_service=selection.external_service, modal_app=None)


def _publish_trace(source, destination, *, skip=()):
    """Same bounded artifact import as UI finalization; original frame tree stays intact."""
    for name in ARTIFACTS:
        if name in ("meta.json", "run_metadata.json", *skip):
            continue
        path = _safe_path(source / name)
        if path.is_file() and path.stat().st_size <= 256 * 1024 * 1024:
            _copy(path, _safe_path(destination / name))


def _read_capture(directory):
    try:
        summary, metrics = _json(directory / "summary.json"), _json(directory / "metrics.json")
        if not isinstance(summary, dict) or not isinstance(metrics, dict):
            raise TypeError("Recording metadata must be objects")
        return summary, metrics, None
    except (OSError, ValueError, TypeError) as exc:
        return {}, {}, type(exc).__name__


def _execution(metrics):
    value = metrics.get("reference_execution")
    return value if isinstance(value, dict) else {}


def _postprocess_failure(result, stage, exc):
    result.update(status="failed", exit_status=1, pipeline_complete=False,
                  postprocess_error=stage, postprocess_error_type=type(exc).__name__)


def _finalize_history(directory, selection, started, result):
    result["pipeline_complete"] = result["exit_status"] == 0
    metadata = {"id": directory.name, "kind": "rollout", "policy": "molmoact2", "task": selection.task,
                "started_at": started, "ended_at": time.time(), "returncode": result["exit_status"],
                "status": "success" if result["exit_status"] == 0 else "failed", "active": False,
                "task_success": None, "log_complete": False,
                "execution_status": result.get("execution_status"),
                "postprocess_error": result.get("postprocess_error")}
    if result.get("fake_hardware") is True:
        metadata.update(hardware_tested=False, motion_approval_received=False, fake_hardware=True)
    _write(directory / "meta.json", metadata)
    _write(directory / "report.json", result)


def _configuration(selection):
    """Identical production CLI configuration; call only inside fake-I/O scope."""
    from lerobot.rollout.configs import RolloutConfig
    from lerobot_robot_yamkit import BiYamFollowerConfig

    from .remote_policy import YamkitRemoteConfig

    # The unchanged runner checks these flags even with explicit fake devices.
    # This is not an approval record: the helper only returns false approval.
    selection = replace(selection, supervised_confirmed=True, mapping_accepted=True)
    selection.validate(motion=True)
    return RolloutConfig(
        robot=BiYamFollowerConfig(rig=selection.rig_path, left="left_follower", right="right_follower", id="yam"),
        policy=YamkitRemoteConfig(profile="molmoact2", backend="external", modal_app="",
                                 external_service=selection.external_service, center_crop=False,
                                 image_encoding="rgb8", jpeg_quality=85, call_mode="http",
                                 execution_mode="cuda_graph10", controller_mode="reference", task=selection.task,
                                 prediction_queue_threshold=None, supervised_confirmed=True, mapping_accepted=True),
        task=selection.task, duration=selection.duration, fps=30, device="cpu", play_sounds=False,
        use_torch_compile=False, return_to_initial_position=False,
    )


@_fake_process_state()
def run_fake_molmoact2(selection, *, artifact_dir, capture_trace=False, upload_repo_id=None,
                      backend_config=None):
    """Run exactly one bounded software replay, never a physical robot operation.

    With RGB capture, originals stay in a new ``.context/rollout-traces/<id>``
    directory and bounded playback artifacts are copied into ``artifact_dir``.
    The normal trace ``execute`` and export are reused unchanged. Without RGB
    capture, the exact normal ``run_remote_rollout`` returns full metrics.
    Exceptions are recorded by type only; local originals survive every failure.
    """
    if (not isinstance(selection, InferenceOptions) or selection.policy != "molmoact2"
            or selection.backend != "external" or selection.controller_mode != "reference"
            or selection.call_mode != "http" or selection.execution_mode != "cuda_graph10"
            or selection.image_encoding != "rgb8" or selection.jpeg_quality != 85
            or selection.center_crop or selection.rtc or selection.prediction_queue_threshold is not None
            or selection.fps != 30 or selection.arms not in ((), ("left_follower", "right_follower"))
            or selection.supervised_confirmed or selection.mapping_accepted
            or type(selection.duration) not in (int, float) or not math.isfinite(selection.duration)
            or not 0 < selection.duration <= 90 or type(capture_trace) is not bool):
        raise WorkflowError("--fake-hardware requires the exact unapproved MolmoAct2 reference selection and 0–90 seconds")
    selection.validate()
    require_prepared_current(selection)  # No stale or foreign-host qualification bypass.
    target = load_target("lambda", "molmoact2", config=backend_config)
    if target.service != selection.external_service:
        raise WorkflowError("Configured fake-input backend differs from the prepared MolmoAct2 service")
    observations = saved_observations(target)
    destination = _local_path(str(artifact_dir), root=ROOT)
    if destination.exists():
        raise WorkflowError("Fake execution requires a new repository-local artifact directory")
    if upload_repo_id is not None:
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(upload_repo_id)
        if upload_repo_id.count("/") != 1:
            raise WorkflowError("An explicit private namespace/dataset destination is required")
        capture_trace = True
    tools = _tools()
    trace_directory = None
    if capture_trace:
        if int(selection.duration) != selection.duration:
            raise WorkflowError("The unchanged MA2 RGB recorder requires an integral number of seconds")
        trace_directory = _local_path(str(Path(ROOT) / ".context/rollout-traces" / uuid.uuid4().hex), root=ROOT)
        tools.artifact_directory(ROOT, trace_directory, "unused")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.time()
    metrics, summary, error_type, code = {}, {}, None, 0
    devices = []
    try:
        with molmo_fake_devices(observations) as devices:
            if capture_trace:
                code = tools.execute(_trace_arguments(selection, trace_directory))
            else:
                from .remote_rollout import run_remote_rollout

                with tools.wall_limit(tools.MAX_ROLLOUT_WALL_S):
                    metrics = run_remote_rollout(_configuration(selection))
    except BaseException as exc:  # noqa: BLE001 — save sanitized software-failure evidence; never retry
        code, error_type = 1, type(exc).__name__
        attached = getattr(exc, "metrics", None)
        if isinstance(attached, dict):
            metrics = attached
    if not isinstance(metrics, dict):
        metrics, code, error_type = {}, 1, "TypeError"
    released = all(device.closed for device in devices)
    if capture_trace:
        summary, captured_metrics, read_error = _read_capture(trace_directory)
        if read_error is None:
            metrics = captured_metrics
        else:
            code, error_type = 1, error_type or read_error
        released = released and summary.get("resources_released") is True
        if summary.get("status") != "TRACE_SAVED":
            code = 1
    if (not released or metrics.get("failed") or len(devices) != 2
            or not (_execution(metrics).get("interpolation_dispatches", 0)
                    or _execution(metrics).get("completed_steps", 0))):
        code = 1
    result = {"status": "completed" if code == 0 else "failed", "exit_status": code,
              "execution_status": "completed" if code == 0 else "failed",
              "hardware_tested": False, "motion_approval_received": False, "physical_task_success": None,
              "fake_hardware": True, "model": "molmoact2", "controller_mode": "reference",
              "task": selection.task, "duration_s": selection.duration, "resources_released": released,
              "error_type": error_type, "artifact_directory": str(destination), "capture_trace": capture_trace,
              "trace_directory": str(trace_directory) if trace_directory is not None else None,
              "fake_sdk_instances": len(devices), "fake_sdk_commands_including_startup_home":
                  sum(len(device.commands) for device in devices),
              "execution": _execution(metrics),
              "saved_observation_count": len(observations),
              "input_scope": "Saved real RGB replay with perfect fake command tracking; not dynamics or scene evolution",
              "recording_overhead": "Unchanged production trace observer copies, included in control-loop time"
                                    if capture_trace else "No RGB recording instrumentation",
              "qualification_reused": "Current exact host qualification checked; this replay does not promote new proof"}
    if not destination.is_dir():
        destination.mkdir(parents=True, exist_ok=False)
    summary.update(synthetic_fixture=True, hardware_tested=False, fake_hardware=True,
                   motion_approval_received=False, task_success=None, resources_released=released,
                   input_scope=result["input_scope"], recording_overhead=result["recording_overhead"])
    if trace_directory is not None and trace_directory.is_dir():
        try:
            _write(trace_directory / "summary.json", summary)
            _write(trace_directory / "metrics.json", {**metrics, "fake_execution": result})
            if (trace_directory / "plan.json").is_file():
                plan = _json(trace_directory / "plan.json")
                plan.update(hardware_opened=False, fake_hardware=True, motion_approval_received=False,
                            run_effects="Software-only replay under explicit fake SDK/camera guards; no hardware operation")
                _write(trace_directory / "plan.json", plan)
            if released and (trace_directory / "trace.json").is_file():
                try:
                    tools.render_report(trace_directory)  # Existing renderer's explicit synthetic-fixture banner.
                except Exception as exc:  # noqa: BLE001 — retained originals survive offline display failure
                    _postprocess_failure(result, "report_render_failed", exc)
                    summary.update(status="TRACE_SAVED_WITH_EXPORT_ERRORS", render_error_type=type(exc).__name__)
                    _write(trace_directory / "summary.json", summary)
            _publish_trace(trace_directory, destination,
                           skip=("report.html",) if result.get("postprocess_error") == "report_render_failed" else ())
        except Exception as exc:  # noqa: BLE001 — preserve a finalized failure entry if artifact import fails
            _postprocess_failure(result, "artifact_import_failed", exc)
    _write(destination / "metrics.json", {**metrics, "fake_execution": result})
    _write(destination / "summary.json", summary)
    _finalize_history(destination, selection, started, result)
    _write(destination / "run_metadata.json", {
        "model": {"policy": "molmoact2"}, "capture": {"requested": capture_trace, "hardware_tested": False},
        "software": {"fake_hardware": True, "motion_approval_received": False},
        "original_paths": {"trace_dir": str(trace_directory) if trace_directory is not None else None},
        "known_missing_data": [result["input_scope"], "No physical operation or success is established."]})
    if upload_repo_id is not None and result["exit_status"] == 0 and released:
        try:
            uploaded = upload_rollout(package_rollout(destination, trace_dir=trace_directory), repo_id=upload_repo_id)
            result["upload"] = {key: uploaded.get(key) for key in ("status", "repo_id", "revision")}
            if uploaded.get("status") not in ("uploaded", "already_uploaded"):
                _postprocess_failure(result, "upload_failed", ValueError())
        except Exception as exc:  # noqa: BLE001 — no arbitrary credential-bearing service diagnostic
            result["upload"] = {"status": "failed", "repo_id": upload_repo_id, "error_type": type(exc).__name__}
            _postprocess_failure(result, "upload_failed", exc)
    elif upload_repo_id is not None:
        result["upload"] = {"status": "not_attempted", "repo_id": upload_repo_id,
                            "reason": "Execution or recording did not complete with confirmed release"}
    _finalize_history(destination, selection, started, result)
    _write(destination / "fake-result.json", result)
    return sanitize(result)


def run_recorded_molmoact2(selection, *, confirm_supervised=False, accept_mapping=False, upload_repo_id=None):
    """Future explicitly approved physical CLI capture through the existing recorder.

    Not used by software qualification. The normal caller holds the lifetime
    workflow lock, verifies idle hardware ownership and obtains fresh on-site
    approval. This entry additionally rejects missing confirmation before any
    run or device import, and rechecks current exact qualification itself.
    """
    if confirm_supervised is not True or accept_mapping is not True:
        raise WorkflowError("Recorded physical MolmoAct2 requires fresh supervised confirmation and mapping acceptance")
    if (not isinstance(selection, InferenceOptions) or selection.policy != "molmoact2"
            or selection.backend != "external" or selection.controller_mode != "reference"
            or type(selection.duration) not in (int, float) or not math.isfinite(selection.duration)
            or not 0 < selection.duration <= 90 or int(selection.duration) != selection.duration):
        raise WorkflowError("The existing physical recorder requires MolmoAct2 reference and 1–90 integral seconds")
    selection.validate()
    require_prepared_current(selection)
    replace(selection, supervised_confirmed=True, mapping_accepted=True).validate(motion=True)
    if upload_repo_id is not None:
        from huggingface_hub.utils import validate_repo_id

        validate_repo_id(upload_repo_id)
        if upload_repo_id.count("/") != 1:
            raise WorkflowError("An explicit private namespace/dataset destination is required")
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-rollout-" + uuid.uuid4().hex[:8]
    destination = _local_path(str(Path(ROOT) / "outputs/ui/deployments" / run_id), root=ROOT)
    trace_directory = _local_path(str(Path(ROOT) / ".context/rollout-traces" / uuid.uuid4().hex), root=ROOT)
    tools = _tools()
    tools.artifact_directory(ROOT, trace_directory, "unused")
    destination.mkdir(parents=True, exist_ok=False)
    started = time.time()
    code, error_type = 0, None
    try:
        code = tools.execute(_trace_arguments(selection, trace_directory))
    except BaseException as exc:  # noqa: BLE001 — normal runner owns cleanup; retain failure, never retry
        code, error_type = 1, type(exc).__name__
    summary, metrics, read_error = _read_capture(trace_directory)
    if read_error is not None:
        code, error_type = 1, error_type or read_error
    released = summary.get("resources_released") is True
    code = code or int(not released or summary.get("status") != "TRACE_SAVED")
    result = {"status": "completed" if code == 0 else "failed", "exit_status": code,
              "execution_status": "completed" if code == 0 else "failed", "error_type": error_type,
              "artifact_directory": str(destination), "trace_directory": str(trace_directory),
              "resources_released": released, "task_success": None, "controller_mode": "reference",
              "execution": _execution(metrics)}
    try:
        _publish_trace(trace_directory, destination)
    except Exception as exc:  # noqa: BLE001 — a failed copy never hides the completed child lifecycle
        _postprocess_failure(result, "artifact_import_failed", exc)
    _write(destination / "summary.json", summary)
    _write(destination / "run_metadata.json", {
        "model": {"policy": "molmoact2"}, "capture": {"requested": True},
        "original_paths": {"trace_dir": str(trace_directory)}})
    _finalize_history(destination, selection, started, result)
    if upload_repo_id is not None and result["exit_status"] == 0 and released:
        try:
            uploaded = upload_rollout(package_rollout(destination, trace_dir=trace_directory), repo_id=upload_repo_id)
            result["upload"] = {key: uploaded.get(key) for key in ("status", "repo_id", "revision")}
            if uploaded.get("status") not in ("uploaded", "already_uploaded"):
                _postprocess_failure(result, "upload_failed", ValueError())
        except Exception as exc:  # noqa: BLE001 — retained originals, no automatic retry of any kind
            result["upload"] = {"status": "failed", "repo_id": upload_repo_id, "error_type": type(exc).__name__}
            _postprocess_failure(result, "upload_failed", exc)
    elif upload_repo_id is not None:
        result["upload"] = {"status": "not_attempted", "repo_id": upload_repo_id,
                            "reason": "Execution or recording did not complete with confirmed release"}
    _finalize_history(destination, selection, started, result)
    return sanitize(result)
