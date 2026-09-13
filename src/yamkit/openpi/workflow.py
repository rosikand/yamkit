"""One-command preparation and explicit supervised execution of official π0.5."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from yamkit.backend_workflow import (
    WorkflowError,
    _local_path,
    assert_ui_idle,
    connect_configured_runtime,
    load_target,
)
from yamkit.external_ops import _read_private, _save
from yamkit.inference.client import RemoteFault
from yamkit.paths import ROOT

from .admission import load_statistics, rig_contract, validate_qualification
from .interface import CONTRACT_ID
from .transport import OpenPiTransport

MAX_NATURAL_EXPIRY_WAIT_S = 120.0
NATURAL_EXPIRY_CLEANUP_GRACE_S = 1.0


@dataclass(frozen=True)
class OpenPiSelection:
    task: str
    rig_path: str
    duration: float
    external_service: str
    config: Path | None
    qualification_path: Path
    policy: str = "pi05-base"
    backend: str = "lambda"
    controller_mode: str = CONTRACT_ID


def make_transport(target, statistics, *, shutdown_event=None):
    if target.policy != "pi05-base" or target.service != "lambda-openpi" or target.token_file is None:
        raise WorkflowError("Configure the separate lambda-openpi service and its private token file")
    return OpenPiTransport(endpoint_url=target.endpoint, token=_read_private(target.token_file, 257).strip(),
                           statistics_sha256=statistics.metadata()["candidate_statistics_sha256"],
                           shutdown_event=shutdown_event)


def readiness(target, statistics):
    transport = make_transport(target, statistics)
    try:
        return transport.ready(timeout_s=5)
    finally:
        transport.close()


def _renew_near_expiry(target, statistics, metadata, *, task, duration, progress):
    """Wait for one natural software expiry, never stop or extend a live process.

    The bounded wait uses a monotonic deadline, remains interruptible and checks
    UI ownership throughout. Only prepare calls this helper, before motion
    confirmation. An expired prepared physical command is never retried here.
    """
    expiry = metadata.get("session_expires_at")
    if type(expiry) not in (int, float) or not math.isfinite(expiry):
        raise WorkflowError("Official OpenPI service must have a finite authenticated expiry")
    remaining = expiry - time.time()
    if remaining > duration + 60:
        return metadata
    if remaining > MAX_NATURAL_EXPIRY_WAIT_S or not target.ssh or not target.remote:
        raise WorkflowError("Near-expiry official OpenPI needs configured existing-service startup for automatic renewal")
    assert_ui_idle()
    # The finite supervisor needs a brief opportunity to retire its owned child
    # and release its startup lock after expiry. Never extend the 120-second cap.
    wait_s = min(MAX_NATURAL_EXPIRY_WAIT_S, max(0.0, remaining) + NATURAL_EXPIRY_CLEANUP_GRACE_S)
    deadline = time.monotonic() + wait_s
    progress("Waiting for the near-expiry official OpenPI session to end naturally; no process is stopped")
    while time.monotonic() < deadline:
        assert_ui_idle()
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    assert_ui_idle()
    progress("Reconnecting official OpenPI after natural session expiry; software preparation only")
    renewed = connect_configured_runtime(target, task, lambda: readiness(target, statistics), progress=progress)
    renewed_expiry = renewed.get("session_expires_at")
    if (type(renewed_expiry) not in (int, float) or not math.isfinite(renewed_expiry)
            or renewed_expiry <= time.time() + duration + 60
            or renewed.get("instance_id") == metadata.get("instance_id")):
        raise WorkflowError("Official OpenPI renewal did not produce a fresh bounded session; no repeated restart attempted")
    return renewed


def prepare(*, backend, task, rig, duration=60, arms=(), config=None, progress=lambda _value: None, force=False):
    from .qualification import collect, load_saved

    if (backend != "lambda" or type(task) is not str or not task.strip() or len(task) > 512
            or type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 60
            or tuple(arms) not in ((), ("left_follower", "right_follower"))):
        raise WorkflowError("Official OpenPI YAM requires Lambda, an exact task, both followers and 1–60 seconds")
    rig = _local_path(str(rig))
    rig_contract(rig)
    statistics = load_statistics()
    target = load_target(backend, "pi05-base", config=config)
    if not target.saved_observations:
        raise WorkflowError("Configure existing saved observations for automatic hardware-free OpenPI qualification")
    metadata = connect_configured_runtime(target, task, lambda: readiness(target, statistics), progress=progress)
    original_instance = metadata["instance_id"]
    metadata = _renew_near_expiry(target, statistics, metadata, task=task, duration=duration, progress=progress)
    renewal_reasons = ["near_expiry_before_qualification"] if metadata["instance_id"] != original_instance else []
    directory = _local_path(str(ROOT / "data/inference/openpi" / target.service))
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    pointer = directory / "qualification.json"
    report, path = None, None
    if not force and pointer.is_file():
        try:
            from yamkit.external_ops import _read_json

            path = _local_path(_read_json(pointer)["path"])
            if not 0 < path.stat().st_size < 32 * 1024 * 1024:
                raise ValueError("Invalid qualification size")
            report = json.loads(path.read_text())
            validate_qualification(report, metadata, task=task, rig_path=rig, statistics=statistics)
        except (ValueError, OSError, KeyError, TypeError):
            report = None

    def renew(reason):
        nonlocal metadata
        if renewal_reasons:
            raise WorkflowError("Official OpenPI preparation exhausted its one software renewal; no repeated restart attempted")
        # Keep every old qualification (including failed native bounds evidence)
        # untouched. A separate record explains this software-only renewal.
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        record = {"reason": reason, "previous_instance_id": metadata["instance_id"],
                  "previous_expires_at": metadata["session_expires_at"], "checked_at": time.time(),
                  "hardware_activation": False, "automatic_physical_retry": False, "renewed": False}
        _save(path.parent / "preparation-renewal.json", record)
        metadata = _renew_near_expiry(target, statistics, metadata, task=task, duration=duration, progress=progress)
        if metadata["instance_id"] == record["previous_instance_id"]:
            raise WorkflowError("Official OpenPI renewal did not produce a fresh session; no repeated restart attempted")
        renewal_reasons.append(reason)
        record.update(renewed=True, instance_id=metadata["instance_id"], expires_at=metadata["session_expires_at"])
        _save(path.parent / "preparation-renewal.json", record)

    # No recursion, and at most one total renewal, including the initial check.
    for _attempt in range(2):
        reused = report is not None
        if report is None:
            observations = [load_saved(saved) for saved in target.saved_observations]
            evidence = ROOT / ".context/openpi-qualification" / uuid.uuid4().hex
            path = evidence / "qualification.json"
            progress("Warming official pi05_base and qualifying saved inputs with explicit fake arms")
            transport = make_transport(target, statistics)
            failure = None
            try:
                report = collect(transport, statistics=statistics, observations=observations, task=task,
                                 rig_path=rig, directory=evidence, progress=progress)
            except Exception as exc:  # noqa: BLE001 — never persist private transport exception text
                failure = (type(exc).__name__, isinstance(exc, (RemoteFault, TimeoutError)))
            finally:
                transport.close()
            if failure is not None:
                evidence.mkdir(mode=0o700, parents=True, exist_ok=True)
                _save(evidence / "preparation-failure.json", {"reason": "qualification_" + failure[0],
                                                           "hardware_tested": False})
                if failure[1] and metadata["session_expires_at"] <= time.time():
                    renew("authenticated_session_expired_during_qualification")
                    continue
                raise WorkflowError(f"Official OpenPI qualification failed ({failure[0]}); inspect {evidence}") from None
        if not isinstance(report, dict) or report.get("qualified") is not True or report.get("reasons") != []:
            # Expiry is not a license to retry arbitrary native output, mapping,
            # bounds or execution faults. Only a sole transport-expiry failure
            # and an actually elapsed authenticated expiry permit renewal.
            reasons = report.get("reasons") if isinstance(report, dict) else None
            if (reasons in (["qualification_RemoteFault"], ["qualification_InvalidatedRequest"],
                            ["qualification_TimeoutError"])
                    and metadata["session_expires_at"] <= time.time()):
                renew("authenticated_session_expired_during_qualification")
                report = None
                continue
            raise WorkflowError("Official OpenPI software qualification failed; retained evidence: " + str(path))
        if metadata["session_expires_at"] <= time.time():
            renew("authenticated_session_expired_after_qualification")
            report = None
            continue
        # The Stop probe's cancelled transport is never reused. Refresh the
        # authenticated identity even when the qualification itself was cached.
        try:
            fresh = readiness(target, statistics)
        except Exception as exc:  # noqa: BLE001 — sanitize readiness errors as well
            if isinstance(exc, (RemoteFault, TimeoutError)) and metadata["session_expires_at"] <= time.time():
                renew("authenticated_session_expired_during_readiness")
                report = None
                continue
            raise WorkflowError(f"Official OpenPI final readiness failed ({type(exc).__name__}); retained evidence: {path}") from None
        try:
            validate_qualification(report, fresh, task=task, rig_path=rig, statistics=statistics)
        except ValueError:
            raise WorkflowError("Official OpenPI software qualification failed; retained evidence: " + str(path)) from None
        metadata = fresh
        if metadata["session_expires_at"] <= time.time() + duration + 60:
            renew("execution_margin_consumed_by_qualification")
            report = None
            continue
        if not reused:
            _save(pointer, {"path": str(path)})
        break
    else:
        raise WorkflowError("Official OpenPI preparation did not retain its execution margin; no repeated restart attempted")
    selection = OpenPiSelection(task, str(rig), duration, target.service, config, path)
    return selection, {"ready": True, "reused": reused, "hardware_tested": False,
                       "motion_approval_received": False, "policy": "pi05-base", "backend": "lambda",
                       "controller_mode": CONTRACT_ID, "service": target.service,
                       "label": "official frozen pi05_base + documented experimental YAM adapter",
                       "expires_at": metadata["session_expires_at"], "instance_id": metadata["instance_id"],
                       "software_session_renewals": len(renewal_reasons), "software_renewal_reasons": renewal_reasons,
                       "evidence_directory": str(path.parent),
                       "direct_warm_round_trip_s": report["direct_warm_round_trip_s"]}


def run_prepared(selection, *, confirm_supervised=False, accept_mapping=False,
                 artifact_dir=None, capture_trace=False, upload_repo_id=None, fake_hardware=False):
    from .rollout import run_rollout

    if type(fake_hardware) is not bool:
        raise WorkflowError("OpenPI fake_hardware must be an explicit boolean")
    if fake_hardware and (confirm_supervised or accept_mapping):
        raise WorkflowError("Explicit fake OpenPI execution cannot carry physical approval")
    if not fake_hardware and (confirm_supervised is not True or accept_mapping is not True):
        raise WorkflowError("Official OpenPI physical rollout needs fresh supervised confirmation and mapping acceptance")
    target = load_target("lambda", "pi05-base", config=selection.config)
    if target.service != selection.external_service:
        raise WorkflowError("Official OpenPI backend changed after preparation")
    statistics = load_statistics()
    transport = make_transport(target, statistics)
    try:
        metadata = transport.ready()
        path = _local_path(str(selection.qualification_path))
        if not 0 < path.stat().st_size < 32 * 1024 * 1024:
            raise WorkflowError("Invalid OpenPI qualification evidence size")
        report = json.loads(path.read_text())
        validate_qualification(report, metadata, task=selection.task, rig_path=Path(selection.rig_path), statistics=statistics)
        if metadata["session_expires_at"] <= time.time() + selection.duration + 60:
            raise WorkflowError("OpenPI session expired or lacks the bounded execution/release margin")
        destination = _local_path(str(artifact_dir or ROOT / "outputs/ui/deployments" / ("openpi-" + uuid.uuid4().hex)))
        kwargs = {"task": selection.task, "duration_s": selection.duration, "rig_path": Path(selection.rig_path),
                  "statistics": statistics, "qualification": report, "artifact_dir": destination,
                  "capture_trace": capture_trace, "upload_repo_id": upload_repo_id}
        if fake_hardware:
            from .qualification import load_saved

            observations = [load_saved(path) for path in target.saved_observations]
            return run_rollout(transport, fake_hardware=True, fake_observations=observations, **kwargs)
        return run_rollout(transport, confirm_supervised=True, accept_mapping=True, **kwargs)
    except WorkflowError:
        raise
    except Exception as exc:  # noqa: BLE001 — never echo private SDK/transport exception text
        raise WorkflowError(f"Official OpenPI execution failed ({type(exc).__name__}); no automatic retry") from None
    finally:
        transport.close()
