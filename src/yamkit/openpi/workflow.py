"""One-command preparation and explicit supervised execution of official π0.5."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from yamkit.backend_workflow import WorkflowError, _local_path, connect_configured_runtime, load_target
from yamkit.external_ops import _read_private, _save
from yamkit.paths import ROOT

from .admission import load_statistics, rig_contract, validate_qualification
from .interface import CONTRACT_ID
from .transport import OpenPiTransport


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
    if metadata["session_expires_at"] <= time.time() + duration + 60:
        raise WorkflowError("The bounded official OpenPI service expires too soon for this supervised session")
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
    reused = report is not None
    if report is None:
        observations = [load_saved(path) for path in target.saved_observations]
        evidence = ROOT / ".context/openpi-qualification" / uuid.uuid4().hex
        path = evidence / "qualification.json"
        progress("Warming official pi05_base and qualifying saved inputs with explicit fake arms")
        transport = make_transport(target, statistics)
        try:
            report = collect(transport, statistics=statistics, observations=observations, task=task,
                             rig_path=rig, directory=evidence, progress=progress)
            # A cancelled Stop probe cannot be reused for motion. This fresh
            # transport only checks the unchanged service identity afterward.
        except Exception as exc:  # noqa: BLE001 — transport errors can carry private values
            raise WorkflowError(f"Official OpenPI qualification failed ({type(exc).__name__}); inspect {evidence}") from None
        finally:
            transport.close()
        metadata = readiness(target, statistics)
        try:
            validate_qualification(report, metadata, task=task, rig_path=rig, statistics=statistics)
        except ValueError:
            raise WorkflowError("Official OpenPI software qualification failed; retained evidence: " + str(path)) from None
        _save(pointer, {"path": str(path)})
    selection = OpenPiSelection(task, str(rig), duration, target.service, config, path)
    return selection, {"ready": True, "reused": reused, "hardware_tested": False,
                       "motion_approval_received": False, "policy": "pi05-base", "backend": "lambda",
                       "controller_mode": CONTRACT_ID, "service": target.service,
                       "label": "official frozen pi05_base + documented experimental YAM adapter",
                       "expires_at": metadata["session_expires_at"], "instance_id": metadata["instance_id"],
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
