"""Native π0.5 backend/qualification glue; never use MA2 or base-policy fallback."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .backend_workflow import WorkflowError, _local_path, connect_configured_runtime, load_target
from .paths import ROOT


@dataclass(frozen=True)
class Pi05Selection:
    task: str
    rig_path: str
    duration: float
    external_service: str
    config: Path | None
    qualification_path: Path
    policy: str = "pi05-yam"
    backend: str = "lambda"
    controller_mode: str = "pi05_reference"


def _directory(service):
    from .external_ops import _name

    target = _local_path(str(ROOT / "data/inference/native" / _name(service)))
    target.mkdir(mode=0o700, parents=True, exist_ok=True)
    return target


def _readiness(target, token):
    from .inference.http_transport import HttpTransport
    from .inference.identity import external_service_binding, http_ingress_binding, inference_build_id
    from .pi05.transport import validate_readiness

    transport = HttpTransport(target.service, "pi05", endpoint_url=target.endpoint, token=token,
                              http_ingress="ssh", http_session_expires_at=time.time() + 30)
    try:
        metadata = transport._invoke("ready", None, 5)
    finally:
        transport.close()
    validate_readiness(metadata)
    binding = http_ingress_binding(metadata, endpoint_url=target.endpoint)
    external = external_service_binding(metadata)
    if (binding["http_ingress"] != "ssh" or external["service_id"] != target.service or external["provider"] != "lambda"
            or metadata.get("inference_build_id") != inference_build_id()):
        raise WorkflowError("π0.5 service identity or endpoint differs from its configured backend")
    return metadata


def _transport(target, token, metadata):
    from .pi05.transport import Pi05Transport

    return Pi05Transport(target.service, endpoint_url=target.endpoint, token=token, http_ingress="ssh",
                         http_session_expires_at=metadata["http_session_expires_at"])


def prepare_pi05(*, backend, task, rig, duration=60, arms=(), config=None, progress=lambda _value: None, force=False):
    from .external_ops import _read_json, _read_private, _save
    from .pi05.admission import passive_target_validator, validate_qualification
    from .pi05.qualification import collect_qualification, load_saved_observation, save_qualification
    from .rollout_artifacts import sanitize

    if (backend != "lambda" or not isinstance(task, str) or not task.strip() or len(task) > 2048
            or type(duration) not in (int, float) or not math.isfinite(duration) or not 0 < duration <= 90
            or tuple(arms) not in ((), ("left_follower", "right_follower"))):
        raise WorkflowError("Native π0.5 requires Lambda, an exact task, both named followers, and a duration of 1–90 seconds")
    rig = _local_path(str(rig))
    passive_target_validator(rig)  # Saved configuration/SDK bound files only; no hardware.
    target = load_target(backend, "pi05-yam", config=config)
    if target.token_file is None:
        raise WorkflowError("Configure the native π0.5 service's private token-file path")
    token = _read_private(target.token_file, 257).strip()
    directory = _directory(target.service)
    if not target.saved_observations and not (directory / "qualification.json").is_file():
        raise WorkflowError("Configure saved_observations with existing recording NPZ files for native π0.5 qualification; "
                            "no live observation capture is performed")
    try:
        metadata = connect_configured_runtime(target, task, lambda: _readiness(target, token), progress=progress)
    except WorkflowError as exc:
        raise WorkflowError(str(exc) + "; π0.5 also requires authorized access to google/paligemma-3b-pt-224. "
                            "If access is denied, accept its license with the configured HF account; never substitute a tokenizer/model") from None
    if time.time() + duration + 60 >= metadata["http_session_expires_at"]:
        raise WorkflowError("Native π0.5 model session expires too soon; let its bounded service end and prepare a new session")
    _save(directory / "receipt.json", sanitize({"status": "ready", "profile": "pi05-yam", "service": target.service,
                                               "metadata": metadata}, secrets=(token,)))
    report, report_path = None, None
    if not force:
        try:
            pointer = _read_json(directory / "qualification.json")
            report_path = _local_path(pointer["path"])
            if report_path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("Oversized qualification")
            report = json.loads(report_path.read_text())
            validate_qualification(report, metadata, task=task, rig_path=rig)
        except (OSError, ValueError, KeyError, TypeError):
            report = None
    reused = report is not None
    if report is None:
        if not target.saved_observations:
            raise WorkflowError("Configure saved_observations with existing recording NPZ files for native π0.5 qualification; "
                                "no live observation capture is performed")
        for path in target.saved_observations:
            if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024 * 1024:
                raise WorkflowError("Each saved π0.5 observation must be an existing bounded repository-local NPZ file")
        observations = [load_saved_observation(path) for path in target.saved_observations]
        evidence = _local_path(str(ROOT / ".context/pi05-qualification" / uuid.uuid4().hex))
        report_path = evidence / "qualification.json"
        progress("Qualifying native π0.5 with saved real observations and fake arms (50 warm samples)")
        transport = _transport(target, token, metadata)
        try:
            report = collect_qualification(transport, observations=observations, task=task,
                                           requests=50, rig_path=rig)
        finally:
            transport.close()
        report = sanitize(report, secrets=(token,))
        save_qualification(report, report_path, observation_paths=list(target.saved_observations))
        try:
            validate_qualification(report, metadata, task=task, rig_path=rig)
        except ValueError:
            raise WorkflowError("Native π0.5 qualification did not pass; inspect its saved report. No motion was started") from None
        _save(directory / "qualification.json", {"path": str(report_path)})
    selection = Pi05Selection(task, str(rig), duration, target.service, config, report_path)
    result = {"ready": True, "reused": reused, "hardware_tested": False, "motion_approval_received": False,
              "backend": backend, "policy": "pi05-yam", "controller_mode": "pi05_reference",
              "service": target.service, "instance_id": metadata["instance_id"],
              "expires_at": metadata["http_session_expires_at"], "evidence_directory": str(report_path.parent),
              "direct_warm_round_trip_s": report["direct_warm_round_trip_s"],
              "integrated": {key: report["integrated"].get(key) for key in (
                  "predicted_rows", "completed_rows", "dropped_rows", "modified_commands", "coherence_violations",
                  "completed_chunks", "execution_rate_hz", "faults")}}
    return selection, result


def run_prepared_pi05(selection, *, confirm_supervised, accept_mapping):
    """Called only inside the CLI's lifetime lock after fresh terminal approval."""
    from .external_ops import _read_private
    from .pi05.rollout import run_rollout

    if confirm_supervised is not True or accept_mapping is not True:
        raise WorkflowError("Native π0.5 requires fresh mapping acceptance and supervised confirmation")
    target = load_target("lambda", "pi05-yam", config=selection.config)
    if target.service != selection.external_service or target.token_file is None:
        raise WorkflowError("Native π0.5 backend selection changed after preparation")
    token = _read_private(target.token_file, 257).strip()
    metadata = _readiness(target, token)
    qualification = json.loads(selection.qualification_path.read_text())
    transport = _transport(target, token, metadata)
    try:
        return run_rollout(transport, task=selection.task, duration_s=selection.duration,
                           rig_path=Path(selection.rig_path), qualification=qualification,
                           confirm_supervised=True, accept_mapping=True,
                           artifact_dir=ROOT / "outputs/inference" / ("pi05-" + uuid.uuid4().hex))
    finally:
        transport.close()
