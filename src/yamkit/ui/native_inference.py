"""Additive native-policy UI boundary; imports and preflight never open devices or HTTP."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..deployment import InferenceOptions
from ..policy_selection import OPENPI_YAM_BLOCKER, canonical_policy

NATIVE_POLICIES = frozenset({"pi05-yam", "pi05_yam", "pi05-base", "pi05_base"})


def is_native(policy):
    return policy in NATIVE_POLICIES


@dataclass(frozen=True)
class NativeUIOptions(InferenceOptions):
    """Independent validation; no changes to frozen MA2 options or controller paths."""

    backend: str = "lambda"
    controller_mode: str = "pi05_reference"
    async_chunks: bool = False
    call_mode: str = "http"
    capture_trace: bool = False
    upload_repo_id: str | None = None

    def validate(self, *, motion=False):
        if canonical_policy(self.policy) == "pi05-base":
            raise ValueError(OPENPI_YAM_BLOCKER)
        if (not is_native(self.policy) or self.backend != "lambda"
                or self.controller_mode != "pi05_reference" or self.execution_mode != "eager"
                or self.call_mode != "http" or self.image_encoding != "rgb8"
                or self.jpeg_quality != 85 or self.prediction_queue_threshold is not None
                or self.center_crop is not False or self.rtc is not False or self.async_chunks is not False
                or self.modal_app is not None or self.fps != 30
                or tuple(self.arms) not in ((), ("left_follower", "right_follower"))):
            raise ValueError("YAM π0.5 requires Lambda, its native pi05_reference FIFO controller, full RGB HTTP, both followers and 30 Hz; no MA2 graph/interpolation/RTC settings")
        if (not isinstance(self.task, str) or not self.task.strip() or len(self.task) > 2048
                or type(self.duration) not in (int, float) or not math.isfinite(self.duration)
                or not 0 < self.duration <= 90):
            raise ValueError("Native π0.5 requires an exact task and a duration of 1–90 seconds")
        if any(type(value) is not bool for value in (self.mapping_accepted, self.supervised_confirmed, self.capture_trace)):
            raise ValueError("Native approval and recording fields must be booleans")
        if (not isinstance(self.device, str) or not re.fullmatch(r"(?:cpu|mps|cuda(?::[0-9]+)?)", self.device)
                or not isinstance(self.gpu, str) or not re.fullmatch(r"[A-Za-z0-9!_.-]{1,40}", self.gpu)
                or self.external_service is not None and (not isinstance(self.external_service, str)
                    or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", self.external_service))):
            raise ValueError("Native device display preferences or configured service name are invalid")
        if self.capture_trace and self.duration not in (5, 10, 20, 30, 45, 60, 90):
            raise ValueError("Native recording supports 5, 10, 20, 30, 45, 60 or 90 seconds")
        if self.upload_repo_id is not None:
            from huggingface_hub.utils import validate_repo_id

            validate_repo_id(self.upload_repo_id)
            if self.upload_repo_id.count("/") != 1 or not self.capture_trace:
                raise ValueError("Native upload requires local recording and a namespace/repository private dataset ID")
        if motion and (self.mapping_accepted is not True or self.supervised_confirmed is not True):
            raise ValueError("Native π0.5 requires fresh mapping acceptance and supervised confirmation")
        return self

    @property
    def operation_key(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


def preparation_context(options):
    """Only configured paths, hashes and saved schema; never read a credential value."""
    from ..backend_workflow import CONFIG_RELATIVE, _local_path, load_target
    from ..paths import ROOT
    from ..pi05.admission import passive_target_validator
    from ..pi05.contract import build_id

    options.validate()
    rig = _local_path(options.rig_path)
    passive_target_validator(rig)
    target = load_target("lambda", "pi05-yam")
    if options.external_service not in (None, target.service):
        raise ValueError("Selected native service differs from the configured π0.5 backend")
    if target.token_file is None or not target.token_file.is_file():
        raise ValueError("Configure the native π0.5 service's existing private token-file path")
    config = _local_path(str(ROOT / CONFIG_RELATIVE))
    return {"selection_key": options.operation_key, "service": target.service,
            "pi05_build_id": build_id(), "rig_sha256": hashlib.sha256(rig.read_bytes()).hexdigest(),
            "backend_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()}


def retained_selection(options):
    """Validate local retained proof. No live service contact, preparation, or directory writes."""
    from ..backend_workflow import _local_path, load_target
    from ..external_ops import _read_json
    from ..inference.identity import external_service_binding, http_ingress_binding, inference_build_id
    from ..paths import ROOT
    from ..pi05.admission import validate_qualification
    from ..pi05_workflow import Pi05Selection

    context = preparation_context(options)
    target = load_target("lambda", "pi05-yam")
    directory = _local_path(str(ROOT / "data/inference/native" / target.service))
    receipt = _read_json(directory / "receipt.json")
    metadata = receipt.get("metadata", {})
    external = external_service_binding(metadata)
    # prepare_pi05 validates the live advertised endpoint, then deliberately
    # strips endpoint keys when persisting sanitized metadata. This local-only
    # view uses the configured canonical loopback origin for that missing field;
    # a present but different/empty origin is still invalid. The shared workflow
    # checks live advertised origin/instance again before any physical startup.
    cached_ingress = dict(metadata)
    if "http_endpoint" not in cached_ingress:
        cached_ingress["http_endpoint"] = target.endpoint
    binding = http_ingress_binding(cached_ingress, endpoint_url=target.endpoint)
    if (receipt.get("status") != "ready" or receipt.get("profile") != "pi05-yam"
            or receipt.get("service") != target.service or external["service_id"] != target.service
            or external["provider"] != "lambda" or binding["http_ingress"] != "ssh"
            or metadata.get("inference_build_id") != inference_build_id()):
        raise ValueError("Native π0.5 local service identity differs from the configured backend; prepare this task")
    pointer = _read_json(directory / "qualification.json")
    report_path = _local_path(pointer["path"])
    if not 0 < report_path.stat().st_size <= 16 * 1024 * 1024:
        raise ValueError("Native π0.5 qualification evidence is unavailable or oversized")
    report = json.loads(report_path.read_text())
    validate_qualification(report, metadata, task=options.task, rig_path=Path(options.rig_path))
    checked = time.time()
    expires = min(metadata["http_session_expires_at"], report["completed_at"] + 86400)
    if checked + options.duration + 60 >= expires:
        raise ValueError("Native π0.5 session expires too soon; prepare again before supervised Start")
    selection = Pi05Selection(options.task, options.rig_path, options.duration, target.service, None, report_path)
    return selection, {"ready": True, "reason": "Native π0.5 qualified for this exact task; supervised confirmation is still required",
                       "selection_key": context["selection_key"], "checked_at": checked, "expires_at": expires,
                       "external_service": target.service, "hardware_tested": False}


def execute_request(request_path, *, motion=False):
    """Managed child entry point; preparation cannot call the physical runner."""
    import contextlib
    import io
    import os

    from ..backend_workflow import WorkflowError, _local_path
    from ..paths import ROOT
    from ..pi05_workflow import prepare_pi05, run_prepared_pi05
    from ..workflow_lock import workflow_lock

    class DiscardDiagnostics(io.TextIOBase):
        def write(self, value):
            return len(value)

    result = {"ready": False, "hardware_tested": None if motion else False, "selection_key": None}
    if not motion:
        result["motion_approval_received"] = False
    directory = None
    try:
        request_path = _local_path(str(request_path))
        candidate = request_path.parent
        parent = ROOT / ".context" / ("native-inference" if motion else "inference-preparation")
        if (request_path.name != "request.json" or candidate.parent != parent
                or len(candidate.name) != 32 or any(c not in "0123456789abcdef" for c in candidate.name)
                or not 0 < request_path.stat().st_size <= 16384):
            raise ValueError("Invalid native UI request location or size")
        request = json.loads(request_path.read_text())
        expected_keys = {"options", "expected", "trace_dir"} if motion else {"options", "expected"}
        if not isinstance(request, dict) or set(request) != expected_keys:
            raise ValueError("Invalid native UI request schema")
        values = dict(request["options"])
        values["arms"] = tuple(values.get("arms", ()))
        options = NativeUIOptions(**values).validate(motion=motion)
        if not motion and (options.mapping_accepted or options.supervised_confirmed):
            raise ValueError("Software preparation cannot carry motion approval")
        directory = candidate
        result["selection_key"] = options.operation_key
        if preparation_context(options) != request["expected"]:
            raise ValueError("Native selection/source/configuration changed after the UI request")
        # An exact approved request is consumed once, even if startup or export later fails.
        # Never replay embedded confirmation flags by re-running an old helper request.
        with (directory / "claimed.json").open("x") as output:
            json.dump({"selection_key": options.operation_key, "motion": motion, "claimed_at": time.time()}, output)
        with workflow_lock(root=ROOT, wait_s=10):
            if motion:
                selection, _ready = retained_selection(options)
                trace = _local_path(request["trace_dir"])
                if (trace.parent != ROOT / ".context/rollout-traces" or len(trace.name) != 32
                        or any(c not in "0123456789abcdef" for c in trace.name) or trace.exists()):
                    raise ValueError("Native rollout requires its own fresh recording directory")
                # Parent owns optional post-release upload; never upload twice.
                report = run_prepared_pi05(selection, confirm_supervised=True, accept_mapping=True,
                                           artifact_dir=trace, capture_trace=options.capture_trace,
                                           upload_repo_id=None, artifact_metadata=asdict(options))
                status = report.get("status")
                tested = report.get("hardware_tested")
                result.update(ready=False, hardware_tested=tested if type(tested) is bool else None,
                              status=status if status in ("completed", "stopped", "failed", "release_failed") else "unknown",
                              released=report.get("released") is True, artifact_directory=str(trace))
                if (status not in ("completed", "stopped") or report.get("released") is not True
                        or report.get("exit_status", 0) not in ((0, 130) if status == "stopped" else (0,))
                        or report.get("artifact_status", "TRACE_SAVED") != "TRACE_SAVED"):
                    raise WorkflowError("Native rollout or recording did not complete successfully; inspect " + str(trace)
                                        + ". No automatic physical retry was started")
            else:
                os.environ.pop("YAMKIT_PREVIEW_SESSION", None)
                os.environ.pop("YAMKIT_PREVIEW_TOKEN", None)
                print("[yamkit-prepare] warming_and_qualifying", flush=True)
                # Backend/library diagnostics are not a public protocol. Retain only the
                # shared workflow's sanitized evidence and controlled result fields.
                with contextlib.redirect_stdout(DiscardDiagnostics()), contextlib.redirect_stderr(DiscardDiagnostics()):
                    _selection, prepared = prepare_pi05(
                        backend="lambda", task=options.task, rig=options.rig_path, duration=options.duration,
                        arms=options.arms, own_preparation_dir=directory)
                if preparation_context(options) != request["expected"]:
                    raise ValueError("Native configuration changed during preparation")
                _selection, current = retained_selection(options)
                result.update(current, reused=prepared["reused"], evidence_directory=prepared["evidence_directory"])
                print("[yamkit-prepare] ready", flush=True)
        exit_code = 130 if motion and result.get("status") == "stopped" else 0
    except KeyboardInterrupt:
        result["reason"] = "Native UI operation cancelled; no automatic retry was started"
        exit_code = 130
    except Exception as exc:  # noqa: BLE001 — never leak private transport diagnostics
        result["reason"] = str(exc) if isinstance(exc, WorkflowError) else "Native UI validation or operation failed; inspect retained local evidence and prepare again"
        result["error_type"] = type(exc).__name__
        exit_code = 2
    if directory is not None:
        try:
            with (directory / "result.json").open("x") as output:
                json.dump(result, output, indent=2, allow_nan=False)
                output.write("\n")
        except (OSError, ValueError):
            result.update(ready=False, reason="Native result could not be retained; no automatic retry was started")
            exit_code = 2
    print("[yamkit-result] " + json.dumps(result, allow_nan=False), flush=True)
    return exit_code
