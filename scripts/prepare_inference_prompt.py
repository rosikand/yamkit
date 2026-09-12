"""UI-owned, software-only prompt warmup and reference qualification. Never launches motion."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path


class SanitizedOutput(io.TextIOBase):
    """Line-buffer optional diagnostics without reflecting credentials or huge result payloads."""

    def __init__(self, destination, secrets):
        self.destination, self.secrets, self.pending = destination, secrets, ""

    def write(self, value):
        from yamkit.rollout_artifacts import sanitize_text

        self.pending += value
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            self.destination.write(sanitize_text(line, secrets=self.secrets)[:2048] + "\n")
        if len(self.pending) > 32768:
            self.pending = "[oversized diagnostic omitted]"
        return len(value)

    def flush(self):
        self.destination.flush()


def phase(value):
    print("[yamkit-prepare] " + value, flush=True)


def execute(request_path: Path) -> int:
    from yamkit.config import RigConfig
    from yamkit.deployment import InferenceOptions
    from yamkit.external_ops import _probe_ready, _validated_metadata, http_credentials, owned_service
    from yamkit.inference.qualification import (
        MAX_AGE_S,
        _settings_path,
        settings_from_rig,
        validate_qualification,
    )
    from yamkit.modal_qualification import collect_qualification
    from yamkit.paths import ROOT
    from yamkit.rollout_artifacts import _read, _safe_path, sanitize, sanitize_text
    from yamkit.ui.server import _capture_memory_preflight, _prompt_preparation_context

    directory = request_path.parent
    result = {"ready": False, "hardware_tested": False, "selection_key": None,
              "evidence_directory": str(directory)}
    private_values = ()
    safe_directory = False

    def save(name, value):
        if not safe_directory:
            raise ValueError("Evidence directory was not validated")
        with (directory / name).open("x") as output:
            json.dump(sanitize(value, secrets=private_values), output, indent=2, allow_nan=False)
            output.write("\n")

    try:
        phase("validating")
        root = ROOT.resolve()
        _safe_path(request_path)
        if (request_path.name != "request.json" or directory.parent != root / ".context" / "inference-preparation"
                or len(directory.name) != 32 or any(value not in "0123456789abcdef" for value in directory.name)
                or request_path.stat().st_size > 16384):
            raise ValueError("Prompt preparation requires its own bounded repository-local request")
        safe_directory = True
        request = json.loads(_read(request_path))
        if not isinstance(request, dict) or set(request) != {"options", "expected", "capture_trace"}:
            raise ValueError("Prompt preparation request schema differs from the UI selection")
        values = dict(request["options"])
        values["arms"] = tuple(values.get("arms", ()))
        options = InferenceOptions(**values)
        result["selection_key"] = options.operation_key
        if options.rig_path is None or not _safe_path(Path(options.rig_path)).is_relative_to(root):
            raise ValueError("Prompt preparation requires this repository's rig configuration")
        rig = RigConfig.load(options.rig_path)
        context = _prompt_preparation_context(options, rig)
        if context != request["expected"]:
            raise ValueError("Prompt selection or attached service changed before preparation")
        if request["capture_trace"] and not _capture_memory_preflight(int(options.duration))["admission_passes"]:
            raise ValueError("Insufficient available memory to save this recording")
        credentials = http_credentials(options.external_service)
        private_values = (credentials["token"],)
        # The helper cannot become a camera-owning child. Qualification also replaces
        # all SDK and camera factories with generated fixtures before integrated work.
        os.environ.pop("YAMKIT_PREVIEW_SESSION", None)
        os.environ.pop("YAMKIT_PREVIEW_TOKEN", None)
        receipt = owned_service(options.external_service)
        readiness = _validated_metadata(options.external_service, credentials["endpoint_url"],
                                        _probe_ready(options.external_service, credentials["endpoint_url"], credentials["token"]),
                                        receipt["provider"])
        if any(readiness.get(key) != receipt["metadata"].get(key) for key in (
                "instance_id", "inference_build_id", "external_service", "runtime_provenance",
                "execution_identity", "http_session_expires_at")):
            raise ValueError("The live model service changed; refresh its attachment before preparing a prompt")
        save("previous-attachment.json", receipt)
        previous_path = _settings_path({"profile": "molmoact2", "backend": "external",
                                       "external_service_name": options.external_service, "controller_mode": "reference"})
        previous = json.loads(_read(previous_path)) if previous_path.exists() else {"present": False}
        save("previous-qualification.json", previous)
        phase("warming_and_qualifying")
        with contextlib.redirect_stdout(SanitizedOutput(sys.stdout, private_values)), contextlib.redirect_stderr(SanitizedOutput(sys.stderr, private_values)):
            record = collect_qualification(
                "molmoact2", requests=50, rig_path=Path(options.rig_path), backend="external",
                external_service=options.external_service, image_encoding="rgb8", jpeg_quality=85,
                call_mode="http", center_crop=False, prediction_queue_threshold=None,
                execution_mode="cuda_graph10", controller_mode="reference", task=options.task)
        save("qualification.json", record)
        if record.get("hardware_tested") is not False or record.get("assessment", {}).get("qualified") is not True:
            reasons = record.get("assessment", {}).get("reasons", [])
            raise ValueError("Prompt qualification did not pass: " + "; ".join(str(reason) for reason in reasons[:4]))
        phase("checking")
        current = _prompt_preparation_context(options, RigConfig.load(options.rig_path))
        if current != context:
            raise ValueError("Attached model identity changed during prompt preparation")
        qualified = validate_qualification(settings_from_rig(options))
        expires = min(current["expires_at"], qualified["created_unix_s"] + MAX_AGE_S)
        if time.time() + options.duration + 60 >= expires:
            raise ValueError("The model session expires too soon for this rollout")
        result.update(ready=True, expires_at=expires, reason="Prompt ready; supervised Start is still required")
        phase("ready")
        code = 0
    except KeyboardInterrupt:
        result["reason"] = "Prompt preparation cancelled; no motion was started"
        phase("cancelled")
        code = 130
    except Exception as exc:  # noqa: BLE001 — private service exceptions must not print tracebacks or credentials.
        result["reason"] = (sanitize_text(str(exc), secrets=private_values)[:512] if isinstance(exc, ValueError)
                            else f"Prompt preparation did not complete ({type(exc).__name__}); no motion was started")
        phase("failed")
        code = 1
    try:
        save("result.json", result)
    except (OSError, ValueError):
        result.update(ready=False, reason="Prompt result could not be saved; no motion was started")
        phase("failed")
        code = 1
    print("[yamkit-result] " + json.dumps(sanitize(result, secrets=private_values)), flush=True)
    return code


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Use the UI-managed prompt preparation request")
    raise SystemExit(execute(Path(sys.argv[1])))
