"""Host-bound evidence for Modal performance; never grants hardware permission.

Records live under git-ignored data/qualifications. A passing record still needs
the independent hardware mapping checks and explicit supervised-run confirmation.
Copying a cloud diagnostic to the robot host does not qualify that host's network.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path

from yamkit.paths import DATA_DIR, ROOT

from .performance import percentile_summary
from .profiles import LEROBOT_VERSION, get_profile

MAX_AGE_S = 24 * 60 * 60
MIN_WARM_SAMPLES = 50
_RUNNER_CONTEXT = ContextVar("yamkit_validated_remote_runner", default=False)


class QualificationError(ValueError):
    """The current host and settings lack sufficient recent performance evidence."""


@contextmanager
def validated_runner_context():
    token = _RUNNER_CONTEXT.set(True)
    try:
        yield
    finally:
        _RUNNER_CONTEXT.reset(token)


def require_runner_context():
    from .performance import QUALIFICATION_GATE_ENABLED

    if QUALIFICATION_GATE_ENABLED and not _RUNNER_CONTEXT.get():
        raise QualificationError("Physical remote policies require yamkit rollout's validated hardware and Stop path")


def host_identity() -> dict:
    hostname = socket.gethostname()
    machine_file = Path("/etc/machine-id")
    machine = machine_file.read_text().strip() if machine_file.is_file() else hostname
    return {"hostname": hostname,
            "machine_fingerprint": hashlib.sha256(f"{hostname}:{machine}".encode()).hexdigest()}


def is_cloud_host() -> bool:
    return str(ROOT).startswith("/home/vercel-sandbox/") or any(
        os.environ.get(key) for key in ("CONDUCTOR_WORKSPACE_ID", "VERCEL_SANDBOX_ID"))


def qualification_settings(profile, *, modal_app: str, call_mode: str = "remote",
                           image_encoding: str = "rgb8", jpeg_quality: int = 85,
                           image_hw=(480, 640), crop: str = "none", requested_region: str = "us-west",
                           observed_region: str, routing_region: str = "us-west",
                           prediction_queue_threshold: int | None = None,
                           max_observation_age_s: float = 2.0) -> dict:
    profile = get_profile(profile)
    if not modal_app or call_mode not in ("remote", "spawn") or image_encoding not in ("rgb8", "jpeg"):
        raise QualificationError("Explicit Modal app, supported call path and image encoding are required")
    if (type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100 or len(image_hw) != 2
            or any(type(value) is not int or value <= 0 for value in image_hw)):
        raise QualificationError("Invalid image dimensions or JPEG quality")
    if crop not in ("none", "center_16_9"):
        raise QualificationError("Unknown image crop")
    if (any(not isinstance(region, str) or not region for region in (requested_region, observed_region, routing_region))
            or observed_region == "unknown"):
        raise QualificationError("Requested compute/routing and reliably observed compute placement are required")
    threshold = profile.chunk_size if prediction_queue_threshold is None else prediction_queue_threshold
    if type(threshold) is not int or not 0 <= threshold <= profile.chunk_size:
        raise QualificationError("Invalid prediction queue threshold")
    if type(max_observation_age_s) not in (int, float) or not 0 < max_observation_age_s <= 2.0:
        raise QualificationError("Qualification must retain the production observation-age guard")
    return {"profile": profile.id, "model_revision": profile.revision,
            "dependency_revision": profile.dependency_revision, "modal_app": modal_app,
            "call_mode": call_mode, "image_encoding": image_encoding,
            "jpeg_quality": jpeg_quality if image_encoding == "jpeg" else None,
            "image_hw": list(image_hw), "crop": crop, "requested_region": requested_region,
            "observed_region": observed_region, "routing_region": routing_region,
            "fps": profile.fps, "chunk_steps": profile.chunk_size,
            "prediction_queue_threshold": threshold, "max_observation_age_s": max_observation_age_s,
            "protocol_version": 1, "lerobot_version": LEROBOT_VERSION,
            "jpeg_subsampling": 2 if image_encoding == "jpeg" else None,
            "image_boundary_version": "saved-policy-transform-v1"}


def current_settings(config, *, image_hw, metadata=None) -> dict:
    """Resolve placement from the owned service and fresh readiness when supplied."""
    from yamkit.modal_ops import owned_service

    profile = get_profile(getattr(config, "profile", None) or config.policy)
    receipt = owned_service() or {}
    if (receipt.get("status") != "ready" or receipt.get("profile_id") != profile.id
            or receipt.get("revision") != profile.revision
            or (getattr(config, "modal_app", None) and receipt.get("app_name") != config.modal_app)):
        raise QualificationError("Prepare the matching owned Modal service before qualifying or rolling out")
    metadata = receipt.get("metadata", {}) if metadata is None else metadata
    if (metadata.get("requested_compute_region") != receipt.get("region")
            or metadata.get("routing_region") != receipt.get("routing_region")):
        raise QualificationError("Current service placement differs from its ownership receipt")
    return qualification_settings(
        profile, modal_app=receipt["app_name"], call_mode=config.call_mode,
        image_encoding=config.image_encoding, jpeg_quality=config.jpeg_quality,
        image_hw=image_hw, crop="center_16_9" if config.center_crop else "none",
        requested_region=metadata.get("requested_compute_region"),
        observed_region=metadata.get("compute_region"), routing_region=metadata.get("routing_region"),
        prediction_queue_threshold=config.prediction_queue_threshold,
        max_observation_age_s=getattr(config, "max_observation_age_s", 2.0))


def settings_from_policy(config, metadata=None) -> dict:
    shapes = {tuple(feature.shape[-2:]) for name, feature in config.input_features.items()
              if name.startswith("observation.images.")}
    if len(shapes) != 1:
        raise QualificationError("Qualification requires exact, equal camera dimensions")
    return current_settings(config, image_hw=next(iter(shapes)), metadata=metadata)


def settings_from_rig(options) -> dict:
    from yamkit.config import RigConfig
    from yamkit.paths import DEFAULT_RIG

    rig = RigConfig.load(options.rig_path or DEFAULT_RIG)
    profile = get_profile(options.policy)
    dimensions = {(rig.cameras[name].get("height"), rig.cameras[name].get("width")) for name in profile.image_keys}
    if len(dimensions) != 1:
        raise QualificationError("Qualification requires exact, equal configured camera dimensions")
    return current_settings(options, image_hw=next(iter(dimensions)))


def _number(value, field):
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise QualificationError(f"Missing or invalid {field}")
    return value


def _has_inference_experiment(value, profile):
    """Reject diagnostic overrides wherever the collector preserved their evidence."""
    if isinstance(value, list):
        return any(_has_inference_experiment(item, profile) for item in value)
    if not isinstance(value, dict):
        return False
    if any(value.get(key) is not None for key in ("diagnostic_num_inference_steps", "diagnostic_cuda_graph")):
        return True
    if any(value.get(key) not in (None, False) for key in ("experiment_only", "experimental")):
        return True
    # Graphs remain disabled in the qualified production runtime. A diagnostic
    # request can leave its graph cache populated, which alone does not mean use.
    if any(value.get(key) not in (None, False) for key in ("cuda_graph_enabled", "cuda_graph_used")):
        return True
    effective = value.get("effective_num_inference_steps")
    if effective is not None:
        expected = 10 if profile == "molmoact2" else value.get("default_num_inference_steps")
        if type(effective) is not int or effective != expected:
            return True
    return any(_has_inference_experiment(item, profile) for item in value.values())


def _assess(settings, direct, integrated, requested):
    if type(requested) is not int or not MIN_WARM_SAMPLES <= requested <= 500:
        raise QualificationError("Qualification requires 50–500 warm requests")
    reasons = []
    if any(_has_inference_experiment(report, settings["profile"]) for report in (direct, integrated)):
        reasons.append("Diagnostic inference experiments cannot qualify the unchanged production policy")

    def counter(value, name, *, minimum=0):
        if type(value) is not int or value < minimum:
            reasons.append(f"Missing or invalid integer counter {name}")
            return None
        return value

    samples = direct.get("samples", [])
    if not isinstance(samples, list):
        samples = []
        reasons.append("Raw direct request samples are missing")
    durations = []
    identity = (direct.get("readiness") or {}).get("instance_id")
    for sequence, sample in enumerate(samples):
        if not isinstance(sample, dict):
            reasons.append("Malformed raw direct request sample")
            continue
        if type(sample.get("sequence_id")) is not int or sample["sequence_id"] != sequence:
            reasons.append("Raw request sequence is not contiguous from zero")
        if not isinstance(identity, str) or not identity or sample.get("instance_id") != identity:
            reasons.append("Raw requests lack a stable readiness-matching container")
        try:
            durations.append(_number(sample.get("round_trip_s"), "raw request round trip"))
        except QualificationError as exc:
            reasons.append(str(exc))
    measured_rpc = percentile_summary(durations[1:])
    for key in ("p50", "p95", "p99"):
        reported = direct.get("warm_round_trip_s", {}).get(key)
        measured = measured_rpc.get(key)
        if (type(reported) not in (float, int) or not math.isfinite(reported)
                or measured is None or not math.isclose(reported, measured, rel_tol=1e-9, abs_tol=1e-9)):
            reasons.append(f"Reported warm {key} does not match raw request samples")
    if counter(direct.get("warm_sample_count"), "warm_sample_count") != max(0, len(samples) - 1):
        reasons.append("Reported warm count does not match raw request samples")
    completed = []
    for event in integrated.get("prediction_samples", []):
        accepted = counter(event.get("accepted_steps"), "accepted_steps")
        if event.get("error") is None and accepted is not None and accepted > 0:
            completed.append(event)
    warm = completed[1:]
    p95 = measured_rpc.get("p95")
    ages = []
    measured_horizons = []
    for event in warm:
        try:
            ages.append(_number(event.get("observation_age_at_return_s"), "integrated observation age"))
            measured_horizons.append(_number(event.get("remaining_valid_action_horizon_s"), "actual merged horizon"))
        except QualificationError as exc:
            reasons.append(str(exc))
    age_p95 = percentile_summary(ages).get("p95")
    horizon = min(settings["chunk_steps"] / settings["fps"], settings["max_observation_age_s"])
    usable_horizon = max(0.0, horizon - age_p95) if age_p95 is not None else 0.0
    # p05 of the actual merge horizons is the tail corresponding to p95 age.
    # This also accounts for shorter returned chunks and postprocessing time.
    actual_horizon_p05 = -percentile_summary(-value for value in measured_horizons)["p95"] if measured_horizons else 0.0
    usable_horizon = min(usable_horizon, actual_horizon_p05)
    if "real Modal" not in direct.get("measurement", "") or "real Modal" not in integrated.get("source", ""):
        reasons.append("Qualification requires real Modal measurements through the final integrated path")
    for report in (direct, integrated):
        if report.get("measurement_host") != host_identity():
            reasons.append("Measurements originated on another or unknown host")
        metadata = report.get("readiness") or {}
        if (metadata.get("profile") != settings["profile"]
                or metadata.get("model_revision") != settings["model_revision"]
                or metadata.get("requested_compute_region") != settings["requested_region"]
                or metadata.get("compute_region") != settings["observed_region"]
                or metadata.get("routing_region") != settings["routing_region"]
                or report.get("image_hw") != settings["image_hw"]):
            reasons.append("Measured model, image dimensions or placement do not match the requested qualification")
    direct_readiness = direct.get("readiness") or {}
    integrated_readiness = integrated.get("readiness") or {}
    if (direct_readiness.get("instance_id") is None
            or direct_readiness.get("instance_id") != integrated_readiness.get("instance_id")):
        reasons.append("Direct and integrated measurements used different or unknown containers")
    for key in ("image_encoding", "call_mode", "crop"):
        if direct.get(key) != settings[key]:
            reasons.append(f"Direct measurement {key} differs from the qualification settings")
    policy_options = integrated.get("policy_options", {})
    for key in ("image_encoding", "call_mode"):
        if policy_options.get(key) != settings[key]:
            reasons.append(f"Integrated measurement {key} differs from the qualification settings")
    if settings["image_encoding"] == "jpeg" and (
            direct.get("jpeg_quality") != settings["jpeg_quality"]
            or policy_options.get("jpeg_quality") != settings["jpeg_quality"]):
        reasons.append("Measured JPEG quality differs from the qualification settings")
    if bool(policy_options.get("center_crop")) != (settings["crop"] == "center_16_9"):
        reasons.append("Integrated crop differs from the qualification settings")
    measured_threshold = policy_options.get("prediction_queue_threshold")
    measured_threshold = settings["chunk_steps"] if measured_threshold is None else measured_threshold
    if measured_threshold != settings["prediction_queue_threshold"]:
        reasons.append("Integrated prediction scheduling differs from the qualification settings")
    if (integrated.get("fps") != settings["fps"] or integrated.get("chunk_steps") != settings["chunk_steps"]):
        reasons.append("Integrated action cadence or nominal chunk differs from the qualification settings")
    if len(durations) - 1 < requested or len(warm) < requested:
        reasons.append("Insufficient completed warm requests in the direct and integrated paths")
    if direct.get("terminated") != "request_limit" or counter(direct.get("container_instance_count"), "containers") != 1:
        reasons.append("Direct measurements did not finish on one stable container")
    if (integrated.get("failed") is not False
            or counter(integrated.get("executed_actions"), "executed_actions", minimum=1) is None
            or integrated.get("all_fake_robots_released") is not True):
        reasons.append("Integrated fake-robot execution failed or did not release normally")
    for key in ("underruns", "expired_chunks", "expired_queued_actions", "expired_before_dispatch"):
        if counter(integrated.get(key), key) != 0:
            reasons.append(f"Integrated execution reported {key} or omitted its measurement")
    if counter(integrated.get("minimum_execution_queue_depth"), "minimum_execution_queue_depth", minimum=1) is None:
        reasons.append("The executing queue drained")
    if (integrated.get("stop_requested_during_inflight_rpc") is not True
            or counter(integrated.get("commands_after_stop"), "commands_after_stop") != 0):
        reasons.append("Stop during in-flight inference did not prove zero late SDK commands")
    if any(failure.get("reason") != "InvalidatedRequest" for failure in integrated.get("failures", [])):
        reasons.append("A non-Stop request failure occurred during integrated execution")
    if p95 is None or not usable_horizon or p95 > usable_horizon * 0.8:
        reasons.append("Warm RPC p95 does not fit the effective usable horizon with 20% margin")
    return {"qualified": not reasons, "reasons": reasons, "requested_warm_samples": requested,
            "completed_integrated_warm_samples": len(warm), "nominal_action_horizon_s": horizon,
            "integrated_observation_age_p95_s": age_p95,
            "integrated_merged_horizon_p05_s": actual_horizon_p05,
            "effective_usable_action_horizon_s": usable_horizon,
            "required_margin_fraction": 0.2, "maximum_qualifying_rpc_p95_s": usable_horizon * 0.8,
            "warm_round_trip_s": measured_rpc}


def build_qualification(settings: dict, *, direct: dict, integrated: dict,
                        requested_warm_samples: int = MIN_WARM_SAMPLES) -> dict:
    """Store both passing and failing evidence without changing any hardware gate."""
    created = time.time()
    assessment = _assess(settings, direct, integrated, requested_warm_samples)
    return {"schema_version": 1, "created_unix_s": created,
            "created_at": datetime.fromtimestamp(created, UTC).isoformat(), "host": host_identity(),
            "scope": "same host and settings only; mapping and supervised confirmation required",
            "source_environment": "cloud" if is_cloud_host() else "robot-host candidate",
            "status": ("READY_FOR_LENOVO_QUALIFICATION" if is_cloud_host() else "QUALIFIED_FOR_THIS_HOST")
            if assessment["qualified"] else "STILL_TOO_SLOW",
            "hardware_tested": False, "settings": settings, "assessment": assessment,
            "direct": direct, "integrated": integrated}


def _path(profile: str) -> Path:
    return DATA_DIR / "qualifications" / f"modal-{get_profile(profile).id}.json"


def save_qualification(record: dict, path: Path | None = None) -> Path:
    path = _path(record["settings"]["profile"]) if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def validate_qualification(settings: dict, *, path: Path | None = None, now: float | None = None) -> dict:
    """Validate current-host evidence; callers must still enforce hardware guards."""
    path = _path(settings["profile"]) if path is None else Path(path)
    try:
        if path.stat().st_size > 10_000_000:
            raise QualificationError("Qualification record exceeds its bounded size")
        def reject_constant(value):
            raise QualificationError(f"Qualification JSON contains nonfinite {value}")

        record = json.loads(path.read_text(), parse_constant=reject_constant)
        if record.get("schema_version") != 1 or record.get("hardware_tested") is not False:
            raise QualificationError("Unsupported qualification record")
        if record.get("host") != host_identity():
            raise QualificationError("Qualification belongs to another host; rerun on the robot host")
        age = (time.time() if now is None else now) - _number(record.get("created_unix_s"), "record timestamp")
        if not 0 <= age <= MAX_AGE_S:
            raise QualificationError("Qualification is expired or has a future timestamp; rerun within 24 hours")
        if record.get("settings") != settings:
            raise QualificationError("Qualification settings changed; rerun on this host with the current settings")
        assessment = _assess(settings, record["direct"], record["integrated"],
                             record["assessment"]["requested_warm_samples"])
        if not assessment["qualified"]:
            raise QualificationError("; ".join(assessment["reasons"]))
        record["assessment"] = assessment
        return record
    except (OSError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise QualificationError("No valid local qualification record; run yamkit modal-qualify on this host") from exc
