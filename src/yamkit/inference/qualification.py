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


def qualification_settings(profile, *, modal_app: str | None = None, call_mode: str = "remote",
                           image_encoding: str = "rgb8", jpeg_quality: int = 85,
                           image_hw=(480, 640), crop: str = "none", requested_region: str = "us-west",
                           observed_region: str | None = None, routing_region: str = "us-west",
                           prediction_queue_threshold: int | None = None,
                           max_observation_age_s: float = 2.0, execution_mode: str = "eager",
                           task: str | None = None, metadata: dict | None = None,
                           endpoint_url: str | None = None, backend: str = "modal",
                           external_service: str | None = None, controller_mode: str = "async") -> dict:
    profile = get_profile(profile)
    if controller_mode not in ("async", "reference"):
        raise QualificationError("Unknown remote controller mode")
    if controller_mode == "reference" and (profile.id != "molmoact2" or call_mode != "http"
            or execution_mode != "cuda_graph10" or image_encoding != "rgb8" or crop != "none"):
        raise QualificationError("Reference qualification requires MolmoAct2 HTTP graph10, raw RGB and no crop")
    if backend not in ("modal", "external"):
        raise QualificationError("Unknown remote inference backend")
    if backend == "external":
        from yamkit.external_ops import _name

        from .identity import external_service_binding

        _name(external_service)
        external = external_service_binding(metadata or {})
        if external["service_id"] != external_service or call_mode != "http" or modal_app:
            raise QualificationError("External qualification requires its explicit service and HTTP path")
    if ((backend == "modal" and not modal_app) or call_mode not in ("remote", "spawn", "http")
            or image_encoding not in ("rgb8", "jpeg")):
        raise QualificationError("Explicit Modal app, supported call path and image encoding are required")
    if (type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100 or len(image_hw) != 2
            or any(type(value) is not int or value <= 0 for value in image_hw)):
        raise QualificationError("Invalid image dimensions or JPEG quality")
    if crop not in ("none", "center_16_9"):
        raise QualificationError("Unknown image crop")
    if backend == "modal" and (
            any(not isinstance(region, str) or not region for region in (requested_region, observed_region, routing_region))
            or observed_region == "unknown"):
        raise QualificationError("Requested compute/routing and reliably observed compute placement are required")
    threshold = profile.chunk_size if prediction_queue_threshold is None else prediction_queue_threshold
    if type(threshold) is not int or not 0 <= threshold <= profile.chunk_size:
        raise QualificationError("Invalid prediction queue threshold")
    if type(max_observation_age_s) not in (int, float) or not 0 < max_observation_age_s <= 2.0:
        raise QualificationError("Qualification must retain the production observation-age guard")
    result = {"profile": profile.id, "model_revision": profile.revision,
            "dependency_revision": profile.dependency_revision, "modal_app": modal_app,
            "call_mode": call_mode, "controller_mode": controller_mode, "image_encoding": image_encoding,
            "jpeg_quality": jpeg_quality if image_encoding == "jpeg" else None,
            "image_hw": list(image_hw), "crop": crop, "requested_region": requested_region,
            "observed_region": observed_region, "routing_region": routing_region,
            "fps": profile.fps, "chunk_steps": profile.chunk_size,
            "prediction_queue_threshold": threshold, "max_observation_age_s": max_observation_age_s,
            "protocol_version": 1, "lerobot_version": LEROBOT_VERSION,
            "jpeg_subsampling": 2 if image_encoding == "jpeg" else None,
            "image_boundary_version": "saved-policy-transform-v1"}
    if backend == "external":
        for key in ("modal_app", "requested_region", "observed_region", "routing_region"):
            result.pop(key)
        result.update(backend="external", external_service_name=external_service, external_service=external)
    if call_mode == "http":
        from .identity import http_runtime_binding

        try:
            if endpoint_url is None:
                raise ValueError("HTTP qualification requires the measured endpoint")
            result.update(http_runtime_binding(profile, metadata or {}, execution_mode=execution_mode,
                                               task=task, image_hw=image_hw, crop=crop,
                                               image_encoding=image_encoding, jpeg_quality=jpeg_quality,
                                               endpoint_url=endpoint_url))
            if not result.get("http_endpoint"):
                raise ValueError("HTTP qualification requires the measured endpoint")
            if (result.get("http_ingress") == "ssh") != (backend == "external"):
                raise ValueError("HTTP ingress does not match the selected remote provider")
        except (ValueError, TypeError) as exc:
            raise QualificationError(str(exc)) from None
    elif execution_mode != "eager":
        raise QualificationError("Production CUDA graphs require the reviewed HTTP path")
    return result


def current_settings(config, *, image_hw, metadata=None) -> dict:
    """Resolve placement from the owned service and fresh readiness when supplied."""
    profile = get_profile(getattr(config, "profile", None) or config.policy)
    backend = getattr(config, "backend", "modal")
    if backend == "external":
        from yamkit.external_ops import http_credentials, owned_service

        name = getattr(config, "external_service", None)
        receipt = owned_service(name) or {}
        http_credentials(name)
    elif backend == "modal":
        from yamkit.modal_ops import owned_service

        receipt = owned_service() or {}
    else:
        raise QualificationError("Unknown remote inference backend")
    if (receipt.get("status") != "ready" or receipt.get("profile_id") != profile.id
            or receipt.get("revision") != profile.revision
            or (backend == "modal" and getattr(config, "modal_app", None)
                and receipt.get("app_name") != config.modal_app)):
        raise QualificationError("Attach or prepare the matching remote service before qualifying or rolling out")
    metadata = receipt.get("metadata", {}) if metadata is None else metadata
    if backend == "external" and any(metadata.get(key) != receipt.get("metadata", {}).get(key)
                                     for key in ("external_service", "runtime_provenance")):
        raise QualificationError("Current external host or runtime differs from its attachment")
    if backend == "modal" and (metadata.get("requested_compute_region") != receipt.get("region")
            or metadata.get("routing_region") != receipt.get("routing_region")):
        raise QualificationError("Current service placement differs from its ownership receipt")
    if config.call_mode == "http" and (
            receipt.get("transport") != "http"
            or not receipt.get("http_endpoint")
            or receipt.get("execution_mode") != getattr(config, "execution_mode", "eager")):
        raise QualificationError("Current HTTP execution differs from the owned service")
    if config.call_mode == "http":
        from .identity import http_ingress_binding

        try:
            current = http_ingress_binding(metadata, endpoint_url=receipt.get("http_endpoint"))
            owned = http_ingress_binding(receipt.get("metadata", {}), endpoint_url=receipt.get("http_endpoint"))
            if (current != owned or current["http_ingress"] != receipt.get("http_ingress", "asgi")
                    or current["http_session_expires_at"] != receipt.get("http_session_expires_at")
                    or metadata.get("instance_id") != receipt.get("metadata", {}).get("instance_id")):
                raise ValueError("Current HTTP ingress, expiry or instance differs from the owned service")
        except (ValueError, TypeError) as exc:
            raise QualificationError(str(exc)) from None
    return qualification_settings(
        profile, modal_app=receipt.get("app_name") if backend == "modal" else None, call_mode=config.call_mode,
        image_encoding=config.image_encoding, jpeg_quality=config.jpeg_quality,
        image_hw=image_hw, crop="center_16_9" if config.center_crop else "none",
        requested_region=metadata.get("requested_compute_region"),
        observed_region=metadata.get("compute_region"), routing_region=metadata.get("routing_region"),
        prediction_queue_threshold=config.prediction_queue_threshold,
        max_observation_age_s=getattr(config, "max_observation_age_s", 2.0),
        execution_mode=getattr(config, "execution_mode", "eager"), task=getattr(config, "task", None),
        controller_mode=getattr(config, "controller_mode", "async"),
        metadata=metadata, endpoint_url=receipt.get("http_endpoint"),
        **({"backend": backend, "external_service": name} if backend == "external" else {}))


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
    try:
        valid = type(value) in (float, int) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise QualificationError(f"Missing or invalid {field}")
    return value


def _same_value(actual, expected):
    """JSON identity comparison without Python's True == 1 == 1.0 coercion."""
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(_same_value(actual[key], value)
                                                       for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(_same_value(a, b) for a, b in zip(actual, expected, strict=True))
    return actual == expected


def _mapping(value, label, reasons):
    if not isinstance(value, dict):
        reasons.append(f"Missing or malformed {label}")
        return {}
    return value


def _rows(value, label, reasons):
    if not isinstance(value, list):
        reasons.append(f"Missing or malformed {label}")
        return []
    result = []
    for row in value:
        if not isinstance(row, dict):
            reasons.append(f"Malformed {label} row")
        else:
            result.append(row)
    return result


def _has_inference_experiment(value, profile, *, production_graph=False):
    """Reject diagnostic overrides wherever the collector preserved their evidence."""
    if isinstance(value, list):
        return any(_has_inference_experiment(item, profile, production_graph=production_graph) for item in value)
    if not isinstance(value, dict):
        return False
    if any(value.get(key) is not None for key in ("diagnostic_num_inference_steps", "diagnostic_cuda_graph")):
        return True
    if any(value.get(key) not in (None, False) for key in ("experiment_only", "experimental")):
        return True
    # Only the explicitly bound HTTP graph runtime may use production graphs.
    # Per-request diagnostic overrides above are always rejected.
    if not production_graph and any(value.get(key) not in (None, False)
                                    for key in ("cuda_graph_enabled", "cuda_graph_used")):
        return True
    effective = value.get("effective_num_inference_steps")
    if effective is not None:
        expected = 10 if profile == "molmoact2" else value.get("default_num_inference_steps")
        if type(effective) is not int or effective != expected:
            return True
    return any(_has_inference_experiment(item, profile, production_graph=production_graph) for item in value.values())


def _check_http_evidence(settings, direct, integrated, reasons, requested):
    """Independently validate every new execution binding, including raw samples."""
    from .http_wire import MAX_MESSAGE_BYTES, WIRE_CODEC, WIRE_VERSION
    from .identity import http_runtime_binding

    graph = settings.get("execution_mode") == "cuda_graph10"
    expected_execution = {
        "execution_mode": settings.get("execution_mode"), "execution_identity": settings.get("execution_identity"),
        "configured_model_dtype": "bfloat16", "parameter_dtype_numel": {"torch.bfloat16": 5442196208},
        "default_num_inference_steps": 10, "effective_num_inference_steps": 10,
        "production_cuda_graph_configured": graph, "cuda_graph_enabled": graph, "cuda_graph_used": graph,
    }

    def readiness_binding(metadata):
        return http_runtime_binding(settings["profile"], metadata,
                                    execution_mode=settings.get("execution_mode"), task=settings.get("task"),
                                    image_hw=settings["image_hw"], crop=settings["crop"],
                                    image_encoding=settings["image_encoding"], endpoint_url=settings.get("http_endpoint"))

    for report in (direct, integrated):
        try:
            metadata = _mapping(report.get("readiness"), "HTTP readiness", reasons)
            binding = readiness_binding(metadata)
            if any(not _same_value(settings.get(key), value) for key, value in binding.items()):
                reasons.append("HTTP readiness execution, warmup or container binding changed")
            execution = _mapping(metadata.get("model_execution"), "HTTP readiness model execution", reasons)
            if (metadata.get("ready") is not True or any(
                    not _same_value(execution.get(key), expected_execution[key])
                    for key in ("configured_model_dtype", "parameter_dtype_numel", "default_num_inference_steps",
                                "production_cuda_graph_configured", "cuda_graph_enabled"))):
                reasons.append("HTTP readiness did not prove the bound execution configuration")
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            reasons.append("HTTP readiness lacks the current reviewed runtime and exact task warmup")
    if direct.get("execution_mode") != settings.get("execution_mode") or direct.get("task") != settings.get("task"):
        reasons.append("Direct HTTP task or execution mode differs from qualification")
    options = _mapping(integrated.get("policy_options"), "integrated policy options", reasons)
    if options.get("execution_mode") != settings.get("execution_mode") or options.get("task") != settings.get("task"):
        reasons.append("Integrated HTTP task or execution mode differs from qualification")
    raw_image_bytes = len(get_profile(settings["profile"]).image_keys) * math.prod(settings["image_hw"]) * 3
    ready = _mapping(direct.get("readiness"), "direct HTTP readiness", reasons)
    expected_warm = _mapping(ready.get("graph_warmup"), "direct graph warmup", reasons) if graph else {}
    for label, rows in (("direct", direct.get("samples")), ("integrated", integrated.get("samples"))):
        if not isinstance(rows, list) or len(rows) < requested + 1:
            reasons.append(f"Missing raw {label} HTTP execution evidence")
            continue
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                reasons.append(f"Malformed raw {label} HTTP execution evidence")
                continue
            if any(not _same_value(row.get(key), settings.get(key))
                   for key in ("instance_id", "task", "execution_mode", "execution_identity")):
                reasons.append(f"A {label} HTTP response changed execution identity")
                break
            execution = _mapping(row.get("model_execution"), f"{label} model execution", reasons)
            if any(not _same_value(execution.get(key), value) for key, value in expected_execution.items()):
                reasons.append(f"A {label} HTTP response did not use its bound execution mode")
                break
            if graph:
                warm = _mapping(row.get("graph_warmup"), f"{label} graph warmup", reasons)
                if (warm.get("ready") is not True
                        or not _same_value(warm.get("signature"), expected_warm.get("signature"))
                        or warm.get("signature_sha256") != settings.get("graph_signature_sha256")
                        or warm.get("cache_key_sha256") != settings.get("graph_cache_key_sha256")
                        or execution.get("graph_cache_key_sha256") != settings.get("graph_cache_key_sha256")):
                    reasons.append(f"A {label} HTTP response used another graph warmup")
                    break
                capture = execution.get("graph_capture_required")
                if type(capture) is not bool or (capture and (label != "direct" or index != 0)):
                    reasons.append(f"A warm {label} HTTP request captured another graph")
                    break
            timing = _mapping(row.get("transport_timing"), f"{label} HTTP transport timing", reasons)
            expected_route = {"http_ingress": settings.get("http_ingress", "asgi"),
                              "http_session_expires_at": settings.get("http_session_expires_at"),
                              "http_endpoint_sha256": hashlib.sha256(settings["http_endpoint"].encode()).hexdigest()}
            if any(not _same_value(timing.get(key), value) for key, value in expected_route.items()
                   if settings.get("http_ingress") in ("tunnel", "ssh") or key in timing):
                reasons.append(f"A {label} HTTP request used another ingress, endpoint or session expiry")
                break
            size = row.get("wire_payload_bytes")
            response_size = timing.get("wire_response_bytes")
            if (type(size) is not int or not raw_image_bytes < size <= MAX_MESSAGE_BYTES
                    or not _same_value(timing.get("wire_request_bytes"), size)
                    or type(response_size) is not int or not 0 < response_size <= MAX_MESSAGE_BYTES
                    or timing.get("call_mode") != "http" or timing.get("wire_codec") != WIRE_CODEC
                    or not _same_value(timing.get("wire_version"), WIRE_VERSION)
                    or timing.get("wire_compression") != "none"
                    or row.get("image_encoding") != "rgb8"
                    or not _same_value(row.get("payload_bytes"), raw_image_bytes)
                    or (label == "direct" and not _same_value(row.get("image_hw"), settings["image_hw"]))):
                reasons.append(f"A {label} HTTP response omitted its bounded raw RGB wire measurement")
                break


def _check_reference_holds(settings, integrated, completed, reasons, counter):
    """Prove overlapping SDK sends only maintained one unchanged endpoint."""
    proof = _mapping(integrated.get("reference_execution"), "reference execution", reasons)
    guard = _mapping(integrated.get("command_shaping"), "reference command guard", reasons)
    predictions = _rows(integrated.get("prediction_samples"), "reference prediction samples", reasons)
    executed = counter(integrated.get("executed_actions"), "reference total commands", minimum=1)
    names = set(get_profile(settings["profile"]).action_names)

    def action(value):
        try:
            return (isinstance(value, dict) and set(value) == names
                    and all(type(item) in (int, float) and math.isfinite(item) for item in value.values())
                    and all(0 <= value[name] <= 1 for name in names if "gripper" in name))
        except (OverflowError, TypeError):
            return False

    total = 0
    last_dispatch = -1
    for index, event in enumerate(predictions):
        count = counter(event.get("maintenance_holds_during_prediction"), "reference maintenance count")
        samples = _rows(event.get("maintenance_hold_samples"), "reference maintenance samples", reasons)
        if count != len(samples) or len(samples) > 64:
            reasons.append("Reference maintenance count lacks complete bounded send evidence")
        if counter(event.get("maintenance_hold_samples_dropped"), "reference maintenance samples dropped") != 0:
            reasons.append("Reference maintenance evidence was truncated")
        if counter(event.get("actions_executed_during_prediction"), "reference policy inference overlap") != 0:
            reasons.append("Policy interpolation overlapped reference inference")
        anchor = event.get("maintenance_hold_anchor")
        if not action(anchor) and not (index == 0 and anchor is None and count == 0):
            reasons.append("Reference maintenance anchor is missing or invalid")
        try:
            started = _number(event.get("prediction_started_monotonic_s"), "reference prediction start")
            observed = _number(event.get("observation_timestamp_monotonic_s"), "reference observation time")
            wait_deadline = _number(event.get("inference_wait_deadline_monotonic_s"), "reference inference deadline")
            if not observed <= started < wait_deadline <= observed + settings["max_observation_age_s"] + 1e-8:
                reasons.append("Reference maintenance lacks its original bounded inference deadline")
        except QualificationError as exc:
            reasons.append(str(exc))
            started = wait_deadline = 0
        previous_at = None
        for sample in samples:
            dispatch = counter(sample.get("dispatch_index"), "reference maintenance dispatch index")
            if dispatch is None or dispatch <= last_dispatch or executed is None or dispatch >= executed:
                reasons.append("Reference maintenance dispatch indices are not unique and ordered")
            if dispatch is not None:
                last_dispatch = dispatch
            if not action(sample.get("sent")) or not _same_value(sample.get("sent"), anchor):
                reasons.append("Reference maintenance changed its cached full-vector endpoint")
            try:
                at = _number(sample.get("monotonic_s"), "reference maintenance time")
                deadline = _number(sample.get("deadline_monotonic_s"), "reference maintenance dispatch deadline")
                # A hold selected before worker completion can be prepared just
                # after its return. The fixed wait lease still binds that send.
                if not started <= at < deadline <= wait_deadline + 1e-8 or deadline - at > 0.1 + 1e-8:
                    reasons.append("Reference maintenance escaped its original wait or dispatch deadline")
                if previous_at is not None and not 0 < at - previous_at <= 0.1 + 1e-8:
                    reasons.append("Reference maintenance intervals exceeded the active stall guard")
                previous_at = at
            except QualificationError as exc:
                reasons.append(str(exc))
        total += len(samples)
    if (counter(proof.get("inference_hold_dispatches"), "reference inference holds") != total
            or counter(guard.get("inference_hold_count"), "guard inference holds") != total):
        reasons.append("Reference maintenance totals disagree with the recorded sends")
    if (counter(proof.get("inference_hold_mismatches"), "reference maintenance mismatches") != 0
            or counter(guard.get("postclamp_modified_count"), "reference postclamp modifications") != 0):
        reasons.append("Reference maintenance or postclamp guard changed a command")
    if counter(guard.get("sample_count"), "reference guard sample count") != integrated.get("executed_actions"):
        reasons.append("Reference guard did not validate every executed command")

    # This independently observed benchmark count includes native warmup. It
    # counts individual successful arm sends strictly inside HTTP, then floors
    # pairs; boundary-crossing pairs and engine pre/postprocessing explain <=.
    transport = _rows(integrated.get("transport_predictions"), "reference transport requests", reasons)
    returned = [event for event in transport if "returned" in event]
    sdk = integrated.get("sdk_commands_during_completed_rpc")
    if not isinstance(sdk, list) or len(sdk) != len(returned):
        reasons.append("Reference maintenance lacks independent SDK overlap counts")
        sdk = []
    matched = 0
    for request, observed_pairs in zip(returned, sdk, strict=False):
        pairs = counter(observed_pairs, "reference observed SDK overlap")
        if request.get("mode") == "native_fixture":
            if pairs != 0:
                reasons.append("SDK commands occurred during pre-hardware reference warmup")
            continue
        if request.get("mode") != "robot" or matched >= len(completed):
            reasons.append("Reference transport requests do not match admitted policy predictions")
            continue
        event = completed[matched]
        matched += 1
        holds = counter(event.get("maintenance_holds_during_prediction"), "reference matched maintenance count")
        if pairs is None or holds is None or pairs > holds:
            reasons.append("Observed SDK overlap exceeds proven stationary reference holds")
        try:
            start = _number(request.get("started"), "reference HTTP request start")
            end = _number(request.get("returned"), "reference HTTP request return")
            prediction_start = _number(event.get("prediction_started_monotonic_s"), "reference prediction start")
            elapsed = _number(event.get("prediction_s"), "reference prediction duration")
            if not prediction_start <= start <= end <= prediction_start + elapsed + 1e-8:
                reasons.append("Reference HTTP overlap measurement belongs to another prediction interval")
        except QualificationError as exc:
            reasons.append(str(exc))
    if matched != len(completed):
        reasons.append("Reference SDK overlap evidence omits completed predictions")
    return total


def _check_reference_evidence(settings, integrated, completed, reasons, counter):
    """Check full response consumption, not just successful RPC completion."""
    proof = _mapping(integrated.get("reference_execution"), "reference execution", reasons)
    chunk_steps = settings["chunk_steps"]
    if (settings.get("call_mode") != "http" or settings.get("execution_mode") != "cuda_graph10"
            or settings.get("image_encoding") != "rgb8" or settings.get("crop") != "none"
            or proof.get("controller_mode") != "reference"):
        reasons.append("Reference evidence lacks its reviewed controller and model execution binding")
    for key in ("expired_prefix_dropped", "overlap_prefix_dropped", "prefix_drop",
                "coherence_violations", "expired_plans", "uncompleted_steps_at_stop"):
        if counter(proof.get(key), f"reference {key}") != 0:
            reasons.append(f"Reference execution reported {key} or omitted its measurement")
    for key in ("expired_prefix_dropped", "overlap_prefix_dropped"):
        if counter(integrated.get(key), key) != 0:
            reasons.append("Reference execution must not discard an action prefix")
    if (proof.get("next_observation_after_full_chunk") is not True
            or proof.get("partial_chunk_at_stop") is not False):
        reasons.append("Reference qualification must finish each full chunk before the next observation and Stop probe")
    for key, expected in (("predicted_steps", len(completed) * chunk_steps),
                          ("admitted_steps", len(completed) * chunk_steps),
                          ("completed_steps", len(completed) * chunk_steps),
                          ("completed_chunks", len(completed))):
        if counter(proof.get(key), f"reference {key}") != expected:
            reasons.append(f"Reference {key} does not match every accepted full response")
    planned_dispatches = 0
    for index, event in enumerate(completed):
        if (counter(event.get("accepted_steps"), "reference accepted_steps") != chunk_steps
                or counter(event.get("completed_chunks_at_start"), "reference completed_chunks_at_start") != index
                or counter(event.get("completed_steps_at_start"), "reference completed_steps_at_start") != index * chunk_steps
                or counter(event.get("actions_executed_during_prediction"), "reference inference overlap") != 0):
            reasons.append("Reference requests did not follow complete, ordered chunk consumption")
        points = counter(event.get("plan_dispatches"), "reference plan_dispatches", minimum=chunk_steps)
        if points is not None:
            planned_dispatches += points
        try:
            age = _number(event.get("observation_age_at_return_s"), "reference observation age")
            duration = _number(event.get("planned_duration_s"), "reference planned duration")
            deadline = _number(event.get("plan_deadline_monotonic_s"), "reference plan deadline")
            started = _number(event.get("prediction_started_monotonic_s"), "reference request start")
            prediction_s = _number(event.get("prediction_s"), "reference prediction duration")
            if (age > settings["max_observation_age_s"] or points is None
                    or not math.isclose(duration, points / settings["fps"], rel_tol=1e-9, abs_tol=1e-9)
                    or not started + prediction_s < deadline <= started + prediction_s + duration * 1.1 + 0.1 + 1e-8):
                reasons.append("Reference request freshness or fixed plan lease is invalid")
        except QualificationError as exc:
            reasons.append(str(exc))
    dispatches = counter(proof.get("interpolation_dispatches"), "reference interpolation_dispatches", minimum=1)
    if dispatches != planned_dispatches:
        reasons.append("Reference dispatch count does not exhaust every accepted interpolation plan")
    holds = _check_reference_holds(settings, integrated, completed, reasons, counter)
    if dispatches is None or dispatches + holds != integrated.get("executed_actions"):
        reasons.append("Reference total commands do not equal interpolation plus stationary inference holds")


def _assess(settings, direct, integrated, requested):
    if type(requested) is not int or not MIN_WARM_SAMPLES <= requested <= 500:
        raise QualificationError("Qualification requires 50–500 warm requests")
    reasons = []
    mode = settings.get("controller_mode", "async")
    reference = mode == "reference"
    if mode not in ("async", "reference"):
        reasons.append("Unknown qualification controller mode")
    direct = _mapping(direct, "direct report", reasons)
    integrated = _mapping(integrated, "integrated report", reasons)
    production_graph = settings.get("call_mode") == "http" and settings.get("execution_mode") == "cuda_graph10"
    if any(_has_inference_experiment(report, settings["profile"], production_graph=production_graph)
           for report in (direct, integrated)):
        reasons.append("Diagnostic inference experiments cannot qualify the unchanged production policy")
    if settings.get("call_mode") == "http":
        _check_http_evidence(settings, direct, integrated, reasons, requested)

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
    direct_readiness = _mapping(direct.get("readiness"), "direct readiness", reasons)
    integrated_readiness = _mapping(integrated.get("readiness"), "integrated readiness", reasons)
    identity = direct_readiness.get("instance_id")
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
    reported_rpc = _mapping(direct.get("warm_round_trip_s"), "warm round trip summary", reasons)
    for key in ("p50", "p95", "p99"):
        reported = reported_rpc.get(key)
        measured = measured_rpc.get(key)
        try:
            valid = measured is not None and math.isclose(_number(reported, f"warm {key}"), measured,
                                                         rel_tol=1e-9, abs_tol=1e-9)
        except QualificationError:
            valid = False
        if not valid:
            reasons.append(f"Reported warm {key} does not match raw request samples")
    if counter(direct.get("warm_sample_count"), "warm_sample_count") != max(0, len(samples) - 1):
        reasons.append("Reported warm count does not match raw request samples")
    completed = []
    for event in _rows(integrated.get("prediction_samples"), "integrated prediction samples", reasons):
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
            if not reference:
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
    if reference:
        # There is deliberately no prediction/actuation overlap. Measure RPC
        # against its admission budget, not the asynchronous action queue.
        horizon = usable_horizon = settings["max_observation_age_s"]
        actual_horizon_p05 = None
    external = settings.get("backend") == "external"
    if external:
        expected_provenance = {"backend": "external", "external_service": settings.get("external_service"),
                               "transport": "http"}
        for report in (direct, integrated):
            if not _same_value(report.get("service_provenance"), expected_provenance):
                reasons.append("External qualification requires measured provider, service and host provenance")
        options = _mapping(integrated.get("policy_options"), "integrated external policy options", reasons)
        if (options.get("backend") != "external"
                or options.get("external_service") != settings.get("external_service_name")):
            reasons.append("Integrated execution selected another external service")
    elif (not isinstance(direct.get("measurement"), str) or "real Modal" not in direct["measurement"]
            or not isinstance(integrated.get("source"), str) or "real Modal" not in integrated["source"]):
        reasons.append("Qualification requires real Modal measurements through the final integrated path")
    for report in (direct, integrated):
        if report.get("measurement_host") != host_identity():
            reasons.append("Measurements originated on another or unknown host")
        metadata = direct_readiness if report is direct else integrated_readiness
        if (metadata.get("profile") != settings["profile"]
                or metadata.get("model_revision") != settings["model_revision"]
                or (not external and (
                    metadata.get("requested_compute_region") != settings["requested_region"]
                    or metadata.get("compute_region") != settings["observed_region"]
                    or metadata.get("routing_region") != settings["routing_region"]))
                or (external and not _same_value(metadata.get("external_service"), settings.get("external_service")))
                or report.get("image_hw") != settings["image_hw"]):
            reasons.append("Measured model, image dimensions or placement do not match the requested qualification")
    if (direct_readiness.get("instance_id") is None
            or direct_readiness.get("instance_id") != integrated_readiness.get("instance_id")):
        reasons.append("Direct and integrated measurements used different or unknown containers")
    for key in ("image_encoding", "call_mode", "crop"):
        if direct.get(key) != settings[key]:
            reasons.append(f"Direct measurement {key} differs from the qualification settings")
    policy_options = _mapping(integrated.get("policy_options"), "integrated policy options", reasons)
    if policy_options.get("controller_mode", "async") != mode:
        reasons.append("Integrated controller mode differs from the qualification settings")
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
    if reference:
        _check_reference_evidence(settings, integrated, completed, reasons, counter)
    elif counter(integrated.get("minimum_execution_queue_depth"), "minimum_execution_queue_depth", minimum=1) is None:
        reasons.append("The executing queue drained")
    if (integrated.get("stop_requested_during_inflight_rpc") is not True
            or counter(integrated.get("commands_after_stop"), "commands_after_stop") != 0):
        reasons.append("Stop during in-flight inference did not prove zero late SDK commands")
    if any(failure.get("reason") != "InvalidatedRequest"
           for failure in _rows(integrated.get("failures"), "integrated failures", reasons)):
        reasons.append("A non-Stop request failure occurred during integrated execution")
    if p95 is None or not usable_horizon or p95 > usable_horizon * 0.8:
        reasons.append("Warm RPC p95 does not fit the effective usable horizon with 20% margin")
    if reference and (age_p95 is None or age_p95 > usable_horizon * 0.8):
        reasons.append("Reference observation-age p95 does not fit its admission budget with 20% margin")
    return {"qualified": not reasons, "reasons": reasons, "requested_warm_samples": requested,
            "controller_mode": mode,
            "latency_budget_basis": "synchronous RPC admission" if reference else "asynchronous remaining queue",
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


def _path(profile: str, *, backend="modal", external_service=None, controller_mode="async") -> Path:
    if controller_mode not in ("async", "reference"):
        raise QualificationError("Unknown qualification controller mode")
    suffix = "-reference" if controller_mode == "reference" else ""
    if backend == "external":
        from yamkit.external_ops import _name

        return DATA_DIR / "qualifications" / f"external-{_name(external_service)}-{get_profile(profile).id}{suffix}.json"
    if backend != "modal":
        raise QualificationError("Unknown qualification backend")
    return DATA_DIR / "qualifications" / f"modal-{get_profile(profile).id}{suffix}.json"


def _settings_path(settings: dict) -> Path:
    return _path(settings["profile"], backend=settings.get("backend", "modal"),
                 external_service=settings.get("external_service_name"),
                 controller_mode=settings.get("controller_mode", "async"))


def save_qualification(record: dict, path: Path | None = None) -> Path:
    path = _settings_path(record["settings"]) if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return path


def validate_qualification(settings: dict, *, path: Path | None = None, now: float | None = None) -> dict:
    """Validate current-host evidence; callers must still enforce hardware guards."""
    path = _settings_path(settings) if path is None else Path(path)
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
        command = "external-qualify" if settings.get("backend") == "external" else "modal-qualify"
        raise QualificationError(f"No valid local qualification record; run yamkit {command} on this host") from exc
