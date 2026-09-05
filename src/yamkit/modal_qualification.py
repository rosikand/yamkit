"""Explicit host-local Modal qualification collection; never opens arms or cameras."""

from __future__ import annotations

import importlib.util
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from .paths import DEFAULT_RIG, ROOT


def _benchmark_module():
    """Load the repository diagnostic explicitly, including from another working directory."""
    spec = importlib.util.spec_from_file_location("yamkit_remote_benchmark", ROOT / "scripts/benchmark_remote.py")
    if spec is None or spec.loader is None:
        raise ValueError("Qualification requires this repository's scripts/benchmark_remote.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_qualification(policy="molmoact2", *, requests=50, modal_app=None, rig_path=DEFAULT_RIG,
                          image_encoding="jpeg", jpeg_quality=85, call_mode="remote", center_crop=False,
                          prediction_queue_threshold=None) -> dict:
    from .config import RigConfig
    from .inference.profiles import get_profile
    from .inference.qualification import build_qualification, qualification_settings, save_qualification
    from .modal_ops import owned_service

    if type(requests) is not int or not 50 <= requests <= 200:
        raise ValueError("Qualification requires 50–200 warm requests plus a first request")
    if image_encoding not in ("jpeg", "rgb8") or type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("Use JPEG quality 1–100 or raw rgb8 encoding")
    if call_mode not in ("remote", "spawn"):
        raise ValueError("call_mode must be remote or spawn")
    profile = get_profile(policy)
    if profile.id != "molmoact2":
        raise ValueError("Only the reviewed MolmoAct2 YAM candidate supports physical qualification")
    if prediction_queue_threshold is not None and (
            type(prediction_queue_threshold) is not int
            or not 0 <= prediction_queue_threshold <= profile.chunk_size):
        raise ValueError("Prediction queue threshold must be between zero and the chunk size")
    receipt = owned_service() or {}
    app_name = modal_app or receipt.get("app_name")
    if not app_name:
        raise ValueError("Prepare a dedicated Modal service first or pass --modal-app")
    # Reading configuration does not enumerate, lease or stream any device.
    rig_path = Path(rig_path)
    rig = RigConfig.load(rig_path)
    dimensions = set()
    for camera in profile.image_keys:
        spec = rig.cameras.get(camera)
        if not isinstance(spec, dict) or type(spec.get("height")) is not int or type(spec.get("width")) is not int:
            raise ValueError(f"Qualification requires configured image dimensions for {camera}")
        dimensions.add((spec["height"], spec["width"]))
    if len(dimensions) != 1:
        raise ValueError("Qualification currently requires equal camera dimensions; no resize is applied")
    image_hw = next(iter(dimensions))
    benchmark = _benchmark_module()

    def transport_factory(stop=None):
        return benchmark.make_benchmark_transport(app_name, profile.id, shutdown_event=stop, call_mode=call_mode)

    direct = benchmark.profile_modal(transport_factory(), profile_name=profile.id, warm_samples=requests,
                                     max_wall_s=min(600, 15 * requests), image_hw=image_hw,
                                     image_encoding=image_encoding, jpeg_quality=jpeg_quality,
                                     center_crop=center_crop)
    readiness = direct.get("readiness") or {}

    def save_failure(reason, integrated=None):
        from .inference.qualification import host_identity

        created = time.time()
        record = {"schema_version": 1, "created_unix_s": created,
                  "created_at": datetime.fromtimestamp(created, UTC).isoformat(), "host": host_identity(),
                  "hardware_tested": False,
                  "settings": {"profile": profile.id, "model_revision": profile.revision, "modal_app": app_name,
                               "call_mode": call_mode, "image_encoding": image_encoding,
                               "jpeg_quality": jpeg_quality, "image_hw": list(image_hw),
                               "crop": "center_16_9" if center_crop else "none",
                               "requested_region": readiness.get("requested_compute_region") or receipt.get("region"),
                               "observed_region": readiness.get("compute_region"),
                               "routing_region": readiness.get("routing_region") or receipt.get("routing_region")},
                  "assessment": {"qualified": False, "reasons": [reason], "requested_warm_samples": requests},
                  "direct": direct, "integrated": integrated or {},
                  "status": "QUALIFICATION_FAILED"}
        return {**record, "qualification_path": str(save_qualification(record))}

    if direct.get("terminated") != "request_limit" or direct.get("warm_sample_count", 0) < requests:
        return save_failure("Direct measurements did not complete; no additional integrated requests were sent")
    # Import the repository fake SDK only inside the diagnostic. Every camera and
    # hardware factory is replaced before run_remote_rollout constructs a robot.
    original_path = list(sys.path)
    try:
        sys.path.insert(0, str(ROOT))
        integrated = benchmark.run_scenario(
            "host_modal_qualification", [0], duration=min(300, requests * 1.5 + 15),
            image_hw=image_hw, transport_factory=transport_factory, target_warm_samples=requests,
            policy_options={"profile": profile.id, "image_encoding": image_encoding, "jpeg_quality": jpeg_quality,
                            "call_mode": call_mode, "center_crop": center_crop,
                            "prediction_queue_threshold": prediction_queue_threshold})
    except Exception as exc:  # noqa: BLE001 — record failure type without SDK data or credentials
        return save_failure(f"Integrated diagnostic failed ({type(exc).__name__})")
    finally:
        sys.path[:] = original_path
    try:
        settings = qualification_settings(
            profile, modal_app=app_name, call_mode=call_mode, image_encoding=image_encoding,
            jpeg_quality=jpeg_quality, image_hw=image_hw, crop="center_16_9" if center_crop else "none",
            requested_region=readiness.get("requested_compute_region"), observed_region=readiness.get("compute_region"),
            routing_region=readiness.get("routing_region"), prediction_queue_threshold=prediction_queue_threshold)
        record = build_qualification(settings, direct=direct, integrated=integrated, requested_warm_samples=requests)
    except ValueError as exc:
        return save_failure(f"Qualification evidence was rejected ({type(exc).__name__})", integrated)
    path = save_qualification(record)
    return {**record, "qualification_path": str(path),
            "notice": "Host-bound measurement only. Physical rollout separately requires a valid local record, "
                      "accepted mapping and explicit supervised confirmation."}
