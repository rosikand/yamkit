"""Profile-aware probe orchestration shared by the CLI and its UI child process.

Readiness completes before live capture. Every prediction uses the same ModelRuntime saved
pre/postprocessing boundary as Modal rollout; neither hardware adapters nor normalization
statistics are replaced here.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from pathlib import Path

from .config import RigConfig
from .deployment import InferenceOptions
from .inference.mapping import CAMERA_RENAME_MAP
from .inference.profiles import get_profile
from .inference.protocol import MAX_TIMEOUT_S, encode_image, validate_request, validate_response
from .probes import (
    ProbeObservation,
    capture_live_observation,
    load_saved_observation,
    preflight_live_probe,
    probe_observation,
)

log = logging.getLogger(__name__)


def _validate_live_rig(rig: RigConfig, arms: Sequence[str] | None, profile) -> list[str]:
    specs, _ = preflight_live_probe(rig, arms, expected_state_names=profile.state_names)
    if set(rig.cameras) != set(profile.image_keys):
        raise ValueError(f"profile cameras must exactly match original rig names {profile.image_keys}")
    for name, config in rig.cameras.items():
        if str(config.get("color_mode", "rgb")).lower() != "rgb":
            raise ValueError(f"{name}: profile probes require RGB camera output")
    if profile.id == "molmoact2":
        for spec in specs:
            if spec.arm_type != "yam" or spec.gripper != "linear_4310":
                raise ValueError(f"{spec.name}: reviewed Molmo mapping requires YAM with a LINEAR_4310 gripper")
    return [spec.name for spec in specs]


def _validate_observation(observation: ProbeObservation, profile) -> None:
    observation.validate()
    if observation.state_names != profile.state_names:
        raise ValueError("observation ordered state names differ from the profile; no padding or reordering")
    if set(observation.images) != set(profile.image_keys):
        raise ValueError(f"observation cameras must exactly match original rig names {profile.image_keys}")
    age = observation.age_s()
    if age is None or age < 0:
        raise ValueError("profile probes require a known nonfuture capture timestamp; observation age is invalid")


def run_profile_probe(
    policy: str,
    rig_path: str | Path,
    *,
    saved: str | Path | None = None,
    live: bool = False,
    approved: bool = False,
    backend: str = "local",
    device: str = "cpu",
    modal_app: str | None = None,
    task: str = "",
    arms: Sequence[str] | None = None,
    center_crop: bool = False,
) -> dict:
    """One diagnostic chunk; a result never authorizes motion or supplies a replayable chunk."""
    if (saved is not None) == live:
        raise ValueError("choose exactly one saved observation or explicit live active-read mode")
    if live and approved is not True:
        raise PermissionError("explicit operator approval required for GRAVITY-COMPENSATION ACTIVE READ")
    options = InferenceOptions(
        policy=policy, task=task, backend=backend, device=device, modal_app=modal_app,
        arms=tuple(arms or ()), center_crop=center_crop,
    ).validate()
    profile = get_profile(options.policy)
    profile.require_robot_mapping()
    rig = None
    observation = None
    selected = None
    if live:
        rig = RigConfig.load(rig_path)
        selected = _validate_live_rig(rig, arms, profile)
    else:
        # Invalid files/mappings must not trigger a paid model load. Loading is hardware-free.
        observation = load_saved_observation(saved)
        _validate_observation(observation, profile)
    started = time.monotonic()
    if backend == "modal":
        from .modal_ops import _validate_ready, call, owned_service, service_handle

        receipt = owned_service()
        app_name = modal_app or (receipt or {}).get("app_name")
        if not app_name:
            raise ValueError("prepare a dedicated Modal service before probing")
        service = service_handle(app_name, profile.id)
        metadata = call(service.ready, timeout=300)
        _validate_ready(metadata, profile)

        def predict_request(request):
            return call(service.predict_chunk, request, timeout=request["timeout_s"])

        def retire(session_id):
            call(service.reset, session_id, timeout=5)
    else:
        from .inference.service import ModelRuntime
        from .modal_ops import _validate_ready

        service = ModelRuntime.load(profile, device=device)
        metadata = service.ready()
        _validate_ready(metadata, profile)
        predict_request = service.predict_chunk
        retire = service.reset
    if metadata.get("ready") is not True or metadata.get("saved_processors") is not True:
        raise ValueError("inference service did not confirm readiness with saved processors")
    if metadata.get("action_units") != "robot":
        raise ValueError("inference service did not confirm robot-unit actions")
    readiness_s = time.monotonic() - started
    if live:
        observation = capture_live_observation(
            rig, selected, approved=approved, expected_state_names=profile.state_names,
        )
        _validate_observation(observation, profile)
    assert observation is not None
    session_id = str(uuid.uuid4())
    response_metadata = {}
    attempted = False

    def predict(current):
        nonlocal attempted
        encoded_at = time.monotonic()
        request = {
            "protocol_version": 1, "profile": profile.id, "model_revision": profile.revision,
            "session_id": session_id, "sequence_id": 0, "observation_time": time.monotonic(),
            "observation_age_s": current.age_s(), "timeout_s": MAX_TIMEOUT_S,
            "task": task, "state": current.state.tolist(), "state_names": list(current.state_names),
            "images": {name: encode_image(frame, encoding="jpeg" if backend == "modal" else "rgb8")
                       for name, frame in current.images.items()},
            "mode": "live_probe" if live else "saved_probe", "crop": "center_16_9" if center_crop else "none",
            "continuation": None,
        }
        request["observation_age_s"] = current.age_s()
        validate_request(request, profile)
        encoding_s = time.monotonic() - encoded_at
        attempted = True
        sent = time.monotonic()
        response = predict_request(request)
        elapsed = time.monotonic() - sent
        if elapsed >= request["timeout_s"]:
            raise TimeoutError("probe response deadline exceeded; prediction discarded")
        validate_response(response, request, profile)
        response_metadata.update(
            server_timing=response["timing"], round_trip_s=elapsed,
            encoding_s=encoding_s, payload_bytes=sum(len(image["data"]) for image in request["images"].values()),
            image_encoding="jpeg" if backend == "modal" else "rgb8",
            jpeg_quality=85 if backend == "modal" else None,
            image_transforms=response.get("transforms", {}),
            saved_postprocessor_clamp=bool(response.get("saved_postprocessor_clamp")),
        )
        if "unclipped_chunk" not in response or response.get("unclipped_action_units") != "robot":
            raise ValueError("service omitted pre-clipping robot-unit diagnostics; a clipped chunk cannot satisfy an action probe")
        return response["unclipped_chunk"]

    try:
        result = probe_observation(
            observation, predict, action_names=profile.action_names,
            profile_id=profile.id, revision=profile.revision, mapping_validated=profile.mapping_verified,
            expected_state_names=profile.state_names,
            transforms=["server crop: " + ("center_16_9" if center_crop else "none"),
                        "server saved preprocessor (including one camera rename_map)",
                        "server saved postprocessor robot-unit conversion; diagnostics before clamp"],
        )
    finally:
        if attempted:
            try:
                retire(session_id)
            except Exception:  # noqa: BLE001 — retirement cannot mask the original inference/Stop error
                # Runtime resets every chunk as well; a failed retirement cannot enable replay.
                log.warning("probe session retirement failed; local probe is closed and its result is never executable")
    result.update(
        backend=backend, readiness_s=readiness_s, metadata=metadata,
        mapping_note=profile.mapping_note, camera_rename_map=CAMERA_RENAME_MAP,
        mapping_validation_basis="source conventions only; not supervised physical validation",
        physical_validation="not performed",
        **response_metadata,
    )
    return result
