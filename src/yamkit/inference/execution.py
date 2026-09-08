"""Stable execution identities and request shapes shared by clients and runtimes."""

from __future__ import annotations

import hashlib
import json

from .profiles import ModelProfile, get_profile

EXECUTION_MODES = ("eager", "cuda_graph10")


def validate_execution_mode(mode: str, profile: str | ModelProfile, device: str | None = None) -> str:
    if mode not in EXECUTION_MODES:
        raise ValueError("Execution mode must be eager or cuda_graph10")
    profile = get_profile(profile)
    if mode == "cuda_graph10" and (profile.id != "molmoact2" or device is not None and not device.startswith("cuda")):
        raise ValueError("cuda_graph10 requires the reviewed MolmoAct2 profile on CUDA")
    return mode


def signature_digest(value: dict | tuple) -> str:
    """Hash JSON-compatible identities without including image or state values."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def request_execution_signature(request: dict, profile: str | ModelProfile, *,
                                execution_mode: str = "cuda_graph10") -> dict:
    """Canonicalize native warm-up and robot observations to the same input contract.

    Callers first validate the wire request. Image bytes and state values must be
    allowed to change on graph replay; task, transformations and ordered shapes
    must match the synthetic request warmed before hardware connects.
    """
    profile = get_profile(profile)
    validate_execution_mode(execution_mode, profile)
    names = profile.native_image_keys if request.get("mode", "robot") == "native_fixture" else profile.image_keys
    images = []
    for name, native_name in zip(names, profile.native_image_keys, strict=True):
        image = request["images"][name]
        images.append({"name": native_name, "height": image["height"], "width": image["width"],
                       "encoding": image["encoding"], "quality": image.get("quality")})
    return {"version": 1, "execution_mode": execution_mode, "profile": profile.id,
            "model_revision": profile.revision, "task": request["task"], "crop": request.get("crop", "none"),
            "state_width": len(request["state"]), "images": images}
