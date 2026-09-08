"""Content identity for the deployed inference boundary and its local consumer."""

from __future__ import annotations

import copy
import hashlib
import math
import time
from pathlib import Path


def inference_build_id() -> str:
    """Bind evidence to source contents, including uncommitted development changes.

    The Modal image contains the same package and pinned requirement file. Git
    metadata, credentials, recordings and machine-specific configuration are not
    included. Reading this identity never imports a model or opens hardware.
    """
    from yamkit.paths import ROOT

    package = Path(__file__).resolve().parent.parent
    files = [*package.joinpath("inference").glob("*.py"),
             *package.joinpath("remote_policy").glob("*.py")]
    files += [package / name for name in (
        "remote_rollout.py", "deployment.py", "modal_ops.py", "modal_qualification.py")]
    entries = [(str(path.relative_to(package)), path) for path in files]
    entries.append(("configs/modal-requirements.txt", Path(ROOT) / "configs/modal-requirements.txt"))
    digest = hashlib.sha256(b"yamkit-inference-build-v1\0")
    for name, path in sorted(entries):
        content = path.read_bytes()
        digest.update(name.encode() + b"\0" + hashlib.sha256(content).digest())
    return digest.hexdigest()


def http_ingress_binding(metadata: dict, *, endpoint_url: str | None = None) -> dict:
    """Bind an explicit tunnel to its exact origin and finite owned lifetime.

    Older ASGI readiness may omit ingress metadata. Tunnel readiness must carry
    every field; a tunnel cannot silently inherit ASGI's unbounded lifetime.
    """
    from .http_transport import validate_endpoint_url

    if not isinstance(metadata, dict):
        raise ValueError("HTTP ingress readiness must be a mapping")  # noqa: TRY004 — uniform validation error
    ingress = metadata.get("http_ingress", "asgi")
    if ingress not in ("asgi", "tunnel"):
        raise ValueError("Unsupported HTTP ingress")
    expires = metadata.get("http_session_expires_at")
    if ingress == "tunnel":
        now = time.time()
        try:
            valid_expiry = type(expires) in (int, float) and math.isfinite(expires) and now < expires <= now + 900
        except OverflowError:
            valid_expiry = False
        if not valid_expiry or not metadata.get("http_endpoint"):
            raise ValueError("HTTP tunnel requires an unexpired bounded session and explicit endpoint")
    elif expires is not None:
        raise ValueError("ASGI readiness cannot inherit a tunnel session expiry")
    advertised = metadata.get("http_endpoint")
    if advertised is not None:
        if validate_endpoint_url(advertised, http_ingress=ingress) != advertised:
            raise ValueError("HTTP readiness endpoint must be canonical")
        if endpoint_url is not None and advertised != validate_endpoint_url(endpoint_url, http_ingress=ingress):
            raise ValueError("HTTP readiness endpoint differs from the requested origin")
    endpoint = endpoint_url if endpoint_url is not None else advertised
    result = {"http_ingress": ingress, "http_session_expires_at": expires}
    if endpoint is not None:
        result["http_endpoint"] = validate_endpoint_url(endpoint, http_ingress=ingress)
    return result


def http_runtime_binding(profile, metadata: dict, *, execution_mode: str, task: str,
                         image_hw, crop: str = "none", image_encoding: str = "rgb8",
                         jpeg_quality: int = 85, require_warmup: bool = True,
                         endpoint_url: str | None = None) -> dict:
    """Check the reviewed HTTP runtime independently of a saved qualification."""
    from .execution import request_execution_signature, signature_digest
    from .http_wire import WIRE_CODEC, WIRE_VERSION
    from .profiles import get_profile
    from .protocol import MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH

    profile = get_profile(profile)
    if (profile.id != "molmoact2" or execution_mode not in ("eager", "cuda_graph10")
            or not isinstance(task, str) or not task.strip() or len(task) > 2048
            or image_encoding != "rgb8"):
        raise ValueError("Reviewed HTTP execution requires MolmoAct2, raw RGB and an explicit task")
    if (not isinstance(image_hw, (list, tuple)) or len(image_hw) != 2
            or any(type(value) is not int for value in image_hw)
            or not 1 <= image_hw[0] <= MAX_IMAGE_HEIGHT or not 1 <= image_hw[1] <= MAX_IMAGE_WIDTH
            or crop not in ("none", "center_16_9")):
        raise ValueError("HTTP runtime requires bounded image dimensions and an explicit supported crop")
    if (not isinstance(metadata, dict) or metadata.get("transport") != "http"
            or metadata.get("execution_mode") != execution_mode
            or metadata.get("inference_build_id") != inference_build_id()
            or type(metadata.get("http_wire_version")) is not int
            or metadata.get("http_wire_version") != WIRE_VERSION
            or metadata.get("http_wire_codec") != WIRE_CODEC):
        raise ValueError("HTTP runtime code, execution mode or wire format differs from this checkout")
    identity = metadata.get("execution_identity")
    expected = {"version": 1, "execution_mode": execution_mode, "profile": profile.id,
                "model_revision": profile.revision, "model_dtype": "bfloat16", "num_inference_steps": 10,
                "cuda_graph": execution_mode == "cuda_graph10", "chunk_size": 30, "action_width": 14,
                "parameter_dtype_numel": {"torch.bfloat16": 5442196208}}
    if (identity != expected or any(type(identity.get(key)) is not type(value) for key, value in expected.items())
            or type(identity["parameter_dtype_numel"]["torch.bfloat16"]) is not int):
        raise ValueError("HTTP execution does not match the pinned Molmo model, dtype, steps or action shape")
    instance = metadata.get("instance_id")
    if not isinstance(instance, str) or not 1 <= len(instance) <= 128:
        raise ValueError("HTTP readiness requires a bounded container instance identity")
    result = {"execution_mode": execution_mode, "execution_identity": copy.deepcopy(identity),
              "inference_build_id": metadata["inference_build_id"], "http_wire_version": WIRE_VERSION,
              "http_wire_codec": WIRE_CODEC, "instance_id": instance, "task": task}
    result.update(http_ingress_binding(metadata, endpoint_url=endpoint_url))
    execution = metadata.get("model_execution") or {}
    graph_enabled = execution_mode == "cuda_graph10"
    if (not isinstance(execution, dict) or execution.get("configured_model_dtype") != "bfloat16"
            or execution.get("default_num_inference_steps") != 10
            or execution.get("parameter_dtype_numel") != expected["parameter_dtype_numel"]
            or execution.get("production_cuda_graph_configured") is not graph_enabled
            or execution.get("cuda_graph_enabled") is not graph_enabled):
        raise ValueError("HTTP runtime execution settings changed")
    if graph_enabled:
        request = {"mode": "native_fixture", "task": task, "crop": crop, "state": [0.0] * 14,
                   "images": {name: {"height": image_hw[0], "width": image_hw[1], "encoding": image_encoding,
                                     **({"quality": jpeg_quality} if image_encoding == "jpeg" else {})}
                              for name in profile.native_image_keys}}
        signature = request_execution_signature(request, profile)
        expected_hash = signature_digest(signature)
        warm = metadata.get("graph_warmup") or {}
        if not isinstance(warm, dict):
            raise ValueError("HTTP graph warmup metadata is malformed")
        if require_warmup and (warm.get("ready") is not True or warm.get("signature") != signature
                               or warm.get("signature_sha256") != expected_hash
                               or not isinstance(warm.get("cache_key_sha256"), str)
                               or len(warm["cache_key_sha256"]) != 64
                               or any(character not in "0123456789abcdef" for character in warm["cache_key_sha256"])):
            raise ValueError("The actual task and image shape have not been warmed on this HTTP runtime")
        result.update(graph_signature_sha256=expected_hash, graph_cache_key_sha256=warm.get("cache_key_sha256"))
    return result
