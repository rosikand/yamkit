"""Bounded authenticated-SDK payload schema; no public endpoint or clock subtraction."""

from __future__ import annotations

import math
import time
import uuid
import warnings
from io import BytesIO
from typing import Any

import numpy as np

from .mapping import validate_order
from .profiles import ModelProfile, get_profile

PROTOCOL_VERSION = 1
MAX_IMAGES = 3
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 720
MAX_PAYLOAD_BYTES = MAX_IMAGES * MAX_IMAGE_WIDTH * MAX_IMAGE_HEIGHT * 3
MAX_TIMEOUT_S = 120.0
MAX_OBSERVATION_AGE_S = 2.0
MAX_TASK_CHARS = 2048
DEFAULT_IMAGE_ENCODING = "jpeg"
DEFAULT_JPEG_QUALITY = 85
IMAGE_ENCODINGS = ("jpeg", "rgb8")


class ProtocolError(ValueError):
    """Invalid wire data; consistently a validation error for caller handling."""


def _number(value: Any, name: str, minimum: float = 0, maximum: float = float("inf")) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a finite number")
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ProtocolError(f"{name} is outside its finite bounds")
    return float(value)


def encode_image(image: np.ndarray, *, encoding: str = DEFAULT_IMAGE_ENCODING,
                 quality: int = DEFAULT_JPEG_QUALITY) -> dict:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ProtocolError("Expected HWC uint8 RGB image")
    h, w = image.shape[:2]
    if not 1 <= h <= MAX_IMAGE_HEIGHT or not 1 <= w <= MAX_IMAGE_WIDTH:
        raise ProtocolError("RGB dimensions exceed protocol limits")
    if encoding not in IMAGE_ENCODINGS:
        raise ProtocolError("Unsupported image encoding")
    result = {"encoding": encoding, "height": int(h), "width": int(w)}
    if encoding == "rgb8":
        result["data"] = np.ascontiguousarray(image).tobytes()
    else:
        from PIL import Image

        if type(quality) is not int or not 1 <= quality <= 100:
            raise ProtocolError("JPEG quality must be an integer from 1 to 100")
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=quality, subsampling=2)
        result.update(data=buffer.getvalue(), quality=quality)
    _validate_image(result)
    return result


def _validate_image(payload: dict) -> None:
    """Validate the envelope and JPEG header without decoding the pixels twice."""
    if not isinstance(payload, dict):
        raise ProtocolError("Malformed RGB payload")
    encoding = payload.get("encoding")
    expected = {"encoding", "height", "width", "data"}
    if encoding == "jpeg":
        expected.add("quality")
    if set(payload) != expected or encoding not in IMAGE_ENCODINGS:
        raise ProtocolError("Malformed image payload or unsupported encoding")
    h, w = payload["height"], payload["width"]
    if type(h) is not int or type(w) is not int or not 1 <= h <= MAX_IMAGE_HEIGHT or not 1 <= w <= MAX_IMAGE_WIDTH:
        raise ProtocolError("RGB dimensions exceed protocol limits")
    data = payload["data"]
    if not isinstance(data, bytes) or not 1 <= len(data) <= MAX_PAYLOAD_BYTES:
        raise ProtocolError("Image byte count exceeds protocol limits")
    if encoding == "rgb8":
        if len(data) != h * w * 3:
            raise ProtocolError("RGB encoding or byte count mismatch")
        return
    if type(payload["quality"]) is not int or not 1 <= payload["quality"] <= 100:
        raise ProtocolError("JPEG quality must be an integer from 1 to 100")
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format != "JPEG" or image.mode != "RGB" or image.size != (w, h):
                    raise ProtocolError("JPEG format or dimensions do not match its envelope")
    except (OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ProtocolError("Invalid JPEG image") from exc


def decode_image(payload: dict) -> np.ndarray:
    _validate_image(payload)
    if payload["encoding"] == "rgb8":
        return np.frombuffer(payload["data"], dtype=np.uint8).reshape(payload["height"], payload["width"], 3)
    from PIL import Image

    try:
        with Image.open(BytesIO(payload["data"])) as image:
            return np.array(image, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise ProtocolError("Invalid JPEG image") from exc


def validate_request(request: dict, profile: ModelProfile) -> None:
    if not isinstance(request, dict):
        raise ProtocolError("Request must be a dictionary")
    if type(request.get("protocol_version")) is not int or request.get("protocol_version") != PROTOCOL_VERSION \
            or request.get("profile") != profile.id:
        raise ProtocolError("Protocol/profile mismatch")
    if request.get("model_revision") != profile.revision:
        raise ProtocolError("Model revision mismatch")
    session = request.get("session_id")
    if not isinstance(session, str) or not 1 <= len(session) <= 128 or not session.isascii():
        raise ProtocolError("Invalid session id")
    sequence = request.get("sequence_id")
    if type(sequence) is not int or not 0 <= sequence < 2**53:
        raise ProtocolError("Invalid sequence id")
    mode = request.get("mode", "robot")
    if mode not in ("robot", "saved_probe", "live_probe", "native_fixture"):
        raise ProtocolError("Unknown inference mode")
    if mode != "native_fixture":
        profile.require_robot_mapping()
    _number(request.get("observation_time"), "observation_time")
    _number(request.get("observation_age_s"), "observation_age_s", maximum=MAX_OBSERVATION_AGE_S
            if mode in ("robot", "live_probe") else float("inf"))
    _number(request.get("timeout_s"), "timeout_s", minimum=0.01, maximum=MAX_TIMEOUT_S)
    task = request.get("task")
    if not isinstance(task, str) or not task.strip() or len(task) > MAX_TASK_CHARS:
        raise ProtocolError("Task must be nonempty and at most 2048 characters")
    validate_order(request.get("state_names", []), profile.state_names)
    state = request.get("state")
    if not isinstance(state, list) or len(state) != len(profile.state_names):
        raise ProtocolError("State dimension mismatch; physical vectors are never padded/truncated")
    for value in state:
        _number(value, "state", minimum=-float("inf"))
    keys = profile.native_image_keys if mode == "native_fixture" else profile.image_keys
    images = request.get("images")
    if not isinstance(images, dict) or set(images) != set(keys) or len(images) > MAX_IMAGES:
        raise ProtocolError(f"Image names must be exactly {keys}")
    payload_bytes = 0
    for image in images.values():
        _validate_image(image)
        payload_bytes += len(image["data"])
    if payload_bytes > MAX_PAYLOAD_BYTES:
        raise ProtocolError("Image payload exceeds request bound")
    if request.get("crop", "none") not in ("none", "center_16_9"):
        raise ProtocolError("Unsupported crop")
    # Raw/relative/normalized prefixes and anchor conversion have not been qualified for v0.
    if request.get("continuation") is not None:
        raise ProtocolError("Guided RTC/prefix continuation is unsupported; use unguided async")
    if request.get("diagnostic_seed") is not None:
        seed = request["diagnostic_seed"]
        if mode != "native_fixture" or profile.id != "molmoact2" or type(seed) is not int or not 0 <= seed < 2**32:
            raise ProtocolError("diagnostic_seed is a uint32 available only for Molmo native fixtures")


def validate_response(response: dict, request: dict, profile: ModelProfile) -> None:
    if not isinstance(response, dict):
        raise ProtocolError("Response must be a dictionary")
    for key in ("protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time"):
        if response.get(key) != request[key]:
            raise ProtocolError(f"Response {key} mismatch")
    if request.get("diagnostic_seed") is not None and response.get("diagnostic_seed") != request["diagnostic_seed"]:
        raise ProtocolError("Response diagnostic seed mismatch")
    units = "checkpoint_native" if request.get("mode", "robot") == "native_fixture" else "robot"
    if response.get("action_units") != units:
        raise ProtocolError("Action units mismatch; numerical processing must occur exactly once")
    validate_order(response.get("action_names", []), profile.action_names)
    rows = response.get("chunk")
    if not isinstance(rows, list) or not 1 <= len(rows) <= profile.chunk_size or any(
        not isinstance(row, list) or len(row) != len(profile.action_names) for row in rows
    ):
        raise ProtocolError("Malformed action chunk shape")
    for row in rows:
        for value in row:
            _number(value, "action chunk", minimum=-float("inf"))
    chunk = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(chunk).all():
        raise ProtocolError("Nonfinite action chunk")
    timing = response.get("timing")
    if not isinstance(timing, dict):
        raise ProtocolError("Missing server timing")
    for key in ("preprocess_s", "inference_s", "postprocess_s", "total_s"):
        _number(timing.get(key), key)


def native_fixture_request(profile: str | ModelProfile, *, sequence_id: int = 0,
                           session_id: str | None = None, crop: str = "none",
                           encoding: str = DEFAULT_IMAGE_ENCODING, quality: int = DEFAULT_JPEG_QUALITY,
                           diagnostic_seed: int | None = None) -> dict:
    """A diagnostic fixture, never a robot observation and never executable as a rollout chunk."""
    profile = get_profile(profile)
    h, w = profile.native_image_hw
    rng = np.random.default_rng(sequence_id)
    return {
        "protocol_version": PROTOCOL_VERSION, "profile": profile.id, "model_revision": profile.revision,
        "session_id": session_id or str(uuid.uuid4()), "sequence_id": sequence_id,
        "observation_time": time.monotonic(), "observation_age_s": 0.0, "timeout_s": MAX_TIMEOUT_S,
        "task": "pick up the red cube", "state": [0.0] * len(profile.state_names),
        "state_names": list(profile.state_names), "images": {
            key: encode_image(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), encoding=encoding, quality=quality)
            for key in profile.native_image_keys
        }, "mode": "native_fixture", "crop": crop, "continuation": None, "diagnostic_seed": diagnostic_seed,
    }
