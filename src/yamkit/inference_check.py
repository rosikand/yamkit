"""Hardware-free fresh-chunk checks shared by CLI and UI jobs."""

from __future__ import annotations

import time
import uuid

import numpy as np

from .inference.profiles import get_profile
from .inference.protocol import native_fixture_request, validate_response


def percentiles(samples: list[float]) -> dict:
    return {"sample_count": len(samples), **({f"p{q}_s": float(np.percentile(samples, q))
            for q in (50, 95, 99)} if samples else {})}


def run_check(policy: str, *, backend: str = "local", device: str = "cpu", task: str = "pick up the object",
              steps: int = 3, modal_app: str | None = None, center_crop: bool = False) -> dict:
    if not 3 <= steps <= 20:
        raise ValueError("fresh-chunk check requires 3–20 predictions")
    profile = get_profile(policy)
    started = time.monotonic()
    if backend == "modal":
        from .modal_ops import _validate_ready, call, owned_service, service_handle

        receipt = owned_service()
        app_name = modal_app or (receipt or {}).get("app_name")
        if not app_name:
            raise ValueError("prepare a dedicated Modal service first with yamkit modal-prepare")
        service = service_handle(app_name, profile.id)
        metadata = call(service.ready, timeout=300)
        _validate_ready(metadata, profile)
        predict = lambda request: call(service.predict_chunk, request, timeout=request["timeout_s"])
    elif backend == "local":
        from .inference.service import ModelRuntime

        service = ModelRuntime.load(profile, device=device)
        metadata = service.ready()
        predict = service.predict_chunk
    else:
        raise ValueError("backend must be local or modal")
    readiness_s = time.monotonic() - started
    samples = []
    session_id = str(uuid.uuid4())
    for sequence in range(steps):
        encoded_at = time.monotonic()
        request = native_fixture_request(profile, sequence_id=sequence, session_id=session_id,
                                         crop="center_16_9" if center_crop else "none")
        request["task"] = task
        encoding_s = time.monotonic() - encoded_at
        sent = time.monotonic()
        response = predict(request)  # predict_action_chunk on every call; never select_action cached pops
        elapsed = time.monotonic() - sent
        validate_response(response, request, profile)
        chunk = np.asarray(response["chunk"])
        samples.append({"sequence_id": sequence, "round_trip_s": elapsed, "encoding_s": encoding_s,
                        "payload_bytes": sum(len(i["data"]) for i in request["images"].values()),
                        "shape": list(chunk.shape), "finite": bool(np.isfinite(chunk).all()),
                        "min": float(chunk.min()), "max": float(chunk.max()), "server": response["timing"],
                        "transforms": response.get("transforms"), "observation_age_s": elapsed + encoding_s})
    return {"profile_id": profile.id, "revision": profile.revision, "backend": backend,
            "source": "checkpoint-native synthetic fixture; no physical YAM compatibility implied",
            "action_units": "checkpoint_native", "mapping_verified": profile.mapping_verified,
            "mapping_note": profile.mapping_note, "readiness_s": readiness_s, "metadata": metadata,
            "fresh_chunks": samples, "cold": percentiles([readiness_s + samples[0]["round_trip_s"]]),
            "warm": percentiles([x["round_trip_s"] for x in samples[1:]]),
            "queue_depth": None, "underruns": None,
            "measurement_note": "Small diagnostic sample; cold includes readiness. No hardware queue exercised."}
