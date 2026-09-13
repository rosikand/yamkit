"""Single-flight official OpenPI HTTP client with bounded local Stop invalidation.

Only explicit loopback SSH origins are accepted. No retries, preparation,
hardware access, credential logging, or model-output projection occur here.
"""

from __future__ import annotations

import copy
import http.client
import math
import re
import threading
import time
import uuid

import numpy as np

from yamkit.inference.client import InvalidatedRequest, RemoteFault

from .contract import CHECKPOINT, MODEL_CONFIG, UPSTREAM_REVISION, identity
from .runtime import validate_normalized_chunk
from .service import (
    MAX_BODY_BYTES,
    MAX_SESSION_SECONDS,
    WIRE_VERSION,
    NativeRuntime,
    build_id,
    decode_json,
    encode_json,
)
from .yam_candidate import CANDIDATE_ID


def validate_readiness(metadata: dict, *, statistics_sha256: str,
                       expected_source_sha: str | None = None, now: float | None = None) -> None:
    now = time.time() if now is None else now
    expected = {
        "wire_version": WIRE_VERSION, "policy": "pi05-base", "checkpoint": CHECKPOINT,
        "runtime_revision": UPSTREAM_REVISION, "manifest_sha256": identity()["manifest_sha256"],
        "model_config": dict(MODEL_CONFIG), "num_inference_steps": 10,
        "service_id": "lambda-openpi", "openpi_service_build_id": build_id(),
        "adapter_id": CANDIDATE_ID, "statistics_sha256": statistics_sha256,
        "native_output_shape": [50, 32], "model_weights_modified": False,
        "native_inference_modified": False, "hardware_tested": False, "ready": True,
    }
    if type(metadata) is not dict or any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("Official OpenPI service identity differs from the pinned local interface")
    if any(type(metadata.get(key)) is not type(value) for key, value in expected.items()):
        raise ValueError("Official OpenPI service identity has ambiguous value types")
    expires = metadata.get("session_expires_at")
    if (type(expires) not in (int, float) or not math.isfinite(expires)
            or not now < expires <= now + MAX_SESSION_SECONDS
            or type(metadata.get("instance_id")) is not str
            or re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", metadata["instance_id"]) is None
            or type(metadata.get("source_sha")) is not str
            or re.fullmatch(r"[0-9a-f]{40}", metadata["source_sha"]) is None
            or type(metadata.get("statistics_file_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", metadata["statistics_file_sha256"]) is None
            or (expected_source_sha is not None and metadata["source_sha"] != expected_source_sha)):
        raise ValueError("Official OpenPI service session, source or statistics-file identity is invalid")
    image_hw = metadata.get("image_hw")
    if (type(image_hw) is not list or len(image_hw) != 2
            or any(type(value) is not int for value in image_hw)
            or not 1 <= image_hw[0] <= 720 or not 1 <= image_hw[1] <= 1280):
        raise ValueError("Official OpenPI service RGB geometry is invalid")


class OpenPiTransport:
    call_mode = "http"
    profile = "pi05-base"

    def __init__(self, *, endpoint_url: str, token: str, statistics_sha256: str,
                 shutdown_event: threading.Event | None = None, expected_source_sha: str | None = None):
        endpoint = (re.fullmatch(r"http://127\.0\.0\.1:([1-9][0-9]{0,4})", endpoint_url)
                    if type(endpoint_url) is str else None)
        if endpoint is None or not 1 <= int(endpoint[1]) <= 65535:
            raise ValueError("Official OpenPI transport requires an explicit bare loopback SSH origin")
        if type(token) is not str or re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token) is None:
            raise ValueError("Official OpenPI transport requires a bounded bearer credential")
        if type(statistics_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", statistics_sha256) is None:
            raise ValueError("Official OpenPI transport requires an explicit statistics identity")
        self._port, self._token = int(endpoint[1]), token
        self.statistics_sha256 = statistics_sha256
        self.expected_source_sha = expected_source_sha
        self._shutdown_event = shutdown_event
        self._state_lock, self._busy = threading.Lock(), threading.Lock()
        self._generation, self._sequence = 0, 0
        self._closed = False
        self._cancel_event: threading.Event | None = None
        self._deadline = None
        self._expiry_timer = None
        self._client_id = uuid.uuid4().hex
        self.metadata: dict | None = None
        self.instance_id: str | None = None
        self.last_timing: dict = {}
        self.request_sent = threading.Event()

    def __repr__(self):
        return "OpenPiTransport(policy='pi05-base', ingress='loopback-ssh')"

    def cancel(self) -> None:
        """Invalidate current work immediately; remote inference may finish unused."""
        with self._state_lock:
            self._generation += 1
            if self._cancel_event is not None:
                self._cancel_event.set()

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            self._generation += 1
            if self._cancel_event is not None:
                self._cancel_event.set()
            timer, self._expiry_timer = self._expiry_timer, None
        if timer is not None:
            timer.cancel()

    def ensure_session_active(self) -> None:
        if self._closed or (self._shutdown_event is not None and self._shutdown_event.is_set()):
            raise InvalidatedRequest("Official OpenPI execution is stopped or closed")
        if self._deadline is not None and (time.monotonic() >= self._deadline
                                           or time.time() >= self.metadata["session_expires_at"]):
            raise RemoteFault("Official OpenPI session expired")

    def _invoke(self, route: str, payload: dict | None, timeout_s: float) -> dict:
        maximum = {"/ready": 30.0, "/warm": 120.0, "/predict": 2.0}.get(route)
        if (maximum is None or type(timeout_s) not in (int, float) or not math.isfinite(timeout_s)
                or not 0 < timeout_s <= maximum):
            raise ValueError("Official OpenPI request deadline is outside its finite bound")
        started = time.monotonic()
        deadline = started + timeout_s
        if self._deadline is not None:
            deadline = min(deadline, self._deadline)
        self.ensure_session_active()
        done, cancel = threading.Event(), threading.Event()
        result = {}
        timing = {"call_mode": "http", "remote_cancellation_supported": False,
                  "network_only_s": None, "request_retry_count": 0}
        self.last_timing = timing
        with self._state_lock:
            self.ensure_session_active()
            if not self._busy.acquire(blocking=False):
                raise RemoteFault("A previous official OpenPI request is still in flight")
            generation = self._generation
            self._cancel_event = cancel
            self.request_sent.clear()

        def invalidated():
            return cancel.is_set() or (self._shutdown_event is not None and self._shutdown_event.is_set())

        def work():
            connection = None
            try:
                if invalidated() or time.monotonic() >= deadline:
                    return
                body = None if payload is None else encode_json(payload)
                encoded = time.monotonic()
                timing.update(serialization_s=encoded - started, wire_request_bytes=len(body or b""))
                if invalidated() or encoded >= deadline:
                    return
                connection = http.client.HTTPConnection("127.0.0.1", self._port, timeout=deadline - encoded)
                connection.request("GET" if payload is None else "POST", route, body=body,
                                   headers={"Authorization": "Bearer " + self._token,
                                            "Content-Type": "application/json", "Accept-Encoding": "identity"})
                timing["request_sent_monotonic_s"] = time.monotonic()
                self.request_sent.set()
                response = connection.getresponse()
                if response.status != 200:
                    result["status"] = response.status
                    return
                declared = response.getheader("Content-Length")
                if (response.getheader("Content-Type") != "application/json"
                        or response.getheader("Content-Encoding", "identity") != "identity"
                        or response.getheader("Transfer-Encoding") is not None
                        or declared is None or re.fullmatch(r"[0-9]{1,8}", declared) is None
                        or not 0 < int(declared) <= MAX_BODY_BYTES):
                    raise ValueError("Official OpenPI response framing is invalid")
                body = response.read(int(declared) + 1)
                received = time.monotonic()
                timing.update(wire_response_bytes=len(body), http_request_s=received - encoded)
                if len(body) != int(declared):
                    raise ValueError("Official OpenPI response length differs")
                if invalidated() or received >= deadline:
                    return
                envelope = decode_json(body)
                if set(envelope) != {"ok", "result"} or envelope["ok"] is not True or type(envelope["result"]) is not dict:
                    raise ValueError("Official OpenPI response envelope differs")
                result["value"] = envelope["result"]
                timing["deserialization_s"] = time.monotonic() - received
            except Exception:  # noqa: BLE001 — remote exceptions may contain credentials or untrusted content
                result["error"] = True
            finally:
                if connection is not None:
                    connection.close()
                timing["worker_total_s"] = time.monotonic() - started
                with self._state_lock:
                    if self._cancel_event is cancel:
                        self._cancel_event = None
                    self._busy.release()
                    done.set()

        try:
            threading.Thread(target=work, daemon=True, name="yamkit-openpi-http").start()
        except Exception:  # noqa: BLE001 — no untrusted thread-start details
            with self._state_lock:
                self._cancel_event = None
                self._busy.release()
            raise RemoteFault("Official OpenPI HTTP worker could not start") from None
        while not done.wait(min(0.01, max(0.0, deadline - time.monotonic()))):
            try:
                self.ensure_session_active()
            except RemoteFault:
                cancel.set()
                raise
            if invalidated():
                cancel.set()
                raise InvalidatedRequest("Official OpenPI request invalidated locally")
            if time.monotonic() >= deadline:
                cancel.set()
                raise RemoteFault("Official OpenPI request deadline exceeded")
        self.ensure_session_active()
        with self._state_lock:
            if invalidated() or generation != self._generation:
                raise InvalidatedRequest("Official OpenPI request invalidated locally")
            if time.monotonic() >= deadline:
                raise RemoteFault("Official OpenPI request deadline exceeded")
            if "value" not in result:
                status = result.get("status")
                suffix = f" (HTTP {status})" if type(status) is int else ""
                raise RemoteFault("Official OpenPI request failed" + suffix) from None
            return result["value"]

    def ready(self, timeout_s: float = 10.0) -> dict:
        value = self._invoke("/ready", None, timeout_s)
        try:
            validate_readiness(value, statistics_sha256=self.statistics_sha256,
                               expected_source_sha=self.expected_source_sha)
            if self.metadata is not None:
                # All stable identity fields, including immutable stats file bytes,
                # must remain unchanged. Task warm caches and busy can vary.
                for key in self.metadata:
                    if key not in ("warm_signatures", "busy", "runtime_provenance") and value.get(key) != self.metadata[key]:
                        raise ValueError("Official OpenPI service identity changed")
            else:
                self._deadline = time.monotonic() + value["session_expires_at"] - time.time()
                if self._shutdown_event is not None:
                    timer = threading.Timer(max(0.0, self._deadline - time.monotonic()), self._shutdown_event.set)
                    timer.daemon = True
                    timer.start()
                    self._expiry_timer = timer
            self.metadata = value
            self.instance_id = value["instance_id"]
            return value
        except (ValueError, RuntimeError):
            self.close()
            raise

    def _prediction(self, request: dict, timeout_s: float, *, warm: bool) -> dict:
        started = time.monotonic()
        if self.metadata is None:
            raise ValueError("Verify official OpenPI service identity before inference")
        if type(request) is not dict or set(request) != {"images", "state", "task"}:
            raise ValueError("Official OpenPI observation request schema differs")
        with self._state_lock:
            self._sequence += 1
            sequence = self._sequence
            generation = self._generation
        request = copy.deepcopy(request)
        payload = {**request, "wire_version": WIRE_VERSION, "client_id": self._client_id,
                   "sequence_id": sequence, "timeout_s": timeout_s, "instance_id": self.instance_id,
                   "statistics_sha256": self.statistics_sha256,
                   "openpi_service_build_id": self.metadata["openpi_service_build_id"]}
        result = self._invoke("/warm" if warm else "/predict", payload, timeout_s)
        try:
            for key, value in self.metadata.items():
                if key not in ("ready", "busy", "warm_signatures", "runtime_provenance") and result.get(key) != value:
                    raise ValueError("Official OpenPI response identity differs")
            if (result.get("client_id") != self._client_id or type(result.get("sequence_id")) is not int
                    or result["sequence_id"] != sequence or result.get("state") != request["state"]
                    or result.get("warm") is not warm
                    or result.get("warm_signature") != NativeRuntime.signature(request["task"], tuple(self.metadata["image_hw"]))
                    or result.get("raw_dtype") not in ("float32", "float64")):
                raise ValueError("Official OpenPI response correlation or measured anchor differs")
            raw = validate_normalized_chunk(np.asarray(result["raw_normalized_chunk"], dtype=result["raw_dtype"]))
            result["raw_normalized_chunk"] = raw
            self.ensure_session_active()
            with self._state_lock:
                if self._generation != generation:
                    raise InvalidatedRequest("Official OpenPI request invalidated during response validation")
            if time.monotonic() - started >= timeout_s:
                raise RemoteFault("Official OpenPI request deadline exceeded during response validation")
            return result
        except (KeyError, TypeError, ValueError):
            self.cancel()
            raise ValueError("Official OpenPI response failed native shape, anchor or identity validation") from None

    def warm(self, request: dict, timeout_s: float = 120.0) -> dict:
        return self._prediction(request, timeout_s, warm=True)

    def predict_chunk(self, request: dict, timeout_s: float = 2.0) -> dict:
        return self._prediction(request, timeout_s, warm=False)
