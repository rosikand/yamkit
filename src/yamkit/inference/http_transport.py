"""Persistent authenticated HTTP with local deadlines and generation invalidation.

Importing this module never opens a connection. Every call has one total local
deadline, including serialization and response parsing. A cancelled or timed-out
call retains the single-flight lock until its underlying HTTP worker retires.
There is no retry and no promise that local cancellation stops remote compute.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
import time
from urllib.parse import urlsplit

from .client import InvalidatedRequest, RemoteFault
from .http_wire import MAX_MESSAGE_BYTES, WIRE_CODEC, WIRE_VERSION, decode_message, encode_message


def validate_endpoint_url(value: str, *, http_ingress: str = "asgi") -> str:
    """Return a canonical HTTPS Modal origin; never include invalid input in errors."""
    if (http_ingress not in ("asgi", "tunnel") or type(value) is not str
            or not 1 <= len(value) <= 512 or not value.isascii() or value != value.strip()):
        raise ValueError("HTTP inference requires a valid HTTPS Modal endpoint")
    try:
        parts = urlsplit(value)
        hostname = parts.hostname or ""
        labels = hostname.split(".")
        provider = (len(labels) >= 3 and labels[-2:] == ["modal", "run"] if http_ingress == "asgi"
                    else len(labels) == 4 and labels[-2:] == ["modal", "host"])
        valid = (parts.scheme == "https" and parts.username is None and parts.password is None
                 and parts.port in (None, 443) and parts.path in ("", "/")
                 and not parts.query and not parts.fragment and "?" not in value and "#" not in value
                 and len(hostname) <= 253 and provider
                 and all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in labels)
                 and not any(char.isspace() for char in value))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("HTTP inference requires a bare HTTPS Modal endpoint without credentials or query")
    return "https://" + hostname


def _make_client(token: str):
    """Private factory seam: tests substitute an in-memory HTTPX transport."""
    import httpx

    limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    return httpx.Client(
        timeout=120.0, follow_redirects=False, trust_env=False,
        limits=limits, transport=httpx.HTTPTransport(retries=0, limits=limits, trust_env=False),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/octet-stream",
                 "Accept": "application/octet-stream", "Accept-Encoding": "identity"},
    )


class HttpTransport:
    """One bounded, authenticated request at a time; never deploys a service."""

    call_mode = "http"

    def __init__(self, app_name: str, profile: str, *, endpoint_url: str, token: str, shutdown_event=None,
                 http_ingress: str = "asgi", http_session_expires_at: float | None = None):
        if type(app_name) is not str or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", app_name):
            raise ValueError("An explicit bounded inference app name is required")
        if profile not in ("smolvla", "molmoact2", "pi05"):
            raise ValueError("A reviewed inference profile is required")
        self._endpoint_url = validate_endpoint_url(endpoint_url, http_ingress=http_ingress)
        from .identity import http_ingress_binding

        self._http_binding = http_ingress_binding(
            {"http_ingress": http_ingress, "http_session_expires_at": http_session_expires_at,
             "http_endpoint": self._endpoint_url}, endpoint_url=self._endpoint_url)
        self.http_ingress = http_ingress
        self.http_session_expires_at = http_session_expires_at
        self._session_deadline_monotonic = (None if http_session_expires_at is None
                                            else time.monotonic() + http_session_expires_at - time.time())
        if type(token) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token):
            raise ValueError("A bounded inference bearer credential is required")
        self.app_name, self.profile = app_name, profile
        self._token = token
        self._shutdown_event = shutdown_event
        self._busy = threading.Lock()
        self._state_lock = threading.Lock()
        self._generation = 0
        self._closed = False
        self._current_cancel: threading.Event | None = None
        self._client = None
        self.last_timing: dict = {}
        self._expiry_timer = None
        if self._session_deadline_monotonic is not None and shutdown_event is not None:
            # Startup homing already watches this same event, before the final
            # rollout dispatch guard exists. The timer never opens any resource.
            timer = threading.Timer(max(0.0, self._session_deadline_monotonic - time.monotonic()), shutdown_event.set)
            timer.daemon = True
            self._expiry_timer = timer
            try:
                timer.start()
            except RuntimeError:
                self._closed = True
                self._expiry_timer = None
                raise RemoteFault("HTTP session expiry guard could not start") from None

    def __repr__(self):
        return f"HttpTransport(app_name={self.app_name!r}, profile={self.profile!r})"

    def cancel(self) -> None:
        """Invalidate this generation without waiting for network I/O or closing the client."""
        with self._state_lock:
            self._generation += 1
            if self._current_cancel is not None:
                self._current_cancel.set()

    @staticmethod
    def _close_client(client) -> None:
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001, S110 — cleanup must not expose HTTP client or credential details
                pass

    def close(self) -> None:
        """Permanently stop this transport; never wait for HTTP or pool cleanup."""
        client = None
        with self._state_lock:
            if not self._closed:
                self._closed = True
                self._generation += 1
                if self._current_cancel is not None:
                    self._current_cancel.set()
            if not self._busy.locked():
                client, self._client = self._client, None
            timer, self._expiry_timer = self._expiry_timer, None
        if timer is not None:
            timer.cancel()
        if client is not None:
            # Even idle pool cleanup stays off the Stop caller's thread. A
            # running worker takes sole ownership of cleanup in its finally.
            try:
                threading.Thread(target=self._close_client, args=(client,), daemon=True,
                                 name="yamkit-http-close").start()
            except Exception:  # noqa: BLE001 — a later close can retry if no cleanup thread could start
                with self._state_lock:
                    self._client = client

    def _stopped(self, cancel: threading.Event) -> bool:
        return cancel.is_set() or (self._shutdown_event is not None and self._shutdown_event.is_set())

    def ensure_session_active(self) -> None:
        """Nonblocking dispatch guard; clock changes can never extend a tunnel."""
        if self._closed:
            raise InvalidatedRequest("HTTP transport is closed")
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            raise InvalidatedRequest("Local execution is stopped")
        if self._session_deadline_monotonic is not None and (
                time.monotonic() >= self._session_deadline_monotonic
                or time.time() >= self.http_session_expires_at):
            raise RemoteFault("HTTP session expired; prepare and qualify a new owned session")

    def _invoke(self, method: str, payload: dict | None, timeout_s: float) -> dict:
        begin = time.monotonic()
        maximum = 900.0 if method == "ready" else 120.0
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= maximum:
            raise ValueError("HTTP request timeout is outside its finite bounds")
        if method not in ("ready", "predict_chunk", "reset"):
            raise ValueError("Unsupported HTTP inference operation")
        if ((method == "ready" and payload is not None)
                or (method != "ready" and type(payload) is not dict)):
            raise ValueError("HTTP inference payload is malformed")
        self.ensure_session_active()
        remaining = (None if self.http_session_expires_at is None else min(
            self.http_session_expires_at - time.time(), self._session_deadline_monotonic - begin))
        deadline = begin + (timeout_s if remaining is None else min(timeout_s, remaining))
        done, cancel = threading.Event(), threading.Event()
        result = {}
        timing = {"call_mode": self.call_mode, "wire_codec": WIRE_CODEC, "wire_version": WIRE_VERSION,
                  "http_ingress": self.http_ingress, "http_session_expires_at": self.http_session_expires_at,
                  "http_endpoint_sha256": hashlib.sha256(self._endpoint_url.encode()).hexdigest(),
                  "wire_compression": "none", "remote_cancellation_supported": False,
                  "network_only_s": None, "modal_queue_s": None,
                  "note": "Measured HTTP time includes routing and network; image pixels are unchanged"}
        self.last_timing = timing
        with self._state_lock:
            if self._closed:
                raise InvalidatedRequest("HTTP transport is closed")
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                raise InvalidatedRequest("Local execution is stopped")
            if not self._busy.acquire(blocking=False):
                raise RemoteFault("A previous HTTP request is still in flight")
            generation = self._generation
            self._current_cancel = cancel

        def work():
            try:
                if self._stopped(cancel) or time.monotonic() >= deadline:
                    return
                wire = encode_message({"method": method, "payload": payload})
                encoded = time.monotonic()
                timing.update(wire_request_bytes=len(wire), serialization_s=encoded - begin,
                              client_reused=self._client is not None)
                if self._stopped(cancel) or encoded >= deadline:
                    return
                if self._client is None:
                    self._client = _make_client(self._token)
                self.ensure_session_active()
                remaining = deadline - time.monotonic()
                if self._stopped(cancel) or remaining <= 0:
                    return
                sent = time.monotonic()
                with self._client.stream("POST", self._endpoint_url + "/rpc", content=wire,
                                         timeout=remaining) as response:
                    if response.status_code != 200:
                        raise RemoteFault("HTTP inference request was rejected")
                    if (response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                            != "application/octet-stream"):
                        raise RemoteFault("HTTP inference response type is invalid")
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise RemoteFault("HTTP inference response compression is unsupported")
                    declared = response.headers.get("content-length")
                    if declared is not None and (not re.fullmatch(r"[0-9]{1,10}", declared)
                                                  or not 1 <= int(declared) <= MAX_MESSAGE_BYTES):
                        raise RemoteFault("HTTP inference response length is invalid")
                    body = bytearray()
                    # Do not return the connection to the pool while its body is
                    # in flight. Local Stop returns independently of this worker.
                    for part in response.iter_raw():
                        if len(body) + len(part) > MAX_MESSAGE_BYTES:
                            raise RemoteFault("HTTP inference response exceeds its bound")
                        body.extend(part)
                    if declared is not None and len(body) != int(declared):
                        raise RemoteFault("HTTP inference response length mismatch")
                received = time.monotonic()
                timing.update(wire_response_bytes=len(body), http_request_s=received - sent)
                if self._stopped(cancel) or received >= deadline:
                    return
                value = decode_message(bytes(body))
                timing["deserialization_s"] = time.monotonic() - received
                if type(value) is not dict or set(value) != {"ok", "result"} or value["ok"] is not True \
                        or type(value["result"]) is not dict:
                    raise RemoteFault("HTTP inference response envelope is invalid")
                result["value"] = value["result"]
            except Exception:  # noqa: BLE001 — HTTP failures may contain credentials, bodies or endpoint URLs
                result["error"] = True
            finally:
                timing["worker_total_s"] = time.monotonic() - begin
                closing_client = None
                with self._state_lock:
                    if self._current_cancel is cancel:
                        self._current_cancel = None
                    if self._closed:
                        closing_client, self._client = self._client, None
                    self._busy.release()
                    done.set()
                self._close_client(closing_client)

        try:
            threading.Thread(target=work, daemon=True, name="yamkit-http-request").start()
        except Exception:  # noqa: BLE001 — release ownership if the local worker cannot start
            with self._state_lock:
                self._current_cancel = None
                self._busy.release()
            raise RemoteFault("HTTP inference worker could not start") from None
        while not done.wait(min(0.01, max(0.0, deadline - time.monotonic()))):
            try:
                self.ensure_session_active()
            except RemoteFault:
                cancel.set()
                raise
            if self._stopped(cancel):
                cancel.set()
                raise InvalidatedRequest("HTTP request invalidated locally")
            if time.monotonic() >= deadline:
                cancel.set()
                raise RemoteFault("HTTP request deadline exceeded")
        self.last_timing = timing
        self.ensure_session_active()
        with self._state_lock:
            if self._stopped(cancel) or self._generation != generation:
                raise InvalidatedRequest("HTTP request invalidated locally")
            if time.monotonic() >= deadline:
                raise RemoteFault("HTTP request deadline exceeded")
            if result.get("error") or "value" not in result:
                raise RemoteFault("HTTP inference request failed; verify the owned service") from None
            return result["value"]

    def ready(self, timeout_s: float) -> dict:
        from .identity import http_ingress_binding

        metadata = self._invoke("ready", None, timeout_s)
        try:
            binding = http_ingress_binding(metadata, endpoint_url=self._endpoint_url)
            if binding != self._http_binding:
                raise ValueError("HTTP ingress changed")
        except (ValueError, TypeError):
            self.close()
            raise RemoteFault("HTTP readiness endpoint, ingress or session expiry changed") from None
        return metadata

    def predict_chunk(self, request: dict, timeout_s: float) -> dict:
        return self._invoke("predict_chunk", request, timeout_s)

    def reset(self, session_id: str, timeout_s: float = 5.0) -> dict:
        return self._invoke("reset", {"session_id": session_id}, timeout_s)
