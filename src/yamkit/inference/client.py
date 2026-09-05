"""Authenticated Modal SDK transport and locally timed request invalidation.

This module is safe to import without Modal, torch, credentials, or a running pool.
Every operation is explicit. Robot execution never calls this transport directly.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any


class RemoteFault(RuntimeError):
    """A remote result must not be used for motion."""

    def __init__(self, message: str, *, metrics: dict | None = None):
        super().__init__(message)
        self.metrics = metrics or {}


class InvalidatedRequest(RemoteFault):
    """A response belongs to an earlier pause/reset/stop generation."""


class ModalTransport:
    """One bounded outbound SDK call at a time; never deploys an application."""

    def __init__(self, app_name: str, profile: str, *, shutdown_event=None):
        if not app_name or len(app_name) > 100:
            raise ValueError("An explicit dedicated Modal app name is required")
        self.app_name = app_name
        self.profile = profile
        self._shutdown_event = shutdown_event
        self._busy = threading.Lock()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        # Nonblocking: the caller may be the local Stop handler.
        self._cancel.set()

    def _invoke(self, method: str, payload: dict | None, timeout_s: float) -> dict:
        if self._shutdown_event is not None and self._shutdown_event.is_set():
            raise InvalidatedRequest("Local execution is stopped")
        if not self._busy.acquire(blocking=False):
            raise RemoteFault("A previous Modal request is still in flight")
        done = threading.Event()
        cancel = threading.Event()
        self._cancel = cancel
        result: dict[str, Any] = {}
        deadline = time.monotonic() + timeout_s

        def work():
            call = None
            try:
                import modal

                service = modal.Cls.from_name(self.app_name, "PolicyService")()
                remote_method = getattr(service, method)
                if cancel.is_set():
                    return
                call = remote_method.spawn() if payload is None else remote_method.spawn(payload)
                remaining = deadline - time.monotonic()
                if cancel.is_set() or remaining <= 0:
                    call.cancel()
                    return
                result["value"] = call.get(timeout=remaining)
            except Exception:  # noqa: BLE001 — SDK exceptions may contain credentials/payloads
                # SDK exceptions may contain request details; keep secrets/payloads out of logs.
                result["error"] = True
            finally:
                if call is not None and (cancel.is_set() or time.monotonic() >= deadline):
                    try:
                        call.cancel()
                    except Exception:  # noqa: BLE001 — no server exception data reaches logs
                        result["cancel_failed"] = True
                self._busy.release()
                done.set()

        threading.Thread(target=work, daemon=True, name="yamkit-modal-request").start()
        while not done.wait(min(0.05, max(0.0, deadline - time.monotonic()))):
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                cancel.set()
            if cancel.is_set():
                raise InvalidatedRequest("Modal request invalidated locally")
            if time.monotonic() >= deadline:
                cancel.set()
                raise RemoteFault("Modal request deadline exceeded")
        if cancel.is_set():
            raise InvalidatedRequest("Modal request invalidated locally")
        if time.monotonic() >= deadline:
            raise RemoteFault("Modal request deadline exceeded")
        if result.get("error") or "value" not in result:
            raise RemoteFault("Modal SDK request failed; verify the dedicated app and server logs")
        return result["value"]

    def ready(self, timeout_s: float) -> dict:
        return self._invoke("ready", None, timeout_s)

    def predict_chunk(self, request: dict, timeout_s: float) -> dict:
        return self._invoke("predict_chunk", request, timeout_s)


class RemoteSession:
    """Strict protocol checks with a new identity after every reset.

    ``observation_time`` is generated and checked only on this host. A server's
    clock is never subtracted from it. Timing samples are bounded in memory.
    """

    def __init__(self, transport, profile, *, timeout_s: float = 10.0, max_observation_age_s: float = 2.0):
        if not 0 < timeout_s <= 120 or not 0 < max_observation_age_s <= 120:
            raise ValueError("Request and freshness deadlines must be positive and at most 120 seconds")
        self.transport = transport
        self.profile = profile
        self.timeout_s = timeout_s
        self.max_observation_age_s = max_observation_age_s
        self._lock = threading.Lock()
        self._flight_lock = threading.Lock()
        self.instance_id = None
        self.samples: deque[dict] = deque(maxlen=1000)
        self.request_count = 0
        self.failed_request_count = 0
        self.failures: deque[dict] = deque(maxlen=50)
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.session_id = uuid.uuid4().hex
            self.sequence_id = 0
            self._closed = False
        self.transport.cancel()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.session_id = uuid.uuid4().hex
        self.transport.cancel()

    def predict(self, *, state: list[float], images: dict, task: str, observation_time: float,
                crop: str = "none", mode: str = "robot") -> dict:
        from .protocol import validate_request, validate_response

        if not self._flight_lock.acquire(blocking=False):
            raise RemoteFault("Only one chunk request may be in flight per session")
        attempt_started = time.monotonic()
        self.request_count += 1
        try:
            with self._lock:
                if self._closed:
                    raise InvalidatedRequest("Remote session is stopped")
                session_id = self.session_id
                sequence_id = self.sequence_id
                self.sequence_id += 1
            now = time.monotonic()
            age = now - observation_time
            if age < 0 or age > self.max_observation_age_s:
                raise RemoteFault("Observation is stale or has an invalid local timestamp")
            request = {
                "protocol_version": 1, "profile": self.profile.id,
                "model_revision": self.profile.revision,
                "session_id": session_id, "sequence_id": sequence_id,
                "observation_time": observation_time, "observation_age_s": age,
                "timeout_s": self.timeout_s, "task": task, "state": state,
                "state_names": list(self.profile.state_names), "images": images,
                "mode": mode, "crop": crop, "continuation": None,
            }
            validate_request(request, self.profile)
            start = time.monotonic()
            response = self.transport.predict_chunk(request, self.timeout_s)
            elapsed = time.monotonic() - start
            with self._lock:
                if self._closed or self.session_id != session_id:
                    raise InvalidatedRequest("Late response rejected after pause/reset/stop")
            if elapsed >= self.timeout_s or time.monotonic() - observation_time > self.max_observation_age_s:
                raise RemoteFault("Response expired before local execution")
            validate_response(response, request, self.profile)
            if self.instance_id is not None and response.get("instance_id") != self.instance_id:
                raise RemoteFault("Remote container restarted; stop and prepare the selected profile again")
            self.samples.append({"round_trip_s": elapsed, "observation_age_s": time.monotonic() - observation_time,
                                 "payload_bytes": sum(len(im["data"]) for im in images.values()),
                                 "server_timing": response.get("timing", {})})
            return response
        except Exception as exc:
            self.failed_request_count += 1
            self.failures.append({"elapsed_s": time.monotonic() - attempt_started,
                                  "reason": type(exc).__name__})
            raise
        finally:
            self._flight_lock.release()

    def metrics(self) -> dict:
        import numpy as np

        samples = list(self.samples)
        values = [s["round_trip_s"] for s in samples]
        return {"sample_count": len(samples), "request_count": self.request_count,
                "failed_request_count": self.failed_request_count, "failures": list(self.failures),
                "first_round_trip_s": values[0] if values else None,
                "max_observation_age_s": max((s["observation_age_s"] for s in samples), default=None),
                "warm_sample_count": max(0, len(values) - 1),
                "warm_round_trip_s": dict(zip(("p50", "p95", "p99"),
                                              np.percentile(values[1:], [50, 95, 99]).tolist()))
                if len(values) > 1 else {}, "samples": samples}
