"""One expiring TLS tunnel and Uvicorn thread for an already loaded runtime.

Neither a stopped HTTP listener nor a timed thread join proves that model worker
threads stopped. The owner must still retire and verify the exact Modal container.
"""

from __future__ import annotations

import math
import threading
import time

STARTUP_TIMEOUT_S = 10.0
REVOKE_TIMEOUT_S = 2.0
JOIN_TIMEOUT_S = 3.0


class HttpTunnelServer:
    def __init__(self, application, *, expires_at: float):
        if type(expires_at) not in (int, float) or not math.isfinite(expires_at) or expires_at <= time.time():
            raise ValueError("A tunnel requires a future finite session expiry")
        self.application, self.expires_at = application, expires_at
        self.endpoint = None
        self.server = self.thread = self._context = self._timer = None
        self._close_lock = threading.Lock()
        self._revoke_started = False
        self._revoked = threading.Event()
        self._revoke_done = threading.Event()
        self._server_failed = False
        self._thread_started = False

    def start(self):
        import modal
        import uvicorn

        from .http_transport import validate_endpoint_url

        if self.server is not None:
            raise RuntimeError("The HTTP tunnel may only start once")
        try:
            self.server = uvicorn.Server(uvicorn.Config(
                self.application, host="0.0.0.0", port=8000, workers=1, loop="asyncio", http="h11",
                access_log=False, log_config=None, log_level="critical", proxy_headers=False,
                timeout_keep_alive=60, timeout_graceful_shutdown=2,
            ))

            def serve():
                try:
                    self.server.run()
                except BaseException:  # noqa: BLE001 — a worker must not log ASGI/model or credential details
                    self._server_failed = True

            self.thread = threading.Thread(target=serve, daemon=True, name="yamkit-http-tunnel")
            self.thread.start()
            self._thread_started = True
            limit = time.monotonic() + min(STARTUP_TIMEOUT_S, max(0, self.expires_at - time.time()))
            while not self.server.started:
                if not self.thread.is_alive() or self._server_failed or time.monotonic() >= limit:
                    raise RuntimeError("HTTP tunnel listener did not start within its bound")
                time.sleep(0.01)
            if time.time() >= self.expires_at:
                raise RuntimeError("HTTP tunnel session expired during startup")
            self._context = modal.forward(8000, unencrypted=False, h2_enabled=False)
            tunnel = self._context.__enter__()
            self.endpoint = validate_endpoint_url(tunnel.url, http_ingress="tunnel")
            self.ensure_running()

            def expire():
                try:
                    self.close()
                except RuntimeError:
                    # The owner verifies container retirement independently.
                    pass

            self._timer = threading.Timer(max(0.0, self.expires_at - time.time()), expire)
            self._timer.daemon = True
            self._timer.start()
            return self
        except BaseException:  # noqa: BLE001 — every failed startup must retire its partial resources
            try:
                self.close()
            except RuntimeError:
                pass
            raise RuntimeError("HTTP tunnel startup failed; container retirement is required") from None

    def ensure_running(self):
        if (time.time() >= self.expires_at or self.endpoint is None or self._server_failed
                or self.thread is None or not self.thread.is_alive()
                or not self.server.started or self.server.should_exit or self._revoke_started):
            raise RuntimeError("The owned HTTP tunnel is unavailable or expired")

    def close(self) -> dict:
        """Revoke first, then request shutdown; never claim unverified cleanup."""
        if not self._close_lock.acquire(blocking=False):
            raise RuntimeError("HTTP tunnel cleanup is still in progress")
        try:
            if self._timer is not None:
                self._timer.cancel()
            try:
                if self._context is not None and not self._revoke_started:
                    self._revoke_started = True

                    def revoke():
                        try:
                            self._context.__exit__(None, None, None)
                        except BaseException:  # noqa: BLE001, S110 — SDK details remain private
                            pass
                        else:
                            self._revoked.set()
                        finally:
                            self._revoke_done.set()

                    threading.Thread(target=revoke, daemon=True, name="yamkit-http-revoke").start()
                if self._context is not None:
                    self._revoke_done.wait(REVOKE_TIMEOUT_S)
            except Exception:  # noqa: BLE001 — listener shutdown must still run if revocation cannot start
                self._revoke_done.set()
            finally:
                # Failed or slow tunnel revocation cannot skip listener shutdown.
                if self.server is not None:
                    self.server.should_exit = True
                if self.thread is not None and self._thread_started:
                    self.thread.join(timeout=JOIN_TIMEOUT_S)
            result = {"tunnel_revoked": self._context is None or self._revoked.is_set(),
                      "server_thread_stopped": self.thread is None or not self.thread.is_alive()}
            if not all(result.values()):
                raise RuntimeError("HTTP tunnel cleanup is unverified; retire the exact Modal container")
            return result
        finally:
            self._close_lock.release()
