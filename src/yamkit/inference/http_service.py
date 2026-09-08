"""Authenticated, bounded ASGI adapter for the same serialized model runtime.

The endpoint accepts inert binary messages only. Runtime work runs in a worker
thread so model execution cannot stall the ASGI event loop.
Disconnect or request cancellation never releases ownership of unfinished work.
"""

from __future__ import annotations

import asyncio
import hmac
import re
import threading
from collections.abc import Callable
from typing import Any

from .http_wire import MAX_WIRE_BYTES, decode_message, encode_message

HTTP_TOKEN_ENV = "YAMKIT_HTTP_TOKEN"


def validate_http_token(token: str) -> None:
    """Require an independently generated bearer secret, without echoing it."""
    if type(token) is not str or re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token) is None:
        raise ValueError("HTTP inference requires a dedicated 32–256 character bearer token")


def _validate_envelope(message: dict) -> tuple[str, dict | None]:
    if set(message) != {"method", "payload"}:
        raise ValueError("Invalid RPC envelope")
    method, payload = message["method"], message["payload"]
    if method == "ready" and payload is None:
        return method, payload
    if method == "predict_chunk" and type(payload) is dict and payload:
        return method, payload
    if (method == "reset" and type(payload) is dict and set(payload) == {"session_id"}
            and type(payload["session_id"]) is str and 1 <= len(payload["session_id"]) <= 128):
        return method, payload
    raise ValueError("Invalid RPC method or payload")


def create_http_app(runtime: Any, *, token: str, ready: Callable[[], dict] | None = None,
                    request_timeout_s: float = 120):
    """Serve ``POST /rpc`` with one active runtime call and no credential logs.

    The caller supplies the already loaded runtime, so this adds no second model
    instance or GPU pool. Errors contain fixed categories, never runtime text.
    """
    validate_http_token(token)
    if (type(request_timeout_s) not in (int, float) or not 0 < request_timeout_s <= 120):
        raise ValueError("HTTP inference requires a finite timeout of at most 120 seconds")
    expected = ("Bearer " + token).encode("ascii")
    ready = ready or runtime.ready
    busy = threading.Lock()
    workers = set()

    def worker_finished(worker):
        workers.discard(worker)
        if not worker.cancelled():
            # A cancelled HTTP await must not produce an unobserved background
            # exception containing model input or credentials in server logs.
            worker.exception()

    async def respond(send, status: int, value: dict):
        body = encode_message(value)
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/octet-stream"),
                                (b"cache-control", b"no-store"),
                                (b"content-length", str(len(body)).encode("ascii")),
                                (b"x-content-type-options", b"nosniff")]})
        await send({"type": "http.response.body", "body": body})

    async def reject(send, status: int, category: str = "Rejected"):
        await respond(send, status, {"ok": False, "error_type": category})

    def invoke(method: str, payload: dict | None) -> bytes:
        try:
            if method == "ready":
                result = ready()
            elif method == "predict_chunk":
                result = runtime.predict_chunk(payload)
            else:
                runtime.reset(payload["session_id"])
                result = {}
            if type(result) is not dict:
                raise ValueError("Invalid runtime result")
            # Encoding is bounded and remains off the event loop as well.
            return encode_message({"ok": True, "result": result})
        finally:
            busy.release()

    async def application(scope, receive, send):
        if scope["type"] == "lifespan":
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif event["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":
            return
        headers = scope.get("headers", [])
        authorization = [value for name, value in headers if name.lower() == b"authorization"]
        # In particular, do not ask ASGI to receive even one body byte before auth.
        if len(authorization) != 1 or not hmac.compare_digest(authorization[0], expected):
            await reject(send, 401)
            return
        if scope["method"] != "POST" or scope["path"] != "/rpc" or scope.get("query_string", b""):
            await reject(send, 404)
            return
        lengths = [value for name, value in headers if name.lower() == b"content-length"]
        if len(lengths) > 1 or (lengths and (not lengths[0].isdigit() or len(lengths[0]) > 10)):
            await reject(send, 400)
            return
        if lengths and int(lengths[0]) > MAX_WIRE_BYTES:
            await reject(send, 413)
            return
        body = bytearray()
        try:
            async with asyncio.timeout(request_timeout_s):
                while True:
                    event = await receive()
                    if event["type"] == "http.disconnect":
                        return
                    if event["type"] != "http.request":
                        await reject(send, 400)
                        return
                    part = event.get("body", b"")
                    if len(body) + len(part) > MAX_WIRE_BYTES:
                        await reject(send, 413)
                        return
                    body.extend(part)
                    if not event.get("more_body", False):
                        break
                if lengths and len(body) != int(lengths[0]):
                    await reject(send, 400)
                    return
                try:
                    method, payload = _validate_envelope(decode_message(bytes(body)))
                except ValueError:
                    await reject(send, 400)
                    return
                if not busy.acquire(blocking=False):
                    await reject(send, 409, "Busy")
                    return
                # The worker owns release. A timeout/cancel may retire this await,
                # but cannot admit another model call before the old worker ends.
                worker = asyncio.create_task(asyncio.to_thread(invoke, method, payload))
                workers.add(worker)
                worker.add_done_callback(worker_finished)
                try:
                    # Shield even a queued executor submission: cancelling it
                    # before invoke starts would otherwise strand the busy lock.
                    response = await asyncio.shield(worker)
                except Exception:  # noqa: BLE001 — RPC boundary must never expose runtime exception text.
                    await reject(send, 500, "Failed")
                    return
        except TimeoutError:
            await reject(send, 504, "Deadline")
            return
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/octet-stream"),
                                (b"cache-control", b"no-store"),
                                (b"content-length", str(len(response)).encode("ascii")),
                                (b"x-content-type-options", b"nosniff")]})
        await send({"type": "http.response.body", "body": response})

    return application
