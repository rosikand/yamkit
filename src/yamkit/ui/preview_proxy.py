"""Bounded, authenticated HTTP access to the current child's loopback preview server."""

from __future__ import annotations

import asyncio
import http.client
import json
import math
import socket
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import quote

from starlette.responses import StreamingResponse

from .sessions import PreviewRegistration

MJPEG_MEDIA_TYPE = "multipart/x-mixed-replace; boundary=yamkitframe"
TOKEN_HEADER = "X-Yamkit-Preview-Token"
CONNECT_TIMEOUT_S = 1.0
READ_TIMEOUT_S = 2.0
REQUEST_TIMEOUT_S = 3.0
SEND_TIMEOUT_S = 5.0
MAX_STATUS_BYTES = 65536
STREAM_CHUNK_BYTES = 65536
_stream_permits = threading.BoundedSemaphore(12)
_status_permits = threading.BoundedSemaphore(4)


class PreviewUnavailable(RuntimeError):
    """The child stream is unavailable; direct capture must remain suspended."""


def _shutdown(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass


def _connect(
    reg: PreviewRegistration, path: str, is_current: Callable[[PreviewRegistration], bool],
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse, socket.socket]:
    if type(reg.port) is not int or not 1 <= reg.port <= 65535 or not is_current(reg):
        raise PreviewUnavailable("preview session changed")
    # Host/path are never read from a registration URL or a request parameter. HTTPConnection
    # does not follow redirects and does not read HTTP_PROXY environment settings.
    conn = http.client.HTTPConnection("127.0.0.1", reg.port, timeout=CONNECT_TIMEOUT_S)
    timer = None
    try:
        conn.connect()
        assert conn.sock is not None
        sock = conn.sock
        sock.settimeout(READ_TIMEOUT_S)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, STREAM_CHUNK_BYTES)
        timer = threading.Timer(REQUEST_TIMEOUT_S, _shutdown, args=(sock,))
        timer.daemon = True
        timer.start()
        conn.request("GET", path, headers={TOKEN_HEADER: reg.token, "Connection": "close"})
        response = conn.getresponse()
        if response.status != 200 or not is_current(reg):
            response.close()
            raise PreviewUnavailable("preview endpoint unavailable")
        return conn, response, sock
    except Exception as exc:
        conn.close()
        if isinstance(exc, PreviewUnavailable):
            raise
        raise PreviewUnavailable("preview connection failed") from None
    finally:
        if timer:
            timer.cancel()


def fetch_status(
    reg: PreviewRegistration, is_current: Callable[[PreviewRegistration], bool],
) -> dict[str, dict[str, Any]]:
    """Read a size-limited status response; return only known cameras and safe public fields."""
    if not _status_permits.acquire(blocking=False):
        raise PreviewUnavailable("preview status busy")
    conn = response = timer = None
    started = time.monotonic()
    try:
        conn, response, sock = _connect(reg, "/status", is_current)
        timer = threading.Timer(max(0.01, REQUEST_TIMEOUT_S - (time.monotonic() - started)), _shutdown, args=(sock,))
        timer.daemon = True
        timer.start()
        if response.getheader("Content-Type", "").split(";", 1)[0] != "application/json":
            raise PreviewUnavailable("invalid preview status")
        length = response.getheader("Content-Length")
        if length is None or not length.isdigit() or int(length) > MAX_STATUS_BYTES:
            raise PreviewUnavailable("invalid preview status size")
        body = response.read(MAX_STATUS_BYTES + 1)
        if len(body) > MAX_STATUS_BYTES or not is_current(reg):
            raise PreviewUnavailable("preview session changed")
        payload = json.loads(body)
        if not isinstance(payload, dict) or payload.get("v") != 1 or payload.get("session") != reg.session:
            raise PreviewUnavailable("invalid preview session")
        cameras = payload.get("cameras")
        if not isinstance(cameras, dict):
            raise PreviewUnavailable("invalid preview status")
        public = {}
        for name in reg.cameras:
            status = cameras.get(name)
            if not isinstance(status, dict):
                continue
            state = status.get("state")
            if not isinstance(state, str) or state not in {"idle", "waiting", "live", "stale", "unavailable"}:
                state = "unavailable"
            clean: dict[str, Any] = {"state": state}
            for key in ("seq", "source_seq", "age_s", "viewers", "encoded", "dropped", "errors", "fps"):
                value = status.get(key)
                if type(value) in (int, float) and 0 <= value <= 1e15 and math.isfinite(value):
                    clean[key] = value + (time.monotonic() - started) if key == "age_s" else value
                elif key == "age_s":
                    clean[key] = None
            public[name] = clean
        return public
    except (OSError, ValueError, RecursionError, http.client.HTTPException):
        raise PreviewUnavailable("preview status unavailable") from None
    finally:
        if timer is not None:
            timer.cancel()
        if response is not None:
            response.close()
        if conn is not None:
            conn.close()
        _status_permits.release()


class PreviewStream(Iterator[bytes]):
    """One bounded upstream socket; clients pull chunks without application queues."""

    def __init__(
        self, reg: PreviewRegistration, conn: http.client.HTTPConnection,
        response: http.client.HTTPResponse, sock: socket.socket,
        is_current: Callable[[PreviewRegistration], bool],
    ) -> None:
        self._reg = reg
        self._conn = conn
        self._response = response
        self._socket = sock
        self._is_current = is_current
        self._closed = False
        self._lock = threading.Lock()

    def __iter__(self) -> PreviewStream:
        return self

    def __next__(self) -> bytes:
        if self._closed or not self._is_current(self._reg):
            self.close()
            raise StopIteration
        try:
            chunk = self._response.read1(STREAM_CHUNK_BYTES)
        except (OSError, ValueError, http.client.HTTPException):
            chunk = b""
        if not chunk or self._closed or not self._is_current(self._reg):
            self.close()
            raise StopIteration
        return chunk

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            _shutdown(self._socket)
            self._response.close()
            self._conn.close()
            _stream_permits.release()


def open_stream(
    reg: PreviewRegistration, name: str, is_current: Callable[[PreviewRegistration], bool],
) -> PreviewStream:
    if name not in reg.cameras:
        raise PreviewUnavailable("unknown preview camera")
    if not _stream_permits.acquire(blocking=False):
        raise PreviewUnavailable("too many preview viewers")
    conn = response = None
    try:
        conn, response, sock = _connect(reg, f"/cameras/{quote(name, safe='')}/stream", is_current)
        if response.getheader("Content-Type", "") != MJPEG_MEDIA_TYPE:
            raise PreviewUnavailable("invalid preview stream")
        return PreviewStream(reg, conn, response, sock, is_current)
    except Exception:
        if response is not None:
            response.close()
        if conn is not None:
            conn.close()
        _stream_permits.release()
        raise


class PreviewStreamingResponse(StreamingResponse):
    """Bound downstream send time as well as upstream reads, including ASGI cancellation."""

    def __init__(self, stream: PreviewStream) -> None:
        self.preview_stream = stream
        super().__init__(stream, media_type=MJPEG_MEDIA_TYPE, headers={"Cache-Control": "no-store"})

    async def stream_response(self, send) -> None:
        async def bounded_send(message):
            await asyncio.wait_for(send(message), timeout=SEND_TIMEOUT_S)

        try:
            await super().stream_response(bounded_send)
        except TimeoutError:
            pass
        finally:
            self.preview_stream.close()
