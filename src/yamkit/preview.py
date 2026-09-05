"""Live camera previews published by the process that owns the cameras.

While `lerobot-record` / `-teleoperate` / `-rollout` runs, the V4L2 devices belong to that
process; the web UI cannot open them a second time. Instead the follower plugin hands every
frame it has *already* acquired for the observation to :meth:`PreviewPublisher.offer`, and a
small MJPEG server inside the same process serves the encoded previews on the loopback
interface. The UI (`yamkit ui`) proxies that server behind its usual ``/api/cameras/<name>/stream``.

Design rules (they are what the tests check):

* ``offer()`` runs on the observation thread and must cost next to nothing: it checks viewer
  demand and the preview rate *before* touching the pixels, keeps at most one pending frame per
  camera (a newer frame replaces an older one, never queues), never blocks on the encoder, a
  viewer or the network, never mutates the observation image, and swallows every failure.
  The only pixel work it does is one copy into preview-owned memory. NumPy's OWNDATA flag
  cannot prove that a producer will not reuse an array.
* One background worker converts and JPEG-encodes the selected frames — once per frame, shared
  by every viewer. There is one pending frame and one latest JPEG per camera, one in-flight
  encode globally, and a fixed number of viewers, each retaining at most one JPEG while sending.
* The server binds ``127.0.0.1`` on an OS-assigned port and requires the per-session token
  the parent handed over in the environment (``YAMKIT_PREVIEW_TOKEN``). The child announces
  itself with exactly one prefixed, versioned JSON line on stdout (never containing the token).
* Frames keep their source order and timing: a replayed frame (the camera had nothing new) is
  not re-published and does not reset the age the UI shows.

Activation is opt-in through the environment (see :func:`start_from_env`); without it the
plugin uses :class:`NullPreview` and the observation path is unchanged.
"""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import select
import socket
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote

import numpy as np

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
REGISTRATION_PREFIX = "@yamkit-preview/"  # line = prefix + version + " " + JSON
ENV_SESSION = "YAMKIT_PREVIEW_SESSION"
ENV_TOKEN = "YAMKIT_PREVIEW_TOKEN"
ENV_FPS = "YAMKIT_PREVIEW_FPS"
TOKEN_HEADER = "X-Yamkit-Preview-Token"
MJPEG_BOUNDARY = "yamkitframe"  # shared with the UI's direct streams so parts pass through verbatim
MJPEG_MEDIA_TYPE = f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}"

DEFAULT_FPS = 10.0  # preview rate target, independent of the dataset / control rate
RATE_TOLERANCE = 0.9  # accept a frame once 90 % of the preview period has passed (30 Hz source → every 3rd frame)
JPEG_QUALITY = 70
STALE_S = 1.0  # no new source frame for this long → the camera is reported "stale"
MAX_VIEWERS = 12  # concurrent stream connections (the UI needs one per tile)
CONTROL_CONNECTIONS = 4  # reserve capacity for status/authentication, also bounded before thread creation
SEND_TIMEOUT_S = 5.0  # a viewer that does not drain its socket for this long is dropped
SOCKET_BUFFER_BYTES = 128 * 1024
MAX_FRAME_BYTES = 32 * 1024 * 1024  # reject malformed/unbounded input; normal camera resolutions fit
CLOSE_TIMEOUT_S = 2.0
_LOG_EVERY_S = 10.0  # failures are counted per frame but logged at most this often


# ----------------------------------------------------------------------------------- frames --
@dataclass(frozen=True)
class _Pending:
    source_seq: int
    frame: np.ndarray
    t_src: float  # perf_counter when this frame object was first offered (≈ acquisition)
    t_offer: float


@dataclass(frozen=True)
class Encoded:
    seq: int
    source_seq: int
    jpeg: bytes
    t_src: float
    t_enc: float
    shape: tuple


class _Slot:
    """Per-camera state shared by the observation thread, the encoder worker and the viewers."""

    def __init__(self, name: str, color_mode: str) -> None:
        self.name = name
        self.color_mode = color_mode
        self.viewers = 0
        # observation-thread side (only that thread writes these)
        self.last_accept = -1e9
        self.last_seen: weakref.ReferenceType | None = None
        self.last_seen_t = 0.0
        self.last_accepted_ref: weakref.ReferenceType | None = None
        self.last_accepted_t: float | None = None
        self.source_seq = 0
        self.offered = 0
        self.accepted = 0
        self.copied = 0
        self.rate_skipped = 0
        self.replayed = 0
        self.dropped_pending = 0
        self.dropped_busy = 0
        self.offer_errors = 0
        # handoff slot
        self._lock = threading.Lock()
        self._pending: _Pending | None = None
        # encoded side
        self._cond = threading.Condition()
        self._latest: Encoded | None = None
        self.encoded = 0
        self.encode_errors = 0
        self.encode_errors_since_ok = 0
        self.last_error: str | None = None
        self.last_encode_ms: float | None = None
        self.seq = 0
        self.closed = False

    # -- observation thread -> worker (never blocks)
    def publish(self, p: _Pending) -> bool:
        if not self._lock.acquire(blocking=False):
            self.dropped_busy += 1
            return False
        try:
            if self._pending is not None:
                self.dropped_pending += 1  # the worker is behind: keep only the newest frame
            self._pending = p
        finally:
            self._lock.release()
        return True

    def take(self) -> _Pending | None:
        with self._lock:
            p, self._pending = self._pending, None
        return p

    # -- worker -> viewers
    def set_latest(self, enc: Encoded) -> None:
        with self._cond:
            if self.closed:
                return
            self._latest = enc
            self._cond.notify_all()

    def wake_viewers(self) -> None:
        with self._cond:
            self._cond.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._pending = None
        with self._cond:
            self.closed = True
            self._latest = None
            self._cond.notify_all()
        self.last_seen = self.last_accepted_ref = None

    def wait_for(self, last_seq: int, timeout: float) -> Encoded | None:
        """The newest encoded frame with seq > last_seq, waiting up to `timeout` for one."""
        with self._cond:
            if self._latest is None or self._latest.seq <= last_seq:
                self._cond.wait(timeout)
            enc = self._latest
        return enc if enc is not None and enc.seq > last_seq else None

    @property
    def latest(self) -> Encoded | None:
        return self._latest

    def status(self, now: float) -> dict[str, Any]:
        enc = self._latest
        age = round(now - enc.t_src, 3) if enc is not None else None
        if self.last_error and (enc is None or self.encode_errors_since_ok):
            state = "unavailable"
        elif self.viewers <= 0:
            state = "idle"  # nobody watching → nothing is published (by design)
        elif enc is None:
            state = "waiting"
        elif age is not None and age > STALE_S:
            state = "stale"
        else:
            state = "live"
        return {
            "state": state,
            "viewers": self.viewers,
            "seq": self.seq,
            "source_seq": self.source_seq,
            "age_s": age,
            "offered": self.offered,
            "accepted": self.accepted,
            "copied": self.copied,
            "encoded": self.encoded,
            "rate_skipped": self.rate_skipped,
            "replayed": self.replayed,
            "dropped": self.dropped_pending + self.dropped_busy,
            "errors": self.offer_errors + self.encode_errors,
            "last_error": self.last_error,
            "encode_ms": self.last_encode_ms,
            "shape": list(enc.shape) if enc is not None else None,
        }

class NullPreview:
    """Stand-in when previews are not requested: `offer` is a no-op."""

    enabled = False
    port = None

    def offer(self, key: str, frame: Any, *, source_time: float | None = None) -> None:
        return None

    def status(self) -> dict[str, Any]:
        return {}

    def close(self, timeout: float = CLOSE_TIMEOUT_S) -> None:
        return None


class PreviewPublisher:
    """Encoder worker + loopback MJPEG server for the frames a robot already acquired.

    `cameras` maps camera name → colour mode of the frames that will be offered ("rgb" — LeRobot's
    default — or "bgr"); anything else is treated as "rgb"."""

    enabled = True

    def __init__(
        self,
        session: str,
        token: str,
        cameras: dict[str, str],
        *,
        owner: str | None = None,
        fps: float = DEFAULT_FPS,
        quality: int = JPEG_QUALITY,
        max_viewers: int = MAX_VIEWERS,
    ) -> None:
        if not session or not token:
            raise ValueError("preview session id and token are required")
        self.session = session
        self.owner = owner
        self._token = token
        self.fps = max(1.0, min(float(fps), 30.0))
        self.quality = int(quality)
        self.max_viewers = max(1, int(max_viewers))
        self._viewer_lock = threading.Lock()
        self._min_interval = RATE_TOLERANCE / self.fps
        self._slots: dict[str, _Slot] = {n: _Slot(n, (m or "rgb").lower()) for n, m in cameras.items()}
        self._wake = threading.Event()
        self._stopping = False
        self._server: _Server | None = None
        self._server_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._cv2: Any = None
        self._last_log: dict[str, float] = {}
        self.started_at: float | None = None

    # ----- lifecycle --------------------------------------------------------------------------
    @property
    def port(self) -> int | None:
        return self._server.server_address[1] if self._server is not None else None

    @property
    def cameras(self) -> list[str]:
        return list(self._slots)

    def start(self) -> PreviewPublisher:
        if self._server is not None:
            return self
        if self._stopping:
            raise RuntimeError("a closed preview publisher cannot restart")
        import cv2  # the encoder needs it; fail here, not on the first frame

        self._cv2 = cv2
        srv = _Server(("127.0.0.1", 0), _Handler, max_connections=self.max_viewers + CONTROL_CONNECTIONS)
        srv.publisher = self
        self._server = srv
        self._server_thread = threading.Thread(
            target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True, name="yamkit-preview-http"
        )
        self._server_thread.start()
        self._worker = threading.Thread(target=self._work, daemon=True, name="yamkit-preview-encoder")
        self._worker.start()
        self.started_at = time.time()
        return self

    def registration_line(self) -> str:
        """The single stdout line the parent parses: prefix, protocol version, JSON. No token."""
        body = {
            "v": PROTOCOL_VERSION,
            "session": self.session,
            "port": self.port,
            "cameras": self.cameras,
            "pid": os.getpid(),
            "fps": self.fps,
        }
        if self.owner is not None:
            body["owner"] = self.owner
        return f"{REGISTRATION_PREFIX}{PROTOCOL_VERSION} " + json.dumps(body, separators=(",", ":"))

    def announce(self) -> None:
        """Write the registration line to stdout as one write (survives a replaced sys.stdout)."""
        line = self.registration_line() + "\n"
        try:
            sys.stdout.flush()
            os.write(sys.stdout.fileno(), line.encode())
        except Exception:  # noqa: BLE001 — e.g. stdout captured by a test harness
            print(line, end="", flush=True)

    def close(self, timeout: float = CLOSE_TIMEOUT_S) -> None:
        """Stop publishing, close the server and join the worker — bounded by `timeout`."""
        deadline = time.monotonic() + max(0.0, timeout)
        self._stopping = True
        self._wake.set()
        for s in self._slots.values():
            s.wake_viewers()
        srv, self._server = self._server, None
        if srv is not None:
            srv.request_stop()
            srv.close_connections()
            srv.server_close()
        if self._server_thread is not None and self._server_thread.ident is not None:
            self._server_thread.join(max(0.0, deadline - time.monotonic()))
        if self._worker is not None and self._worker.ident is not None:
            self._worker.join(max(0.0, deadline - time.monotonic()))
        for slot in self._slots.values():
            slot.clear()

    # ----- observation thread -----------------------------------------------------------------
    def offer(self, key: str, frame: Any, *, source_time: float | None = None) -> None:
        """Hand one acquired frame over for preview. Never blocks, never raises, never writes to
        `frame`. Returns immediately when nobody is watching or the preview rate is satisfied.

        `source_time`, when available, is the acquisition timestamp on the perf_counter clock.
        Otherwise identity detects replay (both installed LeRobot camera drivers return the
        same ndarray until the next capture), and age starts at the frame's first offer.
        """
        try:
            slot = self._slots.get(key)
            if slot is None or self._stopping or slot.viewers <= 0:
                return  # demand check: no viewer → not even a timestamp is taken
            now = time.perf_counter()
            if not isinstance(frame, np.ndarray):
                # Conversion belongs on the worker; the installed camera APIs return ndarrays.
                slot.offer_errors += 1
                slot.last_error = "offer: expected a NumPy image"
                return
            if source_time is not None and (not math.isfinite(source_time) or source_time > now):
                slot.offer_errors += 1
                slot.last_error = "offer: invalid acquisition timestamp"
                return
            same_source = (
                slot.last_seen is not None
                and frame is slot.last_seen()
                and (source_time is None or source_time == slot.last_seen_t)
            )
            if same_source:
                t_src = slot.last_seen_t  # the camera had nothing new: keep the frame's original time
                replay = True
            else:
                t_src = now if source_time is None else source_time
                slot.last_seen, slot.last_seen_t, replay = weakref.ref(frame), t_src, False
                slot.source_seq += 1
            slot.offered += 1
            if now - slot.last_accept < self._min_interval:
                slot.rate_skipped += 1  # rate check before any copy
                return
            if (
                slot.last_accepted_ref is not None
                and frame is slot.last_accepted_ref()
                and t_src == slot.last_accepted_t
            ):
                slot.replayed += 1  # already published; re-sending it would fake freshness
                return
            if frame.nbytes > MAX_FRAME_BYTES:
                slot.offer_errors += 1
                slot.last_error = "offer: image exceeds preview memory limit"
                return
            if not slot._lock.acquire(blocking=False):
                slot.dropped_busy += 1
                return
            try:
                if self._stopping:
                    return
                # Even an owning array can be overwritten by its producer after offer returns.
                # Explicit ndarray.copy avoids a subclass's arbitrary overridden copy method.
                arr = np.ndarray.copy(frame, order="C")
                slot.copied += 1
                if slot._pending is not None:
                    slot.dropped_pending += 1
                slot._pending = _Pending(slot.source_seq, arr, t_src, now)
                slot.accepted += 1
                slot.last_accept = now
                slot.last_accepted_ref = weakref.ref(frame)
                slot.last_accepted_t = t_src
                if replay:
                    slot.replayed += 1
            finally:
                slot._lock.release()
            self._wake.set()
        except Exception:  # noqa: BLE001 — previews must never break the observation loop
            try:
                s = self._slots.get(key)
                if s is not None:
                    s.offer_errors += 1
                    s.last_error = "offer: frame handoff failed"
            except Exception:  # noqa: BLE001, S110 — no logging on the observation thread
                pass

    # ----- worker -----------------------------------------------------------------------------
    def _work(self) -> None:
        while not self._stopping:
            if not self._wake.wait(0.5):
                continue
            self._wake.clear()
            for slot in self._slots.values():
                if self._stopping:
                    return
                p = slot.take()
                if p is None:
                    continue
                t0 = time.perf_counter()
                try:
                    jpeg = self._encode(p.frame, slot.color_mode)
                except Exception as e:  # noqa: BLE001
                    slot.encode_errors += 1
                    slot.encode_errors_since_ok += 1
                    slot.last_error = f"encode: {e}"
                    self._log_throttled(f"encode:{slot.name}", "preview encode failed for %s: %s", slot.name, e)
                    continue
                if self._stopping:
                    return
                slot.seq += 1
                slot.encoded += 1
                slot.encode_errors_since_ok = 0
                slot.last_error = None
                slot.last_encode_ms = round((time.perf_counter() - t0) * 1e3, 2)
                slot.set_latest(Encoded(slot.seq, p.source_seq, jpeg, p.t_src, time.perf_counter(), tuple(p.frame.shape)))

    def _encode(self, a: np.ndarray, color_mode: str) -> bytes:
        cv2 = self._cv2
        if a.dtype != np.uint8:
            raise ValueError(f"unsupported dtype {a.dtype} (previews need uint8 images)")
        if a.ndim == 3 and a.shape[2] == 3:
            if color_mode != "bgr":
                a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)  # a new array; the observation image is untouched
        elif a.ndim != 2:
            raise ValueError(f"unsupported frame shape {a.shape}")
        ok, buf = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return buf.tobytes()

    def _log_throttled(self, key: str, msg: str, *args: Any) -> None:
        now = time.monotonic()
        if now - self._last_log.get(key, -1e9) >= _LOG_EVERY_S:
            self._last_log[key] = now
            log.warning(msg, *args)

    # ----- server side ------------------------------------------------------------------------
    def check_token(self, presented: str | None) -> bool:
        try:
            return presented is not None and hmac.compare_digest(presented, self._token)
        except TypeError:
            return False

    def slot(self, name: str) -> _Slot | None:
        return self._slots.get(name)

    @property
    def viewers(self) -> int:
        with self._viewer_lock:
            return sum(s.viewers for s in self._slots.values())

    def acquire_viewer(self, slot: _Slot) -> bool:
        """Atomically reserve a stream before returning its HTTP response."""
        with self._viewer_lock:
            if self._stopping or sum(s.viewers for s in self._slots.values()) >= self.max_viewers:
                return False
            slot.viewers += 1
            return True

    def release_viewer(self, slot: _Slot) -> None:
        with self._viewer_lock:
            slot.viewers -= 1

    def status(self) -> dict[str, Any]:
        now = time.perf_counter()
        return {
            "v": PROTOCOL_VERSION,
            "session": self.session,
            "fps": self.fps,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "cameras": {n: s.status(now) for n, s in self._slots.items()},
        }

    @property
    def stopping(self) -> bool:
        return self._stopping


def part_header(enc: Encoded, now: float) -> bytes:
    """Multipart part header for one encoded frame (the JPEG bytes follow, then CRLF)."""
    return (
        f"--{MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(enc.jpeg)}\r\n"
        f"X-Yamkit-Seq: {enc.seq}\r\nX-Yamkit-Source-Seq: {enc.source_seq}\r\n"
        f"X-Yamkit-Age-Ms: {max(0.0, (now - enc.t_src) * 1e3):.0f}\r\n\r\n"
    ).encode()


class _Server(ThreadingHTTPServer):
    """Bound connections before spawning threads, including clients with incomplete headers."""

    allow_reuse_address = False
    daemon_threads = True
    block_on_close = False
    request_queue_size = MAX_VIEWERS + CONTROL_CONNECTIONS
    publisher: PreviewPublisher

    def __init__(self, *args: Any, max_connections: int, **kwargs: Any) -> None:
        self._permits = threading.BoundedSemaphore(max_connections)
        self._connections: set[socket.socket] = set()
        self._connections_lock = threading.Lock()
        self._stop = threading.Event()
        super().__init__(*args, **kwargs)

    def serve_forever(self, poll_interval: float = 0.05) -> None:
        # A stop event avoids BaseServer.shutdown's unbounded wait, even if thread startup fails.
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self], [], [], poll_interval)
            except (OSError, ValueError):
                return  # close() can close the listener while select is waiting
            if readable and not self._stop.is_set():
                self._handle_request_noblock()

    def request_stop(self) -> None:
        self._stop.set()

    def get_request(self) -> tuple[socket.socket, Any]:
        sock, address = super().get_request()
        sock.settimeout(SEND_TIMEOUT_S)  # header reads as well as stream writes
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCKET_BUFFER_BYTES)
        return sock, address

    def process_request(self, request: socket.socket, client_address: Any) -> None:
        if not self._permits.acquire(blocking=False):
            self.shutdown_request(request)
            return
        with self._connections_lock:
            if self._stop.is_set():
                self._permits.release()
                self.shutdown_request(request)
                return
            self._connections.add(request)
        try:
            super().process_request(request, client_address)
        except Exception:
            with self._connections_lock:
                self._connections.discard(request)
            self._permits.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: Any) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._connections_lock:
                self._connections.discard(request)
            self._permits.release()

    def close_connections(self) -> None:
        with self._connections_lock:
            connections = tuple(self._connections)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def handle_error(self, request: socket.socket, client_address: Any) -> None:
        return  # malformed requests/disconnects must not flood recorder stderr


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"  # one response per connection; the stream ends with the socket
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:
        return  # no per-request / per-frame logging

    def _deny(self, code: int, text: str) -> None:
        body = (text + "\n").encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        pub = self.server.publisher
        if not pub.check_token(self.headers.get(TOKEN_HEADER)):
            self._deny(401, "unauthorized")
            return
        path = self.path.split("?", 1)[0]
        if path == "/status":
            body = json.dumps(pub.status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[0] == "cameras" and parts[2] == "stream":
            slot = pub.slot(unquote(parts[1]))
            if slot is None:
                self._deny(404, "no such camera")
                return
            if not pub.acquire_viewer(slot):
                self._deny(503, "too many viewers")
                return
            self._stream(slot)
            return
        self._deny(404, "not found")

    def _stream(self, slot: _Slot) -> None:
        pub = self.server.publisher
        try:
            self.connection.settimeout(SEND_TIMEOUT_S)
            self.send_response(200)
            self.send_header("Content-Type", MJPEG_MEDIA_TYPE)
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            last_seq = 0
            latest = slot.latest
            if latest is not None:  # show what we have, with its true age; the next frame follows
                self._write_part(latest)
                last_seq = latest.seq
            while not pub.stopping:
                enc = slot.wait_for(last_seq, timeout=0.25)
                if enc is None:
                    # No frame writes means no broken pipe to reveal disconnects. Peeking for
                    # socket readability catches EOF without replaying stale images as a heartbeat.
                    if select.select([self.connection], [], [], 0)[0]:
                        return
                    continue
                last_seq = enc.seq
                self._write_part(enc)
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            pass  # the viewer went away (or stalled for SEND_TIMEOUT_S)
        finally:
            pub.release_viewer(slot)

    def _write_part(self, enc: Encoded) -> None:
        w = self.wfile
        w.write(part_header(enc, time.perf_counter()))
        w.write(enc.jpeg)  # the shared bytes object — no per-viewer copy
        w.write(b"\r\n")


# ------------------------------------------------------------------------------- activation --
def from_env(
    cameras: dict[str, str], environ: dict[str, str] | None = None, *, owner: str | None = None
) -> PreviewPublisher | NullPreview:
    """A publisher when the parent asked for previews (session id + token in the environment),
    else a NullPreview. Nothing is started yet."""
    env = os.environ if environ is None else environ
    session, token = env.get(ENV_SESSION, ""), env.get(ENV_TOKEN, "")
    if not session or not token or not cameras:
        return NullPreview()
    try:
        fps = float(env.get(ENV_FPS, DEFAULT_FPS))
    except ValueError:
        fps = DEFAULT_FPS
    return PreviewPublisher(session, token, cameras, fps=fps, owner=owner)


def start_from_env(
    cameras: dict[str, str], environ: dict[str, str] | None = None, *, owner: str | None = None
) -> PreviewPublisher | NullPreview:
    """`from_env` + start + announce. A failure is logged and yields a NullPreview: the recording
    must never depend on the preview."""
    pub = from_env(cameras, environ, owner=owner)
    if not pub.enabled:
        return pub
    try:
        pub.start()
        pub.announce()
        log.info("live previews on 127.0.0.1:%s for %s", pub.port, ", ".join(pub.cameras))
        return pub
    except Exception as e:  # noqa: BLE001
        log.warning("live previews disabled: %s", e)
        try:
            pub.close(0.5)
        except Exception:  # noqa: BLE001, S110 — startup failure must not prevent recording
            pass
        return NullPreview()
