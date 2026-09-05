"""MJPEG preview streams for the rig cameras (configs/rig.yaml `cameras:`).

Devices are opened lazily when a browser asks for a stream and released as soon as nobody is
watching, or when a session that needs the cameras itself (record / teleoperate / rollout)
starts — V4L2 devices are exclusive, and the LeRobot child process must win.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from typing import Any

log = logging.getLogger(__name__)

BOUNDARY = b"--yamkitframe"
IDLE_RELEASE_S = 5.0
STOP_JOIN_S = 10.0  # a capture thread blocked in cap.read() must have released the device before a recorder opens it
REOPEN_DELAY_S = 1.0  # after a failed open/read: the next stream client retries (RealSense needs a moment after another process lets go)


class _Camera:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.cfg = cfg
        self.device = cfg.get("index_or_path")
        self.fps = float(cfg.get("fps") or 15)
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.frame: bytes | None = None
        self.frame_t = 0.0
        self.error: str | None = None
        self.clients = 0
        self.last_client_t = 0.0
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._allowed = True

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ensure_running(self) -> bool:
        with self.lock:
            if not self._allowed:
                return False
            if not self.running:
                self._stop.clear()
                self.error = None
                self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{self.name}")
                self._thread.start()
            return True

    def stop(self, join: bool = False, *, disable: bool = False, timeout: float = STOP_JOIN_S) -> bool:
        with self.cond:
            if disable:
                self._allowed = False
            self._stop.set()
            self.cond.notify_all()
        if join and self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                log.warning("camera %s: capture thread did not stop in %.0fs — the device may still be busy", self.name, STOP_JOIN_S)
        return not self.running

    def allow(self) -> None:
        with self.lock:
            self._allowed = True

    def _loop(self) -> None:
        try:
            import cv2
        except ImportError as e:
            self.error = f"opencv not available: {e}"
            return
        dev = self.device
        cap = cv2.VideoCapture(int(dev) if isinstance(dev, int) or str(dev).isdigit() else str(dev))
        try:
            if not cap.isOpened():
                self.error = f"could not open {dev}"
                self._stop.wait(REOPEN_DELAY_S)  # the next stream client starts a fresh attempt
                return
            if self.cfg.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg["width"]))
            if self.cfg.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg["height"]))
            period = 1.0 / max(self.fps, 1.0)
            failures = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    if failures < 5:  # a RealSense returns a few empty reads right after opening
                        time.sleep(0.05)
                        continue
                    self.error = f"read failed on {dev}"
                    self._stop.wait(REOPEN_DELAY_S)
                    break
                failures = 0
                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    with self.cond:
                        self.frame = jpg.tobytes()
                        self.frame_t = time.time()
                        self.error = None
                        self.cond.notify_all()
                if self.clients == 0 and time.time() - self.last_client_t > IDLE_RELEASE_S:
                    break  # nobody watching → release the device
                time.sleep(period)
        finally:
            cap.release()
            del cap  # make sure OpenCV drops the device handle right now, not at some later GC
            with self.cond:
                self.cond.notify_all()

    def frames(self, stop_flag: threading.Event) -> Iterator[bytes]:
        """Yield multipart JPEG parts until the camera stops or the client goes away."""
        self.clients += 1
        self.last_client_t = time.time()
        try:
            if stop_flag.is_set() or not self.ensure_running():
                return
            last_sent = 0.0
            while not stop_flag.is_set() and not self._stop.is_set():
                with self.cond:
                    if self.frame is None or self.frame_t <= last_sent:
                        self.cond.wait(timeout=1.0)
                    frame, t = self.frame, self.frame_t
                if not self.running:
                    break
                if frame is not None and t > last_sent:
                    last_sent = t
                    yield (
                        BOUNDARY
                        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                self.last_client_t = time.time()
        finally:
            self.clients -= 1
            self.last_client_t = time.time()

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "device": str(self.device),
            "width": self.cfg.get("width"),
            "height": self.cfg.get("height"),
            "fps": self.cfg.get("fps"),
            "streaming": self.running,
            "clients": self.clients,
            "error": self.error,
            "frame_age_s": round(time.time() - self.frame_t, 1) if self.frame_t else None,
        }


class CameraHub:
    """All rig cameras + a suspend switch used while a recording session owns the devices."""

    def __init__(self, cameras: dict[str, dict[str, Any]]) -> None:
        self.cams = {name: _Camera(name, dict(cfg)) for name, cfg in (cameras or {}).items()}
        self.suspended_by: str | None = None
        self._lock = threading.RLock()
        self._closing = False

    def get(self, name: str) -> _Camera | None:
        return self.cams.get(name)

    def reload(self, cameras: dict[str, dict[str, Any]]) -> None:
        """Apply a new `cameras:` section (after the rig file was saved): unchanged entries keep
        streaming, changed/removed ones are stopped, new ones are added."""
        cameras = cameras or {}
        with self._lock:
            if self._closing:
                raise RuntimeError("camera hub is closing")
            for name, cam in list(self.cams.items()):
                if cameras.get(name) != cam.cfg:
                    if not cam.stop(join=True, disable=True):
                        raise RuntimeError(f"camera {name} has not released its device")
                    del self.cams[name]
            for name, cfg in cameras.items():
                if name not in self.cams:
                    cam = _Camera(name, dict(cfg))
                    if self.suspended_by is not None:
                        cam.stop(disable=True)
                    self.cams[name] = cam
        log.info("camera list reloaded: %s", ", ".join(self.cams) or "none")

    def suspend(self, reason: str) -> bool:
        """Disable all starts before joining; success confirms every device was released."""
        with self._lock:
            if self._closing:
                return False
            if self.suspended_by not in (None, reason):
                return False
            self.suspended_by = reason
            for c in self.cams.values():
                c.stop(disable=True)
            deadline = time.monotonic() + STOP_JOIN_S
            released = True
            for c in self.cams.values():
                if not c.stop(join=True, disable=True, timeout=max(0.0, deadline - time.monotonic())):
                    released = False
            log.info("camera streams suspended (%s); released=%s", reason, released)
            return released

    def resume(self, owner: str | None = None) -> bool:
        with self._lock:
            if self._closing or self.suspended_by != owner:
                return False
            self.suspended_by = None
            for c in self.cams.values():
                c.allow()  # streams restart lazily on the next client
            return True

    def close(self) -> bool:
        """Permanently block direct capture, including delayed session release callbacks."""
        with self._lock:
            self._closing = True
            for c in self.cams.values():
                c.stop(disable=True)
            deadline = time.monotonic() + STOP_JOIN_S
            released = True
            for c in self.cams.values():
                if not c.stop(join=True, disable=True, timeout=max(0.0, deadline - time.monotonic())):
                    released = False
            return released

    def statuses(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{**c.status(), "suspended_by": self.suspended_by} for c in self.cams.values()]
