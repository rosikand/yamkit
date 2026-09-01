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

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ensure_running(self) -> None:
        with self.lock:
            if not self.running:
                self._stop.clear()
                self.error = None
                self._thread = threading.Thread(target=self._loop, daemon=True, name=f"cam-{self.name}")
                self._thread.start()

    def stop(self, join: bool = False) -> None:
        self._stop.set()
        with self.cond:
            self.cond.notify_all()
        if join and self._thread is not None:
            self._thread.join(timeout=3.0)

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
                return
            if self.cfg.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.cfg["width"]))
            if self.cfg.get("height"):
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg["height"]))
            period = 1.0 / max(self.fps, 1.0)
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.error = f"read failed on {dev}"
                    break
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
            with self.cond:
                self.cond.notify_all()

    def frames(self, stop_flag: threading.Event) -> Iterator[bytes]:
        """Yield multipart JPEG parts until the camera stops or the client goes away."""
        self.clients += 1
        self.last_client_t = time.time()
        try:
            self.ensure_running()
            last_sent = 0.0
            while not stop_flag.is_set():
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

    def get(self, name: str) -> _Camera | None:
        return self.cams.get(name)

    def suspend(self, reason: str) -> None:
        self.suspended_by = reason
        for c in self.cams.values():
            c.stop(join=True)
        log.info("camera streams suspended (%s)", reason)

    def resume(self) -> None:
        self.suspended_by = None  # streams restart lazily on the next client

    def statuses(self) -> list[dict[str, Any]]:
        return [{**c.status(), "suspended_by": self.suspended_by} for c in self.cams.values()]
