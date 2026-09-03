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
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

BOUNDARY = b"--yamkitframe"
IDLE_RELEASE_S = 5.0
STOP_JOIN_S = 10.0  # a capture thread blocked in cap.read() must have released the device before a recorder opens it
REOPEN_DELAY_S = 1.0  # after a failed open/read: the next poll retries (RealSense needs a moment after another process lets go)


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
            self._thread.join(timeout=STOP_JOIN_S)
            if self._thread.is_alive():
                log.warning("camera %s: capture thread did not stop in %.0fs — the device may still be busy", self.name, STOP_JOIN_S)

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
                self._stop.wait(REOPEN_DELAY_S)  # the next poll starts a fresh attempt
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

    def snapshot(self) -> bytes | None:
        """The newest JPEG (starts the capture thread if needed; polling keeps it alive)."""
        self.last_client_t = time.time()
        self.ensure_running()
        return self.frame

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


def file_frames(path: Path, stop: threading.Event, hz: float = 10.0, stale_s: float = 5.0, alive=lambda: True) -> Iterator[bytes]:
    """Multipart JPEG parts from a file that another process rewrites (see yamkit.frames).

    Ends when `stop` is set, when `alive()` turns false (the session gave the cameras back), or when
    nothing was ever published within `stale_s`."""
    last_mtime = -1.0
    started = time.time()
    while not stop.is_set() and alive():
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = -1.0
        if mtime > last_mtime:
            try:
                data = path.read_bytes()
            except OSError:
                data = b""
            if data.startswith(b"\xff\xd8"):  # a complete JPEG (writes are atomic renames)
                last_mtime = mtime
                yield BOUNDARY + b"\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n"
        elif last_mtime < 0 and time.time() - started > stale_s:
            return  # the session never published anything: let the tile fall back to "no signal"
        time.sleep(1.0 / hz)


class CameraHub:
    """All rig cameras + a suspend switch used while a recording session owns the devices."""

    def __init__(self, cameras: dict[str, dict[str, Any]], frames_dir: Path | None = None) -> None:
        self.cams = {name: _Camera(name, dict(cfg)) for name, cfg in (cameras or {}).items()}
        self.suspended_by: str | None = None
        self.frames_dir = frames_dir  # previews published by the session that owns the cameras

    def get(self, name: str) -> _Camera | None:
        return self.cams.get(name)

    def reload(self, cameras: dict[str, dict[str, Any]]) -> None:
        """Apply a new `cameras:` section (after the rig file was saved): unchanged entries keep
        streaming, changed/removed ones are stopped, new ones are added."""
        cameras = cameras or {}
        for name, cam in list(self.cams.items()):
            if cameras.get(name) != cam.cfg:
                cam.stop(join=True)
                del self.cams[name]
        for name, cfg in cameras.items():
            if name not in self.cams:
                self.cams[name] = _Camera(name, dict(cfg))
        log.info("camera list reloaded: %s", ", ".join(self.cams) or "none")

    def suspend(self, reason: str) -> None:
        self.suspended_by = reason
        for c in self.cams.values():
            c.stop(join=True)
        log.info("camera streams suspended (%s)", reason)

    def resume(self) -> None:
        self.suspended_by = None  # streams restart lazily on the next client

    def statuses(self) -> list[dict[str, Any]]:
        return [{**c.status(), "suspended_by": self.suspended_by} for c in self.cams.values()]
