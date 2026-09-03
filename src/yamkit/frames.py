"""Latest-frame publishing from a recording/rollout process to the web UI.

While `lerobot-record` / `lerobot-rollout` own the cameras, the UI cannot open them. Instead the
follower plugin drops the newest frame of every camera as a JPEG into ``$YAMKIT_FRAMES_DIR`` (a
few times per second, atomically), and the UI streams those files. Off unless the variable is set,
so plain `lerobot-*` runs are untouched.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ENV_VAR = "YAMKIT_FRAMES_DIR"
DEFAULT_HZ = 8.0
JPEG_QUALITY = 70


class FramePublisher:
    """`publish(name, rgb_frame)` writes `<dir>/<name>.jpg` at most `hz` times per second per camera."""

    def __init__(self, directory: str | os.PathLike[str] | None = None, hz: float = DEFAULT_HZ) -> None:
        d = directory if directory is not None else os.environ.get(ENV_VAR)
        self.dir = Path(d) if d else None
        self.period = 1.0 / max(hz, 0.1)
        self._last: dict[str, float] = {}
        self._failed = False
        if self.dir is not None:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                log.warning("frame publishing disabled: %s", e)
                self.dir = None

    @property
    def enabled(self) -> bool:
        return self.dir is not None and not self._failed

    def publish(self, name: str, frame: Any) -> bool:
        if not self.enabled or frame is None:
            return False
        now = time.monotonic()
        if now - self._last.get(name, 0.0) < self.period:
            return False
        self._last[name] = now
        try:
            import cv2
            import numpy as np

            img = np.asarray(frame)
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # LeRobot cameras hand out RGB
            ok, jpg = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                return False
            final = self.dir / f"{name}.jpg"
            tmp = final.with_suffix(".jpg.tmp")
            tmp.write_bytes(jpg.tobytes())
            os.replace(tmp, final)  # readers never see a half-written file
            return True
        except Exception as e:  # noqa: BLE001 — never let the UI preview break a recording
            if not self._failed:
                log.warning("frame publishing stopped: %s", e)
            self._failed = True
            return False

    def clear(self) -> None:
        if self.dir is None:
            return
        for p in self.dir.glob("*.jpg"):
            try:
                p.unlink()
            except OSError:
                pass
