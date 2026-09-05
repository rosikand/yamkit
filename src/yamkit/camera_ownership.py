"""The UI and camera-owning child agree on exclusive access before opening devices."""

from __future__ import annotations

import json
import os
import secrets
import select
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TextIO

PREFIX = "@yamkit-cameras/1 "
ENV_SESSION = "YAMKIT_PREVIEW_SESSION"
ENV_TOKEN = "YAMKIT_PREVIEW_TOKEN"
CLAIM_TIMEOUT_S = 15.0
MAX_MESSAGE = 8192


@contextmanager
def retain_camera_readers(camera, readers: list):
    """Remember pinned camera readers before internal cleanup can detach them.

    OpenCV and RealSense clear ``thread`` even after a failed bounded join, including
    during connect/warmup. RealSense can retry with another reader in the same call.
    Observe that existing instance method without changing its cleanup or retry logic.
    """
    def remember():
        reader = getattr(camera, "thread", None)
        if reader is not None and all(reader is not previous for previous in readers):
            readers.append(reader)

    def check_detached():
        remember()
        current = getattr(camera, "thread", None)
        if any(reader is not current and reader.is_alive() for reader in readers):
            raise RuntimeError("camera release could not be confirmed: detached reader is still alive")

    remember()
    stop = getattr(camera, "_stop_read_thread", None)
    if not callable(stop):
        try:
            yield
            check_detached()
        finally:
            remember()
        return

    missing = object()
    original = vars(camera).get("_stop_read_thread", missing)

    def tracked_stop(*args, **kwargs):
        remember()
        return stop(*args, **kwargs)

    camera._stop_read_thread = tracked_stop
    try:
        yield
        check_detached()
    finally:
        try:
            remember()
        finally:
            if original is missing:
                del camera._stop_read_thread
            else:
                camera._stop_read_thread = original


@dataclass
class CameraLease:
    session: str = ""
    owner: str = ""
    output: TextIO | None = None
    _released: bool = False

    def release(self) -> None:
        """Call only after every owned camera's disconnect has completed successfully."""
        if self._released or not self.owner:
            return
        self._released = True
        try:
            _emit({"v": 1, "session": self.session, "owner": self.owner, "event": "release"}, self.output)
        except (OSError, ValueError):
            # A disappeared parent also loses the stream; process-group cleanup is its fallback.
            pass


def _emit(message: dict, output: TextIO | None = None) -> None:
    output = output or sys.stdout
    output.write(PREFIX + json.dumps(message, separators=(",", ":")) + "\n")
    output.flush()


def claim_from_env(
    cameras: list[str],
    *,
    environ: dict[str, str] | None = None,
    timeout: float = CLAIM_TIMEOUT_S,
    input: TextIO | None = None,
    output: TextIO | None = None,
) -> CameraLease:
    """Fail closed before camera acquisition unless the UI confirms direct capture stopped.

    Outside a UI session this is a no-op. The acknowledgement uses stdin inherited by the
    LeRobot subprocess; no image data or authentication token is sent on either pipe.
    """
    env = os.environ if environ is None else environ
    session = env.get(ENV_SESSION, "")
    if not cameras or not session or not env.get(ENV_TOKEN):
        return CameraLease()
    lease = CameraLease(session, secrets.token_hex(16), output)
    _emit({"v": 1, "session": session, "owner": lease.owner, "event": "acquire", "cameras": cameras}, output)
    source = input or sys.stdin
    deadline = time.monotonic() + timeout
    pending = bytearray()
    try:
        fd = source.fileno()
        while (remaining := deadline - time.monotonic()) > 0:
            if not select.select([fd], [], [], remaining)[0]:
                break
            data = os.read(fd, 1)
            if not data:
                break
            pending.extend(data)
            if len(pending) > MAX_MESSAGE:
                raise RuntimeError("camera ownership acknowledgement is too large")
            if data != b"\n":
                continue
            try:
                message = json.loads(pending)
            except (ValueError, RecursionError):
                pending.clear()
                continue
            pending.clear()
            if not isinstance(message, dict):
                continue
            if message.get("session") != session or message.get("owner") != lease.owner:
                continue
            if message.get("ok") is True:
                return lease
            raise RuntimeError("camera acquisition denied: direct preview did not release the devices")
    except (OSError, ValueError) as exc:
        raise RuntimeError("camera ownership acknowledgement failed") from exc
    raise RuntimeError("timed out waiting for the UI to release cameras")
