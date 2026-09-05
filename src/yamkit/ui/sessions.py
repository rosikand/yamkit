"""Hardware sessions for the web UI: managed `yamkit <cmd>` child processes.

The UI never drives the arms itself — Start Teleop / Start Recording / rollout spawn the same
CLI a user would run in a terminal (`yamkit teleop`, `yamkit record`, ...) and parse its output
for display. One session at a time; stopping sends SIGINT (the CLIs already shut down cleanly on
Ctrl-C) and escalates to SIGTERM/SIGKILL only if the child hangs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Modes that energise motors (gravity-comp on connect; teleop/record/rollout also move them).
HARDWARE_MODES = frozenset({"read", "teleop", "teleoperate", "record", "rollout", "rest", "policy-probe-live"})

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
# `yamkit read`:  `    left_follower q=[+0.001 -0.512 ...] grip=0.98 btn=10`
_READ_RE = re.compile(r"^\s*(\S+) q=\[([^\]]*)\] grip=(\S+)(?: btn=([01]+))?")
# `yamkit teleop`: `[ 99.8Hz] left_leader->left_follower: ENGAGED err=0.012rad grip=0.98 | ...`
_TELEOP_HZ_RE = re.compile(r"\[\s*([\d.]+)Hz\]")
_TELEOP_PAIR_RE = re.compile(r"(\S+->\S+): (ENGAGED|idle)\s*err=\s*([-+.\dnaif]+)rad grip=(\S+)")
# lerobot-record progress (message wording varies between versions; match loosely)
_EPISODE_RE = re.compile(r"[Rr]ecord(?:ing)?\s+episode\s+(\d+)")
_RESET_RE = re.compile(r"[Rr]eset the environment")
_UPLOAD_RE = re.compile(r"\[yamkit\] recording finished")  # recorder exited; only the upload is left
# `yamkit policy-check` table rows
_FIRST_CALL_RE = re.compile(r"first call.*?([\d.]+)\s*ms")
_NEXT_CALLS_RE = re.compile(r"next calls.*?│([^│]*)")
_OWNERSHIP_PREFIX = "@yamkit-cameras/1 "
_PREVIEW_PREFIX = "@yamkit-preview/1 "
_MAX_CONTROL_LINE = 8192
PREVIEW_START_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class PreviewRegistration:
    session: str
    owner: str
    port: int
    cameras: tuple[str, ...]
    token: str = field(repr=False)


def _camera_names(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not 1 <= len(value) <= 32:
        return None
    if any(not isinstance(n, str) or not n or len(n) > 128 or "/" in n or "\\" in n for n in value):
        return None
    if len(set(value)) != len(value):
        return None
    return tuple(value)


def _group_alive(pid: int) -> bool:
    """The launcher can exit before its recorder. Zombies no longer hold devices."""
    proc_root = Path("/proc")
    if proc_root.is_dir():
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                if int(fields[2]) == pid and fields[0] != "Z":
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def parse_line(line: str, parsed: dict[str, Any]) -> None:
    """Update the shared parsed-state dict from one line of child output (in place)."""
    line = _ANSI_RE.sub("", line)
    if line.startswith("[yamkit-result] ") and len(line) <= 65536:
        result = json.loads(line[len("[yamkit-result] "):])
        if isinstance(result, dict):
            parsed["result"] = result
        return
    m = _READ_RE.match(line)
    if m:
        name, qs, grip, btn = m.groups()
        try:
            q = [float(x) for x in qs.split()]
        except ValueError:
            return
        parsed.setdefault("arms", {})[name] = {
            "q": q,
            "gripper": None if grip == "-" else float(grip),
            "buttons": btn,
            "t": time.time(),
        }
        return
    m = _TELEOP_HZ_RE.search(line)
    if m:
        parsed["rate_hz"] = float(m.group(1))
        for name, state, err, grip in _TELEOP_PAIR_RE.findall(line):
            try:
                err_f = float(err)
            except ValueError:
                err_f = None
            parsed.setdefault("pairs", {})[name] = {
                "engaged": state == "ENGAGED",
                "error_rad": err_f,
                "gripper": None if grip == "-" else _maybe_float(grip),
            }
        return
    m = _EPISODE_RE.search(line)
    if m:
        parsed["episode"] = int(m.group(1))
        parsed["phase"] = "record"
        parsed["phase_since"] = time.time()
        return
    if _RESET_RE.search(line):
        parsed["phase"] = "reset"
        parsed["phase_since"] = time.time()
        return
    if _UPLOAD_RE.search(line):
        parsed["phase"] = "upload"
        parsed["phase_since"] = time.time()
        return
    m = _FIRST_CALL_RE.search(line)
    if m:
        parsed["first_call_ms"] = float(m.group(1))
        return
    m = _NEXT_CALLS_RE.search(line)
    if m:
        vals = re.findall(r"([\d.]+)\s*ms", m.group(1))
        if vals:
            parsed["step_call_ms"] = [float(v) for v in vals]


def _maybe_float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


class SessionManager:
    """Runs at most one yamkit child process; keeps a log ring buffer and parsed live state."""

    def __init__(
        self,
        *,
        python: str = sys.executable,
        log_lines: int = 400,
        on_start: Callable[[str], None] | None = None,
        on_exit: Callable[[dict[str, Any]], None] | None = None,
        on_phase: Callable[[str, str], None] | None = None,
        on_camera_acquire: Callable[[str], bool | None] | None = None,
        on_camera_release: Callable[[str], Any] | None = None,
    ) -> None:
        self._python = python
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self.on_start = on_start
        self.on_exit = on_exit
        self.on_phase = on_phase
        self.on_camera_acquire = on_camera_acquire
        self.on_camera_release = on_camera_release
        self._session = ""
        self._token = ""
        self._finishing = False
        self._camera_owner: str | None = None
        self._camera_names: tuple[str, ...] = ()
        self._owner_confirmed = False
        self._camera_claimed_at: float | None = None
        self._seen_owners: set[str] = set()
        self._preview: PreviewRegistration | None = None
        self._preview_registered = False
        self.preview_generation = 0
        self.log: deque[str] = deque(maxlen=log_lines)
        self.parsed: dict[str, Any] = {}
        self.mode: str | None = None
        self.meta: dict[str, Any] = {}
        self.started_at: float | None = None
        self.ended_at: float | None = None
        self.returncode: int | None = None
        self.stopping = False

    # ----- command building -------------------------------------------------------------------
    def yamkit_argv(self, *cli_args: str) -> list[str]:
        return [self._python, "-m", "yamkit.cli", *cli_args]

    # ----- lifecycle --------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._proc is not None and (self._proc.poll() is None or self._finishing)

    @property
    def cameras_owned(self) -> bool:
        return self._camera_owner is not None

    @property
    def preview_starting(self) -> bool:
        return (self._camera_claimed_at is not None
                and time.monotonic() - self._camera_claimed_at < PREVIEW_START_TIMEOUT_S)

    def preview_registration(self, name: str | None = None) -> PreviewRegistration | None:
        with self._lock:
            reg = self._preview
            if reg is None or not self.active or (name is not None and name not in reg.cameras):
                return None
            return reg

    def preview_is_current(self, reg: PreviewRegistration) -> bool:
        with self._lock:
            return self.active and self._preview is reg and self._camera_owner == reg.owner

    def start(self, mode: str, argv: list[str], meta: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.active or (self._reader is not None and self._reader.is_alive()):
                raise RuntimeError(f"a {self.mode!r} session is already running (stop it first)")
            env = dict(os.environ, PYTHONUNBUFFERED="1", COLUMNS="300", NO_COLOR="1")
            session, token = secrets.token_hex(16), secrets.token_urlsafe(32)
            env.update(YAMKIT_PREVIEW_SESSION=session, YAMKIT_PREVIEW_TOKEN=token)
            env.pop("YAMKIT_OPENAI_API_KEY", None)
            env.pop("DATABASE_URL", None)
            # LeRobot's recorder grabs the keyboard system-wide when it sees a display (Esc / arrows /
            # n / r / q in *any* window would end or skip an episode). Sessions started from the UI
            # are controlled by the UI's buttons only.
            env.pop("DISPLAY", None)
            env.pop("WAYLAND_DISPLAY", None)
            if self.on_start:
                self.on_start(mode)
            self.log.clear()
            self.parsed = {}
            self.mode = mode
            self.meta = meta or {}
            self.started_at = time.time()
            self.ended_at = self.returncode = None
            self.stopping = False
            self._session, self._token = session, token
            self._preview = None
            self._preview_registered = False
            self._seen_owners.clear()
            self.preview_generation += 1
            self.log.append("$ " + " ".join(argv))
            try:
                self._proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                    start_new_session=True,  # own group → signals reach LeRobot children too
                )
            except OSError:
                self._proc = None
                self.ended_at = time.time()
                self.returncode = -1
                self._session = self._token = ""
                if self.on_exit:
                    self.on_exit(self.status())
                raise RuntimeError("could not start the requested CLI process") from None
            self._finishing = True
            group_gone = threading.Event()
            threading.Thread(
                target=self._watch_process, args=(self._proc, group_gone), daemon=True,
            ).start()
            self._reader = threading.Thread(
                target=self._read_output, args=(self._proc, session, group_gone), daemon=True,
            )
            self._reader.start()
        return self.status()

    def _watch_process(self, proc: subprocess.Popen, group_gone: threading.Event) -> None:
        proc.wait()
        # A crashed launcher can leave a recorder holding devices and/or stdout open. Terminate
        # that group before declaring ownership released. Never fall back to direct capture early.
        if _group_alive(proc.pid):
            self._signal(proc, signal.SIGTERM)
            deadline = time.monotonic() + 2.0
            while _group_alive(proc.pid) and time.monotonic() < deadline:
                time.sleep(0.05)
            if _group_alive(proc.pid):
                self._signal(proc, signal.SIGKILL)
        while _group_alive(proc.pid):
            time.sleep(0.1)
        group_gone.set()

    def _read_output(self, proc: subprocess.Popen, session: str, group_gone: threading.Event) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            with self._lock:
                if self._proc is not proc or self._session != session:
                    continue
                if self._control_line(line, proc):
                    continue
                if line.strip():
                    line = line.replace(self._token, "[redacted]") if self._token else line
                    self.log.append(line)
                    before = self.parsed.get("phase")
                    try:
                        parse_line(line, self.parsed)
                    except Exception:
                        log.debug("unparseable session line", exc_info=True)
                    if self.on_phase and self.parsed.get("phase") != before:
                        try:
                            self.on_phase(self.mode or "", self.parsed.get("phase") or "")
                        except Exception:
                            log.exception("session on_phase hook failed")
        group_gone.wait()
        with self._lock:
            if self._proc is not proc or self._session != session:
                return
            self._release_cameras()
            self._finishing = False
            self.ended_at = time.time()
            self.returncode = proc.returncode
            self._token = ""
            if proc.stdin:
                proc.stdin.close()
            proc.stdout.close()
            if self.on_exit:
                try:
                    self.on_exit(self.status())
                except Exception:
                    log.exception("session on_exit hook failed")

    def _control_line(self, line: str, proc: subprocess.Popen) -> bool:
        """Consume the two explicit control protocols; never place protocol data in logs."""
        if not line.startswith(("@yamkit-preview/", "@yamkit-cameras/")):
            return False
        prefix = next((p for p in (_OWNERSHIP_PREFIX, _PREVIEW_PREFIX) if line.startswith(p)), None)
        if prefix is None or len(line) > _MAX_CONTROL_LINE:
            return True
        try:
            msg = json.loads(line[len(prefix):])
        except (ValueError, RecursionError):
            return True
        if not isinstance(msg, dict) or msg.get("v") != 1 or msg.get("session") != self._session:
            return True
        owner = msg.get("owner")
        if not isinstance(owner, str) or not 1 <= len(owner) <= 128:
            return True
        if prefix == _PREVIEW_PREFIX:
            names, port = _camera_names(msg.get("cameras")), msg.get("port")
            if (
                owner != self._camera_owner or not self._owner_confirmed or self._preview_registered
                or names is None or set(names) != set(self._camera_names)
                or type(port) is not int or not 1 <= port <= 65535
            ):
                return True
            self._preview = PreviewRegistration(self._session, owner, port, names, self._token)
            self._preview_registered = True
            self.preview_generation += 1
            return True
        event = msg.get("event")
        if event == "release":
            if owner == self._camera_owner and self._owner_confirmed:
                self._release_cameras()
            return True
        if event != "acquire":
            return True
        names = _camera_names(msg.get("cameras"))
        ok = False
        if names and self._camera_owner is None and owner not in self._seen_owners:
            self._seen_owners.add(owner)
            self._camera_owner, self._camera_names = owner, names
            self._camera_claimed_at = time.monotonic()
            self._preview_registered = False
            self.preview_generation += 1
            try:
                ok = self.on_camera_acquire is None or self.on_camera_acquire(owner) is not False
            except Exception:
                log.warning("camera release confirmation failed", exc_info=True)
            self._owner_confirmed = ok
        try:
            assert proc.stdin is not None
            proc.stdin.write(json.dumps({"session": self._session, "owner": owner, "ok": ok}) + "\n")
            proc.stdin.flush()
        except (OSError, ValueError):
            pass  # Retain ownership until the child/group is confirmed gone.
        return True

    def _release_cameras(self) -> None:
        owner = self._camera_owner
        self._preview = None
        self._preview_registered = False
        if owner is None:
            return
        if self.on_camera_release:
            try:
                self.on_camera_release(owner)
            except Exception:
                log.exception("camera resume hook failed")
        self._camera_owner = None
        self._camera_names = ()
        self._owner_confirmed = False
        self._camera_claimed_at = None
        self.preview_generation += 1

    def stop(self, grace_s: float | None = None) -> dict[str, Any]:
        """SIGINT the child's process group (= Ctrl-C), escalate in the background if it hangs.

        The grace period covers the arms' slow return to home (30 s), or a recording's upload to the
        Hub afterwards (10 min); calling stop() again sends a second SIGINT, which makes the CLIs
        release the arms immediately."""
        proc = self._proc
        if proc is None or not self.active:
            return self.status()
        if grace_s is None:
            grace_s = 600.0 if self.mode in ("record", "push", "pull") else 30.0
        self.stopping = True
        self._signal(proc, signal.SIGINT)
        threading.Thread(target=self._escalate, args=(proc, grace_s), daemon=True).start()
        return self.status()

    def _escalate(self, proc: subprocess.Popen, grace_s: float) -> None:
        for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 4.0)):
            deadline = time.monotonic() + wait
            while _group_alive(proc.pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            if not _group_alive(proc.pid):
                return
            self._signal(proc, sig)

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def wait(self, timeout: float | None = None) -> int | None:
        started = time.monotonic()
        if self._proc is not None:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            if self._reader is not None:
                remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
                self._reader.join(timeout=remaining)
        return self.returncode

    def close(self, timeout: float = 2.0) -> bool:
        """Request normal hardware shutdown, with bounded waiting and unchanged stop grace."""
        self.stop()
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=max(0.0, timeout))
        return not self.active

    # ----- reporting --------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        active = self.active
        now = time.time()
        return {
            "active": active,
            "mode": self.mode,
            "pid": self._proc.pid if self._proc is not None else None,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed_s": (
                round((self.ended_at or now) - self.started_at, 1) if self.started_at else None
            ),
            "returncode": self.returncode,
            "stopping": self.stopping and active,
            "cameras_owned": self.cameras_owned,
            "preview_generation": self.preview_generation,
            "meta": self.meta,
            "parsed": self.parsed,
            "phase_elapsed_s": round(now - self.parsed["phase_since"], 1) if active and self.parsed.get("phase_since") else None,
            "log": list(self.log),
        }


# ------------------------------------------------------------------- deployment run registry --
class DeploymentLog:
    """Records rollout / policy-check runs launched from the UI under outputs/ui/deployments/."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def create(self, status: dict[str, Any]) -> Path:
        import uuid

        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + (status.get("mode") or "run") + "-" + uuid.uuid4().hex[:8]
        d = self.root / run_id
        d.mkdir(parents=True, exist_ok=True)
        self._write_meta(d, status, run_id)
        return d

    def finalize(self, run_dir: Path, status: dict[str, Any]) -> None:
        (run_dir / "log.txt").write_text("\n".join(status.get("log", [])) + "\n")
        self._write_meta(run_dir, status, run_dir.name)

    @staticmethod
    def _write_meta(d: Path, status: dict[str, Any], run_id: str) -> None:
        parsed = status.get("parsed", {})
        rc = status.get("returncode")
        if status.get("active", False) or rc is None:
            outcome, termination = "running", None
        elif status.get("stopping") or rc in (-signal.SIGINT, -signal.SIGTERM, -signal.SIGKILL):
            outcome, termination = "stopped", "stopped by user"
        elif rc == 0:
            outcome, termination = "success", "completed"
        else:
            outcome, termination = "failed", f"exit code {rc}"
        meta = {
            "id": run_id,
            "kind": status.get("mode"),
            "policy": status.get("meta", {}).get("policy"),
            "task": status.get("meta", {}).get("task"),
            "started_at": status.get("started_at"),
            "ended_at": status.get("ended_at"),
            "duration_s": status.get("elapsed_s"),
            "returncode": rc,
            "status": outcome,
            "termination": termination,
            "first_call_ms": parsed.get("first_call_ms"),
            "step_call_ms": parsed.get("step_call_ms"),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
