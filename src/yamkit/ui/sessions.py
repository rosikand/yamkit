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
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Modes whose child process opens the rig cameras (the UI's own streams must let go first).
CAMERA_MODES = frozenset({"record", "teleoperate", "rollout"})
# Modes that energise motors (gravity-comp on connect; teleop/record/rollout also move them).
HARDWARE_MODES = frozenset({"read", "teleop", "teleoperate", "record", "rollout", "rest"})

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


def parse_line(line: str, parsed: dict[str, Any]) -> None:
    """Update the shared parsed-state dict from one line of child output (in place)."""
    line = _ANSI_RE.sub("", line)
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
    ) -> None:
        self._python = python
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self.on_start = on_start
        self.on_exit = on_exit
        self.on_phase = on_phase
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
        return self._proc is not None and self._proc.poll() is None

    def start(self, mode: str, argv: list[str], meta: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            if self.active:
                raise RuntimeError(f"a {self.mode!r} session is already running (stop it first)")
            env = dict(os.environ, PYTHONUNBUFFERED="1", COLUMNS="300", NO_COLOR="1")
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
            self.log.append("$ " + " ".join(argv))
            self._proc = subprocess.Popen(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,  # own process group → signals reach lerobot children too
            )
            self._reader = threading.Thread(target=self._read_output, args=(self._proc,), daemon=True)
            self._reader.start()
        return self.status()

    def _read_output(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if line.strip():
                self.log.append(line)
                before = self.parsed.get("phase")
                try:
                    parse_line(line, self.parsed)
                except Exception:
                    log.debug("unparseable line: %r", line, exc_info=True)
                if self.on_phase and self.parsed.get("phase") != before:
                    try:
                        self.on_phase(self.mode or "", self.parsed.get("phase") or "")
                    except Exception:
                        log.exception("session on_phase hook failed")
        proc.wait()
        self.ended_at = time.time()
        self.returncode = proc.returncode
        if self.on_exit:
            try:
                self.on_exit(self.status())
            except Exception:
                log.exception("session on_exit hook failed")

    def stop(self, grace_s: float | None = None) -> dict[str, Any]:
        """SIGINT the child's process group (= Ctrl-C), escalate in the background if it hangs.

        The grace period covers the arms' slow return to home (30 s), or a recording's upload to the
        Hub afterwards (10 min); calling stop() again sends a second SIGINT, which makes the CLIs
        release the arms immediately."""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return self.status()
        if grace_s is None:
            grace_s = 600.0 if self.mode in ("record", "push", "pull") else 30.0
        self.stopping = True
        self._signal(proc, signal.SIGINT)
        threading.Thread(target=self._escalate, args=(proc, grace_s), daemon=True).start()
        return self.status()

    def _escalate(self, proc: subprocess.Popen, grace_s: float) -> None:
        for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 4.0)):
            try:
                proc.wait(timeout=wait)
                return
            except subprocess.TimeoutExpired:
                self._signal(proc, sig)
        proc.wait()

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def wait(self, timeout: float | None = None) -> int | None:
        if self._proc is not None:
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
            if self._reader is not None:
                self._reader.join(timeout=2.0)
        return self.returncode

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
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + (status.get("mode") or "run")
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
