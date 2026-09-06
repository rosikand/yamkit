#!/usr/bin/env python3
r"""Profile native teleop without changing its commands, validation, or cleanup.

This CAN ENABLE AND MOVE ARMS. Obtain approval for the exact invocation first.
Example (arguments after -- go unchanged to ``yamkit teleop``):
  .venv/bin/python scripts/profile_teleop.py --output .context/teleop-profile.json -- \
      --rig configs/rig.yaml --pair left_follower --duration 30 --no-home --bilateral-kp 0

Only main-thread SDK reads made inside a control step are measured. Startup,
homing, and teardown reads are excluded. Existing SDK loop timing logs are also
collected without enabling verbose console logging. The JSON is written only
after the original CLI cleanup confirms every profiled arm closed. No report is
written for help or a command that fails before entering TeleopSession.run.
"""

# These catches apply only to diagnostics; exceptions from the original control
# methods propagate unchanged through their original cleanup.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LIMIT = 8192
LOG_LIMIT = 128


class Samples:
    def __init__(self):
        self.count = 0
        self.total_ns = 0
        self.max_ns = 0
        self.recent = deque(maxlen=SAMPLE_LIMIT)

    def add(self, elapsed_ns):
        self.count += 1
        self.total_ns += elapsed_ns
        self.max_ns = max(self.max_ns, elapsed_ns)
        self.recent.append(elapsed_ns)

    def summary(self):
        ordered = sorted(self.recent)
        return {
            "count": self.count,
            "mean_ms": self.total_ns / self.count / 1e6 if self.count else None,
            "max_ms": self.max_ns / 1e6 if self.count else None,
            "retained_samples": len(ordered),
            "recent_percentiles_ms": {
                str(p): ordered[min(len(ordered) - 1, (len(ordered) * p + 99) // 100 - 1)] / 1e6
                for p in (50, 95, 99)
            }
            if ordered
            else {},
        }


class SdkTimingHandler(logging.Handler):
    """Bounded, report-only capture of the SDK's existing periodic timing logs."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.records = deque(maxlen=LOG_LIMIT)
        self.count = 0
        self.failed = False

    def emit(self, record):
        try:
            if "/i2rt/" not in record.pathname.replace("\\", "/"):
                return
            message = record.getMessage()
            if not any(
                marker in message
                for marker in (
                    "Grav Comp Control Frequency:",
                    "Gravity compensation control loop is slow",
                    "step_time >",
                    "Total rate:",
                )
            ):
                return
            self.count += 1
            self.records.append(
                {"time": record.created, "level": record.levelname, "message": message[:2048]}
            )
        except Exception:
            self.failed = True  # diagnostic failures must not escape into a motor worker


class Profile:
    def __init__(self, teleop_args):
        self.teleop_args = teleop_args
        self.session = None
        self.samples = {}
        self.timing_failed = False
        self.cli_error = None
        self.sdk = SdkTimingHandler()
        self._in_step = False
        self._thread = threading.get_ident()

    def _clock(self):
        try:
            return time.perf_counter_ns()
        except Exception:
            self.timing_failed = True
            return None

    def _record(self, label, start):
        try:
            end = self._clock()
            if start is not None and end is not None:
                if label not in self.samples:
                    self.samples[label] = Samples()
                self.samples[label].add(max(0, end - start))
        except Exception:
            self.timing_failed = True

    def _timed(self, function, label, *, sdk=False):
        @wraps(function)
        def call(*args, **kwargs):
            measure = not sdk or (self._in_step and threading.get_ident() == self._thread)
            start = self._clock() if measure else None
            try:
                return function(*args, **kwargs)
            finally:
                if measure:
                    self._record(label, start)

        return call

    def run(self, original, session, *args, **kwargs):
        self.session = session
        stack = ExitStack()

        def restore():
            try:
                stack.close()
            except Exception:
                self.timing_failed = True

        try:
            try:
                self._install(stack, session)
            except Exception:
                self.timing_failed = True
                restore()
            return original(session, *args, **kwargs)
        finally:
            restore()

    def _install(self, stack, session):
        step = self._timed(session.step, "session.step")

        def profiled_step():
            self._in_step = True
            try:
                return step()
            finally:
                self._in_step = False

        stack.enter_context(patch.object(session, "step", profiled_step))
        if session.on_tick is not None:
            stack.enter_context(
                patch.object(session, "on_tick", self._timed(session.on_tick, "session.on_tick"))
            )
        for pair in session.pairs:
            for arm in (pair.leader, pair.follower):
                robot = arm.robot
                stack.enter_context(
                    patch.object(
                        robot,
                        "get_observations",
                        self._timed(
                            robot.get_observations,
                            f"{arm.name}.get_observations",
                            sdk=True,
                        ),
                    )
                )
                if arm.spec.has_handle:
                    chain = robot.motor_chain
                    stack.enter_context(
                        patch.object(
                            chain,
                            "get_same_bus_device_states",
                            self._timed(
                                chain.get_same_bus_device_states,
                                f"{arm.name}.get_same_bus_device_states",
                                sdk=True,
                            ),
                        )
                    )
        logger = logging.getLogger()
        logger.addHandler(self.sdk)  # CLI logging is already configured; console filters stay intact
        stack.callback(logger.removeHandler, self.sdk)

    def write_after_cleanup(self, output):
        if self.session is None:
            return
        try:
            arms = [arm for pair in self.session.pairs for arm in (pair.leader, pair.follower)]
            if not all(arm._closed for arm in arms):
                raise RuntimeError("arm closure was not confirmed; timing report withheld")
            stats = self.session.stats
            report = {
                "command": ["yamkit", "teleop", *self.teleop_args],
                "arms_closed": True,
                "cli_error": self.cli_error,
                "timing_failed": self.timing_failed or self.sdk.failed,
                "scope": "SDK reads on the control thread inside step; step and on_tick measured separately",
                "percentiles_scope": f"most recent {SAMPLE_LIMIT} calls per label; count/mean/max include all calls",
                "stats": {"ticks": stats.ticks, "rate_hz": stats.rate_hz, "overruns": stats.overruns},
                "timings": {label: values.summary() for label, values in self.samples.items()},
                "sdk_timing_record_count": self.sdk.count,
                "sdk_timing_records": list(self.sdk.records),
            }
            output = output.resolve()
            if not output.is_relative_to(ROOT):
                raise ValueError("output must remain inside this repository")
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x") as stream:
                json.dump(report, stream, indent=2)
                stream.write("\n")
            print(f"Timing report written after arm closure: {output}", file=sys.stderr)
        except Exception as error:
            # A full disk, malformed diagnostic record, etc. must not hide the original CLI error.
            try:
                print(f"Timing report unavailable: {type(error).__name__}: {error}", file=sys.stderr)
            except Exception:
                self.timing_failed = True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", type=Path, required=True, help="new JSON file inside this repository")
    parser.add_argument(
        "teleop_args", nargs=argparse.REMAINDER, help="arguments after -- go to yamkit teleop"
    )
    options = parser.parse_args(argv)
    output = options.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("--output must be a new file inside this repository")
    teleop_args = options.teleop_args
    if teleop_args[:1] == ["--"]:
        teleop_args = teleop_args[1:]

    from yamkit import cli
    from yamkit.teleop import TeleopSession

    profile = Profile(teleop_args)
    original_run = TeleopSession.run

    def run(session, *args, **kwargs):
        return profile.run(original_run, session, *args, **kwargs)

    try:
        with patch.object(TeleopSession, "run", run):
            cli.app(args=["teleop", *teleop_args], prog_name="yamkit")
    except BaseException as error:
        if not isinstance(error, SystemExit) or error.code not in (None, 0):
            profile.cli_error = type(error).__name__
        raise
    finally:
        profile.write_after_cleanup(output)


if __name__ == "__main__":
    main()
