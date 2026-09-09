"""Literal reference interpolation and validation, without command retiming.

The linked YAM runner's spatial samples are retained exactly. The runtime owns
its send/Rate.sleep/observation ordering. This module performs no clock reads,
hardware calls, speed/acceleration shaping, endpoint holds or policy inference.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from yamkit.inference.command_shaping import ACTION_NAMES, FIRST_DT, JOINT_NAMES, MAX_SAMPLES, TOLERANCE
from yamkit.validation import finite_scalar

GRIPPER_NAMES = tuple(name for name in ACTION_NAMES if name not in JOINT_NAMES)


class ReferenceInterpolationFault(ValueError):
    """A reference target, response or execution lifecycle is invalid."""


def _action(value):
    if not isinstance(value, Mapping) or set(value) != set(ACTION_NAMES):
        raise ReferenceInterpolationFault("Reference action requires exactly 14 named values")
    result = {name: finite_scalar(value[name], f"reference {name}") for name in ACTION_NAMES}
    if any(not 0 <= result[name] <= 1 for name in GRIPPER_NAMES):
        raise ReferenceInterpolationFault("Reference gripper target must be in [0, 1]")
    return result


def _bounds(joint_limits, gripper_max_step):
    if not isinstance(joint_limits, Mapping) or set(joint_limits) != set(JOINT_NAMES):
        raise ReferenceInterpolationFault("Reference bounds require all 12 follower joints")
    lower = np.array([finite_scalar(joint_limits[n]["lower"], "joint lower bound") for n in JOINT_NAMES])
    upper = np.array([finite_scalar(joint_limits[n]["upper"], "joint upper bound") for n in JOINT_NAMES])
    if np.any(lower >= upper):
        raise ReferenceInterpolationFault("Invalid reference joint bounds")
    # Preserve the shared caller schema, but these speed-derived values do not
    # alter literal reference points or timing. Other controller modes use them.
    if not isinstance(gripper_max_step, Mapping) or set(gripper_max_step) != set(GRIPPER_NAMES):
        raise ReferenceInterpolationFault("Reference configuration requires both gripper step values")
    for name in JOINT_NAMES:
        finite_scalar(joint_limits[name]["max_step"], "configured joint step", positive=True)
    for name in GRIPPER_NAMES:
        finite_scalar(gripper_max_step[name], "configured gripper step", positive=True)
    return lower, upper


def _check_positions(action, lower, upper):
    q = np.array([action[n] for n in JOINT_NAMES])
    if np.any(q < lower) or np.any(q > upper):
        raise ReferenceInterpolationFault("Original reference target is outside robot joint bounds")
    return q


def _difference(start, target):
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.array([target[n] for n in ACTION_NAMES]) - np.array([start[n] for n in ACTION_NAMES])
    if not np.all(np.isfinite(delta)):
        raise ReferenceInterpolationFault("Reference displacement is not finite")
    return delta


def reference_row(start: Mapping, target: Mapping) -> tuple[dict[str, float], ...]:
    """Literal ``min(int(max14delta/.01),100)`` / endpoint-inclusive linspace.

    The reference sends the target directly when the computed count is 0 or 1.
    This oracle performs no retiming and has no per-rig joint bounds.
    """
    first, last = _action(start), _action(target)
    largest = float(np.max(np.abs(_difference(first, last))))
    count = 100 if largest >= 1 else int(largest / .01)
    if count <= 1:
        return (last,)
    points = np.linspace([first[n] for n in ACTION_NAMES], [last[n] for n in ACTION_NAMES], count)
    return tuple(dict(zip(ACTION_NAMES, row.tolist(), strict=True)) for row in points)


@dataclass(frozen=True)
class ReferenceRowPlan:
    commands: tuple[dict[str, float], ...]
    progress: tuple[float, ...]
    reference_count: int
    period_s: float
    motion_intervals: int

    @property
    def ticks(self):
        return len(self.commands)

    @property
    def added_ticks(self):
        return self.ticks - self.reference_count

    @property
    def duration_s(self):
        """Nominal tick count only; actual reference send/sleep ordering differs."""
        return self.ticks * self.period_s


def plan_reference_row(start: Mapping, target: Mapping, *, joint_limits: Mapping,
                       gripper_max_step: Mapping, period_s: float = FIRST_DT) -> ReferenceRowPlan:
    """Validate then return the exact reference samples, with no extra points."""
    first, last = _action(start), _action(target)
    lower, upper = _bounds(joint_limits, gripper_max_step)
    _check_positions(first, lower, upper)
    _check_positions(last, lower, upper)
    period = finite_scalar(period_s, "reference nominal period", positive=True)
    commands = reference_row(first, last)
    for command in commands:
        _check_positions(command, lower, upper)
    count = len(commands)
    progress = tuple(np.linspace(0, 1, count).tolist()) if count > 1 else (1.0,)
    return ReferenceRowPlan(commands, progress, count, period, max(0, count - 1))


@dataclass(frozen=True)
class ReferenceCommandStep:
    generation: int
    monotonic_s: float
    dt_s: float | None
    requested: dict
    shaped: dict
    previous_position: np.ndarray
    previous_velocity: np.ndarray


class ReferenceCommandGuard:
    """Validate unchanged targets and lifecycle; slopes are metrics, not limits."""

    def __init__(self, initial_position: Mapping, joint_limits: Mapping, gripper_max_step: Mapping,
                 *, period_s: float = FIRST_DT):
        self.lower, self.upper = _bounds(joint_limits, gripper_max_step)
        self.period_s = finite_scalar(period_s, "reference nominal period", positive=True)
        self.valid, self.generation, self.anchor_initialized = True, 0, False
        self.last_at, self._pending, self._pending_target = None, None, None
        self._wait_deadline, self._wait_record, self._clock_floor = None, None, None
        self.samples, self.inference_waits = deque(maxlen=MAX_SAMPLES), deque(maxlen=128)
        self.postclamp_modified_count = 0
        self.maximum_command_velocity_rad_s = self.maximum_command_acceleration_rad_s2 = 0.0
        self._velocity_known = False
        self.initialize_position(initial_position)

    @property
    def last_action(self):
        return dict(self._last_action)

    def invalidate(self):
        self.valid = False

    def _fault(self, message):
        self.invalidate()
        raise ReferenceInterpolationFault(message)

    def initialize_position(self, initial_position):
        if not self.valid or self.generation or self._pending is not None or self._wait_deadline is not None:
            self._fault("Cannot reinitialize a used or invalidated reference guard")
        action = _action(initial_position)
        self.position = _check_positions(action, self.lower, self.upper)
        self.initial_position = self.position.copy()
        self.initial_action = dict(action)
        self.velocity = np.zeros(len(JOINT_NAMES))
        self._last_action = action
        self.anchor_initialized = True

    def begin_inference_wait(self, deadline):
        if not self.valid or self._pending is not None or self._wait_deadline is not None:
            self._fault("Reference inference wait requires a valid committed command")
        try:
            limit = finite_scalar(deadline, "inference wait deadline", positive=True)
        except ValueError:
            self.invalidate()
            raise
        floor = self.last_at if self.last_at is not None else self._clock_floor
        if floor is not None and limit <= floor:
            self._fault("Reference inference wait deadline already elapsed")
        self._wait_deadline = limit
        self._wait_record = {"last_dispatch_monotonic_s": self.last_at,
                             "deadline_monotonic_s": limit, "resumed_monotonic_s": None,
                             "command_dispatches_during_wait": 0}
        self.inference_waits.append(self._wait_record)

    def end_inference_wait(self, now):
        try:
            at = finite_scalar(now, "inference wait end")
        except ValueError:
            self.invalidate()
            raise
        floor = self.last_at if self.last_at is not None else self._clock_floor
        if (not self.valid or self._pending is not None or self._wait_deadline is None or at >= self._wait_deadline
                or (floor is not None and at < floor)):
            self._fault("Reference inference wait expired or became invalid")
        self._wait_record["resumed_monotonic_s"] = at
        self._wait_deadline, self._wait_record = None, None
        self._clock_floor = at
        # Keep the original cached command and its actual dispatch time. The
        # reference does not add a zero-velocity hold or restart its trajectory.

    def prepare(self, requested: Mapping, *, now: float) -> ReferenceCommandStep:
        if not self.valid or self._pending is not None or self._wait_deadline is not None:
            self._fault("Reference command is invalid, pending, or paused for inference")
        try:
            action = _action(requested)
            _check_positions(action, self.lower, self.upper)
            at = finite_scalar(now, "reference dispatch time")
            if (self._clock_floor is not None and at < self._clock_floor) or (
                    self.last_at is not None and at <= self.last_at):
                self._fault("Reference dispatch clock must move strictly forwards")
            dt = None if self.last_at is None else at - self.last_at
            step = ReferenceCommandStep(self.generation, at, dt, dict(action), dict(action),
                                        self.position.copy(), self.velocity.copy())
            if not self.valid:
                self._fault("Reference command invalidated during preparation")
            self._pending, self._pending_target = step, dict(action)
            return step
        except ValueError:
            self.invalidate()
            raise

    def commit(self, step: ReferenceCommandStep, sent: Mapping, *, deadline_monotonic_s: float | None):
        if (not self.valid or self._pending is not step or step.generation != self.generation
                or step.requested != self._pending_target or step.shaped != self._pending_target):
            self._fault("Cannot commit an invalid, mutated or superseded reference command")
        try:
            actual = _action(sent)
            position = _check_positions(actual, self.lower, self.upper)
        except ValueError:
            self.invalidate()
            raise
        modified = any(abs(actual[n] - self._pending_target[n]) > TOLERANCE for n in ACTION_NAMES)
        velocity = np.zeros(len(JOINT_NAMES)) if step.dt_s is None else (position - step.previous_position) / step.dt_s
        acceleration = ((velocity - step.previous_velocity) / step.dt_s
                        if step.dt_s is not None and self._velocity_known else None)
        if step.dt_s is not None:
            self.maximum_command_velocity_rad_s = max(self.maximum_command_velocity_rad_s, float(np.abs(velocity).max()))
        if acceleration is not None:
            self.maximum_command_acceleration_rad_s2 = max(self.maximum_command_acceleration_rad_s2, float(np.abs(acceleration).max()))
        self.postclamp_modified_count += int(modified)
        self.samples.append({"dispatch_index": self.generation, "monotonic_s": step.monotonic_s,
                             "dispatch_role": "interpolation", "deadline_monotonic_s": deadline_monotonic_s,
                             "dt_s": step.dt_s, "requested": dict(self._pending_target),
                             "shaped": dict(self._pending_target), "sent": actual,
                             "postclamp_modified": modified, "postclamp_bounds_exceeded": False})
        self.position, self.velocity, self.last_at = position, velocity, step.monotonic_s
        self._velocity_known = step.dt_s is not None
        self._clock_floor = step.monotonic_s
        self._last_action, self._pending, self._pending_target = actual, None, None
        self.generation += 1
        if modified:
            self._fault("Postclamp reference command changed the literal target; stopping")

    def metrics(self):
        return {"settings": {"mode": "reference_literal", "custom_joint_shaping": False,
                             "yamarm_speed_clamp": False, "max_joint_velocity_rad_s": None,
                             "max_joint_acceleration_rad_s2": None, "max_dispatch_gap_s": None,
                             "nominal_period_s": self.period_s, "timing_deviation": "none in interpolation points",
                             "requested_basis": "literal reference interpolated command; original model rows require plan metadata",
                             "timestamp_basis": "host command preparation; slopes include actual RPC gaps, are not limits or motor dynamics; first interval unknown"},
                "initial_joint_position": dict(zip(JOINT_NAMES, self.initial_position.tolist(), strict=True)),
                "initial_action": dict(self.initial_action),
                "sample_count": self.generation, "samples_dropped": max(0, self.generation - len(self.samples)),
                "postclamp_modified_count": self.postclamp_modified_count,
                "maximum_command_velocity_rad_s": self.maximum_command_velocity_rad_s,
                "maximum_command_acceleration_rad_s2": self.maximum_command_acceleration_rad_s2,
                "samples": list(self.samples), "inference_waits": [dict(wait) for wait in self.inference_waits]}
