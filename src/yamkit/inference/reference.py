"""Pure whole-vector reference interpolation and command validation.

``reference_row`` reproduces the linked YAM runner's spatial samples. The
production plan preserves that straight path and every model-row endpoint, but
retimes it with one shared smoothstep and endpoint holds for existing safety
bounds. It is deliberately not literal reference timing. No clocks, sleeping,
robot reads, network calls or hardware operations occur in this module.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from yamkit.inference.command_shaping import (
    ACTION_NAMES,
    FIRST_DT,
    JOINT_NAMES,
    MAX_ACCELERATION,
    MAX_DISPATCH_GAP,
    MAX_SAMPLES,
    MAX_TIME_BUDGET,
    MAX_VELOCITY,
    TOLERANCE,
    CommandStep,
)
from yamkit.validation import finite_scalar

GRIPPER_NAMES = tuple(name for name in ACTION_NAMES if name not in JOINT_NAMES)
PLAN_ACCELERATION = 1.0  # Headroom for ordinary dispatch jitter; actual checks still apply.
MAX_PLAN_TICKS = 10000


class ReferenceInterpolationFault(ValueError):
    """A coordinated reference command cannot satisfy the unchanged bounds."""


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
    step = np.array([finite_scalar(joint_limits[n]["max_step"], "joint step", positive=True)
                     for n in JOINT_NAMES])
    if np.any(lower >= upper):
        raise ReferenceInterpolationFault("Invalid reference joint bounds")
    if not isinstance(gripper_max_step, Mapping) or set(gripper_max_step) != set(GRIPPER_NAMES):
        raise ReferenceInterpolationFault("Reference bounds require both gripper step limits")
    grips = {n: finite_scalar(gripper_max_step[n], "gripper step", positive=True) for n in GRIPPER_NAMES}
    return lower, upper, step, grips


def _check_positions(action, lower, upper):
    q = np.array([action[n] for n in JOINT_NAMES])
    if np.any(q < lower) or np.any(q > upper):
        raise ReferenceInterpolationFault("Original reference target is outside robot joint bounds")
    return q


def _period(period_s):
    period = finite_scalar(period_s, "reference period", positive=True)
    if period > MAX_DISPATCH_GAP:
        raise ReferenceInterpolationFault("Reference period exceeds the maximum dispatch gap")
    return period


def _difference(start, target):
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.array([target[n] for n in ACTION_NAMES]) - np.array([start[n] for n in ACTION_NAMES])
    if not np.all(np.isfinite(delta)):
        raise ReferenceInterpolationFault("Reference displacement is not finite")
    return delta


def reference_row(start: Mapping, target: Mapping) -> tuple[dict[str, float], ...]:
    """Literal ``min(int(max14delta/.01),100)`` / endpoint-inclusive linspace.

    The reference sends the target directly when the computed count is 0 or 1.
    This oracle performs no safety retiming and has no per-rig joint bounds.
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
        """Nominal time from the preceding committed command, including holds."""
        return self.ticks * self.period_s


def plan_reference_row(start: Mapping, target: Mapping, *, joint_limits: Mapping,
                       gripper_max_step: Mapping, period_s: float = FIRST_DT) -> ReferenceRowPlan:
    """A shared scalar path, exact endpoint, and zero-slope row transitions.

    All 14 coordinates use ``s(u)=3u²-2u³``. Its continuous derivative bounds
    (1.5 and 6) choose sufficient nominal intervals for joint speed/acceleration
    and every existing joint/gripper per-command cap. A terminal duplicate
    endpoint makes the last discrete command slope zero. Actual timing and
    post-clamp feedback must additionally pass ``ReferenceCommandGuard``.
    """
    first, last = _action(start), _action(target)
    lower, upper, step, grips = _bounds(joint_limits, gripper_max_step)
    q0, q1 = _check_positions(first, lower, upper), _check_positions(last, lower, upper)
    period = _period(period_s)
    delta = _difference(first, last)
    reference_count = len(reference_row(first, last))
    if not np.any(delta):
        return ReferenceRowPlan((dict(last),), (1.0,), reference_count, period, 0)
    distance = np.abs(q1 - q0)
    velocity = np.minimum(MAX_VELOCITY, step / MAX_TIME_BUDGET)
    requirements = [1.0, reference_count - 1,
                    float(np.max(1.5 * distance / (velocity * period))),
                    float(np.max(1.5 * distance / step)),
                    math.sqrt(6 * float(distance.max()) / PLAN_ACCELERATION) / period,
                    *[1.5 * abs(last[n] - first[n]) / grips[n] for n in GRIPPER_NAMES]]
    if not all(math.isfinite(value) for value in requirements) or max(requirements) > MAX_PLAN_TICKS - 2:
        raise ReferenceInterpolationFault("Reference row exceeds the bounded interpolation capacity")
    intervals = max(1, math.ceil(max(requirements)))
    linear = np.linspace(0, 1, intervals + 1)
    progress = np.clip(linear * linear * (3 - 2 * linear), 0, 1).tolist() + [1.0]
    commands = []
    for fraction in progress:
        # Preserve exact endpoint values instead of accumulating floating-point deltas.
        command = (dict(first) if fraction == 0 else dict(last) if fraction == 1 else
                   {n: first[n] + fraction * (last[n] - first[n]) for n in ACTION_NAMES})
        _check_positions(command, lower, upper)
        commands.append(command)
    return ReferenceRowPlan(tuple(commands), tuple(progress), reference_count, period, intervals)


class ReferenceCommandGuard:
    """Validate preplanned commands unchanged; never reshape or drop coordinates."""

    def __init__(self, initial_position: Mapping, joint_limits: Mapping, gripper_max_step: Mapping,
                 *, period_s: float = FIRST_DT):
        self.lower, self.upper, self.max_step, self.gripper_max_step = _bounds(joint_limits, gripper_max_step)
        self.max_velocity = np.minimum(MAX_VELOCITY, self.max_step / MAX_TIME_BUDGET)
        self.period_s = _period(period_s)
        self.valid, self.generation, self.anchor_initialized = True, 0, False
        self.last_at, self._pending, self._wait_deadline = None, None, None
        self._wait_anchor, self._wait_record = None, None
        self._clock_floor = None
        self.samples, self.inference_waits = deque(maxlen=MAX_SAMPLES), deque(maxlen=128)
        self.inference_hold_count = 0
        self.postclamp_modified_count = 0
        self.maximum_command_velocity_rad_s = self.maximum_command_acceleration_rad_s2 = 0.0
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
        self.velocity = np.zeros(len(JOINT_NAMES))
        self._velocity_all = np.zeros(len(ACTION_NAMES))
        self._last_action = action
        self.anchor_initialized = True

    def begin_inference_wait(self, deadline):
        if (not self.valid or self._pending is not None or self._wait_deadline is not None
                or np.any(np.abs(self._velocity_all) > TOLERANCE)):
            self._fault("Reference inference wait requires a valid stationary committed endpoint")
        try:
            limit = finite_scalar(deadline, "inference wait deadline", positive=True)
        except ValueError:
            self.invalidate()
            raise
        floor = self.last_at if self.last_at is not None else self._clock_floor
        if floor is not None and limit <= floor:
            self._fault("Reference inference wait deadline already elapsed")
        self._wait_deadline = limit
        self._wait_anchor = dict(self._last_action)
        self._wait_record = {"last_dispatch_monotonic_s": self.last_at,
                             "deadline_monotonic_s": limit, "resumed_monotonic_s": None,
                             "stationary": True, "hold_count": 0}
        self.inference_waits.append(self._wait_record)

    def end_inference_wait(self, now):
        try:
            at = finite_scalar(now, "inference wait end")
        except ValueError:
            self.invalidate()
            raise
        floor = self.last_at if self.last_at is not None else self._clock_floor
        if (not self.valid or self._pending is not None or self._wait_deadline is None or at >= self._wait_deadline
                or (floor is not None and at < floor)
                or np.any(np.abs(self._velocity_all) > TOLERANCE)):
            self._fault("Reference inference wait expired or became invalid")
        if self.generation and (self.last_at is None or at - self.last_at > MAX_DISPATCH_GAP + TOLERANCE):
            self._fault("Reference inference wait command clock stalled; no catch-up")
        self._wait_record["resumed_monotonic_s"] = at
        self._wait_deadline = None
        self._wait_anchor, self._wait_record = None, None
        self._clock_floor = at
        # Preserve actual dispatch time across maintained waits. Only the first
        # ever wait has no committed command and therefore keeps last_at=None.

    def _violations(self, action, step):
        position = _check_positions(action, self.lower, self.upper)
        delta = position - step.previous_position
        velocity = delta / step.dt_s
        reasons = []
        if np.any(np.abs(delta) > self.max_step + TOLERANCE):
            reasons.append("joint step")
        if np.any(np.abs(velocity) > self.max_velocity + TOLERANCE):
            reasons.append("joint velocity")
        if np.any(np.abs(velocity - step.previous_velocity) > MAX_ACCELERATION * min(step.dt_s, MAX_TIME_BUDGET) + TOLERANCE):
            reasons.append("joint acceleration")
        if any(abs(action[n] - self._last_action[n]) > self.gripper_max_step[n] + TOLERANCE for n in GRIPPER_NAMES):
            reasons.append("gripper step")
        return position, velocity, reasons

    def prepare(self, requested: Mapping, *, now: float) -> CommandStep:
        if not self.valid or self._pending is not None:
            self._fault("Reference command is invalid or pending")
        try:
            action = _action(requested)
            _check_positions(action, self.lower, self.upper)
            at = finite_scalar(now, "reference dispatch time")
            if self._wait_deadline is not None:
                if not self.generation:
                    self._fault("First reference inference is paused until its initial pose is captured")
                if at >= self._wait_deadline:
                    self._fault("Reference inference hold deadline expired")
                if action != self._wait_anchor or action != self._last_action:
                    self._fault("Reference inference hold must equal the exact committed stationary endpoint")
            if self._clock_floor is not None and at < self._clock_floor:
                self._fault("Reference dispatch clock moved backwards after inference wait")
            dt = self.period_s if self.last_at is None else at - self.last_at
            if not self.period_s - TOLERANCE <= dt <= MAX_DISPATCH_GAP + TOLERANCE:
                self._fault("Reference dispatch clock is early, stalled or backwards; no catch-up")
            step = CommandStep(self.generation, at, dt, dict(action), dict(action),
                               self.position.copy(), self.velocity.copy())
            _, _, reasons = self._violations(action, step)
            if reasons:
                self._fault("Reference command violates unchanged bounds: " + ", ".join(reasons))
            if not self.valid:
                self._fault("Reference command invalidated during preparation")
            self._pending = step
            return step
        except ValueError:
            self.invalidate()
            raise

    def commit(self, step: CommandStep, sent: Mapping, *, deadline_monotonic_s: float | None):
        if not self.valid or self._pending is not step or step.generation != self.generation:
            self._fault("Cannot commit an invalid or superseded reference command")
        try:
            actual = _action(sent)
            position, velocity, reasons = self._violations(actual, step)
        except ValueError:
            self.invalidate()
            raise
        holding = self._wait_deadline is not None
        # A stationary hold must not accumulate even tiny postclamp changes.
        modified = (actual != self._wait_anchor or actual != step.shaped if holding else
                    any(abs(actual[n] - step.shaped[n]) > TOLERANCE for n in ACTION_NAMES))
        acceleration = (velocity - step.previous_velocity) / step.dt_s
        self.postclamp_modified_count += int(modified)
        self.maximum_command_velocity_rad_s = max(self.maximum_command_velocity_rad_s, float(np.abs(velocity).max()))
        self.maximum_command_acceleration_rad_s2 = max(self.maximum_command_acceleration_rad_s2, float(np.abs(acceleration).max()))
        self.samples.append({"dispatch_index": self.generation, "monotonic_s": step.monotonic_s,
                             "dispatch_role": "inference_hold" if holding else "interpolation",
                             "deadline_monotonic_s": deadline_monotonic_s, "dt_s": step.dt_s,
                             "requested": step.requested, "shaped": step.shaped, "sent": actual,
                             "postclamp_modified": modified, "postclamp_bounds_exceeded": bool(reasons)})
        self._velocity_all = np.array([actual[n] - self._last_action[n] for n in ACTION_NAMES]) / step.dt_s
        self.position, self.velocity, self.last_at = position, velocity, step.monotonic_s
        self._clock_floor = step.monotonic_s
        self._last_action, self._pending = actual, None
        self.generation += 1
        if holding:
            self.inference_hold_count += 1
            self._wait_record["hold_count"] += 1
        if modified or reasons:
            self._fault("Postclamp reference command changed coordination or violated bounds; stopping")

    def metrics(self):
        return {"settings": {"mode": "reference_coordinated", "max_joint_velocity_rad_s": MAX_VELOCITY,
                             "max_joint_acceleration_rad_s2": MAX_ACCELERATION,
                             "nominal_plan_acceleration_rad_s2": PLAN_ACCELERATION,
                             "per_joint_velocity_rad_s": dict(zip(JOINT_NAMES, self.max_velocity.tolist(), strict=True)),
                             "gripper_max_step": dict(self.gripper_max_step), "minimum_period_s": self.period_s,
                             "max_dispatch_gap_s": MAX_DISPATCH_GAP,
                             "requested_basis": "coordinated interpolated command; original model rows require reference plan metadata",
                             "timestamp_basis": "host command preparation; command-space slopes, not motor dynamics",
                             "timing_deviation": "shared smoothstep time dilation and endpoint holds; not literal reference timing"},
                "initial_joint_position": dict(zip(JOINT_NAMES, self.initial_position.tolist(), strict=True)),
                "sample_count": self.generation, "samples_dropped": max(0, self.generation - len(self.samples)),
                "inference_hold_count": self.inference_hold_count,
                "postclamp_modified_count": self.postclamp_modified_count,
                "maximum_command_velocity_rad_s": self.maximum_command_velocity_rad_s,
                "maximum_command_acceleration_rad_s2": self.maximum_command_acceleration_rad_s2,
                "samples": list(self.samples), "inference_waits": [dict(wait) for wait in self.inference_waits]}
