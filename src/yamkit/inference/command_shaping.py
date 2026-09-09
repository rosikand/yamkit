"""Causal position-command shaping for the remote bimanual rollout only.

No hardware reads, sleeps, queue changes or timestamp extensions. Bounds apply
to generated command slopes, not measured robot acceleration or collision risk.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from yamkit.validation import finite_scalar

ACTION_NAMES = tuple(f"{side}_{name}.pos" for side in ("left", "right")
                     for name in (*(f"joint_{i}" for i in range(1, 7)), "gripper"))
JOINT_NAMES = tuple(name for name in ACTION_NAMES if "gripper" not in name)
MAX_VELOCITY = 0.6
MAX_ACCELERATION = 2.0
MAX_TIME_BUDGET = 0.05
MAX_DISPATCH_GAP = 0.1
FIRST_DT = 1 / 30
MAX_SAMPLES = 1000
TOLERANCE = 1e-8


class CommandShapingFault(ValueError):
    """A fresh, bounded and acceleration-compatible command cannot be produced."""


def _action(value) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(ACTION_NAMES):
        raise CommandShapingFault("Remote shaping requires exactly 14 named bimanual targets")
    result = {name: finite_scalar(value[name], f"{name}: shaping target") for name in ACTION_NAMES}
    if any(not 0 <= result[name] <= 1 for name in ACTION_NAMES if "gripper" in name):
        raise CommandShapingFault("Remote gripper target must be in [0, 1]")
    return result


@dataclass(frozen=True)
class CommandStep:
    generation: int
    monotonic_s: float
    dt_s: float
    requested: dict[str, float]
    shaped: dict[str, float]
    previous_position: np.ndarray
    previous_velocity: np.ndarray


class JointCommandShaper:
    """Prepare without changing state; commit only the successful postclamp result."""

    def __init__(self, initial_position: Mapping, joint_limits: Mapping):
        initial = _action(initial_position)
        if not isinstance(joint_limits, Mapping) or set(joint_limits) != set(JOINT_NAMES):
            raise CommandShapingFault("Remote shaping requires bounds for every follower joint")
        self.lower = np.array([finite_scalar(joint_limits[name]["lower"], "joint lower bound")
                               for name in JOINT_NAMES])
        self.upper = np.array([finite_scalar(joint_limits[name]["upper"], "joint upper bound")
                               for name in JOINT_NAMES])
        self.max_step = np.array([finite_scalar(joint_limits[name]["max_step"], "joint command step", positive=True)
                                  for name in JOINT_NAMES])
        if np.any(self.lower >= self.upper):
            raise CommandShapingFault("Invalid joint bounds for remote shaping")
        self.max_velocity = np.minimum(MAX_VELOCITY, self.max_step / MAX_TIME_BUDGET)
        self.position = np.array([initial[name] for name in JOINT_NAMES])
        self.velocity = np.zeros(len(JOINT_NAMES))
        self._check_bounds(self.position)
        self.last_at = None
        self.valid = True
        self.generation = 0
        self.modified_count = 0
        self.postclamp_modified_count = 0
        self.maximum_tracking_lag_rad = 0.0
        self.maximum_command_velocity_rad_s = 0.0
        self.maximum_command_acceleration_rad_s2 = 0.0
        self.samples = deque(maxlen=MAX_SAMPLES)

    def _check_bounds(self, positions):
        if not np.all(np.isfinite(positions)) or np.any(positions < self.lower) or np.any(positions > self.upper):
            raise CommandShapingFault("Remote joint target is outside the original robot bounds")

    def invalidate(self):
        # Permanent invalidation is safe from the inference worker. Do not
        # mutate velocity while the dispatch thread may be preparing a step;
        # no invalidated instance can be resumed or committed again.
        self.valid = False

    def prepare(self, requested: Mapping, *, now: float) -> CommandStep:
        if not self.valid:
            raise CommandShapingFault("Remote command shaper is invalidated")
        action = _action(requested)
        target = np.array([action[name] for name in JOINT_NAMES])
        self._check_bounds(target)  # Never hide invalid original targets through smoothing.
        now = finite_scalar(now, "command timestamp")
        dt = FIRST_DT if self.last_at is None else now - self.last_at
        if not 0 < dt <= MAX_DISPATCH_GAP:
            raise CommandShapingFault("Remote command clock stalled or moved backwards; no catch-up")
        budget = min(dt, MAX_TIME_BUDGET)
        velocity_limit = np.minimum(self.max_velocity, self.max_step / dt)

        # Keep enough joint-boundary room to stop even at the longest permitted
        # dispatch interval. At a 100ms interval only 50ms of acceleration budget
        # is usable, so conservative braking acceleration is 1 rad/s². The
        # additional full interval covers discrete command/hold timing.
        braking = MAX_ACCELERATION * MAX_TIME_BUDGET / MAX_DISPATCH_GAP
        hold = dt + MAX_DISPATCH_GAP
        positive_room = np.maximum(0, self.upper - self.position)
        negative_room = np.maximum(0, self.position - self.lower)
        positive_limit = np.sqrt((braking * hold) ** 2 + 2 * braking * positive_room) - braking * hold
        negative_limit = np.sqrt((braking * hold) ** 2 + 2 * braking * negative_room) - braking * hold
        lower = np.maximum.reduce((-velocity_limit, -negative_limit,
                                   self.velocity - MAX_ACCELERATION * budget))
        upper = np.minimum.reduce((velocity_limit, positive_limit,
                                   self.velocity + MAX_ACCELERATION * budget))
        if np.any(lower > upper + TOLERANCE):
            raise CommandShapingFault("Insufficient joint-boundary or step-budget room for bounded braking")
        delta = target - self.position
        distance = np.abs(delta)
        # Brake for the requested pose as well as the physical boundaries.
        # v*hold + v²/(2*braking) <= distance reserves discrete stopping room;
        # the equivalent quotient avoids cancellation close to the target.
        target_velocity = 2 * braking * distance / (
            np.sqrt((braking * hold) ** 2 + 2 * braking * distance) + braking * hold
        )
        desired_velocity = np.sign(delta) * np.minimum(distance / dt, target_velocity)
        # This is a soft target, not another hard bound: a suddenly nearer or
        # reversed target may require unavoidable overshoot while decelerating.
        # Existing acceleration, step and physical-bound protections still win.
        velocity = np.minimum(np.maximum(desired_velocity, lower), upper)
        position = self.position + velocity * dt
        self._check_bounds(position)
        shaped = dict(action)  # Grippers pass through unchanged to their existing arm clamp.
        shaped.update(zip(JOINT_NAMES, position.tolist(), strict=True))
        if not self.valid:
            raise CommandShapingFault("Remote command shaper was invalidated during preparation")
        return CommandStep(self.generation, now, dt, action, shaped, self.position.copy(), self.velocity.copy())

    def commit(self, step: CommandStep, sent: Mapping, *, deadline_monotonic_s: float | None):
        if not self.valid or step.generation != self.generation:
            raise CommandShapingFault("Cannot commit an invalid or superseded remote command")
        actual = _action(sent)
        position = np.array([actual[name] for name in JOINT_NAMES])
        self._check_bounds(position)
        velocity = (position - step.previous_position) / step.dt_s
        acceleration = (velocity - step.previous_velocity) / step.dt_s
        # A downstream intervention is never hidden. This is checked after a
        # successful send; the caller then faults/releases and does not continue
        # with a fictitious filter state if the existing arm clamp changed it.
        exceeds = (np.any(np.abs(velocity) > self.max_velocity + TOLERANCE)
                   or np.any(np.abs(velocity - step.previous_velocity)
                             > MAX_ACCELERATION * min(step.dt_s, MAX_TIME_BUDGET) + TOLERANCE))
        modified = any(abs(step.requested[name] - step.shaped[name]) > TOLERANCE for name in JOINT_NAMES)
        intervened = any(abs(actual[name] - step.shaped[name]) > TOLERANCE for name in JOINT_NAMES)
        lag = max(abs(step.requested[name] - actual[name]) for name in JOINT_NAMES)
        self.modified_count += int(modified)
        self.postclamp_modified_count += int(intervened)
        self.maximum_tracking_lag_rad = max(self.maximum_tracking_lag_rad, lag)
        self.maximum_command_velocity_rad_s = max(self.maximum_command_velocity_rad_s, float(np.abs(velocity).max()))
        self.maximum_command_acceleration_rad_s2 = max(
            self.maximum_command_acceleration_rad_s2, float(np.abs(acceleration).max()))
        self.samples.append({"dispatch_index": self.generation, "monotonic_s": step.monotonic_s,
                             "deadline_monotonic_s": deadline_monotonic_s, "dt_s": step.dt_s,
                             "requested": step.requested, "shaped": step.shaped, "sent": actual,
                             "tracking_lag_max_rad": lag, "postclamp_modified": intervened,
                             "postclamp_bounds_exceeded": bool(exceeds)})
        self.position, self.velocity, self.last_at = position, velocity, step.monotonic_s
        self.generation += 1
        if exceeds:
            self.invalidate()
            raise CommandShapingFault("Postclamp command violated remote shaping bounds; stopping")

    def metrics(self):
        return {"settings": {"max_joint_velocity_rad_s": MAX_VELOCITY,
                             "max_joint_acceleration_rad_s2": MAX_ACCELERATION,
                             "per_joint_velocity_rad_s": dict(zip(JOINT_NAMES, self.max_velocity.tolist(), strict=True)),
                             "max_time_budget_s": MAX_TIME_BUDGET, "max_dispatch_gap_s": MAX_DISPATCH_GAP,
                             "tracking_lag_basis": "original requested minus sent joint command; not measured tracking error",
                             "timestamp_basis": "host command preparation; not per-motor dispatch or exposure time",
                             "grippers": "unchanged", "scope": "remote policy position commands only"},
                "modified_count": self.modified_count, "postclamp_modified_count": self.postclamp_modified_count,
                "sample_count": self.generation, "samples_dropped": max(0, self.generation - len(self.samples)),
                "maximum_tracking_lag_rad": self.maximum_tracking_lag_rad,
                "maximum_command_velocity_rad_s": self.maximum_command_velocity_rad_s,
                "maximum_command_acceleration_rad_s2": self.maximum_command_acceleration_rad_s2,
                "samples": list(self.samples)}
