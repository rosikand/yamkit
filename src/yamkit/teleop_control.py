"""Small shared operator state for native teleop and LeRobot recording.

No connection, camera, or outer control loop lives here. Inputs and commands use the
aligned follower frame already supplied by YamArm; grippers remain normalized 0..1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import numpy as np

from .config import N_JOINTS
from .validation import finite_scalar, finite_vector

JOINT_KEYS = tuple(f"joint_{index}.pos" for index in range(1, N_JOINTS + 1))


def position_vector(q, gripper=None) -> np.ndarray:
    joints = finite_vector(q, N_JOINTS, "operator joints")
    if gripper is None:
        return joints
    grip = finite_scalar(gripper, "operator gripper", minimum=0)
    if grip > 1:
        raise ValueError("operator gripper must be <= 1")
    return np.append(joints, grip)


def vector_action(vector: np.ndarray) -> dict[str, float]:
    keys = JOINT_KEYS + (("gripper.pos",) if len(vector) == N_JOINTS + 1 else ())
    return dict(zip(keys, map(float, vector), strict=True))


def action_vector(action: dict, prefix: str, *, gripper: bool) -> np.ndarray:
    return position_vector([action[prefix + name] for name in JOINT_KEYS],
                           action[prefix + "gripper.pos"] if gripper else None)


def disconnect_home(home: bool | None) -> bool:
    """Normal Stop/Ctrl-C parks; errors and SystemExit release without a new move."""
    return home if home is not None else sys.exc_info()[0] in (None, KeyboardInterrupt)


class LeaderAction(dict):
    """Raw positions with button metadata outside public action/dataset keys."""

    def __init__(self, values, *, buttons):
        super().__init__(values)
        self.buttons = buttons  # prefix -> tuple of teaching-handle button states


class GatedAction(dict):
    """LeRobot's label object, acknowledged after the follower's safety clamps.

    Pinned LeRobot 0.6.1 records the teleop processor's dictionary and discards
    send_action's return value. Its identity robot processor preserves this object.
    The YAM plugin updates it with the sent values before LeRobot builds the frame.
    This explicit acknowledgment also covers immediate measured holds on release.
    """

    def __init__(self, values, *, capture_hold, on_sent=None):
        super().__init__(values)
        self.capture_hold = frozenset(capture_hold)
        self.on_sent = on_sent

    def acknowledge(self, sent):
        if self.keys() != sent.keys():
            raise ValueError("sent operator action changed the dataset schema")
        self.update(sent)
        if self.on_sent is not None:
            self.on_sent(sent)


@dataclass(frozen=True)
class PairGate:
    engaged: bool = False
    button_prev: bool = False
    hold: np.ndarray | None = None
    origin: np.ndarray | None = None
    target: np.ndarray | None = None
    elapsed: float = 0.0
    duration: float = 0.0
    previous_t: float | None = None

    @property
    def syncing(self) -> bool:
        return self.engaged and self.target is not None

    def acknowledge_hold(self, follower, *, joint_speed, gripper_speed) -> PairGate:
        hold = finite_vector(follower, len(follower), "captured follower hold")
        if self.syncing and self.elapsed == 0:
            delta = np.abs(self.target - hold)
            duration = max(self.duration, float(np.max(delta[:N_JOINTS])) / joint_speed)
            if len(hold) > N_JOINTS:
                duration = max(duration, float(delta[-1]) / gripper_speed)
            return replace(self, hold=hold, origin=hold.copy(), duration=duration)
        return replace(self, hold=hold)

    def advance(self, leader, follower, *, pressed: bool, now: float, period: float,
                sync_seconds: float, joint_speed: float, gripper_speed: float,
                engage: bool | None = None) -> tuple[PairGate, np.ndarray, bool]:
        """Return next state, intended target, and whether to replace an old hold.

        A button rising edge toggles engagement. Synchronization captures the leader
        once and earns at most one loop period per tick, even following a long stall.
        YamArm.command remains the final speed/position safety boundary in both paths.
        """
        size = len(follower)
        if size not in (N_JOINTS, N_JOINTS + 1):
            raise ValueError("operator follower must have six joints and optional gripper")
        leader = finite_vector(leader, size, "leader target")
        follower = finite_vector(follower, size, "follower state")
        now = finite_scalar(now, "operator timestamp", minimum=0)
        period = finite_scalar(period, "operator period", positive=True)
        sync_seconds = finite_scalar(sync_seconds, "sync_seconds", minimum=0)
        joint_speed = finite_scalar(joint_speed, "joint speed", positive=True)
        gripper_speed = finite_scalar(gripper_speed, "gripper speed", positive=True)
        if self.previous_t is not None and now < self.previous_t:
            raise ValueError("operator clock moved backwards")
        wanted = (not self.engaged if pressed and not self.button_prev else self.engaged) if engage is None else engage
        gate = replace(self, button_prev=pressed, previous_t=now)
        capture_hold = self.hold is None or (self.engaged and not wanted)
        if not wanted:
            hold = follower.copy() if capture_hold else self.hold
            return replace(gate, engaged=False, hold=hold, origin=None, target=None), hold.copy(), capture_hold
        if not self.engaged:
            delta = np.abs(leader - follower)
            duration = max(sync_seconds, float(np.max(delta[:N_JOINTS])) / joint_speed)
            if size > N_JOINTS:
                duration = max(duration, float(delta[-1]) / gripper_speed)
            gate = replace(gate, engaged=True, hold=follower.copy(), origin=follower.copy(),
                           target=leader.copy(), elapsed=0.0, duration=duration)
        if gate.target is not None:
            dt = 0.0 if not self.engaged or self.previous_t is None else min(now - self.previous_t, period)
            elapsed = min(gate.duration, gate.elapsed + dt)
            alpha = elapsed / gate.duration if gate.duration > 0 else 1.0
            command = gate.origin + alpha * (gate.target - gate.origin)
            gate = replace(gate, elapsed=elapsed, target=None if alpha >= 1 else gate.target)
            return gate, command, capture_hold
        return gate, leader, capture_hold
