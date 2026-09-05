"""Small Robot observation/action boundary for the experimental LLM controller.

The current YAM plugin cannot supply verifiable frame freshness or fault cleanup
without homing. Live construction is deliberately blocked before plugin import.
The fixture is a hardware-free LeRobot-shaped object, not a physical simulator.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import RigConfig

JOINT_KEYS = tuple(f"joint_{i}.pos" for i in range(1, 7))
GRIPPER_KEY = "gripper.pos"
STATE_KEYS = (*JOINT_KEYS, GRIPPER_KEY)
METADATA_KEY = "__yamkit_agent_observation__"
OBSERVATION_CONTRACT = "yamkit.agent.observation.v1"

LIVE_BLOCKER = (
    "Live agent execution is disabled: YamFollower.disconnect() has no public no-home option, "
    "homes the arm by default, and a camera disconnect failure can skip arm cleanup. "
    "YamFollower.get_observation() uses camera.read_latest() and exposes no acquisition "
    "timestamps or frame sequence, so fresh post-action images/state cannot be verified. "
    "YamFollower.connect() enables motors, may calibrate the gripper, and homes by default; "
    "its camera-error cleanup can also home. No arm or camera was opened."
)


class ObservationError(ValueError):
    """Observation is malformed, stale, or lacks acquisition evidence."""


class LiveIntegrationError(RuntimeError):
    """The public hardware interface cannot meet this controller's contract."""


@dataclass(frozen=True)
class Observation:
    state: dict[str, float]
    images: dict[str, np.ndarray]
    captured_at: float
    sequence: int
    source: str


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")  # noqa: TRY004 - uniform validation error
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _state(values: Mapping[str, Any]) -> dict[str, float]:
    if not isinstance(values, Mapping) or set(values) != set(STATE_KEYS):
        raise ValueError("state/action must contain exactly six joint positions and gripper.pos")
    result = {key: _finite_number(values[key], key) for key in STATE_KEYS}
    if not 0.0 <= result[GRIPPER_KEY] <= 1.0:
        raise ValueError("gripper.pos must be between 0 and 1")
    return result


class RobotAdapter:
    """Read only through get_observation and send only through send_action.

    Test sources must explicitly implement ``OBSERVATION_CONTRACT`` in the
    reserved metadata field. ``captured_at`` is the *oldest acquisition time*
    across state and all images, in the injected monotonic clock domain.
    ``sequence`` advances only after *every* component is acquired again. Merely
    timestamping the return of a cached plugin observation does not implement
    this contract. The adapter never invents acquisition metadata.

    This version owns only FixtureRobot cleanup. Other injected test doubles
    are borrowed; close never calls an unverified physical disconnect method.
    Real ownership remains disabled by make_live_robot.
    """

    def __init__(
        self,
        robot: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_age_s: float = 1.0,
    ) -> None:
        self.robot = robot
        self.clock = clock
        self.max_age_s = _finite_number(max_age_s, "max_age_s")
        if self.max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        self._last: Observation | None = None
        self._closed = False

    def observe(self, after: float | None = None) -> Observation:
        if self._closed:
            raise RuntimeError("robot adapter is closed")
        if after is not None:
            after = _finite_number(after, "after")
        raw = self.robot.get_observation()
        try:
            result = self._validate_observation(raw, after)
        except (ValueError, TypeError, KeyError) as exc:
            raise ObservationError(str(exc)) from exc
        self._last = result
        return result

    def _validate_observation(self, raw: Any, after: float | None) -> Observation:
        if not isinstance(raw, Mapping):
            raise TypeError("Robot.get_observation() must return a mapping")
        metadata = raw.get(METADATA_KEY)
        if not isinstance(metadata, Mapping) or metadata.get("contract") != OBSERVATION_CONTRACT:
            raise ValueError("observation lacks an explicit acquisition freshness contract")
        if set(metadata) != {"contract", "source", "captured_at", "sequence"}:
            raise ValueError("invalid acquisition metadata fields")
        source = metadata["source"]
        if not isinstance(source, str) or not source.strip() or len(source) > 80:
            raise ValueError("observation source must be a nonempty label of at most 80 characters")
        captured_at = _finite_number(metadata["captured_at"], "captured_at")
        now = _finite_number(self.clock(), "clock")
        if captured_at > now:
            raise ValueError("observation acquisition time is in the future")
        if now - captured_at > self.max_age_s:
            raise ValueError("observation is stale")
        if after is not None and captured_at <= after:
            raise ValueError("observation was not acquired after the requested boundary")
        sequence = metadata["sequence"]
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("observation sequence must be a nonnegative integer")
        if self._last is not None:
            if source != self._last.source:
                raise ValueError("observation source changed during the episode")
            if sequence <= self._last.sequence or captured_at <= self._last.captured_at:
                raise ValueError("observation acquisition did not advance; cached feedback is not fresh")
        state = _state({key: raw[key] for key in STATE_KEYS})
        images = {}
        for key, frame in raw.items():
            if key in STATE_KEYS or key == METADATA_KEY:
                continue
            if not isinstance(key, str) or not key.strip() or len(key) > 80:
                raise ValueError("camera names must be nonempty strings of at most 80 characters")
            if (
                not isinstance(frame, np.ndarray)
                or frame.dtype != np.uint8
                or frame.ndim != 3
                or frame.shape[2] != 3
                or min(frame.shape[:2]) < 1
            ):
                raise ValueError(f"camera {key!r} must supply a nonempty uint8 RGB image")
            images[key] = frame.copy()
        if not images:
            raise ValueError("agent observations require at least one named RGB image")
        return Observation(state, images, captured_at, sequence, source)

    def send(self, target: Mapping[str, Any]) -> dict[str, float]:
        if self._closed:
            raise RuntimeError("robot adapter is closed")
        action = _state(target)
        # Returned positions are commanded positions, never measured completion.
        return _state(self.robot.send_action(action))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if isinstance(self.robot, FixtureRobot):
            self.robot.close()


class FixtureRobot:
    """Deterministic synthetic fixture with perfect tracking for paid/mock tests.

    No camera, CAN, SDK, plugin, calibration, homing, or physical teardown is
    reachable here. Sending an action only changes an in-memory dictionary.
    Dry-run operations may send these in-memory targets; no physical action is possible.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.state = {**dict.fromkeys(JOINT_KEYS, 0.0), GRIPPER_KEY: 0.5}
        self.commands: list[dict[str, float]] = []
        self.closed = False
        self.sequence = 0

    def get_observation(self) -> dict[str, Any]:
        if self.closed:
            raise RuntimeError("fixture robot is closed")
        # Rebuild state and images on every read; these are labeled synthetic
        # acquisitions, not cached camera frames dressed with a new timestamp.
        captured_at = self.clock()
        self.sequence += 1
        image = _fixture_image(self.sequence)
        return {
            **self.state,
            "fixture_top": image,
            METADATA_KEY: {
                "contract": OBSERVATION_CONTRACT,
                "source": "fixture",
                "captured_at": captured_at,
                "sequence": self.sequence,
            },
        }

    def send_action(self, action: Mapping[str, Any]) -> dict[str, float]:
        if self.closed:
            raise RuntimeError("fixture robot is closed")
        self.state = _state(action)
        self.commands.append(self.state.copy())
        return self.state.copy()

    def close(self) -> None:
        self.closed = True


def _fixture_image(sequence: int) -> np.ndarray:
    """Tiny RGB scene with a literal FIXTURE label, rendered without camera code."""
    frame = np.full((120, 160, 3), 24, dtype=np.uint8)
    frame[40:80, 25:65] = (230, 25, 25)
    frame[45:95, 100:145] = (25, 50, 225)
    frame[105:110, 5:5 + sequence % 150] = (25, 220, 25)
    glyphs = {
        "F": ("111", "100", "110", "100", "100"),
        "I": ("111", "010", "010", "010", "111"),
        "X": ("101", "101", "010", "101", "101"),
        "T": ("111", "010", "010", "010", "010"),
        "U": ("101", "101", "101", "101", "111"),
        "R": ("110", "101", "110", "101", "101"),
        "E": ("111", "100", "110", "100", "111"),
    }
    for index, letter in enumerate("FIXTURE"):
        for y, row in enumerate(glyphs[letter]):
            for x, pixel in enumerate(row):
                if pixel == "1":
                    left, top = 5 + index * 16 + x * 4, 5 + y * 4
                    frame[top:top + 4, left:left + 4] = 255
    return frame


def validate_rig(rig_path: str | Path, arm: str) -> RigConfig:
    """Validate the selected single follower without opening hardware."""
    try:
        rig = RigConfig.load(rig_path)
        problems = rig.validate()
    except (yaml.YAMLError, AttributeError, TypeError):
        raise ValueError("invalid rig YAML/structure") from None
    if problems:
        raise ValueError("invalid rig: " + "; ".join(problems))
    spec = rig.arm(arm)
    if spec.role != "follower":
        raise ValueError(f"{arm!r} is a {spec.role}; agent requires one follower arm")
    if not spec.has_motor_gripper:
        raise ValueError("agent requires a follower with a motor gripper")
    if not rig.cameras:
        raise ValueError("agent requires at least one camera configured in the rig")
    if not isinstance(rig.cameras, dict):
        raise ValueError("rig cameras must map names to camera configuration objects")  # noqa: TRY004
    for name, config in rig.cameras.items():
        if (
            not isinstance(name, str) or not name.strip() or len(name) > 80
            or name in STATE_KEYS or name == METADATA_KEY or not isinstance(config, dict)
        ):
            raise ValueError("rig cameras must map names to camera configuration objects")
    for name in ("max_joint_speed", "max_gripper_speed"):
        value = _finite_number(getattr(rig.control, name), f"control.{name}")
        if value <= 0:
            raise ValueError(f"control.{name} must be positive")
    return rig


def make_live_robot(rig_path: str | Path, arm: str) -> RobotAdapter:
    """Fail before plugin construction until public freshness/cleanup support exists."""
    validate_rig(rig_path, arm)
    raise LiveIntegrationError(LIVE_BLOCKER)
