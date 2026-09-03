"""YAM follower arms as LeRobot `Robot`s.

Observation / action keys: ``joint_1.pos`` … ``joint_6.pos`` (rad) and ``gripper.pos`` (0 closed … 1
open); bimanual variant prefixes ``left_`` / ``right_``. Camera frames are added under their rig
names. Targets are speed-clamped inside `yamkit.arm.YamArm.command`, so a policy or leader that
is far from the follower produces a bounded-speed move instead of a jump.
"""

from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from yamkit.arm import YamArm, resolve_channel
from yamkit.cameras import camera_configs_from_dicts
from yamkit.config import N_JOINTS, RigConfig

from .config_yam_follower import BiYamFollowerConfig, YamFollowerConfig

logger = logging.getLogger(__name__)
JOINT_NAMES = [f"joint_{i}" for i in range(1, N_JOINTS + 1)]


class _FollowerHandle:
    """One follower arm: rig lookup, feature names, connect/observe/act."""

    def __init__(self, rig: RigConfig, arm_name: str, max_joint_speed: float | None, max_gripper_speed: float | None) -> None:
        self.spec = rig.arm(arm_name)
        if self.spec.role != "follower":
            raise ValueError(f"{arm_name!r} is a {self.spec.role}, expected a follower")
        self.names = JOINT_NAMES + (["gripper"] if self.spec.has_motor_gripper else [])
        self.max_joint_speed = rig.control.max_joint_speed if max_joint_speed is None else max_joint_speed
        self.max_gripper_speed = rig.control.max_gripper_speed if max_gripper_speed is None else max_gripper_speed
        self.home_speed = rig.control.home_speed  # arms park at home on connect/disconnect (0 = off)
        self.arm: YamArm | None = None

    @property
    def features(self) -> dict[str, type]:
        return {f"{n}.pos": float for n in self.names}

    def connect(self) -> None:
        self.arm = YamArm.connect(self.spec, resolve_channel(self.spec), max_joint_speed=self.max_joint_speed, max_gripper_speed=self.max_gripper_speed)
        if self.home_speed > 0:
            self.arm.go_home(self.home_speed)

    def observation(self) -> dict[str, float]:
        st = self.arm.read()
        obs = {f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, st.q)}
        if self.spec.has_motor_gripper:
            obs["gripper.pos"] = float(st.gripper if st.gripper is not None else 1.0)
        return obs

    def send(self, action: dict[str, float]) -> dict[str, float]:
        q = np.array([float(action[f"{n}.pos"]) for n in JOINT_NAMES], dtype=float)
        g = action.get("gripper.pos")
        sent = self.arm.command(q, None if g is None else float(g))
        return {f"{n}.pos": float(v) for n, v in zip(self.names, sent)}

    def disconnect(self) -> None:
        if self.arm is None:
            return
        try:
            if self.home_speed > 0:
                self.arm.go_home(self.home_speed)
        except KeyboardInterrupt:
            logger.warning("%s: home move aborted — releasing here", self.spec.name)
        finally:
            self.arm.close()
            self.arm = None


def _rig_cameras(rig: RigConfig, config) -> dict:
    if config.cameras:
        return dict(config.cameras)
    return camera_configs_from_dicts(rig.cameras) if config.use_rig_cameras else {}


class YamFollower(Robot):
    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        self._h = _FollowerHandle(self.rig, config.arm, config.max_joint_speed, config.max_gripper_speed)
        self.camera_configs = _rig_cameras(self.rig, config)
        self.cameras = make_cameras_from_configs(self.camera_configs)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return self._h.features

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.camera_configs.items()}

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._h.arm is not None and all(c.is_connected for c in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._h.connect()
        try:
            for cam in self.cameras.values():
                cam.connect()
        except Exception:
            self._h.disconnect()
            raise
        logger.info("%s connected on %s", self, self._h.arm.channel)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:  # gripper limits / zero offsets live in the rig file & motor flash
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = self._h.observation()
        for key, cam in self.cameras.items():
            obs[key] = cam.read_latest()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        return self._h.send(action)

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        self._h.disconnect()
        logger.info("%s disconnected", self)


class BiYamFollower(Robot):
    """Two YAM followers; keys prefixed ``left_`` / ``right_``; cameras unprefixed."""

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        self._sides = {
            "left": _FollowerHandle(self.rig, config.left, config.max_joint_speed, config.max_gripper_speed),
            "right": _FollowerHandle(self.rig, config.right, config.max_joint_speed, config.max_gripper_speed),
        }
        self.camera_configs = _rig_cameras(self.rig, config)
        self.cameras = make_cameras_from_configs(self.camera_configs)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{side}_{k}": v for side, h in self._sides.items() for k, v in h.features.items()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {k: (c.height, c.width, 3) for k, c in self.camera_configs.items()}

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return all(h.arm is not None for h in self._sides.values()) and all(c.is_connected for c in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        connected = []
        try:
            for h in self._sides.values():
                h.connect()
                connected.append(h)
            for cam in self.cameras.values():
                cam.connect()
        except Exception:
            for h in connected:
                h.disconnect()
            raise
        logger.info("%s connected (%s)", self, ", ".join(f"{s}={h.arm.channel}" for s, h in self._sides.items()))

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        obs: dict = {}
        for side, h in self._sides.items():
            obs.update({f"{side}_{k}": v for k, v in h.observation().items()})
        for key, cam in self.cameras.items():
            obs[key] = cam.read_latest()
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        out: dict = {}
        for side, h in self._sides.items():
            sub = {k.removeprefix(f"{side}_"): v for k, v in action.items() if k.startswith(f"{side}_")}
            out.update({f"{side}_{k}": v for k, v in h.send(sub).items()})
        return out

    @check_if_not_connected
    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        for h in self._sides.values():
            h.disconnect()
        logger.info("%s disconnected", self)
