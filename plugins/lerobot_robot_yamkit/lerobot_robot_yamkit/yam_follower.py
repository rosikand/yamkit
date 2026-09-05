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

from yamkit.arm import YamArm, go_home_all, resolve_channel
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

    @property
    def home_job(self) -> tuple[YamArm, dict] | None:
        return (self.arm, {"speed": self.home_speed}) if self.arm is not None and self.home_speed > 0 else None

    def connect(self, home: bool = True) -> None:
        self.arm = YamArm.connect(self.spec, resolve_channel(self.spec), max_joint_speed=self.max_joint_speed, max_gripper_speed=self.max_gripper_speed)
        if home and self.home_job:
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

    def disconnect(self, home: bool = True) -> None:
        if self.arm is None:
            return
        try:
            if home and self.home_job:
                self.arm.go_home(self.home_speed)
        except KeyboardInterrupt:
            logger.warning("%s: home move aborted — releasing here", self.spec.name)
        finally:
            self.arm.close()
            self.arm = None


def _home_together(handles, stop=None) -> None:
    """Park several arms at the same time (used by the bimanual robot/teleoperator)."""
    jobs = [h.home_job for h in handles if h.home_job]
    if jobs:
        try:
            go_home_all(jobs, stop=stop)
        except KeyboardInterrupt:
            logger.warning("home move aborted — releasing the arms where they are")


def _rig_cameras(rig: RigConfig, config) -> dict:
    if config.cameras:
        return dict(config.cameras)
    return camera_configs_from_dicts(rig.cameras) if config.use_rig_cameras else {}


def _check_session_stop(config) -> None:
    # A transient in-process hook for remote rollout's upstream context builder.
    stop = getattr(config, "_session_shutdown_event", None)
    if stop is not None and stop.is_set():
        raise RuntimeError("Rollout stopped before hardware activation completed")


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
        # In-process lifecycle handle: lets a caller release partially built upstream
        # rollout contexts. This is deliberately not a serialized config field.
        config._runtime_robot = self

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
        _check_session_stop(self.config)
        try:
            self._h.connect(home=False)
            _check_session_stop(self.config)
            if self._h.home_job:
                self._h.arm.go_home(self._h.home_speed, stop=getattr(self.config, "_session_shutdown_event", None))
            _check_session_stop(self.config)
            for cam in self.cameras.values():
                cam.connect()
                _check_session_stop(self.config)
        except Exception:
            self.disconnect_no_home()
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

    def disconnect_no_home(self) -> None:
        """Release after a remote stop/fault, including partially connected cameras."""
        try:
            for cam in self.cameras.values():
                if cam.is_connected:
                    try:
                        cam.disconnect()
                    except Exception:  # noqa: BLE001 — continue releasing every arm despite camera failure
                        logger.warning("Camera cleanup failed during release")
        finally:
            self._h.disconnect(home=False)


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
        config._runtime_robot = self

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
                _check_session_stop(self.config)
                h.connect(home=False)
                connected.append(h)
            _check_session_stop(self.config)
            _home_together(connected, stop=getattr(self.config, "_session_shutdown_event", None))
            _check_session_stop(self.config)
            for cam in self.cameras.values():
                cam.connect()
                _check_session_stop(self.config)
        except Exception:
            self.disconnect_no_home()
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
        _home_together(self._sides.values())  # both arms park at the same time
        for h in self._sides.values():
            h.disconnect(home=False)
        logger.info("%s disconnected", self)

    def disconnect_no_home(self) -> None:
        """Release all followers without a return-to-start or home move."""
        try:
            for cam in self.cameras.values():
                if cam.is_connected:
                    try:
                        cam.disconnect()
                    except Exception:  # noqa: BLE001 — continue releasing every arm despite camera failure
                        logger.warning("Camera cleanup failed during release")
        finally:
            errors = []
            for handle in self._sides.values():
                try:
                    handle.disconnect(home=False)
                except Exception as exc:  # noqa: BLE001 — release remaining arms before surfacing the error
                    errors.append(exc)
            if errors:
                raise errors[0]
