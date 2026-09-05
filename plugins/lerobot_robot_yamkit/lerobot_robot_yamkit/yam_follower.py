"""YAM follower arms as LeRobot `Robot`s.

Observation / action keys: ``joint_1.pos`` … ``joint_6.pos`` (rad) and ``gripper.pos`` (0 closed … 1
open); bimanual variant prefixes ``left_`` / ``right_``. Camera frames are added under their rig
names. Targets are speed-clamped inside `yamkit.arm.YamArm.command`, so a policy or leader that
is far from the follower produces a bounded-speed move instead of a jump.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from functools import cached_property

import numpy as np
from lerobot.cameras import make_cameras_from_configs
from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from yamkit.arm import YamArm, go_home_all, resolve_channel
from yamkit.camera_ownership import claim_from_env
from yamkit.cameras import camera_configs_from_dicts
from yamkit.config import N_JOINTS, RigConfig
from yamkit.preview import NullPreview, start_from_env
from yamkit.validation import finite_scalar

from .config_yam_follower import BiYamFollowerConfig, YamFollowerConfig

logger = logging.getLogger(__name__)
JOINT_NAMES = [f"joint_{i}" for i in range(1, N_JOINTS + 1)]


def _validate_rig(rig: RigConfig) -> None:
    problems = rig.validate()
    if problems:
        raise ValueError("invalid rig: " + "; ".join(problems))


def _validate_action_keys(action: Mapping, expected: Mapping) -> None:
    if not isinstance(action, Mapping):
        raise TypeError("action must be a mapping of named scalar targets")
    missing, extra = set(expected) - set(action), set(action) - set(expected)
    if missing or extra:
        raise ValueError(f"action fields do not match: missing={sorted(missing, key=str)}, extra={sorted(extra, key=str)}")


class _FollowerHandle:
    """One follower arm: rig lookup, feature names, connect/observe/act."""

    def __init__(self, rig: RigConfig, arm_name: str, max_joint_speed: float | None, max_gripper_speed: float | None) -> None:
        self.spec = rig.arm(arm_name)
        if self.spec.role != "follower":
            raise ValueError(f"{arm_name!r} is a {self.spec.role}, expected a follower")
        self.names = JOINT_NAMES + (["gripper"] if self.spec.has_motor_gripper else [])
        control = replace(
            rig.control,
            max_joint_speed=rig.control.max_joint_speed if max_joint_speed is None else max_joint_speed,
            max_gripper_speed=rig.control.max_gripper_speed if max_gripper_speed is None else max_gripper_speed,
        )
        self.max_joint_speed = control.max_joint_speed
        self.max_gripper_speed = control.max_gripper_speed
        self.home_speed = rig.control.home_speed  # arms park at home on connect/disconnect (0 = off)
        self.arm: YamArm | None = None

    @property
    def features(self) -> dict[str, type]:
        return {f"{n}.pos": float for n in self.names}

    @property
    def home_job(self) -> tuple[YamArm, dict] | None:
        return (self.arm, {"speed": self.home_speed}) if self.arm is not None and self.home_speed > 0 else None

    def connect(self, home: bool = True) -> None:
        if self.arm is not None:
            raise RuntimeError(f"{self.spec.name}: already open")
        self.arm = YamArm.connect(self.spec, resolve_channel(self.spec), max_joint_speed=self.max_joint_speed, max_gripper_speed=self.max_gripper_speed)
        if home and self.home_job:
            self.arm.go_home(self.home_speed)

    def observation(self) -> dict[str, float]:
        st = self.arm.read()
        obs = {f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, st.q)}
        if self.spec.has_motor_gripper:
            if st.gripper is None:
                raise ValueError(f"{self.spec.name}: measured gripper is missing")
            obs["gripper.pos"] = float(st.gripper)
        return obs

    def target(self, action: dict[str, float]) -> tuple[np.ndarray, float | None]:
        _validate_action_keys(action, self.features)
        values = [finite_scalar(action[name], f"{name}: target") for name in self.features]
        return np.asarray(values[:N_JOINTS]), values[-1] if self.spec.has_motor_gripper else None

    def send(self, action: dict[str, float]) -> dict[str, float]:
        q, g = self.target(action)
        sent = self.arm.command(q, g)
        return {f"{n}.pos": float(v) for n, v in zip(self.names, sent)}

    def disconnect(self, home: bool = True) -> None:
        if self.arm is None:
            return
        arm = self.arm
        try:
            if home and self.home_speed > 0:
                arm.go_home(self.home_speed)
        finally:
            try:
                arm.close()
            finally:
                if arm._closed:
                    self.arm = None


def _home_together(handles, stop=None) -> None:
    """Park several arms at the same time (used by the bimanual robot/teleoperator)."""
    jobs = [h.home_job for h in handles if h.home_job]
    if jobs:
        go_home_all(jobs, stop=stop)


def _disconnect(handles, disconnect_cameras, *, home: bool) -> None:
    """Attempt every resource even if homing, camera cleanup or cancellation fails."""
    errors: list[BaseException] = []
    try:
        disconnect_cameras()
    except BaseException as e:  # noqa: BLE001 — release every arm even if camera teardown fails
        errors.append(e)
    if home and not errors:
        try:
            _home_together(handles)
        except BaseException as e:  # noqa: BLE001 — re-raised after every resource is attempted
            errors.append(e)
    for h in handles:
        try:
            h.disconnect(home=False)
        except BaseException as e:  # noqa: BLE001 — re-raised after every resource is attempted
            errors.append(e)
    if errors:
        for e in errors[1:]:
            logger.error("additional cleanup failure: %s", e, exc_info=(type(e), e, e.__traceback__))
        raise errors[0]


def _rig_cameras(rig: RigConfig, config) -> dict:
    if config.cameras:
        return dict(config.cameras)
    return camera_configs_from_dicts(rig.cameras) if config.use_rig_cameras else {}


def _check_session_stop(config) -> None:
    # A transient in-process hook for remote rollout's upstream context builder.
    stop = getattr(config, "_session_shutdown_event", None)
    if stop is not None and stop.is_set():
        raise RuntimeError("Rollout stopped before hardware activation completed")


class _CameraPreview:
    """The existing observation owns acquisition; previews only receive its frames."""

    def _init_preview(self) -> None:
        self._preview = NullPreview()
        self._camera_lease = None
        self._opened_cameras = []

    def _connect_cameras(self) -> None:
        # A UI child waits here until every direct capture has released its device.
        # Camera-free commands never claim ownership, regardless of command name.
        self._camera_lease = claim_from_env(list(self.cameras))
        try:
            for cam in self.cameras.values():
                self._opened_cameras.append(cam)  # include a partially failed connect in cleanup
                _check_session_stop(self.config)
                cam.connect()
                _check_session_stop(self.config)
        except BaseException:
            try:
                self._disconnect_cameras()
            except BaseException:  # noqa: BLE001 — preserve the original acquisition failure
                logger.warning("camera startup cleanup incomplete; ownership retained until process exit")
            raise
        try:
            modes = {key: getattr(cfg, "color_mode", "rgb") for key, cfg in self.camera_configs.items()}
            self._preview = start_from_env(modes, owner=self._camera_lease.owner)
        except Exception:  # noqa: BLE001 — optional previews cannot prevent acquisition
            logger.warning("camera previews unavailable; continuing acquisition")

    def _camera_observation(self, obs: dict) -> None:
        for key, cam in self.cameras.items():
            frame = cam.read_latest()
            obs[key] = frame
            try:
                source_time = None
                # LeRobot 0.6.1 publishes pixels and perf_counter timestamp together.
                # Read metadata only if immediately available and still for this frame.
                lock = getattr(cam, "frame_lock", None) if self._preview.enabled else None
                if lock is not None and lock.acquire(blocking=False):
                    try:
                        if (getattr(cam, "latest_frame", None) is frame
                                or getattr(cam, "latest_color_frame", None) is frame):
                            source_time = getattr(cam, "latest_timestamp", None)
                    finally:
                        lock.release()
                self._preview.offer(key, frame, source_time=source_time)
            except Exception:  # noqa: BLE001, S110 — no logging or failures on observation thread
                pass  # even a replacement hook must not interrupt recording

    def _disconnect_cameras(self) -> None:
        failure = None
        failed_cameras = []
        for cam in self._opened_cameras:
            reader = getattr(cam, "thread", None)
            try:
                try:
                    cam.disconnect()
                except DeviceNotConnectedError:
                    pass  # LeRobot may have cleaned up a partially failed connect.
                if cam.is_connected or (reader is not None and reader.is_alive()):
                    raise RuntimeError("camera release could not be confirmed")
            except BaseException as exc:  # noqa: BLE001 — disconnect all cameras, then re-raise
                failure = failure or exc
                failed_cameras.append(cam)
        self._opened_cameras[:] = failed_cameras
        try:
            self._preview.close()
        except Exception:  # noqa: BLE001, S110 — preview cleanup must not retain camera ownership
            pass
        self._preview = NullPreview()
        if failure is not None:
            # The parent keeps direct capture suspended until process exit in this case.
            raise failure
        self._opened_cameras.clear()
        if self._camera_lease is not None:
            self._camera_lease.release()
            self._camera_lease = None


class YamFollower(_CameraPreview, Robot):
    config_class = YamFollowerConfig
    name = "yam_follower"

    def __init__(self, config: YamFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        _validate_rig(self.rig)
        self._h = _FollowerHandle(self.rig, config.arm, config.max_joint_speed, config.max_gripper_speed)
        self.camera_configs = _rig_cameras(self.rig, config)
        self.cameras = make_cameras_from_configs(self.camera_configs)
        self._init_preview()
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
        _validate_rig(self.rig)
        if self._h.arm is not None or self._opened_cameras:
            raise RuntimeError("previous resources remain open; call disconnect(home=False) before reconnecting")
        _check_session_stop(self.config)
        try:
            self._h.connect(home=False)
            _check_session_stop(self.config)
            if self._h.home_job:
                self._h.arm.go_home(self._h.home_speed, stop=getattr(self.config, "_session_shutdown_event", None))
            _check_session_stop(self.config)
            self._connect_cameras()
        except BaseException:
            try:
                self.disconnect(home=False)
            except BaseException:
                logger.exception("cleanup after follower startup failure")
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
        self._camera_observation(obs)
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        return self._h.send(action)

    def disconnect(self, *, home: bool = True) -> None:
        """Release all resources; ``home=False`` skips the normal return-home move."""
        _disconnect([self._h], self._disconnect_cameras, home=home)
        logger.info("%s disconnected", self)

    def disconnect_no_home(self) -> None:
        """Compatibility hook for rollout; use the common hardened teardown."""
        self.disconnect(home=False)


class BiYamFollower(_CameraPreview, Robot):
    """Two YAM followers; keys prefixed ``left_`` / ``right_``; cameras unprefixed."""

    config_class = BiYamFollowerConfig
    name = "bi_yam_follower"

    def __init__(self, config: BiYamFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        _validate_rig(self.rig)
        if config.left == config.right:
            raise ValueError("bimanual follower needs two different arms")
        self._sides = {
            "left": _FollowerHandle(self.rig, config.left, config.max_joint_speed, config.max_gripper_speed),
            "right": _FollowerHandle(self.rig, config.right, config.max_joint_speed, config.max_gripper_speed),
        }
        self.camera_configs = _rig_cameras(self.rig, config)
        self.cameras = make_cameras_from_configs(self.camera_configs)
        self._init_preview()
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
        _validate_rig(self.rig)
        if any(h.arm is not None for h in self._sides.values()) or self._opened_cameras:
            raise RuntimeError("previous resources remain open; call disconnect(home=False) before reconnecting")
        try:
            for h in self._sides.values():
                _check_session_stop(self.config)
                h.connect(home=False)
            _check_session_stop(self.config)
            _home_together(self._sides.values(), stop=getattr(self.config, "_session_shutdown_event", None))
            _check_session_stop(self.config)
            self._connect_cameras()
        except BaseException:
            try:
                self.disconnect(home=False)
            except BaseException:
                logger.exception("cleanup after bimanual follower startup failure")
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
        self._camera_observation(obs)
        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        _validate_action_keys(action, self.action_features)
        actions = {
            side: {k: action[f"{side}_{k}"] for k in h.features}
            for side, h in self._sides.items()
        }
        # Check every target and every arm's measured/previous state before either can move.
        # command() still revalidates its own state immediately before sending to the SDK.
        for side, h in self._sides.items():
            h.arm.validate_command(*h.target(actions[side]))
        out: dict = {}
        for side, h in self._sides.items():
            out.update({f"{side}_{k}": v for k, v in h.send(actions[side]).items()})
        return out

    def disconnect(self, *, home: bool = True) -> None:
        """Release all resources; ``home=False`` skips the normal return-home move."""
        _disconnect(self._sides.values(), self._disconnect_cameras, home=home)
        logger.info("%s disconnected", self)

    def disconnect_no_home(self) -> None:
        """Compatibility hook for rollout; use the common hardened teardown."""
        self.disconnect(home=False)
