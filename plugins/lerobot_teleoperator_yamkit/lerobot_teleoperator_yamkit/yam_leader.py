"""YAM leader arms (gravity-compensated, teaching handle) as LeRobot `Teleoperator`s.

Action keys: ``joint_1.pos`` … ``joint_6.pos`` (rad) and ``gripper.pos`` from the handle trigger
(1 = released/open, 0 = squeezed/closed) — the same keys the follower consumes.
"""

from __future__ import annotations

import logging
from functools import cached_property

from lerobot.lerobot_types import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from yamkit.arm import YamArm, resolve_channel
from yamkit.config import N_JOINTS, RigConfig

from .config_yam_leader import BiYamLeaderConfig, YamLeaderConfig

logger = logging.getLogger(__name__)
JOINT_NAMES = [f"joint_{i}" for i in range(1, N_JOINTS + 1)]


class _LeaderHandle:
    def __init__(self, rig: RigConfig, arm_name: str) -> None:
        self.spec = rig.arm(arm_name)
        if self.spec.role != "leader":
            raise ValueError(f"{arm_name!r} is a {self.spec.role}, expected a leader")
        self.names = JOINT_NAMES + (["gripper"] if self.spec.has_handle else [])
        self.home_speed = rig.control.leader_home_speed if rig.control.home_speed > 0 else 0.0  # leaders park (compliantly, gently) on connect/disconnect
        self.arm: YamArm | None = None

    @property
    def features(self) -> dict[str, type]:
        return {f"{n}.pos": float for n in self.names}

    def connect(self) -> None:
        self.arm = YamArm.connect(self.spec, resolve_channel(self.spec))
        if self.home_speed > 0:
            self.arm.go_home(self.home_speed, compliant=True, release=True)

    def action(self) -> dict[str, float]:
        st = self.arm.read()
        act = {f"{n}.pos": float(v) for n, v in zip(JOINT_NAMES, st.q)}
        if self.spec.has_handle:
            act["gripper.pos"] = float(st.gripper if st.gripper is not None else 1.0)
        return act

    def disconnect(self) -> None:
        if self.arm is None:
            return
        try:
            if self.home_speed > 0:
                self.arm.go_home(self.home_speed, compliant=True, release=True)
        except KeyboardInterrupt:
            logger.warning("%s: home move aborted — releasing here", self.spec.name)
        finally:
            self.arm.close()
            self.arm = None


class YamLeader(Teleoperator):
    config_class = YamLeaderConfig
    name = "yam_leader"

    def __init__(self, config: YamLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        self._h = _LeaderHandle(self.rig, config.arm)

    @cached_property
    def action_features(self) -> dict:
        return self._h.features

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._h.arm is not None

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._h.connect()
        logger.info("%s connected on %s", self, self._h.arm.channel)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return self._h.action()

    def send_feedback(self, feedback: dict) -> None:  # bilateral feedback is handled by `yamkit teleop`
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        self._h.disconnect()
        logger.info("%s disconnected", self)


class BiYamLeader(Teleoperator):
    config_class = BiYamLeaderConfig
    name = "bi_yam_leader"

    def __init__(self, config: BiYamLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        self._sides = {"left": _LeaderHandle(self.rig, config.left), "right": _LeaderHandle(self.rig, config.right)}

    @cached_property
    def action_features(self) -> dict:
        return {f"{side}_{k}": v for side, h in self._sides.items() for k, v in h.features.items()}

    @cached_property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return all(h.arm is not None for h in self._sides.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        connected = []
        try:
            for h in self._sides.values():
                h.connect()
                connected.append(h)
        except Exception:
            for h in connected:
                h.disconnect()
            raise
        logger.info("%s connected", self)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        return {f"{side}_{k}": v for side, h in self._sides.items() for k, v in h.action().items()}

    def send_feedback(self, feedback: dict) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        for h in self._sides.values():
            h.disconnect()
        logger.info("%s disconnected", self)
