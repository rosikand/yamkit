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

from yamkit.arm import YamArm, go_home_all, resolve_channel
from yamkit.config import N_JOINTS, RigConfig
from yamkit.teleop_control import LeaderAction, disconnect_home, position_vector, vector_action

from .config_yam_leader import BiYamLeaderConfig, YamLeaderConfig

logger = logging.getLogger(__name__)
JOINT_NAMES = [f"joint_{i}" for i in range(1, N_JOINTS + 1)]


def _validate_rig(rig: RigConfig) -> None:
    problems = rig.validate()
    if problems:
        raise ValueError("invalid rig: " + "; ".join(problems))


class _LeaderHandle:
    def __init__(self, rig: RigConfig, arm_name: str) -> None:
        self.spec = rig.arm(arm_name)
        if self.spec.role != "leader":
            raise ValueError(f"{arm_name!r} is a {self.spec.role}, expected a leader")
        self.names = JOINT_NAMES + (["gripper"] if self.spec.has_handle else [])
        self.max_joint_speed = rig.control.max_joint_speed
        self.max_gripper_speed = rig.control.max_gripper_speed
        self.home_speed = rig.control.leader_home_speed if rig.control.home_speed > 0 else 0.0  # leaders park (compliantly, gently) on connect/disconnect
        self.arm: YamArm | None = None

    @property
    def features(self) -> dict[str, type]:
        return {f"{n}.pos": float for n in self.names}

    @property
    def home_job(self) -> tuple[YamArm, dict] | None:
        return (self.arm, {"speed": self.home_speed, "compliant": True, "release": True}) if self.arm is not None and self.home_speed > 0 else None

    def connect(self, home: bool = True) -> None:
        if self.arm is not None:
            raise RuntimeError(f"{self.spec.name}: already open")
        self.arm = YamArm.connect(self.spec, resolve_channel(self.spec), max_joint_speed=self.max_joint_speed, max_gripper_speed=self.max_gripper_speed)
        if home and self.home_job:
            self.arm.go_home(self.home_speed, compliant=True, release=True)

    def action(self) -> dict[str, float]:
        st = self.arm.read()
        if self.spec.has_handle and st.gripper is None:
            raise ValueError(f"{self.spec.name}: teaching-handle trigger is missing")
        return LeaderAction(vector_action(position_vector(st.q, st.gripper if self.spec.has_handle else None)),
                            buttons={"": st.buttons})

    def disconnect(self, home: bool = True) -> None:
        if self.arm is None:
            return
        arm = self.arm
        try:
            if home and self.home_speed > 0:
                arm.go_home(self.home_speed, compliant=True, release=True)
        finally:
            try:
                arm.close()
            finally:
                if arm._closed:
                    self.arm = None


def _home_together(handles) -> None:
    """Park several leaders at the same time (bimanual teleoperator)."""
    jobs = [h.home_job for h in handles if h.home_job]
    if jobs:
        go_home_all(jobs)


def _disconnect(handles, *, home: bool) -> None:
    """Finish every close, including after a failed or cancelled home move."""
    errors: list[BaseException] = []
    if home:
        try:
            _home_together(handles)
        except BaseException as e:  # noqa: BLE001 — re-raised after every arm is closed
            errors.append(e)
    for h in handles:
        try:
            h.disconnect(home=False)
        except BaseException as e:  # noqa: BLE001 — re-raised after every arm is closed
            errors.append(e)
    if errors:
        for e in errors[1:]:
            logger.error("additional cleanup failure: %s", e, exc_info=(type(e), e, e.__traceback__))
        raise errors[0]


class YamLeader(Teleoperator):
    config_class = YamLeaderConfig
    name = "yam_leader"

    def __init__(self, config: YamLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        _validate_rig(self.rig)
        self._h = _LeaderHandle(self.rig, config.arm)
        config._runtime_teleop = self

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
        _validate_rig(self.rig)
        if self._h.arm is not None:
            raise RuntimeError("previous arm remains open; call disconnect(home=False) before reconnecting")
        try:
            self._h.connect()
        except BaseException:
            try:
                self.disconnect(home=False)
            except BaseException:
                logger.exception("cleanup after leader startup failure")
            raise
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

    def disconnect(self, *, home: bool | None = None) -> None:
        """Release all resources; ``home=False`` skips the normal return-home move."""
        _disconnect([self._h], home=disconnect_home(home))
        logger.info("%s disconnected", self)


class BiYamLeader(Teleoperator):
    config_class = BiYamLeaderConfig
    name = "bi_yam_leader"

    def __init__(self, config: BiYamLeaderConfig) -> None:
        super().__init__(config)
        self.config = config
        self.rig = RigConfig.load(config.rig)
        _validate_rig(self.rig)
        if config.left == config.right:
            raise ValueError("bimanual leader needs two different arms")
        self._sides = {"left": _LeaderHandle(self.rig, config.left), "right": _LeaderHandle(self.rig, config.right)}
        config._runtime_teleop = self

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
        _validate_rig(self.rig)
        if any(h.arm is not None for h in self._sides.values()):
            raise RuntimeError("previous arms remain open; call disconnect(home=False) before reconnecting")
        try:
            for h in self._sides.values():
                h.connect(home=False)
            _home_together(self._sides.values())  # both leaders park at the same time
        except BaseException:
            try:
                self.disconnect(home=False)
            except BaseException:
                logger.exception("cleanup after bimanual leader startup failure")
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
        actions = {side: handle.action() for side, handle in self._sides.items()}
        return LeaderAction(
            {f"{side}_{key}": value for side, action in actions.items() for key, value in action.items()},
            buttons={f"{side}_": action.buttons[""] for side, action in actions.items()},
        )

    def send_feedback(self, feedback: dict) -> None:
        pass

    def disconnect(self, *, home: bool | None = None) -> None:
        """Release all resources; ``home=False`` skips the normal return-home move."""
        _disconnect(self._sides.values(), home=disconnect_home(home))
        logger.info("%s disconnected", self)
