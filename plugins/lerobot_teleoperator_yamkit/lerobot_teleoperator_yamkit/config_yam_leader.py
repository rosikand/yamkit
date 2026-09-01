"""LeRobot `TeleoperatorConfig`s for YAM leaders (teaching handle). Select with `--teleop.type=yam_leader` /
`bi_yam_leader`; the arm name(s) refer to entries in the rig file."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("yam_leader")
@dataclass(kw_only=True)
class YamLeaderConfig(TeleoperatorConfig):
    id: str | None = "yam_leader"
    rig: str = "configs/rig.yaml"
    arm: str = "right_leader"  # arm name in the rig


@TeleoperatorConfig.register_subclass("bi_yam_leader")
@dataclass(kw_only=True)
class BiYamLeaderConfig(TeleoperatorConfig):
    id: str | None = "bi_yam_leader"
    rig: str = "configs/rig.yaml"
    left: str = "left_leader"
    right: str = "right_leader"
