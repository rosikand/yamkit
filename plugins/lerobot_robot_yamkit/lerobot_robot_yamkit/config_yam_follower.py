"""LeRobot `RobotConfig`s for YAM followers. Select with `--robot.type=yam_follower` / `bi_yam_follower`;
hardware identity comes from the rig file, so the CLI only needs the arm name(s) and optional cameras."""

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@dataclass(kw_only=True)
class _YamFollowerCommon:
    # Rig file (relative to the yamkit repo root) holding the CAN-adapter mapping, cameras and control knobs.
    rig: str = "configs/rig.yaml"
    # Cameras. Empty → use the `cameras:` section of the rig file (unless use_rig_cameras=false).
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_rig_cameras: bool = True
    # Per-tick target clamps (rad/s and normalised-units/s); None → rig `control` values.
    max_joint_speed: float | None = None
    max_gripper_speed: float | None = None


@RobotConfig.register_subclass("yam_follower")
@dataclass(kw_only=True)
class YamFollowerConfig(RobotConfig, _YamFollowerCommon):
    id: str | None = "yam_follower"
    arm: str = "right_follower"  # arm name in the rig


@RobotConfig.register_subclass("bi_yam_follower")
@dataclass(kw_only=True)
class BiYamFollowerConfig(RobotConfig, _YamFollowerCommon):
    id: str | None = "bi_yam_follower"
    left: str = "left_follower"
    right: str = "right_follower"
