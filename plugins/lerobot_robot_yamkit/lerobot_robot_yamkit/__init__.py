"""LeRobot robot plugin: I2RT YAM follower arm(s) via yamkit. Auto-discovered by `lerobot-*` CLIs."""

from .config_yam_follower import BiYamFollowerConfig, YamFollowerConfig
from .yam_follower import BiYamFollower, YamFollower

__all__ = ["BiYamFollower", "BiYamFollowerConfig", "YamFollower", "YamFollowerConfig"]
