"""yamkit — self-contained toolkit for I2RT YAM arms.

CAN discovery, arm wrapper over the vendored i2rt SDK, leader/follower teleop, and
LeRobot plugins (recording, training, policy rollout). Keep this module import-light:
it is imported at interpreter start via ``yamkit_env.pth``.
"""

__version__ = "0.1.0"
