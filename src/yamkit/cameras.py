"""Camera configs from the rig file → LeRobot `CameraConfig` objects."""

from __future__ import annotations

import logging
from typing import Any


def camera_configs_from_dicts(cams: dict[str, dict[str, Any]]) -> dict:
    """Decode `{name: {type: opencv, index_or_path: ..., width, height, fps}}` into CameraConfig objects."""
    if not cams:
        return {}
    import draccus
    from lerobot.cameras import CameraConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401 — registers "opencv"

    try:
        from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401 — registers "intelrealsense"
    except Exception as e:  # noqa: BLE001 — pyrealsense2 is optional
        logging.getLogger(__name__).debug("realsense camera config unavailable: %s", e)
    out = {}
    for name, cfg in cams.items():
        cfg = dict(cfg)
        cfg.setdefault("type", "opencv")
        out[name] = draccus.decode(CameraConfig, cfg)
    return out
