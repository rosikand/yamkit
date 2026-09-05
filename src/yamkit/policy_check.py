"""Load a LeRobot policy/VLA against this rig's feature spec and run it on a synthetic frame.

Mirrors what `lerobot-rollout` does at start-up (policy from checkpoint, pre/post processors,
inference frame built from the robot's observation features) so problems with a checkpoint, the
CPU-only setup or the feature mapping show up *before* an arm is energised.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PolicyCheckResult:
    policy_type: str
    device: str
    state_dim: int
    action_dim: int
    image_keys: list[str]
    chunk_size: int | None
    first_call_s: float
    step_call_s: list[float] = field(default_factory=list)
    action: dict[str, float] = field(default_factory=dict)


def _synthetic_stats(ds_features: dict[str, dict]) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for key, ft in ds_features.items():
        if ft["dtype"] in ("video", "image"):
            c = 3
            stats[key] = {
                "mean": np.full((c, 1, 1), 0.5, np.float32),
                "std": np.full((c, 1, 1), 0.25, np.float32),
                "min": np.zeros((c, 1, 1), np.float32),
                "max": np.ones((c, 1, 1), np.float32),
            }
        elif ft["dtype"] in ("float32", "float64"):
            d = int(np.prod(ft["shape"]))
            stats[key] = {
                "mean": np.zeros(d, np.float32),
                "std": np.ones(d, np.float32),
                "min": -np.ones(d, np.float32),
                "max": np.ones(d, np.float32),
            }
    return stats


def robot_features_from_rig(rig_path: str, arms: list[str] | None, fake_camera: tuple[int, int] | None) -> tuple[dict, dict, str]:
    """(observation_features, action_features, robot_type) for the rig's follower(s) without connecting."""
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig

    from .config import RigConfig

    rig = RigConfig.load(rig_path)
    pairs = rig.pairs if not arms else [p for p in rig.pairs if p.follower in arms or rig.arm(p.follower).side in arms]
    cams: dict[str, Any] = {}
    if fake_camera and not rig.cameras:
        h, w = fake_camera
        cams["top"] = OpenCVCameraConfig(index_or_path=0, fps=30, width=w, height=h)
    if len(pairs) == 1:
        cfg = YamFollowerConfig(rig=str(rig.path), arm=pairs[0].follower, cameras=cams)
    else:
        cfg = BiYamFollowerConfig(rig=str(rig.path), left=pairs[0].follower, right=pairs[1].follower, cameras=cams)
    from lerobot.robots.utils import make_robot_from_config

    robot = make_robot_from_config(cfg)
    return dict(robot.observation_features), dict(robot.action_features), robot.name


def run_policy_check(
    policy_path: str,
    *,
    rig_path: str,
    arms: list[str] | None,
    task: str,
    device: str = "cpu",
    n_steps: int = 3,
    fake_camera: tuple[int, int] | None = (480, 640),
    use_robot_features: bool = True,
) -> PolicyCheckResult:
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.pipeline_features import (
        aggregate_pipeline_dataset_features,
        create_initial_features,
    )
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.processor import make_default_processors
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
    from lerobot.utils.feature_utils import combine_feature_dicts

    obs_ft, act_ft, robot_type = robot_features_from_rig(rig_path, arms, fake_camera)
    teleop_proc, _robot_proc, obs_proc = make_default_processors()
    ds_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(pipeline=teleop_proc, initial_features=create_initial_features(action=act_ft), use_videos=False),
        aggregate_pipeline_dataset_features(pipeline=obs_proc, initial_features=create_initial_features(observation=obs_ft), use_videos=False),
    )
    stats = _synthetic_stats(ds_features)
    ds_meta = SimpleNamespace(features=ds_features, stats=stats)

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device
    if use_robot_features:
        # Take input/output features from *this* robot (what lerobot-train does when fine-tuning):
        # VLAs pad state/action to a fixed width and their vision towers are camera-name agnostic.
        cfg.input_features = {}
    t0 = time.perf_counter()
    policy = make_policy(cfg, ds_meta=ds_meta)
    policy.eval()
    pre, post = make_pre_post_processors(cfg, pretrained_path=policy_path, dataset_stats=stats, preprocessor_overrides={"device_processor": {"device": device}})
    log.info("policy %s loaded in %.1fs (%d params)", cfg.type, time.perf_counter() - t0, sum(p.numel() for p in policy.parameters()))

    image_keys = [k for k in ds_features if k.startswith(OBS_IMAGES)]
    raw_obs: dict[str, Any] = {k: 0.0 for k in obs_ft if isinstance(obs_ft[k], type)}
    for k, shape in obs_ft.items():
        if isinstance(shape, tuple):
            raw_obs[k] = np.random.randint(0, 255, size=shape, dtype=np.uint8)

    def infer() -> tuple[Any, float]:
        frame = build_inference_frame(dict(raw_obs), torch.device(device), ds_features, task, robot_type)
        t = time.perf_counter()
        with torch.inference_mode():
            chunk = post(policy.predict_action_chunk(pre(frame)))
        if chunk.ndim != 3 or chunk.shape[0] != 1 or not torch.isfinite(chunk).all():
            raise ValueError("policy-check produced a malformed or nonfinite fresh action chunk")
        return chunk[:, 0, :], time.perf_counter() - t

    policy.reset()
    action, first = infer()
    steps = []
    for _ in range(n_steps):
        action, dt = infer()
        steps.append(dt)
    robot_action = make_robot_action(action, ds_features)
    return PolicyCheckResult(
        policy_type=cfg.type,
        device=device,
        state_dim=int(np.prod(ds_features[OBS_STATE]["shape"])),
        action_dim=int(np.prod(ds_features[ACTION]["shape"])),
        image_keys=image_keys,
        chunk_size=getattr(cfg, "n_action_steps", None),
        first_call_s=first,
        step_call_s=steps,
        action=robot_action,
    )
