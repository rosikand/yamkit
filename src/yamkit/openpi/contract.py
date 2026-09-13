"""Pinned official model identity, never an invented YAM embodiment contract."""

from __future__ import annotations

import hashlib
from pathlib import Path

POLICY_KEY = "pi05-base"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base"
UPSTREAM_URL = "https://github.com/Physical-Intelligence/openpi.git"
UPSTREAM_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"
MANIFEST_PATH = Path(__file__).with_name("pi05_base_manifest.json")
MODEL_CONFIG = {
    "pi05": True, "dtype": "bfloat16", "paligemma_variant": "gemma_2b",
    "action_expert_variant": "gemma_300m", "action_dim": 32, "action_horizon": 50,
    "max_token_len": 200, "discrete_state_input": True,
}
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
SAVED_IMAGE_MAP = dict(zip(("top", "left_wrist", "right_wrist"), IMAGE_KEYS, strict=True))
YAM_BLOCKERS = (
    "Official pi05_base contains no YAM normalization asset or documented YAM embodiment config.",
    "YAM joint coordinates, gripper direction/range and absolute/delta action semantics are not established.",
    "YAM execution frequency, committed horizon and replanning contract are not established.",
)


def physical_block_reason() -> str:
    return "Official pi05_base physical rollout is blocked. " + " ".join(YAM_BLOCKERS)


def require_yam_contract() -> None:
    """Unconditional: a loaded base model or fake replay cannot authorize YAM motion."""
    raise ValueError(physical_block_reason())


def identity() -> dict:
    return {
        "policy": POLICY_KEY, "checkpoint": CHECKPOINT, "runtime": "official OpenPI JAX",
        "runtime_revision": UPSTREAM_REVISION,
        "manifest_sha256": hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest(),
        "model_config": dict(MODEL_CONFIG), "num_inference_steps": 10,
        "physical_ready": False, "hardware_tested": False, "yam_contract_established": False,
        "blockers": list(YAM_BLOCKERS),
    }
