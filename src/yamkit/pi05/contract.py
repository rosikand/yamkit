"""Source-reviewed YAM π0.5 contract, independent of MolmoAct2 execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yamkit.inference.mapping import MOLMO_NAMES, YAM_NAMES
from yamkit.inference.profiles import ModelProfile

CHECKPOINT = "Jiafei1224/molmoact2-yam-pi05"
REVISION = "51ab2720d7e56d51410407f98ea64bbea97feb2e"
LEROBOT_REVISION = "7e241bd630a3719a56157a497ce5d08f244784f1"
CONTRACT_ID = "pi05_reference"
CAMERA_MAP = {"observation.images.top": "observation.images.top",
              "observation.images.left_wrist": "observation.images.left",
              "observation.images.right_wrist": "observation.images.right"}
STATISTICS_SHA256 = {
    "policy_preprocessor_step_3_normalizer_processor.safetensors":
        "1a7fc7b7242ffc81bf8000e02012034c352aa44af9a1a1ef9e2ec50d66c4e97e",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors":
        "96e084c0ad4d272270c5173baa955ab39ad6b507342d8dd44ddb0ad8c99dfb24",
}
MODEL_SHA256 = "a777861c627234f9aa54a1bb7bdee29101ee6513f4773ef0a581d9c5527981e4"
PROFILE = ModelProfile(
    "pi05-yam", CHECKPOINT, REVISION, "pi05", YAM_NAMES, YAM_NAMES,
    ("top", "left_wrist", "right_wrist"), ("top", "left", "right"), 30, 30, True,
    "Pinned YAM checkpoint and dataset agree on left six absolute joint radians then continuous "
    "gripper 0 closed/1 open, followed by right. Source-reviewed mapping only: physical calibration, "
    "side and camera alignment still require operator verification. Native full FIFO chunks, no "
    "Molmo interpolation, no relative conversion, no RTC.",
    "google/paligemma-3b-pt-224", "35e4f46485b4d07967e7e9935bc3786aad50687c", (360, 640),
)
CONTRACT = {
    "id": CONTRACT_ID, "version": 1, "checkpoint": CHECKPOINT, "revision": REVISION,
    "lerobot_version": "0.6.1", "lerobot_revision": LEROBOT_REVISION,
    "camera_order": list(PROFILE.native_image_keys), "state_names": list(MOLMO_NAMES),
    "action_names": list(MOLMO_NAMES), "action_mode": "absolute",
    "gripper_range": [0.0, 1.0], "gripper_convention": "0 closed, 1 open; continuous",
    "chunk_size": 30, "n_action_steps": 30, "fps": 30, "num_inference_steps": 10,
    "state_action_normalization": "saved QUANTILES q01/q99, exactly once",
    "image_transform": "native PI05 resize-with-padding to 224x224 then [0,1] to [-1,1]",
    "boundary_crop": "none", "model_dtype": "bfloat16", "rtc": False,
    "queue": "native synchronous FIFO: all 30 rows once, in order, then reobserve/replan",
    "interpolation": False, "prefix_drop": False, "command_shaping": False,
    "state_source": "measured observation at chunk boundary",
    "physical_validation": "not performed",
}


def validate_checkpoint_config(config: dict) -> None:
    """Reject changed semantics before allocating a model or opening a device."""
    expected = {
        "type": "pi05", "n_obs_steps": 1, "chunk_size": 30, "n_action_steps": 30,
        "max_state_dim": 32, "max_action_dim": 32, "num_inference_steps": 10,
        "use_relative_actions": False, "rtc_config": None, "dtype": "bfloat16",
        "empty_cameras": 0, "image_resolution": [224, 224], "tokenizer_max_length": 200,
        "action_feature_names": list(MOLMO_NAMES),
        "normalization_mapping": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
    }
    if any(config.get(key) != value or type(config.get(key)) is not type(value)
           for key, value in expected.items()):
        raise ValueError("Pinned YAM π0.5 configuration changed; native contract does not match")
    inputs = config.get("input_features", {})
    images = [name for name, feature in inputs.items() if feature.get("type") == "VISUAL"]
    if images != [f"observation.images.{name}" for name in PROFILE.native_image_keys]:
        raise ValueError("YAM π0.5 camera order must be top, left, right")
    if (set(inputs) != {*images, "observation.state"}
            or inputs["observation.state"] != {"type": "STATE", "shape": [14]}
            or any(inputs[name] != {"type": "VISUAL", "shape": [3, 360, 640]} for name in images)
            or config.get("output_features") != {"action": {"type": "ACTION", "shape": [14]}}):
        raise ValueError("YAM π0.5 checkpoint feature schema changed")


def validate_snapshot(snapshot: Path) -> dict:
    """Verify saved processing statistics; Hub revision pins the weight download."""
    config = json.loads((snapshot / "config.json").read_text())
    validate_checkpoint_config(config)
    for filename, expected in STATISTICS_SHA256.items():
        if hashlib.sha256((snapshot / filename).read_bytes()).hexdigest() != expected:
            raise ValueError("YAM π0.5 saved normalization statistics differ from the pinned checkpoint")
    for name, expected in (("policy_preprocessor.json", ["rename_observations_processor", "to_batch_processor",
                           "relative_actions_processor", "normalizer_processor",
                           "pi05_prepare_state_tokenizer_processor_step", "tokenizer_processor", "device_processor"]),
                           ("policy_postprocessor.json", ["unnormalizer_processor", "absolute_actions_processor",
                            "device_processor"])):
        pipeline = json.loads((snapshot / name).read_text())
        if [step["registry_name"] for step in pipeline["steps"]] != expected:
            raise ValueError("YAM π0.5 saved processor order changed")
        for step in pipeline["steps"]:
            if (step["registry_name"] in ("relative_actions_processor", "absolute_actions_processor")
                    and step["config"].get("enabled") is not False):
                raise ValueError("Pinned YAM π0.5 actions are absolute, never relative")
    return config


def build_id() -> str:
    """Separate PI identity including its reused wire/runtime boundary, not MA2 readiness."""
    from yamkit.inference.identity import inference_build_id

    digest = hashlib.sha256(b"yamkit-pi05-reference-v1\0" + inference_build_id().encode())
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()
