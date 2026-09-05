"""Reviewed immutable checkpoint profiles. Catalog access is entirely offline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .mapping import YAM_NAMES

LEROBOT_VERSION = "0.6.1"
MOLMO_DATASET_REVISION = "e9f21ae15074330839f2ac25ed4b49d76dfa1f9c"


@dataclass(frozen=True)
class ModelProfile:
    id: str
    repo_id: str
    revision: str
    policy_type: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    image_keys: tuple[str, ...]
    native_image_keys: tuple[str, ...]
    chunk_size: int
    fps: int
    mapping_verified: bool
    mapping_note: str
    dependency_repo: str
    dependency_revision: str
    native_image_hw: tuple[int, int] = (256, 256)
    supports_rtc: bool = False

    def require_robot_mapping(self) -> None:
        if not self.mapping_verified:
            raise ValueError(f"{self.id}: no verified physical YAM mapping. {self.mapping_note}")

    def metadata(self) -> dict:
        from .performance import physical_modal_status

        result = asdict(self)
        result.update(
            profile=self.id, profile_id=self.id, model_revision=self.revision, model=self.repo_id,
            action_units="robot" if self.mapping_verified
            else "checkpoint_native", max_chunk_steps=self.chunk_size, lerobot_version=LEROBOT_VERSION,
            physical_validation="not performed",
            mapping_validation="source conventions only; physical calibration/alignment and cameras unvalidated",
            **physical_modal_status(),
        )
        return result


_PROFILES = (
    ModelProfile(
        "smolvla", "lerobot/smolvla_base", "c83c3163b8ca9b7e67c509fffd9121e66cb96205", "smolvla",
        tuple(f"native_state_{i}" for i in range(6)), tuple(f"native_action_{i}" for i in range(6)),
        ("camera1", "camera2", "camera3"), ("camera1", "camera2", "camera3"), 50, 30, False,
        "Six native dimensions have no published YAM joint/gripper mapping; forward-pass fixtures only.",
        "HuggingFaceTB/SmolVLM2-500M-Video-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    ),
    ModelProfile(
        "molmoact2", "lerobot/MolmoAct2-BimanualYAM-LeRobot", "fdade02d1f1c1dd819114b0478f735072fb6b212",
        "molmoact2", YAM_NAMES, YAM_NAMES, ("top", "left_wrist", "right_wrist"), ("top", "left", "right"),
        30, 30, True,
        "Source-defined left then right, six joints in radians then gripper 0 closed / 1 open; absolute "
        "joint targets. Dataset joint_0..5 maps to yamkit joint_1..6. Requires matching I2RT zero frame, "
        "calibrated LINEAR_4310 grippers and physical camera/side verification; this is not robot validation.",
        "allenai/MolmoAct2-BimanualYAM", "8dcbed66f2380e4393189c303ea72488eb9e63c2", (360, 640),
    ),
    ModelProfile(
        "pi05", "lerobot/pi05_base", "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba", "pi05",
        tuple(f"native_state_{i}" for i in range(32)), tuple(f"native_action_{i}" for i in range(32)),
        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), 50, 30, False,
        "32 native dimensions and empty saved normalizer features/statistics; no YAM physical-unit mapping. "
        "Forward-pass fixtures only.",
        "google/paligemma-3b-pt-224", "35e4f46485b4d07967e7e9935bc3786aad50687c", (224, 224),
    ),
)


def get_profile(name: str | ModelProfile) -> ModelProfile:
    if isinstance(name, ModelProfile):
        return name
    for profile in _PROFILES:
        if name in (profile.id, profile.repo_id):
            return profile
    raise ValueError("Unknown reviewed inference profile; use smolvla, molmoact2, or pi05. "
                     "Custom checkpoints remain available through local inference.")


def list_profiles() -> list[dict]:
    return [profile.metadata() for profile in _PROFILES]
