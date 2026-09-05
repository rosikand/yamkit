"""Validate the effective LeRobot configuration before entering its unchanged CLI.

The genuine upstream parser handles nested flags, overrides and checkpoint config
loading first. Validation therefore sees the policy actually selected by LeRobot,
including local checkpoint paths and arbitrary Hub aliases. No weights or hardware
are loaded by this preflight.
"""

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

from lerobot.configs import parser
from lerobot.rollout.configs import RolloutConfig
from lerobot.rollout.inference import RTCInferenceConfig


def validate_local_rollout(cfg: RolloutConfig) -> None:
    from lerobot.robots import make_robot_from_config

    policy = cfg.policy
    if policy.type == "yamkit_remote":
        raise ValueError("Remote proxies require yamkit rollout --backend modal")
    if policy.type == "molmoact2" and (isinstance(cfg.inference, RTCInferenceConfig)
            or getattr(getattr(policy, "rtc_config", None), "enabled", False)):
        # Pinned upstream Molmo declares RTC support in continuous mode. Physical
        # prefix guidance and its failure handling have not been qualified here.
        raise ValueError("Local MolmoAct2 RTC execution is not qualified for this physical profile; "
                         "select local sync; physical Modal rollout is currently blocked")
    if not math.isfinite(cfg.fps) or cfg.fps <= 0:
        raise ValueError("Local rollout requires a finite positive FPS")
    if not math.isfinite(cfg.duration) or cfg.duration < 0:
        raise ValueError("Local duration must be finite and nonnegative; zero means unlimited")
    robot = make_robot_from_config(cfg.robot)  # Constructors expose schemas without connect.
    action_names = [name for name in robot.action_features if name.endswith((".pos", ".vel"))]
    state_names = [name for name, value in robot.observation_features.items()
                   if value is float and name.endswith((".pos", ".vel"))]
    action = (policy.output_features or {}).get("action")
    state = (policy.input_features or {}).get("observation.state")
    if action is None or tuple(action.shape) != (len(action_names),):
        raise ValueError("Checkpoint action dimensions do not match the selected physical robot; "
                         "padding or truncation is forbidden")
    if state is not None and tuple(state.shape) != (len(state_names),):
        raise ValueError("Checkpoint state dimensions do not match the selected physical robot; "
                         "padding or truncation is forbidden")
    configured_order = getattr(policy, "action_feature_names", None)
    if configured_order and list(configured_order) != action_names:
        raise ValueError("Checkpoint action names/order do not match the selected physical robot")
    if policy.type == "molmoact2":
        _validate_molmo_yam(cfg, robot, action_names, state_names)


def _validate_molmo_yam(cfg, robot, action_names, state_names):
    from .inference.mapping import CAMERA_RENAME_MAP
    from .inference.profiles import get_profile
    from .probes import preflight_live_probe

    profile = get_profile("molmoact2")
    policy = cfg.policy
    if cfg.robot.type != "bi_yam_follower":
        raise ValueError("The reviewed MolmoAct2 mapping requires the bimanual YAM follower plugin")
    if action_names != list(profile.action_names) or state_names != list(profile.state_names):
        raise ValueError("MolmoAct2 requires exact left-then-right YAM state/action order")
    if cfg.fps != profile.fps or cfg.interpolation_multiplier != 1:
        raise ValueError("The reviewed MolmoAct2 mapping requires 30 Hz without interpolation")
    if policy.chunk_size != profile.chunk_size:
        raise ValueError("The reviewed MolmoAct2 profile requires its saved 30-step chunk size")
    if (policy.control_mode != "absolute joint pose" or policy.norm_tag != "yam_dual_molmoact2"
            or policy.setup_type != "bimanual yam robotic arms in molmoact2"):
        raise ValueError("This MolmoAct2 checkpoint does not declare the reviewed absolute bimanual YAM mapping")
    if policy.joint_signs is not None or policy.joint_offsets is not None:
        raise ValueError("Unreviewed MolmoAct2 joint-frame overrides require a separately validated profile")
    selected = [cfg.robot.left, cfg.robot.right]
    specs, _ = preflight_live_probe(robot.rig, selected, expected_state_names=profile.state_names)
    for side, spec in zip(("left", "right"), specs, strict=True):
        if spec.side != side or spec.arm_type != "yam" or spec.gripper != "linear_4310":
            raise ValueError("MolmoAct2 requires physically verified left/right standard YAM + LINEAR_4310 arms")
    if set(robot.camera_configs) != set(profile.image_keys):
        raise ValueError("MolmoAct2 requires top, left_wrist and right_wrist rig cameras")
    for camera in robot.camera_configs.values():
        color = getattr(camera, "color_mode", "rgb")
        if str(getattr(color, "value", color)).lower() != "rgb":
            raise ValueError("MolmoAct2 inference requires RGB camera configuration")
    if cfg.rename_map and cfg.rename_map != CAMERA_RENAME_MAP:
        raise ValueError("MolmoAct2 camera rename_map must match the reviewed top/left/right mapping")
    cfg.rename_map = dict(CAMERA_RENAME_MAP)
    if str(policy.checkpoint_path) != profile.dependency_repo:
        nested = Path(policy.checkpoint_path)
        if not nested.is_dir() or nested.name != profile.dependency_revision:
            raise ValueError("MolmoAct2 nested checkpoint must be the reviewed model or its pinned local snapshot")
    if policy.checkpoint_revision not in (None, profile.dependency_revision):
        raise ValueError("MolmoAct2 nested model revision differs from the reviewed profile")
    if str(policy.pretrained_path) == profile.repo_id and policy.pretrained_revision not in (None, profile.revision):
        raise ValueError("MolmoAct2 base model revision differs from the reviewed profile")


def prepare_local_molmo_bundle(cfg: RolloutConfig) -> Path:
    """Pin saved processor and model dependencies without replacing user checkpoint weights.

    Upstream context has no override hook for a saved processor's nested revision.
    Write only derived JSON under data; unchanged checkpoint assets are symlinked.
    The normal upstream factories then load this bundle without an extra runtime.
    """
    import draccus
    from huggingface_hub import snapshot_download
    from lerobot.configs import PreTrainedConfig

    from .inference.profiles import get_profile
    from .paths import ROOT

    policy = cfg.policy
    profile = get_profile("molmoact2")
    source = Path(policy.pretrained_path)
    if not source.is_dir():
        revision = profile.revision if str(policy.pretrained_path) == profile.repo_id else policy.pretrained_revision
        source = Path(snapshot_download(str(policy.pretrained_path), revision=revision,
                                        allow_patterns=["*.json", "*.safetensors"]))
        # HF snapshots use their resolved immutable commit SHA as directory name.
        if len(source.name) == 40 and all(c in "0123456789abcdef" for c in source.name):
            policy.pretrained_revision = source.name
    source = source.resolve()
    dependency = Path(snapshot_download(profile.dependency_repo, revision=profile.dependency_revision,
                                        allow_patterns=["*.json", "*.txt", "*.model", "*.jinja",
                                                        "tokenizer*", "*.safetensors"])).resolve()
    policy.checkpoint_path = str(dependency)
    policy.checkpoint_revision = profile.dependency_revision
    # Saved YAM inference is continuous, with normal local precision defaults.
    policy.enable_inference_cuda_graph = False
    preprocessor = json.loads((source / "policy_preprocessor.json").read_text())
    pack_steps = [step for step in preprocessor["steps"] if step.get("registry_name") == "molmoact2_pack_inputs"]
    if len(pack_steps) != 1:
        raise ValueError("MolmoAct2 saved processor must contain exactly one input-packing step")
    pack_steps[0]["config"].update(checkpoint_path=str(dependency),
                                  checkpoint_revision=profile.dependency_revision, allow_image_key_fallback=False)
    if pack_steps[0]["config"].get("control_mode") != "absolute joint pose":
        raise ValueError("MolmoAct2 saved processor does not declare absolute joint actions")
    pack_config = pack_steps[0]["config"]
    if (pack_config.get("env_action_dim") != len(profile.action_names)
            or pack_config.get("chunk_size") != profile.chunk_size
            or pack_config.get("image_keys") != [f"observation.images.{k}" for k in profile.native_image_keys]):
        raise ValueError("MolmoAct2 saved processor dimensions/camera order differ from the reviewed profile")
    if any(step.get("registry_name") == "relative_actions_processor"
           and step.get("config", {}).get("enabled", True) for step in preprocessor["steps"]):
        raise ValueError("Relative-action local sync processing is unsupported by LeRobot 0.6.1")
    encoded = draccus.encode(policy)
    identity = json.dumps({"source": str(source), "policy": encoded, "processor": preprocessor}, sort_keys=True)
    bundle = ROOT / "data" / "local_policy_bundles" / hashlib.sha256(identity.encode()).hexdigest()[:24]
    bundle.mkdir(parents=True, exist_ok=True)
    for original in source.iterdir():
        if original.name in ("config.json", "policy_preprocessor.json") or original.suffix not in (".json", ".safetensors"):
            continue
        target = bundle / original.name
        try:
            target.symlink_to(original.resolve())
        except FileExistsError:
            if target.resolve() != original.resolve():
                raise ValueError("Existing local policy bundle points to a different checkpoint") from None
    _write_json_atomic(bundle / "policy_preprocessor.json", preprocessor)
    policy.pretrained_path = bundle
    _write_json_atomic(bundle / "config.json", draccus.encode(policy, PreTrainedConfig))
    return bundle


def _write_json_atomic(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as temporary:
        json.dump(value, temporary, indent=2)
        temporary.write("\n")
    os.replace(temporary.name, path)


@parser.wrap()
def rollout(cfg: RolloutConfig):
    validate_local_rollout(cfg)
    if cfg.policy.type == "molmoact2":
        prepare_local_molmo_bundle(cfg)
    from lerobot.scripts.lerobot_rollout import rollout as upstream_rollout

    return upstream_rollout(cfg)


def main():
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    rollout()


if __name__ == "__main__":
    main()
