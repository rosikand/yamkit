"""Actual upstream config parsing and hardware-free effective-config preflight."""

import json
import sys

import pytest

from yamkit.deployment import InferenceOptions
from yamkit.inference.profiles import get_profile


@pytest.fixture
def local_config(rig, tmp_path):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.rollout.configs import RolloutConfig
    from lerobot_robot_yamkit import BiYamFollowerConfig

    profile = get_profile("molmoact2")
    for spec in rig.arms.values():
        if spec.has_motor_gripper:
            spec.gripper_limits = [0.0, 6.5]
    rig.cameras = {key: {"type": "opencv", "index_or_path": i, "width": 8, "height": 8, "fps": 30}
                   for i, key in enumerate(profile.image_keys)}
    rig.save()
    policy = MolmoAct2Config(device="cpu", checkpoint_path=profile.dependency_repo,
                            control_mode="absolute joint pose", norm_tag="yam_dual_molmoact2",
                            setup_type="bimanual yam robotic arms in molmoact2", action_mode="continuous",
                            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (14,)),
                                            **{f"observation.images.{k}": PolicyFeature(FeatureType.VISUAL, (3, 8, 8))
                                               for k in profile.native_image_keys}},
                            output_features={"action": PolicyFeature(FeatureType.ACTION, (14,))})
    checkpoint = tmp_path / "a-local-checkpoint"
    checkpoint.mkdir()
    policy._save_pretrained(checkpoint)
    policy.pretrained_path = checkpoint
    return RolloutConfig(robot=BiYamFollowerConfig(rig=str(rig.path)), policy=policy,
                         task="pick cube", device="cpu")


def test_local_device_indices_and_unlimited_duration_preserved():
    InferenceOptions("outputs/custom", device="cuda:0", duration=0).validate(motion=True)
    InferenceOptions("outputs/custom", device="cuda:12", duration=7200).validate(motion=True)
    with pytest.raises(ValueError, match="duration"):
        InferenceOptions("molmoact2", backend="modal", duration=0).validate(motion=True)


def test_local_molmo_sync_allowed_rtc_rejected():
    InferenceOptions("molmoact2").validate(motion=True)
    with pytest.raises(ValueError, match="RTC"):
        InferenceOptions("molmoact2", rtc=True).validate(motion=True)


def test_local_molmo_path_validates_actual_config_without_activation(local_config, fake_connect):
    from yamkit.inference.mapping import CAMERA_RENAME_MAP
    from yamkit.local_rollout import validate_local_rollout

    validate_local_rollout(local_config)
    assert local_config.rename_map == CAMERA_RENAME_MAP
    assert not fake_connect


def test_effective_nested_rtc_flag_rejected_by_actual_parser(local_config, monkeypatch, fake_connect):
    from yamkit import local_rollout

    monkeypatch.setattr(sys, "argv", ["yamkit.local_rollout", "--robot.type=bi_yam_follower",
                                      f"--robot.rig={local_config.robot.rig}",
                                      f"--policy.path={local_config.policy.pretrained_path}",
                                      "--inference.type=rtc", "--inference.rtc.enabled=false"])
    monkeypatch.setattr("lerobot.scripts.lerobot_rollout.rollout", lambda cfg: pytest.fail("must reject before rollout"))
    with pytest.raises(ValueError, match="RTC"):
        local_rollout.rollout()
    assert not fake_connect


def test_policy_level_rtc_override_cannot_bypass_sync_preflight(local_config, monkeypatch, fake_connect):
    from yamkit import local_rollout

    monkeypatch.setattr(sys, "argv", ["yamkit.local_rollout", "--robot.type=bi_yam_follower",
                                      f"--robot.rig={local_config.robot.rig}",
                                      f"--policy.path={local_config.policy.pretrained_path}",
                                      "--policy.rtc_config.enabled=true"])
    with pytest.raises(ValueError, match="RTC"):
        local_rollout.rollout()
    assert not fake_connect


def test_actual_parser_keeps_upstream_duplicate_path_semantics(local_config, monkeypatch, fake_connect):
    from yamkit import local_rollout

    seen = []
    monkeypatch.setattr(sys, "argv", ["yamkit.local_rollout", "--robot.type=bi_yam_follower",
                                      f"--robot.rig={local_config.robot.rig}",
                                      f"--policy.path={local_config.policy.pretrained_path}",
                                      "--policy.path=ignored-second-checkpoint", "--duration=0"])
    monkeypatch.setattr(local_rollout, "prepare_local_molmo_bundle", lambda cfg: None)
    monkeypatch.setattr("lerobot.scripts.lerobot_rollout.rollout", lambda cfg: seen.append(cfg))
    local_rollout.rollout()
    assert len(seen) == 1 and seen[0].duration == 0
    assert str(seen[0].policy.pretrained_path) == str(local_config.policy.pretrained_path)
    assert not fake_connect


@pytest.mark.parametrize("field,dimension", [("state", 32), ("action", 6)])
def test_custom_policy_dimensions_checked_before_activation(local_config, fake_connect, field, dimension):
    from lerobot.configs import FeatureType, PolicyFeature

    from yamkit.local_rollout import validate_local_rollout

    if field == "state":
        local_config.policy.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (dimension,))
    else:
        local_config.policy.output_features["action"] = PolicyFeature(FeatureType.ACTION, (dimension,))
    with pytest.raises(ValueError, match="dimensions"):
        validate_local_rollout(local_config)
    assert not fake_connect


def test_local_calibration_and_camera_override_rejected(local_config, fake_connect):
    from yamkit.local_rollout import validate_local_rollout

    local_config.rename_map = {"observation.images.left_wrist": "observation.images.right"}
    with pytest.raises(ValueError, match="rename_map"):
        validate_local_rollout(local_config)
    assert not fake_connect


def test_local_molmo_bundle_pins_saved_processor_and_keeps_unchanged_weights(local_config, monkeypatch, tmp_path):
    from yamkit.local_rollout import prepare_local_molmo_bundle

    source = local_config.policy.pretrained_path
    source.joinpath("model.safetensors").write_bytes(b"fake-test-weights")
    source.joinpath("policy_preprocessor.json").write_text(json.dumps({"steps": [
        {"registry_name": "molmoact2_pack_inputs", "config": {"checkpoint_path": "floating-dependency",
         "checkpoint_revision": None, "control_mode": "absolute joint pose", "env_action_dim": 14,
         "chunk_size": 30, "image_keys": ["observation.images.top", "observation.images.left", "observation.images.right"]}},
    ]}))
    source.joinpath("policy_postprocessor.json").write_text('{"steps": []}')
    dependency = tmp_path / "immutable-dependency"
    dependency.mkdir()
    calls = []

    def download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(dependency)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    monkeypatch.setattr("yamkit.paths.ROOT", tmp_path)
    result = prepare_local_molmo_bundle(local_config)
    profile = get_profile("molmoact2")
    assert calls[0][0] == profile.dependency_repo
    assert calls[0][1]["revision"] == profile.dependency_revision
    saved = json.loads((result / "policy_preprocessor.json").read_text())["steps"][0]["config"]
    assert saved["checkpoint_revision"] == local_config.policy.checkpoint_revision == profile.dependency_revision
    assert saved["checkpoint_path"] == local_config.policy.checkpoint_path == str(dependency)
    assert saved["allow_image_key_fallback"] is False
    assert (result / "model.safetensors").is_symlink()
    assert (result / "model.safetensors").resolve() == (source / "model.safetensors").resolve()
    assert json.loads((source / "policy_preprocessor.json").read_text())["steps"][0]["config"]["checkpoint_revision"] is None
