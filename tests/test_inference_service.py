"""Hardware/cloud-free protocol, numerical pipeline and model state isolation tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yamkit.inference.mapping import (
    CAMERA_RENAME_MAP,
    MOLMO_NAMES,
    YAM_NAMES,
    center_crop_rgb,
    dataset_to_yamkit,
    yamkit_to_dataset,
)
from yamkit.inference.profiles import get_profile, list_profiles
from yamkit.inference.protocol import (
    decode_image,
    encode_image,
    native_fixture_request,
    validate_request,
    validate_response,
)
from yamkit.inference.service import ModelRuntime


def test_bidirectional_mapping_preserves_explicit_vector_order():
    values = dict(zip(YAM_NAMES, range(14), strict=True))
    mapped = yamkit_to_dataset(dict(reversed(list(values.items()))))
    assert list(mapped) == list(MOLMO_NAMES)
    assert list(mapped.values()) == list(range(14))
    assert dataset_to_yamkit(mapped) == values
    assert MOLMO_NAMES[0:7] == tuple(f"left_joint_{i}.pos" for i in range(6)) + ("left_gripper.pos",)
    assert MOLMO_NAMES[7] == "right_joint_0.pos"
    for malformed in ({**values, "unknown": 1}, {k: v for k, v in values.items() if k != YAM_NAMES[2]}):
        with pytest.raises(ValueError, match="Missing or extra"):
            yamkit_to_dataset(malformed)


def test_profiles_offline_and_base_mappings_fail_closed():
    assert len(list_profiles()) == 3
    for name, dim in (("smolvla", 6), ("pi05", 32)):
        profile = get_profile(name)
        assert len(profile.revision) == 40
        assert len(profile.state_names) == dim
        with pytest.raises(ValueError, match="no verified"):
            profile.require_robot_mapping()
    assert get_profile("molmoact2").fps == 30


def test_crop_explicit_without_renaming_or_mutating():
    image = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    no_crop, meta = center_crop_rgb(image)
    assert no_crop is image and meta["crop"] == "none"
    crop, meta = center_crop_rgb(image, "center_16_9")
    assert crop.shape == (360, 640, 3)
    assert meta["offset_yx"] == [60, 0]
    np.testing.assert_array_equal(crop, image[60:420])
    assert CAMERA_RENAME_MAP["observation.images.left_wrist"] == "observation.images.left"
    assert "observation.images.left" not in CAMERA_RENAME_MAP


def test_payload_rejects_shape_byte_and_encoding_mismatches():
    image = np.zeros((360, 640, 3), np.uint8)
    encoded = encode_image(image)
    np.testing.assert_array_equal(decode_image(encoded), image)
    for bad in ({**encoded, "width": 10}, {**encoded, "encoding": "pickle"},
                {**encoded, "height": 100000}, {**encoded, "data": "untrusted"}):
        with pytest.raises(ValueError):
            decode_image(bad)


def robot_request(mode="robot"):
    profile = get_profile("molmoact2")
    request = native_fixture_request(profile)
    request.update(mode=mode, images={k: encode_image(np.zeros((48, 64, 3), np.uint8))
                                    for k in profile.image_keys})
    return request


@pytest.mark.parametrize("change", [
    {"model_revision": "wrong"}, {"state": [0.0] * 13}, {"state": [float("nan")] * 14},
    {"state_names": sorted(YAM_NAMES)}, {"observation_age_s": 3.0}, {"timeout_s": 1000},
    {"sequence_id": -1}, {"continuation": {"actions": [[0.0] * 14]}}, {"crop": "auto"},
])
def test_physical_request_rejects_incompatible_mapping_and_freshness(change):
    request = robot_request()
    request.update(change)
    with pytest.raises(ValueError):
        validate_request(request, get_profile("molmoact2"))


def test_saved_probe_age_is_diagnostic_but_live_probe_requires_freshness():
    request = robot_request("saved_probe")
    request["observation_age_s"] = 3600
    validate_request(request, get_profile("molmoact2"))
    request["mode"] = "live_probe"
    with pytest.raises(ValueError, match="observation_age"):
        validate_request(request, get_profile("molmoact2"))


class FakePolicy:
    def __init__(self, dim=14):
        self.calls = 0
        self.resets = 0
        self.dim = dim

    def reset(self):
        self.resets += 1

    def predict_action_chunk(self, batch):
        self.calls += 1
        return torch.ones(1, 2, self.dim) * 0.5

    def select_action(self, batch):
        raise AssertionError("Fresh chunks must never call select_action")


def runtime():
    policy = FakePolicy()
    return ModelRuntime(get_profile("molmoact2"), policy, lambda frame: frame, lambda x: x,
                        device="cpu")


def test_three_fresh_predictions_reset_and_session_isolation():
    service = runtime()
    request = robot_request()
    for i in range(3):
        request["sequence_id"] = i
        reply = service.predict_chunk(request)
        validate_response(reply, request, service.profile)
    assert service.policy.calls == service.policy.resets == 3
    with pytest.raises(ValueError, match="duplicate"):
        service.predict_chunk(request)
    old_session = request["session_id"]
    service.reset(old_session)
    request["sequence_id"] = 3
    with pytest.raises(ValueError, match="Retired"):
        service.predict_chunk(request)
    request.update(session_id="new-session", sequence_id=0)
    service.predict_chunk(request)
    assert service.policy.calls == 4


def test_server_timing_separates_observable_stages_and_marks_modal_queue_unknown():
    service = runtime()
    timing = service.predict_chunk(robot_request())["timing"]
    assert timing["modal_queue_s"] is None
    names = ("queue_wait_s", "state_reset_s", "image_decode_transform_s", "preprocess_s",
             "inference_s", "postprocess_s", "response_conversion_s")
    assert all(timing[name] >= 0 for name in names)
    assert sum(timing[name] for name in names) <= timing["total_s"]


@pytest.mark.parametrize("field,value", [("session_id", "another"), ("sequence_id", 100),
                                        ("action_units", "normalized"), ("chunk", [[float("inf")] * 14]),
                                        ("chunk", [["0.5"] * 14]), ("chunk", [[False] * 14]),
                                        ("chunk", [[0.0] * 13])])
def test_response_mismatch_rejected(field, value):
    service = runtime()
    request = robot_request()
    response = service.predict_chunk(request)
    response[field] = value
    with pytest.raises(ValueError):
        validate_response(response, request, service.profile)


def test_no_robot_requests_to_base_models():
    for profile_id in ("smolvla", "pi05"):
        profile = get_profile(profile_id)
        request = native_fixture_request(profile)
        validate_request(request, profile)
        request["mode"] = "robot"
        with pytest.raises(ValueError, match="no verified"):
            validate_request(request, profile)


def test_expired_inference_result_discarded(monkeypatch):
    service = runtime()
    request = robot_request()
    clock = [1.0]
    original = service.policy.predict_action_chunk

    def expired(batch):
        clock[0] = 200.0
        return original(batch)

    monkeypatch.setattr(service.policy, "predict_action_chunk", expired)
    monkeypatch.setattr("yamkit.inference.service.time", SimpleNamespace(monotonic=lambda: clock[0]))
    with pytest.raises(TimeoutError, match="discarded"):
        service.predict_chunk(request)


def test_actual_saved_style_masked_normalization_and_unclipped_diagnostics():
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        MolmoAct2ClampActionProcessorStep,
        MolmoAct2MaskedNormalizerProcessorStep,
        MolmoAct2MaskedUnnormalizerProcessorStep,
    )
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    mask = [True] * 6 + [False] + [True] * 6 + [False]
    stats = {"action": {"q01": torch.zeros(14), "q99": torch.ones(14) * 2, "mask": torch.tensor(mask)}}
    kwargs = {"features": features, "norm_map": {FeatureType.ACTION: NormalizationMode.QUANTILES},
              "stats": stats}
    norm = MolmoAct2MaskedNormalizerProcessorStep(**copy.deepcopy(kwargs))
    unnorm = MolmoAct2MaskedUnnormalizerProcessorStep(**copy.deepcopy(kwargs))
    raw = torch.ones(1, 2, 14) * 1.5
    raw[..., [6, 13]] = 0.8
    normalize = PolicyProcessorPipeline(steps=[norm], to_transition=policy_action_to_transition,
                                        to_output=transition_to_policy_action)
    post = PolicyProcessorPipeline(steps=[MolmoAct2ClampActionProcessorStep(), unnorm],
                                   to_transition=policy_action_to_transition,
                                   to_output=transition_to_policy_action)
    normalized = normalize(raw)
    assert normalized[0, 0, 6] == raw[0, 0, 6]
    torch.testing.assert_close(post(normalized), raw)
    policy = FakePolicy()
    policy.predict_action_chunk = lambda _: raw * 2
    service = ModelRuntime(get_profile("molmoact2"), policy, lambda f: f, post, device="cpu")
    reply = service.predict_chunk(robot_request("saved_probe"))
    assert reply["chunk"][0][0] == pytest.approx(2.0)
    assert reply["unclipped_chunk"][0][0] == pytest.approx(4.0)
    assert reply["unclipped_chunk"][0][6] == pytest.approx(1.6)
    assert reply["unclipped_action_units"] == "robot"


def test_modal_factory_bounds_and_no_parameterized_pool(monkeypatch):
    from yamkit.inference.modal_service import create_app

    captured = {}

    class Image:
        @classmethod
        def debian_slim(cls, **kwargs):
            return cls()

        def pip_install_from_requirements(self, path):
            return self

        def env(self, values):
            captured["env"] = values
            return self

        def add_local_dir(self, *args, **kwargs):
            return self

    class App:
        def __init__(self, name):
            self.name = name

        def cls(self, **kwargs):
            captured["config"] = kwargs

            def decorate(cls):
                captured["class"] = cls
                return cls
            return decorate

    fake = SimpleNamespace(
        App=App, Image=Image, Volume=SimpleNamespace(from_name=lambda *a, **k: object()),
        Secret=SimpleNamespace(from_dict=lambda values: captured.setdefault("secret", values)),
        enter=lambda: lambda f: f, method=lambda: lambda f: f,
    )
    monkeypatch.setitem(__import__("sys").modules, "modal", fake)
    monkeypatch.setenv("HF_TOKEN", "private-test-hf-token")
    monkeypatch.setenv("DATABASE_URL", "must-not-forward")
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", "must-not-forward-either")
    app = create_app(development=True)
    assert app.name == "yamkit-policy-smolvla"
    cfg = captured["config"]
    assert (cfg["min_containers"], cfg["max_containers"], cfg["buffer_containers"]) == (0, 1, 0)
    assert (cfg["scaledown_window"], cfg["startup_timeout"], cfg["timeout"], cfg["retries"]) == (15, 240, 90, 0)
    assert "__init__" not in captured["class"].__dict__
    assert captured["secret"] == {"HF_TOKEN": "private-test-hf-token"}
    assert not {"HF_TOKEN", "DATABASE_URL", "YAMKIT_OPENAI_API_KEY", "MODAL_TOKEN_SECRET"} & captured["env"].keys()
