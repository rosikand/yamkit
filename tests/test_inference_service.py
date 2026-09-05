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
    image = np.arange(360 * 640 * 3, dtype=np.uint8).reshape(360, 640, 3)
    encoded = encode_image(image)
    assert encoded["encoding"] == "rgb8" and len(encoded["data"]) == image.nbytes
    np.testing.assert_array_equal(decode_image(encoded), image)
    for bad in ({**encoded, "width": 10}, {**encoded, "encoding": "pickle"},
                {**encoded, "height": 100000}, {**encoded, "data": "untrusted"}):
        with pytest.raises(ValueError):
            decode_image(bad)


@pytest.mark.parametrize("quality", [85, 90, 95])
def test_explicit_jpeg_preserves_rgb_order_dimensions_and_bounds(quality):
    pixels = np.zeros((48, 64, 3), np.uint8)
    pixels[:, :32, 0] = 220
    pixels[:, 32:, 2] = 220
    encoded = encode_image(pixels, encoding="jpeg", quality=quality)
    decoded = decode_image(encoded)
    assert encoded["encoding"] == "jpeg" and encoded["quality"] == quality
    assert decoded.shape == pixels.shape and decoded.dtype == pixels.dtype
    assert decoded[24, 8, 0] > 200 and decoded[24, 8, 2] < 10
    assert decoded[24, 56, 2] > 200 and decoded[24, 56, 0] < 10
    assert len(encoded["data"]) < pixels.nbytes
    for bad in ({**encoded, "width": 65}, {**encoded, "data": b"not JPEG"},
                {**encoded, "quality": True}, {**encoded, "data": encoded["data"][:-20]}):
        with pytest.raises(ValueError):
            decode_image(bad)


def test_jpeg_request_validation_does_not_decode_pixels(monkeypatch):
    from PIL import JpegImagePlugin

    request = robot_request()
    request["images"] = {name: encode_image(decode_image(value), encoding="jpeg")
                         for name, value in request["images"].items()}
    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "load",
                        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Unexpected pixel decode")))
    validate_request(request, get_profile("molmoact2"))


@pytest.mark.parametrize("change", [{"mode": "robot"}, {"diagnostic_seed": True},
                                    {"diagnostic_seed": -1}, {"diagnostic_seed": 2**32}])
def test_diagnostic_seed_never_changes_robot_requests_or_accepts_invalid_seeds(change):
    request = native_fixture_request("molmoact2", diagnostic_seed=17)
    request.update(change)
    with pytest.raises(ValueError):
        validate_request(request, get_profile("molmoact2"))


def test_seeded_fixture_uses_same_noise_without_mutating_global_rng():
    service = runtime()
    generators = []

    def predict(batch, *, generator):
        generators.append(generator)
        return torch.rand((1, 2, 14), generator=generator)

    service.policy.predict_action_chunk = predict
    initial_rng = torch.random.get_rng_state().clone()
    first = service.predict_chunk(native_fixture_request("molmoact2", diagnostic_seed=17))
    second = service.predict_chunk(native_fixture_request("molmoact2", diagnostic_seed=17))
    assert first["chunk"] == second["chunk"]
    assert first["diagnostic_seed"] == second["diagnostic_seed"] == 17
    assert len(generators) == 2 and generators[0] is not generators[1]
    assert torch.equal(initial_rng, torch.random.get_rng_state())


@pytest.mark.parametrize("encoding", ["rgb8", "jpeg"])
@pytest.mark.parametrize("crop", ["none", "center_16_9"])
def test_server_crops_decoded_pixels_once_before_saved_preprocessing(encoding, crop):
    service = runtime()
    seen = {}
    service.pre = lambda frame: seen.update(frame) or frame
    request = robot_request()
    pixels = np.random.default_rng(4310).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    request.update(crop=crop, images={name: encode_image(pixels, encoding=encoding)
                                     for name in service.profile.image_keys})
    response = service.predict_chunk(request)
    for name, image in request["images"].items():
        decoded = decode_image(image)
        expected = decoded[6:42] if crop == "center_16_9" else decoded
        tensor = torch.from_numpy(expected.copy()).permute(2, 0, 1).float() / 255
        torch.testing.assert_close(seen[f"observation.images.{name}"], tensor, rtol=0, atol=0)
        assert response["transforms"][name]["input_hw"] == [48, 64]
        assert response["transforms"][name]["output_hw"] == list(expected.shape[:2])


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
    assert timing["request_deserialization_s"] is timing["response_serialization_s"] is None
    assert timing["request_validation_s"] >= 0
    assert timing["response_validation_s"] >= 0
    assert set(timing["per_camera"]) == set(service.profile.image_keys)
    assert all(value >= 0 for camera in timing["per_camera"].values() for value in camera.values())


def test_server_lifecycle_identifies_first_prediction_and_reuses_loaded_model():
    service = runtime()
    request = robot_request()
    assert service.ready()["prediction_count"] == 0
    first = service.predict_chunk(request)
    request["sequence_id"] += 1
    second = service.predict_chunk(request)
    assert first["instance_id"] == second["instance_id"] == service.ready()["instance_id"]
    assert first["lifecycle"]["first_prediction"] is True
    assert first["lifecycle"]["idle_s"] is None
    assert second["lifecycle"]["first_prediction"] is False
    assert second["lifecycle"]["idle_s"] >= 0
    assert second["lifecycle"]["prediction_count"] == service.ready()["prediction_count"] == 2


def test_saved_processor_hooks_preserve_outputs_and_do_not_accumulate_on_failure():
    from lerobot.policies.molmoact2.processor_molmoact2 import MolmoAct2ClampActionProcessorStep
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    from yamkit.inference.service import _run_processor

    post = PolicyProcessorPipeline(steps=[MolmoAct2ClampActionProcessorStep()],
                                   to_transition=policy_action_to_transition,
                                   to_output=transition_to_policy_action)
    original = torch.tensor([[[-2.0, 0.5, 2.0]]])
    expected = post(original.clone())
    for _ in range(3):
        result, steps = _run_processor(post, original.clone())
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert steps["0:MolmoAct2ClampActionProcessorStep"] >= 0
        assert not post.before_step_hooks and not post.after_step_hooks
    with pytest.raises(ValueError, match="PolicyAction"):
        _run_processor(post, "invalid action")
    assert not post.before_step_hooks and not post.after_step_hooks


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
    assert cfg["region"] == cfg["routing_region"] == "us-west"
    assert cfg["memory"] == 65536
    assert "__init__" not in captured["class"].__dict__
    assert captured["secret"] == {"HF_TOKEN": "private-test-hf-token"}
    assert not {"HF_TOKEN", "DATABASE_URL", "YAMKIT_OPENAI_API_KEY", "MODAL_TOKEN_SECRET"} & captured["env"].keys()
    create_app(development=True, memory_mib=49152)
    assert captured["config"]["memory"] == 49152
    for invalid in (32768, 65537, True):
        with pytest.raises(ValueError, match="Host memory"):
            create_app(memory_mib=invalid)
    create_app(gpu="H100!")
    assert captured["config"]["gpu"] == "H100!"
    assert captured["config"]["max_containers"] == 1
    for invalid in ("H100", "H100:2", "H200", "L40S:2"):
        with pytest.raises(ValueError, match="exact H100"):
            create_app(gpu=invalid)


def test_runtime_reports_measured_host_memory_separately_from_gpu():
    memory = runtime().ready()["memory"]
    assert memory["process_peak_rss_bytes"] > 0
    assert "cuda_peak_allocated_bytes" not in memory
    assert "cgroup_memory_peak_bytes" in memory


def test_runtime_reports_actual_cuda_device_identity_without_gpu(monkeypatch):
    import torch

    from yamkit.inference.service import _memory_metadata

    seen = []

    def device_name(device):
        seen.append(device)
        return "NVIDIA H100 80GB HBM3"

    monkeypatch.setattr(torch.cuda, "get_device_name", device_name)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 2048)
    memory = _memory_metadata("cuda:0")
    assert memory["cuda_device_name"] == "NVIDIA H100 80GB HBM3"
    assert memory["cuda_compute_capability"] == [9, 0]
    assert memory["cuda_peak_allocated_bytes"] == 1024
    assert memory["cuda_peak_reserved_bytes"] == 2048
    assert seen == ["cuda:0"]


@pytest.mark.parametrize("key,value", [("diagnostic_num_inference_steps", 5),
                                        ("diagnostic_cuda_graph", True), ("diagnostic_cuda_graph", False)])
def test_model_experiments_cannot_enter_robot_requests(key, value):
    request = robot_request()
    request[key] = value
    with pytest.raises(ValueError, match="only for Molmo native fixtures"):
        validate_request(request, get_profile("molmoact2"))


@pytest.mark.parametrize("key,value", [("diagnostic_num_inference_steps", 0),
                                        ("diagnostic_num_inference_steps", 6),
                                        ("diagnostic_num_inference_steps", True),
                                        ("diagnostic_cuda_graph", 1)])
def test_model_experiment_values_are_bounded(key, value):
    request = native_fixture_request("molmoact2")
    request[key] = value
    with pytest.raises(ValueError):
        validate_request(request, get_profile("molmoact2"))


def experimental_runtime():
    policy = FakePolicy()
    policy.config = SimpleNamespace(model_dtype="bfloat16", num_inference_steps=None,
                                    enable_inference_cuda_graph=False)
    embedding = torch.nn.Linear(14, 14, dtype=torch.bfloat16)
    expert = SimpleNamespace(action_embed=embedding, parameters=embedding.parameters)
    graph = SimpleNamespace(enabled=False, action_flow_graph=None)
    graph.set_enabled = lambda enabled: setattr(graph, "enabled", enabled)

    def run_action_flow():
        graph.action_flow_graph = object()
        return "replayed"

    graph.run_action_flow = run_action_flow
    backbone = SimpleNamespace(config=SimpleNamespace(flow_matching_num_steps=10,
                               action_expert_config=SimpleNamespace(num_layers=36)),
                               action_expert=expert, action_cuda_graph_manager=graph)
    policy._backbone = lambda: backbone
    policy.parameters = embedding.parameters
    policy.buffers = lambda: iter([torch.tensor([1], dtype=torch.int64)])
    return ModelRuntime(get_profile("molmoact2"), policy, lambda frame: frame, lambda action: action, device="cpu")


def test_half_step_fixture_leaves_default_unchanged_and_observes_real_dtypes():
    service = experimental_runtime()
    steps = []

    def predict(batch, **kwargs):
        steps.append(kwargs.get("num_steps"))
        return service._action_expert.action_embed(torch.ones(1, 2, 14, dtype=torch.bfloat16)).float()

    service.policy.predict_action_chunk = predict
    first = service.predict_chunk(native_fixture_request("molmoact2", diagnostic_num_inference_steps=5))
    second = service.predict_chunk(native_fixture_request("molmoact2"))
    assert steps == [5, None]
    assert service.policy.config.num_inference_steps is None
    assert first["model_execution"]["effective_num_inference_steps"] == 5
    assert second["model_execution"]["effective_num_inference_steps"] == 10
    metadata = second["model_execution"]
    assert metadata["parameter_dtype_numel"] == {"torch.bfloat16": 210}
    assert metadata["buffer_dtype_numel"] == {"torch.int64": 1}
    assert metadata["raw_output_dtype"] == "torch.float32"
    assert metadata["observed_activation_dtypes"]["action_embedding"]["output_dtype"] == "torch.bfloat16"
    assert not service._action_expert.action_embed._forward_hooks


@pytest.mark.parametrize("fail", [False, True])
def test_graph_experiment_restores_disabled_manager_and_hooks_even_on_failure(fail):
    service = experimental_runtime()
    service.device = "cuda"  # Test only the context manager; no CUDA operation is invoked.
    graph = service._action_graph
    original = graph.run_action_flow
    request = native_fixture_request("molmoact2", diagnostic_cuda_graph=True)

    def run():
        with service._observe_execution(request) as metadata:
            assert graph.enabled
            assert graph.run_action_flow() == "replayed"
            if fail:
                raise RuntimeError("model failure")
        assert metadata["cuda_graph_enabled"] is metadata["cuda_graph_used"] is True
        assert metadata["cuda_graph_cache_populated_after"] is True

    if fail:
        with pytest.raises(RuntimeError, match="model failure"):
            run()
    else:
        run()
    assert graph.enabled is False
    assert graph.run_action_flow is original
    assert not service._action_expert.action_embed._forward_hooks
