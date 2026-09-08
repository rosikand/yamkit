"""Production graph contracts using the pinned manager with CPU-only fake capture."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from lerobot.policies.molmoact2.molmoact2_hf_model import inference as upstream

from yamkit.inference.execution import request_execution_signature, signature_digest
from yamkit.inference.profiles import get_profile
from yamkit.inference.protocol import encode_image, native_fixture_request
from yamkit.inference.service import ModelRuntime


def request(*, mode="native_fixture", task="put the cube in the bin", sequence=0, shape=(8, 12), value=20):
    profile = get_profile("molmoact2")
    result = native_fixture_request(profile, sequence_id=sequence, session_id="graph-test")
    names = profile.native_image_keys if mode == "native_fixture" else profile.image_keys
    result.update(mode=mode, execution_mode="cuda_graph10", task=task,
                  images={name: encode_image(np.full((*shape, 3), value, dtype=np.uint8)) for name in names})
    return result


@pytest.fixture
def graph_runtime(monkeypatch):
    events = SimpleNamespace(captures=0, replays=0, loop_calls=0, predict_calls=0,
                             batch=1, context_length=4, eligible=True, fail=False, output_steps=30)

    class Replay:
        def __init__(self, run):
            self.run = run
            self.output = torch.empty_like(run())

        def replay(self):
            events.replays += 1
            self.output.copy_(self.run())

    def capture(run, device, *, after_warmup):
        events.captures += 1
        after_warmup()
        replay = Replay(run)
        return replay, replay.output

    monkeypatch.setattr(upstream, "_capture_cuda_graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr("yamkit.inference.service._memory_metadata", lambda device: {"hardware": False})
    embedding = torch.nn.Linear(14, 14, dtype=torch.bfloat16)
    backbone = SimpleNamespace(config=SimpleNamespace(flow_matching_num_steps=10),
                               action_expert=SimpleNamespace(parameters=embedding.parameters))
    graph = upstream.ActionCudaGraphManager(backbone)
    # Only eligibility is faked. Actual keying, cache replacement and copying of
    # every fresh trajectory/context use the installed upstream manager.
    graph.can_use_action_flow = lambda inputs: graph.enabled and events.eligible
    backbone.action_cuda_graph_manager = graph

    class Policy:
        config = SimpleNamespace(model_dtype="bfloat16", num_inference_steps=None,
                                 n_action_steps=30, enable_inference_cuda_graph=False,
                                 action_mode="continuous", inference_action_mode="continuous")
        parameters = staticmethod(embedding.parameters)
        _backbone = staticmethod(lambda: backbone)

        def reset(self):
            pass

        def predict_action_chunk(self, batch, **kwargs):
            events.predict_calls += 1
            if events.fail:
                raise RuntimeError("fake model failed")
            pixels = next(value for name, value in batch.items() if name.startswith("observation.images."))
            state_value = float(batch["observation.state"].sum())
            context = SimpleNamespace(
                kv_contexts=((torch.full((events.batch, 2, events.context_length, 4), float(pixels.mean()), dtype=torch.bfloat16),
                              torch.ones(events.batch, 2, events.context_length, 4, dtype=torch.bfloat16)),),
                cross_mask=None, self_mask=None, valid_action=None, rope_cache=None)
            modulation = SimpleNamespace(conditioning=torch.zeros(events.batch, 1, dtype=torch.bfloat16),
                                         block_modulations=(), final_modulation=())
            flow = upstream._ActionFlowInputs(
                trajectory=torch.full((events.batch, 30, 14), state_value, dtype=torch.bfloat16),
                context=context, modulations=(modulation,) * 10, action_dim_is_pad=None)

            def run_loop(inputs, steps):
                events.loop_calls += 1
                return inputs.trajectory + inputs.context.kv_contexts[0][0].mean()

            if graph.can_use_action_flow(flow):
                output = graph.run_action_flow(flow, kwargs.get("num_steps", 10), run_loop)
            else:
                output = run_loop(flow, kwargs.get("num_steps", 10))
            return output[:, :events.output_steps].float()

    runtime = ModelRuntime(get_profile("molmoact2"), Policy(), lambda frame: frame, lambda action: action,
                           device="cuda", execution_mode="cuda_graph10")
    return runtime, graph, events


@pytest.mark.parametrize("mode", ["robot", "live_probe", "saved_probe"])
def test_real_modes_match_canonical_fixture_signature_and_require_warmup(graph_runtime, mode):
    runtime, graph, events = graph_runtime
    warm = request()
    real = request(mode=mode, sequence=1)
    assert request_execution_signature(warm, runtime.profile) == request_execution_signature(real, runtime.profile)
    with pytest.raises(ValueError, match="warm-up"):
        runtime.predict_chunk(real)
    assert events.predict_calls == events.captures == 0
    warmed = runtime.predict_chunk(warm)
    assert warmed["execution_mode"] == "cuda_graph10"
    assert warmed["execution_identity"]["model_dtype"] == "bfloat16"
    assert warmed["graph_warmup"]["ready"]
    assert warmed["graph_warmup"]["signature_sha256"] == signature_digest(request_execution_signature(real, runtime.profile))
    assert warmed["model_execution"]["graph_capture_required"]
    cached = graph.action_flow_graph
    response = runtime.predict_chunk(real)
    assert response["model_execution"]["cuda_graph_used"] and not response["model_execution"]["graph_capture_required"]
    assert graph.action_flow_graph is cached and events.captures == 1
    assert np.asarray(response["chunk"]).shape == (30, 14)
    assert runtime.ready()["graph_warmup"]["signature"] == warmed["graph_warmup"]["signature"]


def test_replay_copies_fresh_real_states_and_pixels_into_upstream_cache(graph_runtime):
    runtime, graph, events = graph_runtime
    runtime.predict_chunk(request())
    first = request(mode="robot", sequence=1, value=30)
    first["state"][0] = 0.125
    second = request(mode="robot", sequence=2, value=220)
    second["state"][0] = 0.5
    a, b = runtime.predict_chunk(first), runtime.predict_chunk(second)
    assert np.max(np.asarray(b["chunk"]) - np.asarray(a["chunk"])) > 0.5
    assert events.captures == 1 and events.replays == 3
    assert graph.action_flow_graph.static_inputs.trajectory[0, 0, 0].item() == 0.5
    assert graph.action_flow_graph.static_inputs.context.kv_contexts[0][0].mean().item() > 0.8


@pytest.mark.parametrize("change", ["task", "shape", "crop", "encoding", "mode", "state_width"])
def test_changed_request_contract_rejected_before_prediction(graph_runtime, change):
    runtime, _, events = graph_runtime
    runtime.predict_chunk(request())
    real = request(mode="robot", sequence=1)
    if change == "task":
        real["task"] = "another task with the same image shapes"
    elif change == "shape":
        real = request(mode="robot", sequence=1, shape=(12, 12))
    elif change == "crop":
        real["crop"] = "center_16_9"
    elif change == "encoding":
        real["images"] = {name: encode_image(np.zeros((8, 12, 3), np.uint8), encoding="jpeg")
                          for name in runtime.profile.image_keys}
    elif change == "mode":
        real["execution_mode"] = "eager"
    else:
        real["state"] = [0.0] * 13
    with pytest.raises(ValueError):
        runtime.predict_chunk(real)
    assert events.predict_calls == events.captures == 1


@pytest.mark.parametrize("change", ["batch", "context_length", "cache", "eligibility", "disabled"])
def test_unexpected_graph_changes_fail_before_capture_or_action_loop(graph_runtime, change):
    runtime, graph, events = graph_runtime
    runtime.predict_chunk(request())
    loop_calls = events.loop_calls
    original_run, original_can = graph.run_action_flow, graph.can_use_action_flow
    if change == "batch":
        events.batch = 2
    elif change == "context_length":
        events.context_length = 8
    elif change == "cache":
        graph.action_flow_graph = None
    elif change == "eligibility":
        events.eligible = False
    else:
        graph.enabled = False
    with pytest.raises(ValueError):
        runtime.predict_chunk(request(mode="robot", sequence=1))
    assert events.captures == 1 and events.loop_calls == loop_calls
    assert graph.run_action_flow == original_run and graph.can_use_action_flow == original_can


@pytest.mark.parametrize("override", [{"diagnostic_cuda_graph": False}, {"diagnostic_num_inference_steps": 5}])
def test_diagnostic_override_cannot_reconfigure_production_graph(graph_runtime, override):
    runtime, graph, events = graph_runtime
    runtime.predict_chunk(request())
    native = request(sequence=1)
    native.update(override)
    with pytest.raises(ValueError, match="overrides"):
        runtime.predict_chunk(native)
    assert graph.enabled and runtime.policy.config.enable_inference_cuda_graph
    assert events.predict_calls == events.captures == 1
    assert runtime.ready()["graph_warmup"]["ready"]


def test_matching_diagnostic_overrides_leave_production_mode_enabled(graph_runtime):
    runtime, graph, events = graph_runtime
    native = request()
    native.update(diagnostic_cuda_graph=True, diagnostic_num_inference_steps=10)
    response = runtime.predict_chunk(native)
    assert response["execution_mode"] == "cuda_graph10" and response["graph_warmup"]["ready"]
    assert graph.enabled and runtime.policy.config.enable_inference_cuda_graph
    runtime.predict_chunk(request(mode="robot", sequence=1))
    assert events.captures == 1 and events.replays == 2


@pytest.mark.parametrize("name,value", [("num_inference_steps", 0), ("num_inference_steps", 5),
                                       ("n_action_steps", 2), ("model_dtype", "float32")])
def test_changed_production_configuration_fails_before_prediction(graph_runtime, name, value):
    runtime, _, events = graph_runtime
    runtime.predict_chunk(request())
    setattr(runtime.policy.config, name, value)
    with pytest.raises(ValueError, match="requires unchanged"):
        runtime.predict_chunk(request(mode="robot", sequence=1))
    assert events.predict_calls == events.captures == 1


@pytest.mark.parametrize("failure", ["model", "short_chunk"])
def test_failed_warmup_cannot_publish_or_keep_graph_readiness(graph_runtime, failure):
    runtime, graph, events = graph_runtime
    runtime.predict_chunk(request())
    original_run, original_can = graph.run_action_flow, graph.can_use_action_flow
    if failure == "model":
        events.fail = True
    else:
        events.output_steps = 2
    with pytest.raises((RuntimeError, ValueError)):
        runtime.predict_chunk(request(sequence=1))
    assert not runtime.ready()["graph_warmup"]["ready"]
    assert graph.run_action_flow == original_run and graph.can_use_action_flow == original_can
    assert graph.enabled
    with pytest.raises(ValueError, match="warm-up"):
        runtime.predict_chunk(request(mode="robot", sequence=2))


def test_new_fixture_warms_changed_task_and_reset_does_not_rewarm_old_task(graph_runtime):
    runtime, _, events = graph_runtime
    old = runtime.predict_chunk(request())
    newer = runtime.predict_chunk(request(sequence=1, task="place the object to the left"))
    assert newer["graph_warmup"]["signature_sha256"] != old["graph_warmup"]["signature_sha256"]
    runtime.reset("graph-test")
    real = request(mode="robot", sequence=0)
    real["session_id"] = "next-session"
    with pytest.raises(ValueError, match="warm-up"):
        runtime.predict_chunk(real)
    real["task"] = "place the object to the left"
    runtime.predict_chunk(real)
    assert events.predict_calls == 3


def test_warm_metadata_cannot_be_mutated_by_a_caller(graph_runtime):
    runtime, _, _ = graph_runtime
    response = runtime.predict_chunk(request())
    response["graph_warmup"]["signature"]["task"] = "corrupted"
    ready = runtime.ready()
    ready["graph_warmup"]["signature"]["images"][0]["width"] = 1
    assert runtime.ready()["graph_warmup"]["signature"]["task"] == "put the cube in the bin"
    runtime.predict_chunk(request(mode="robot", sequence=1))


@pytest.mark.parametrize("profile,device,mode", [("molmoact2", "cpu", "cuda_graph10"),
                                                ("smolvla", "cuda", "cuda_graph10"),
                                                ("molmoact2", "cuda", "unknown")])
def test_unsupported_execution_fails_before_loading(profile, device, mode, monkeypatch):
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **kw: pytest.fail("unexpected model download"))
    with pytest.raises(ValueError):
        ModelRuntime.load(profile, device=device, execution_mode=mode)
