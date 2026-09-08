"""Actual LeRobot construction with fake HTTP, cameras and motors only."""

import copy
import threading

import pytest

from tests.test_http_runtime_binding import TASK, runtime_metadata
from tests.test_remote_policy import batch
from yamkit.inference import identity
from yamkit.inference.client import RemoteFault
from yamkit.inference.profiles import get_profile
from yamkit.remote_policy import YamkitRemoteConfig
from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy


@pytest.fixture
def transport(monkeypatch, fake_connect):
    monkeypatch.setattr(identity, "inference_build_id", lambda: "b" * 64)
    monkeypatch.setattr("yamkit.inference.performance.require_physical_modal_rollout", lambda *a, **k: None)
    monkeypatch.setattr("yamkit.inference.qualification.require_runner_context", lambda: None)

    class FakeHttp:
        call_mode = "http"

        def __init__(self):
            self.requests = []
            self.ready_count = 0
            self.closed = False
            self.warm_hook = None
            self.refresh_hook = None
            self.response_hook = None
            self.metadata = runtime_metadata(task="previous pool task", image_hw=(8, 8))
            self.last_timing = {"wire_request_bytes": 2048}

        def cancel(self):
            pass

        def close(self):
            self.closed = True

        def ready(self, timeout_s):
            assert not fake_connect, "Readiness must finish before any motor connection"
            self.ready_count += 1
            result = copy.deepcopy(self.metadata)
            if self.ready_count > 1 and self.refresh_hook:
                self.refresh_hook(result)
            return result

        def predict_chunk(self, request, timeout_s):
            self.requests.append(request)
            native = request["mode"] == "native_fixture"
            if native:
                assert not fake_connect, "Graph warm-up must precede motor connection even on a reused pool"
                image = next(iter(request["images"].values()))
                self.metadata = runtime_metadata(task=request["task"], image_hw=(image["height"], image["width"]))
                if self.warm_hook:
                    self.warm_hook()
            keys = ("protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time")
            response = {**{key: request[key] for key in keys},
                        "action_units": "checkpoint_native" if native else "robot",
                        "action_names": list(get_profile("molmoact2").action_names),
                        "instance_id": "same-container", "execution_mode": "cuda_graph10",
                        "execution_identity": copy.deepcopy(self.metadata["execution_identity"]),
                        "graph_warmup": copy.deepcopy(self.metadata["graph_warmup"]),
                        "model_execution": {"cuda_graph_enabled": True, "cuda_graph_used": True,
                                            "effective_num_inference_steps": 10},
                        "chunk": [[0.2] * 14 for _ in range(30)],
                        "timing": dict.fromkeys(("preprocess_s", "inference_s", "postprocess_s", "total_s"), 0.0)}
            if not native and self.response_hook:
                self.response_hook(response)
            return response

    value = FakeHttp()
    monkeypatch.setattr("yamkit.remote_policy.modeling_yamkit_remote.make_transport", lambda cfg: value)
    return value


@pytest.fixture
def rollout_config(rig, fake_connect, monkeypatch):
    from tests.test_remote_policy import rollout_config as fake_rollout_config

    config = fake_rollout_config.__wrapped__(rig, fake_connect, monkeypatch)
    config.task = TASK
    config.policy = YamkitRemoteConfig(modal_app="fake-http-app", call_mode="http", execution_mode="cuda_graph10",
                                       task=TASK, image_hw=(8, 8))
    return config


@pytest.mark.parametrize("stopped", [False, True])
def test_reused_graph_pool_warms_actual_task_before_hardware_and_stop_prevents_connect(
        transport, rollout_config, fake_connect, stopped):
    from lerobot.rollout import build_rollout_context

    stop = threading.Event()
    rollout_config.policy._session_shutdown_event = stop
    assert transport.metadata["prediction_count"] > 0
    if stopped:
        transport.warm_hook = stop.set
        with pytest.raises(RemoteFault, match="Stop"):
            build_rollout_context(rollout_config, stop)
        assert not fake_connect and transport.closed
    else:
        context = build_rollout_context(rollout_config, stop)
        try:
            assert fake_connect and transport.ready_count == 2
            policy = context.policy.policy
            assert policy.metadata["graph_warmup"]["signature"]["task"] == TASK
            assert policy.session.graph_warmup["ready"]
            assert not policy._actions and not policy.session.samples
        finally:
            context.hardware.robot_wrapper.inner.disconnect_no_home()
            context.policy.policy.close()
    assert len(transport.requests) == 1
    request = transport.requests[0]
    assert request["mode"] == "native_fixture" and request["task"] == TASK
    assert request["execution_mode"] == "cuda_graph10"
    assert set(request["images"]) == set(get_profile("molmoact2").native_image_keys)
    assert all((image["height"], image["width"], image["encoding"]) == (8, 8, "rgb8")
               for image in request["images"].values())
    assert request.get("diagnostic_cuda_graph") is None and request.get("diagnostic_num_inference_steps") is None


@pytest.mark.parametrize("change", ["task", "shape", "instance", "build", "warm_ready"])
def test_changed_refreshed_readiness_never_connects_hardware(transport, rollout_config, fake_connect, change):
    from lerobot.rollout import build_rollout_context

    def corrupt(metadata):
        if change == "task":
            metadata["graph_warmup"]["signature"]["task"] = "some other task"
        elif change == "shape":
            metadata["graph_warmup"]["signature"]["images"][0]["width"] += 1
        elif change == "instance":
            metadata["instance_id"] = "replacement-container"
        elif change == "build":
            metadata["inference_build_id"] = "old-build"
        else:
            metadata["graph_warmup"]["ready"] = False

    transport.refresh_hook = corrupt
    with pytest.raises(RemoteFault):
        build_rollout_context(rollout_config, threading.Event())
    assert len(transport.requests) == 1 and not fake_connect and transport.closed


def test_changed_rollout_task_is_rejected_before_dispatch(transport, rollout_config):
    policy = YamkitRemotePolicy(rollout_config.policy)
    try:
        inputs = batch()
        inputs["task"] = ["different task with identical camera shapes"]
        with pytest.raises(RemoteFault, match="task differs"):
            policy.predict_action_chunk(inputs)
        assert len(transport.requests) == 1 and not policy.session.samples and not policy._actions
    finally:
        policy.close()


def test_matching_real_response_can_populate_only_the_real_action_queue(transport, rollout_config):
    policy = YamkitRemotePolicy(rollout_config.policy)
    try:
        inputs = batch()
        inputs["task"] = [TASK]
        action = policy.select_action(inputs)
        assert tuple(action.shape) == (1, 14) and action[0, 0].item() == pytest.approx(0.2)
        assert len(transport.requests) == 2 and len(policy._actions) == 29
        assert len(policy.session.samples) == 1 and policy.session.samples[0]["task"] == TASK
        assert transport.requests[-1]["mode"] == "robot"
    finally:
        policy.close()


@pytest.mark.parametrize("path,value", [
    (("instance_id",), "replacement-container"), (("execution_mode",), "eager"),
    (("execution_identity", "num_inference_steps"), 5),
    (("graph_warmup", "signature_sha256"), "d" * 64), (("graph_warmup", "cache_key_sha256"), "e" * 64),
    (("graph_warmup", "ready"), False), (("model_execution", "cuda_graph_used"), False),
    (("model_execution", "effective_num_inference_steps"), 5),
])
def test_real_session_rejects_changed_execution_before_queuing_actions(transport, rollout_config, path, value):
    policy = YamkitRemotePolicy(rollout_config.policy)

    def corrupt(response):
        target = response
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    transport.response_hook = corrupt
    try:
        inputs = batch()
        inputs["task"] = [TASK]
        with pytest.raises((RemoteFault, ValueError)):
            policy.select_action(inputs)
        assert len(transport.requests) == 2 and not policy._actions and not policy.session.samples
    finally:
        policy.close()
