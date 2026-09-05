"""Actual pinned LeRobot factories/context/strategy; only transport and hardware are fake."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yamkit.inference.client import InvalidatedRequest, RemoteFault, RemoteSession
from yamkit.inference.profiles import get_profile
from yamkit.inference.protocol import encode_image
from yamkit.remote_policy import YamkitRemoteConfig


class FakeTransport:
    def __init__(self):
        self.requests = []
        self.error = None
        self.hook = None
        self.ready_hook = None

    def ready(self, timeout_s):
        if self.ready_hook:
            self.ready_hook()
        return {**get_profile("molmoact2").metadata(), "ready": True, "instance_id": "fake-instance",
                "fresh_chunk": True, "saved_processors": True, "image_encoding": "rgb8"}

    def cancel(self):
        pass

    def predict_chunk(self, request, timeout_s):
        self.requests.append(request)
        if self.hook:
            self.hook()
        if self.error:
            raise self.error
        keys = ("protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time")
        return {**{k: request[k] for k in keys}, "action_units": "robot", "instance_id": "fake-instance",
                "action_names": list(get_profile("molmoact2").action_names),
                "chunk": [[0.2] * 14 for _ in range(30)],
                "timing": dict.fromkeys(("preprocess_s", "inference_s", "postprocess_s", "total_s"), 0.0)}


@pytest.fixture
def transport(monkeypatch):
    from yamkit.remote_policy import modeling_yamkit_remote

    value = FakeTransport()
    monkeypatch.setattr(modeling_yamkit_remote, "make_transport", lambda cfg: value)
    return value


def batch():
    return {"observation.state": torch.arange(14, dtype=torch.float32).unsqueeze(0), "task": ["pick cube"],
            **{f"observation.images.{name}": torch.full((1, 3, 8, 8), 0.25)
               for name in get_profile("molmoact2").image_keys}}


@pytest.fixture
def rollout_config(rig, fake_connect, monkeypatch):
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.rollout.configs import RolloutConfig
    from lerobot_robot_yamkit import BiYamFollowerConfig, yam_follower

    class Camera:
        is_connected = False

        def connect(self):
            self.is_connected = True

        def disconnect(self):
            self.is_connected = False

        def read_latest(self):
            return np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(yam_follower, "make_cameras_from_configs", lambda configs: {k: Camera() for k in configs})
    rig.control.home_speed = 0
    for spec in rig.arms.values():
        if spec.has_motor_gripper:
            spec.gripper_limits = [0.0, 6.5]
    rig.save()
    cameras = {k: OpenCVCameraConfig(index_or_path=0, width=8, height=8, fps=30)
               for k in get_profile("molmoact2").image_keys}
    return RolloutConfig(
        robot=BiYamFollowerConfig(rig=str(rig.path), cameras=cameras),
        policy=YamkitRemoteConfig(modal_app="fake-app"), device="cpu", task="pick cube", duration=0.15,
        return_to_initial_position=False,
    )


def test_real_policy_factory_registration_processors_and_fresh_chunks(transport, monkeypatch):
    from lerobot.policies.factory import (
        get_policy_class,
        make_policy,
        make_policy_config,
        make_pre_post_processors,
    )
    from lerobot.rollout.inference.rtc import supports_rtc_inference

    def forbidden(*args, **kwargs):
        raise AssertionError("Weights/download/compile must never be used for an RPC proxy")

    monkeypatch.setattr(torch, "compile", forbidden)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", forbidden)
    config = make_policy_config("yamkit_remote", modal_app="fake-app")
    features = {"action": {"dtype": "float32", "shape": (14,), "names": list(get_profile("molmoact2").action_names)},
                "observation.state": {"dtype": "float32", "shape": (14,), "names": list(get_profile("molmoact2").state_names)},
                **{f"observation.images.{k}": {"dtype": "video", "shape": (8, 8, 3),
                                              "names": ["height", "width", "channels"]}
                   for k in get_profile("molmoact2").image_keys}}
    policy = make_policy(config, ds_meta=SimpleNamespace(features=features, stats={}))
    assert type(policy) is get_policy_class("yamkit_remote")
    assert list(policy.parameters()) == []
    assert not supports_rtc_inference(policy)
    pre, post = make_pre_post_processors(config, dataset_stats={"action": {"mean": torch.ones(14) * 100}})
    for _ in range(3):
        observation = pre(batch())
        assert torch.equal(observation["observation.state"], batch()["observation.state"])
        result = post(policy.predict_action_chunk(observation))
        assert result.shape == (1, 30, 14) and torch.isfinite(result).all()
        assert result[0, 0, 0].item() == pytest.approx(0.2)
    assert len(transport.requests) == 3
    policy.reset()
    policy.select_action(pre(batch()))
    policy.select_action(pre(batch()))
    assert len(transport.requests) == 4
    assert transport.requests[-1]["session_id"] != transport.requests[0]["session_id"]


def test_real_rollout_context_readiness_before_connect(transport, rollout_config, fake_connect):
    from lerobot.rollout import build_rollout_context
    from lerobot.rollout.inference import SyncInferenceEngine

    transport.ready_hook = lambda: pytest.fail("readiness was after activation") if fake_connect else None
    ctx = build_rollout_context(rollout_config, threading.Event())
    try:
        assert fake_connect
        assert isinstance(ctx.policy.inference, SyncInferenceEngine)
        result = ctx.policy.inference.get_action({"observation.state": np.zeros(14),
                                                 **{f"observation.images.{k}": np.zeros((8, 8, 3), dtype=np.uint8)
                                                    for k in get_profile("molmoact2").image_keys}})
        assert result.shape == (14,)
    finally:
        ctx.hardware.robot_wrapper.inner.disconnect_no_home()


def test_remote_rollout_runs_actual_strategy_and_no_home(transport, rollout_config, fake_connect, monkeypatch):
    from yamkit.arm import YamArm
    from yamkit.remote_rollout import run_remote_rollout

    monkeypatch.setattr(YamArm, "go_home", lambda *a, **kw: pytest.fail("Remote cleanup attempted homing"))
    result = run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert result["inference"] == "unguided_async"
    assert result["sample_count"] >= 1
    assert result["peak_queue_depth"] > 0
    assert result["last_queue_depth_before_stop"] > 0
    assert all(r.closed and r.commands for r in fake_connect.values())


def test_fault_releases_without_actions_or_homing(transport, rollout_config, fake_connect, monkeypatch):
    from yamkit.arm import YamArm
    from yamkit.remote_rollout import run_remote_rollout

    monkeypatch.setattr(YamArm, "go_home", lambda *a, **kw: pytest.fail("Fault attempted homing"))
    transport.error = RemoteFault("timeout")
    with pytest.raises(RemoteFault) as failure:
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert failure.value.metrics["failed"]
    assert failure.value.metrics["sample_count"] == 0
    assert all(r.closed and not r.commands for r in fake_connect.values())


def test_readiness_failure_never_connects(transport, rollout_config, fake_connect):
    from yamkit.remote_rollout import run_remote_rollout

    transport.ready_hook = lambda: (_ for _ in ()).throw(RemoteFault("not ready"))
    with pytest.raises(RemoteFault):
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert not fake_connect


def test_stop_during_readiness_never_connects(transport, rollout_config, fake_connect):
    from yamkit.remote_rollout import run_remote_rollout

    stop = threading.Event()
    transport.ready_hook = stop.set
    with pytest.raises(RemoteFault, match="Stop"):
        run_remote_rollout(rollout_config, shutdown_event=stop)
    assert not fake_connect


def test_stop_event_interrupts_existing_startup_home(transport, rollout_config, fake_connect, monkeypatch):
    from yamkit.arm import YamArm
    from yamkit.config import RigConfig
    from yamkit.remote_rollout import run_remote_rollout

    rig = RigConfig.load(rollout_config.robot.rig)
    rig.control.home_speed = 1
    rig.save()
    stop = threading.Event()
    original = YamArm.go_home

    def stopped_home(self, *args, **kwargs):
        assert kwargs["stop"] is stop
        stop.set()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(YamArm, "go_home", stopped_home)
    with pytest.raises(RuntimeError, match="stopped"):
        run_remote_rollout(rollout_config, shutdown_event=stop)
    assert all(r.closed and not r.commands for r in fake_connect.values())


def test_context_build_failure_releases_partially_constructed_robot(transport, rollout_config, fake_connect, monkeypatch):
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    from yamkit.remote_rollout import run_remote_rollout

    monkeypatch.setattr(BiYamFollower, "get_observation", lambda self: (_ for _ in ()).throw(ValueError("camera")))
    with pytest.raises(ValueError, match="camera"):
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert all(r.closed and not r.commands for r in fake_connect.values())


@pytest.mark.parametrize("field,value", [("use_torch_compile", True), ("interpolation_multiplier", 2),
                                         ("return_to_initial_position", True), ("fps", 10)])
def test_unsupported_rollout_rejected_before_activation(transport, rollout_config, fake_connect, field, value):
    from yamkit.remote_rollout import run_remote_rollout

    setattr(rollout_config, field, value)
    with pytest.raises(ValueError):
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert not fake_connect and not transport.requests


def test_swapped_rig_sides_rejected_before_activation(transport, rollout_config, fake_connect):
    from yamkit.remote_rollout import run_remote_rollout

    rollout_config.robot.left, rollout_config.robot.right = rollout_config.robot.right, rollout_config.robot.left
    with pytest.raises(ValueError, match="side"):
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert not fake_connect


def session_inputs():
    return {"state": [0.0] * 14, "images": {k: encode_image(np.zeros((8, 8, 3), dtype=np.uint8))
                                            for k in get_profile("molmoact2").image_keys},
            "task": "pick cube", "observation_time": time.monotonic()}


def test_inflight_reset_rejects_late_reply(transport):
    session = RemoteSession(transport, get_profile("molmoact2"))
    transport.hook = session.reset
    with pytest.raises(InvalidatedRequest):
        session.predict(**session_inputs())
    assert not session.samples


def test_deadline_and_old_observation(transport):
    session = RemoteSession(transport, get_profile("molmoact2"), timeout_s=0.01)
    transport.hook = lambda: time.sleep(0.02)
    with pytest.raises(RemoteFault, match="expired"):
        session.predict(**session_inputs())
    inputs = session_inputs()
    inputs["observation_time"] -= 10
    with pytest.raises(RemoteFault, match="stale"):
        session.predict(**inputs)


def test_container_restart_requires_new_preparation(transport):
    session = RemoteSession(transport, get_profile("molmoact2"))
    session.instance_id = "previous-instance"
    with pytest.raises(RemoteFault, match="restarted"):
        session.predict(**session_inputs())
    assert not session.samples


def test_invalidated_queue_never_accepts_old_merge():
    from yamkit.remote_rollout import InvalidatableActionQueue

    queue = InvalidatableActionQueue(max_steps=3, max_age_s=1)
    chunk = torch.ones((2, 14))
    queue.merge(chunk, chunk, 0)
    with pytest.raises(RemoteFault, match="capacity"):
        queue.merge(chunk, chunk, 0)
    queue.invalidate()
    queue.merge(chunk, chunk, 0)
    assert queue.get() is None and queue.qsize() == 0


def test_expired_queue_rejected():
    from yamkit.remote_rollout import InvalidatableActionQueue

    queue = InvalidatableActionQueue(max_steps=3, max_age_s=1, observation_time=lambda: time.monotonic() - 2)
    chunk = torch.ones((2, 14))
    queue.merge(chunk, chunk, 0)
    with pytest.raises(RemoteFault, match="expired"):
        queue.get()


def test_underrun_stops_instead_of_replaying(transport):
    from lerobot.policies.factory import make_pre_post_processors

    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy
    from yamkit.remote_rollout import UnguidedRemoteInferenceEngine

    policy = YamkitRemotePolicy(YamkitRemoteConfig(modal_app="fake-app"))
    pre, post = make_pre_post_processors(policy.config)
    stop = threading.Event()
    engine = UnguidedRemoteInferenceEngine(policy=policy, preprocessor=pre, postprocessor=post,
                                          robot_wrapper=SimpleNamespace(), hw_features={}, task="task", fps=30,
                                          shutdown_event=stop)
    engine._action_queue = engine._new_queue()
    chunk = torch.ones((1, 14))
    engine.action_queue.merge(chunk, chunk, 0)
    assert engine.get_action(None).shape == (14,)
    with pytest.raises(RemoteFault, match="underrun"):
        engine.get_action(None)
    assert engine.underruns == 1 and engine.failed and stop.is_set()
    assert policy.session._closed


def test_stop_releases_before_rpc_completes_and_rejects_late_actions(transport, rollout_config, fake_connect):
    from yamkit.remote_rollout import run_remote_rollout

    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    errors = []
    rollout_config.duration = 5

    def wait_response():
        entered.set()
        assert release.wait(3)

    transport.hook = wait_response

    def run():
        try:
            run_remote_rollout(rollout_config, shutdown_event=stop)
        except RemoteFault as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert entered.wait(2)
        stop.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not all(r.closed for r in fake_connect.values()):
            time.sleep(0.01)
        assert all(r.closed and not r.commands for r in fake_connect.values())
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert all(not r.commands for r in fake_connect.values())


@pytest.mark.parametrize("key,value", [("sequence_id", 99), ("session_id", "wrong"),
                                      ("action_units", "normalized"), ("chunk", [[float("nan")] * 14])])
def test_bad_response_rejected(transport, key, value):
    original = transport.predict_chunk

    def bad_reply(request, timeout_s):
        reply = original(request, timeout_s)
        reply[key] = value
        return reply

    transport.predict_chunk = bad_reply
    session = RemoteSession(transport, get_profile("molmoact2"))
    with pytest.raises(ValueError):
        session.predict(**session_inputs())
    assert not session.samples


def test_upstream_local_sync_strategy_with_fake_robot_is_preserved(rollout_config, fake_connect, monkeypatch):
    """Exercise the unmodified local context/strategy with a tiny genuine local ACT."""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.rollout import build_rollout_context, create_strategy

    local = ACTConfig(device="cpu", input_features={
        "observation.state": rollout_config.policy.input_features["observation.state"],
        "observation.images.top": rollout_config.policy.input_features["observation.images.top"]},
                      output_features=rollout_config.policy.output_features, chunk_size=3, n_action_steps=3,
                      dim_model=32, n_heads=4, dim_feedforward=64, n_encoder_layers=1, n_decoder_layers=1,
                      use_vae=False, pretrained_backbone_weights=None)
    policy = ACTPolicy(local)
    pre, post = make_pre_post_processors(local, dataset_stats={"observation.state": {
        "mean": torch.zeros(14), "std": torch.ones(14)}, "action": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "observation.images.top": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)}})
    # Test seams replace heavy checkpoint disk I/O, not the context or loop.
    monkeypatch.setattr("lerobot.rollout.context._load_pretrained_policy", lambda cfg: policy)
    monkeypatch.setattr("lerobot.rollout.context.make_pre_post_processors", lambda **kw: (pre, post))
    rollout_config.policy = local
    ctx = build_rollout_context(rollout_config, threading.Event())
    strategy = create_strategy(rollout_config.strategy)
    try:
        strategy.setup(ctx)
        strategy.run(ctx)
    finally:
        strategy.teardown(ctx)
    assert all(r.commands and r.closed for r in fake_connect.values())
