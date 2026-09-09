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
                "fresh_chunk": True, "saved_processors": True, "image_encoding": "jpeg",
                "supported_image_encodings": ["jpeg", "rgb8"], "prediction_count": 1}

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
    monkeypatch.setattr("yamkit.inference.performance.require_physical_modal_rollout", lambda *a, **k: None)
    # Factory-only tests replace the entire transport and never activate hardware.
    monkeypatch.setattr("yamkit.inference.qualification.require_runner_context", lambda: None)
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

    # This test fixture already substitutes every camera, motor and transport.
    # Production has no switch that bypasses the independent performance gate.
    monkeypatch.setattr("yamkit.inference.performance.require_physical_modal_rollout", lambda *a, **k: None)

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
        policy=YamkitRemoteConfig(modal_app="fake-app", image_hw=(8, 8)), device="cpu", task="pick cube", duration=0.15,
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
    config = make_policy_config("yamkit_remote", modal_app="fake-app", image_hw=(8, 8))
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


def test_proxy_image_conversion_preserves_exact_pixels_and_per_camera_timings(transport):
    from yamkit.inference.protocol import decode_image
    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy

    inputs = batch()
    pixels = np.random.default_rng(741).random((17, 19, 3), dtype=np.float32)
    pixels[0, 0] = [0.0, 1.0, 0.5]
    # Noncontiguous CHW views are what the real upstream observation path supplies.
    image = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    assert not image.is_contiguous()
    for name in get_profile("molmoact2").image_keys:
        inputs[f"observation.images.{name}"] = image
    policy = YamkitRemotePolicy(YamkitRemoteConfig(modal_app="fake-app", image_encoding="rgb8", image_hw=(17, 19)))
    policy.predict_action_chunk(inputs)
    expected = (pixels * 255).round().astype(np.uint8)
    sample = policy.session.samples[-1]
    for name, encoded in transport.requests[-1]["images"].items():
        assert np.array_equal(decode_image(encoded), expected)
        timing = sample["per_camera_timing"][name]
        assert timing["payload_bytes"] == expected.nbytes
        assert timing["image_encode_s"] >= 0 and timing["tensor_transform_s"] >= 0
        assert timing["jpeg_encode_s"] == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.001, 1.001])
def test_proxy_image_validation_rejects_invalid_pixels_before_request(transport, invalid):
    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy

    policy = YamkitRemotePolicy(YamkitRemoteConfig(modal_app="fake-app", image_hw=(8, 8)))
    inputs = batch()
    name = get_profile("molmoact2").image_keys[0]
    inputs[f"observation.images.{name}"][0, 0, 0, 0] = invalid
    with pytest.raises(RemoteFault, match="finite RGB"):
        policy.predict_action_chunk(inputs)
    assert transport.requests == []


def test_proxy_rejects_different_image_resolution_before_request(transport):
    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy

    policy = YamkitRemotePolicy(YamkitRemoteConfig(modal_app="fake-app", image_hw=(480, 640)))
    with pytest.raises(RemoteFault, match="dimensions differ"):
        policy.predict_action_chunk(batch())
    assert not transport.requests


@pytest.mark.parametrize("count", [None, True, -1, float("nan"), float("inf"), 1.0])
def test_readiness_rejects_missing_or_malformed_prediction_count(transport, count):
    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy

    original = transport.ready
    transport.ready = lambda timeout: {**original(timeout), "prediction_count": count}
    with pytest.raises(RemoteFault, match="prediction count"):
        YamkitRemotePolicy(YamkitRemoteConfig(modal_app="fake-app"))
    assert not transport.requests


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


@pytest.mark.parametrize("stop_during_warmup", [False, True])
def test_cold_model_warmup_is_native_and_precedes_hardware_activation(
        transport, rollout_config, fake_connect, stop_during_warmup):
    from lerobot.rollout import build_rollout_context

    stop = threading.Event()
    rollout_config.policy._session_shutdown_event = stop
    original_ready, original_predict = transport.ready, transport.predict_chunk
    transport.ready = lambda timeout: {**original_ready(timeout), "prediction_count": 0}

    def warm(request, timeout):
        assert not fake_connect, "First-forward warmup must finish before activation"
        assert request["mode"] == "native_fixture"
        assert all((image["height"], image["width"]) == (8, 8) for image in request["images"].values())
        if stop_during_warmup:
            stop.set()
        return {**original_predict(request, timeout), "action_units": "checkpoint_native"}

    transport.predict_chunk = warm
    if stop_during_warmup:
        with pytest.raises(RemoteFault, match="Stop"):
            build_rollout_context(rollout_config, stop)
        assert not fake_connect
    else:
        ctx = build_rollout_context(rollout_config, stop)
        try:
            assert len(transport.requests) == 1
            assert not ctx.policy.policy._actions
            assert not ctx.policy.policy.session.samples
            assert ctx.policy.policy.warmup_s > 0
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
    assert result["duration_completed"] and not result["home_attempted"]
    assert all(r.closed and r.commands for r in fake_connect.values())


def _enable_rollout_home(rollout_config):
    from yamkit.config import RigConfig

    rig = RigConfig.load(rollout_config.robot.rig)
    rig.control.home_speed = 0.25
    rig.save()


def test_normal_duration_cancels_policy_then_homes_concurrently_before_camera_release(
        transport, rollout_config, fake_connect, monkeypatch, caplog):
    from yamkit.arm import YamArm
    from yamkit.remote_rollout import run_remote_rollout

    _enable_rollout_home(rollout_config)
    calls = {}
    both_homing = threading.Barrier(2)
    original = YamArm.go_home

    def home(self, *args, **kwargs):
        calls[self.name] = calls.get(self.name, 0) + 1
        if calls[self.name] == 2:
            robot = rollout_config.robot._runtime_robot
            assert all(camera.is_connected for camera in robot.cameras.values())
            assert all(not fake.closed for fake in fake_connect.values())
            assert rollout_config.policy._session_shutdown_event.is_set() is False
            assert kwargs["speed"] == 0.25
            both_homing.wait(timeout=2)
        return original(self, *args, **kwargs)

    from yamkit import remote_rollout
    original_invalidate = remote_rollout.UnguidedRemoteInferenceEngine.invalidate
    cancelled = threading.Event()

    def invalidate(engine):
        original_invalidate(engine)
        assert not engine.action_queue.valid and engine.action_queue.qsize() == 0
        assert engine._policy.session._closed
        cancelled.set()

    def checked_home(self, *args, **kwargs):
        if calls.get(self.name) == 1:
            assert cancelled.is_set()
        return home(self, *args, **kwargs)

    monkeypatch.setattr(YamArm, "go_home", checked_home)
    monkeypatch.setattr(remote_rollout.UnguidedRemoteInferenceEngine, "invalidate", invalidate)
    with caplog.at_level("INFO", logger="yamkit.remote_rollout"):
        result = run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert calls == {"left_follower": 2, "right_follower": 2}
    assert result["duration_completed"] and result["home_attempted"] and result["home_completed"]
    assert not result["home_aborted"] and not result["failed"]
    assert result["stop_to_robot_release_s"] is None
    assert result["fault_stop_to_robot_release_s"] is None
    assert result["policy_stop_to_home_s"] >= 0
    assert result["policy_stop_to_robot_release_s"] >= result["home_duration_s"] > 0
    assert [record.message for record in caplog.records if "[yamkit-rollout]" in record.message] == [
        "[yamkit-rollout] running", "[yamkit-rollout] returning_home",
        "[yamkit-rollout] releasing", "[yamkit-rollout] released"]
    assert all(fake.closed for fake in fake_connect.values())
    for fake in fake_connect.values():
        np.testing.assert_allclose(fake.pos[:6], 0, atol=1e-9)
        assert fake.pos[6] > 0, "Home must preserve the current gripper target"


@pytest.mark.parametrize("abort", ["operator_stop", "session_expired", "home_timeout"])
def test_return_home_abort_releases_both_without_retry(
        transport, rollout_config, fake_connect, monkeypatch, abort):
    from yamkit import remote_rollout
    from yamkit.arm import YamArm

    _enable_rollout_home(rollout_config)
    stop = threading.Event()
    entered = threading.Barrier(2)
    calls = {}
    original = YamArm.go_home
    if abort == "home_timeout":
        monkeypatch.setattr(remote_rollout, "RETURN_HOME_TIMEOUT_S", 0.05)
    if abort == "session_expired":
        # Keep the fake transport active through startup and the policy phase.
        transport.http_session_expires_at = time.time() + 30
        original_return = remote_rollout._return_home_after_duration

        def soon_expiring(*args, **kwargs):
            transport.http_session_expires_at = time.time() + 0.05
            return original_return(*args, **kwargs)

        monkeypatch.setattr(remote_rollout, "_return_home_after_duration", soon_expiring)

    def home(self, *args, **kwargs):
        calls[self.name] = calls.get(self.name, 0) + 1
        if calls[self.name] == 1:
            return original(self, *args, **kwargs)
        entered.wait(timeout=2)
        if abort == "operator_stop":
            stop.set()
        assert kwargs["stop"].wait(1), "Every home worker must see Stop or the finite deadline"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(YamArm, "go_home", home)
    if abort == "operator_stop":
        result = remote_rollout.run_remote_rollout(rollout_config, shutdown_event=stop)
    else:
        with pytest.raises(RemoteFault, match=abort) as failure:
            remote_rollout.run_remote_rollout(rollout_config, shutdown_event=stop)
        result = failure.value.metrics
        assert result["failed"]
    assert calls == {"left_follower": 2, "right_follower": 2}
    assert result["home_attempted"] and result["home_aborted"] and not result["home_completed"]
    assert result["home_abort_reason"] == abort
    assert 0 <= result["fault_stop_to_robot_release_s"] < 1
    assert all(fake.closed for fake in fake_connect.values())


@pytest.mark.parametrize("error", [RuntimeError("home failed"), KeyboardInterrupt(), SystemExit(1)])
def test_home_error_or_second_interrupt_still_releases_without_retry(
        transport, rollout_config, fake_connect, monkeypatch, error):
    from yamkit import arm
    from yamkit.remote_rollout import run_remote_rollout

    _enable_rollout_home(rollout_config)
    calls = []

    def interrupted(jobs, *, stop):
        calls.append(jobs)
        raise error

    # The plugin retains its original startup helper; only return-home is interrupted.
    monkeypatch.setattr(arm, "go_home_all", interrupted)
    with pytest.raises(RemoteFault if isinstance(error, RuntimeError) else type(error)) as failure:
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert len(calls) == 1
    assert failure.value.metrics["home_aborted"] and not failure.value.metrics["home_completed"]
    assert all(fake.closed for fake in fake_connect.values())


def test_duration_without_dispatched_actions_never_initiates_return_home(
        transport, rollout_config, fake_connect, monkeypatch):
    from yamkit import arm
    from yamkit.remote_rollout import run_remote_rollout

    _enable_rollout_home(rollout_config)
    transport.hook = lambda: time.sleep(0.2)
    monkeypatch.setattr(arm, "go_home_all", lambda *a, **kw: pytest.fail("No successful policy phase to return from"))
    result = run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert result["executed_actions"] == 0
    assert not result["duration_completed"] and not result["home_attempted"]
    assert all(fake.closed for fake in fake_connect.values())


def test_late_prediction_during_return_home_cannot_restore_policy_actions(
        transport, rollout_config, fake_connect, monkeypatch):
    from yamkit import remote_rollout
    from yamkit.arm import YamArm

    _enable_rollout_home(rollout_config)
    pending, allow_reply = threading.Event(), threading.Event()
    original_home = YamArm.go_home
    original_return = remote_rollout._return_home_after_duration
    calls, captured = {}, {}

    def predict():
        if len(transport.requests) == 2:
            pending.set()
            assert allow_reply.wait(3)

    def returning(robot, engine, stop):
        captured["engine"] = engine
        captured["executed_actions"] = engine.executed_actions
        return original_return(robot, engine, stop)

    def home(self, *args, **kwargs):
        calls[self.name] = calls.get(self.name, 0) + 1
        if calls[self.name] == 2:
            assert pending.is_set() and not captured["engine"].action_queue.valid
            allow_reply.set()
        return original_home(self, *args, **kwargs)

    transport.hook = predict
    monkeypatch.setattr(YamArm, "go_home", home)
    monkeypatch.setattr(remote_rollout, "_return_home_after_duration", returning)
    try:
        result = remote_rollout.run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    finally:
        allow_reply.set()
    assert result["home_completed"] and not result["failed"]
    assert result["executed_actions"] == captured["executed_actions"] > 0
    assert result["queue_depth"] == 0 and result["failed_request_count"] == 1
    assert result["prediction_samples"][-1]["error"] == "invalidated"
    assert all(fake.closed for fake in fake_connect.values())


def test_expiry_at_duration_boundary_prevents_home(transport, rollout_config, fake_connect, monkeypatch):
    from lerobot.rollout.strategies.base import BaseStrategy

    from yamkit import arm
    from yamkit.remote_rollout import run_remote_rollout

    _enable_rollout_home(rollout_config)
    ended = threading.Event()
    original_run = BaseStrategy.run

    def run(strategy, ctx):
        original_run(strategy, ctx)
        ended.set()

    def check():
        if ended.is_set():
            raise RemoteFault("session expired at duration boundary")

    transport.ensure_session_active = check
    monkeypatch.setattr(BaseStrategy, "run", run)
    monkeypatch.setattr(arm, "go_home_all", lambda *a, **kw: pytest.fail("Expired session attempted return"))
    with pytest.raises(RemoteFault, match="expired") as failure:
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert not failure.value.metrics["home_attempted"]
    assert all(fake.closed for fake in fake_connect.values())


@pytest.mark.parametrize("reason", ["operator_stop", "session_expired", "remote_fault"])
def test_stop_or_expiry_racing_with_policy_cancellation_cannot_start_home(
        transport, rollout_config, fake_connect, monkeypatch, reason):
    from yamkit import arm, remote_rollout

    _enable_rollout_home(rollout_config)
    stop = threading.Event()
    original_invalidate = remote_rollout.UnguidedRemoteInferenceEngine.invalidate

    def raced_invalidate(engine):
        original_invalidate(engine)
        if reason == "session_expired":
            # The guard already captured this deadline before transport closure.
            time.sleep(0.02)
        elif reason == "remote_fault":
            engine._rtc_error.set()
            stop.set()
        else:
            stop.set()

    if reason == "session_expired":
        original_return = remote_rollout._return_home_after_duration

        def soon_expiring(*args, **kwargs):
            transport.http_session_expires_at = time.time() + 0.01
            return original_return(*args, **kwargs)

        monkeypatch.setattr(remote_rollout, "_return_home_after_duration", soon_expiring)
    monkeypatch.setattr(remote_rollout.UnguidedRemoteInferenceEngine, "invalidate", raced_invalidate)
    monkeypatch.setattr(arm, "go_home_all", lambda *a, **kw: pytest.fail("Cancellation race attempted home"))
    if reason == "operator_stop":
        result = remote_rollout.run_remote_rollout(rollout_config, shutdown_event=stop)
    else:
        with pytest.raises(RemoteFault, match=reason) as failure:
            remote_rollout.run_remote_rollout(rollout_config, shutdown_event=stop)
        result = failure.value.metrics
        assert result["failed"]
    assert result["home_aborted"] and not result["home_attempted"] and not result["home_completed"]
    assert result["home_abort_reason"] == reason
    assert all(fake.closed for fake in fake_connect.values())


def test_partial_strategy_setup_failure_keeps_original_error_and_releases(
        transport, rollout_config, fake_connect, monkeypatch):
    from lerobot.rollout.strategies.base import BaseStrategy

    from yamkit.remote_rollout import run_remote_rollout

    def failed_setup(*args, **kwargs):
        raise RuntimeError("partial strategy setup failed")

    monkeypatch.setattr(BaseStrategy, "setup", failed_setup)
    with pytest.raises(RuntimeError, match="partial strategy setup failed") as failure:
        run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert not failure.value.metrics["home_attempted"]
    assert all(fake.closed for fake in fake_connect.values())


def test_policy_invalidation_failure_cannot_skip_robot_release(
        transport, rollout_config, fake_connect, monkeypatch):
    from yamkit import remote_rollout

    def failed_close():
        raise RuntimeError("transport close failed")

    transport.close = failed_close
    with pytest.raises(RuntimeError, match="transport close failed"):
        remote_rollout.run_remote_rollout(rollout_config, shutdown_event=threading.Event())
    assert all(fake.closed for fake in fake_connect.values())


@pytest.mark.parametrize("clock", ["wall", "monotonic"])
def test_home_guard_retains_both_session_expiry_clocks_after_transport_close(monkeypatch, clock):
    from yamkit import remote_rollout

    now = {"wall": 100.0, "monotonic": 10.0}
    monkeypatch.setattr(remote_rollout, "time", SimpleNamespace(
        time=lambda: now["wall"], monotonic=lambda: now["monotonic"]))
    transport = SimpleNamespace(http_session_expires_at=105.0, _session_deadline_monotonic=15.0)
    event = threading.Event()
    stop = remote_rollout._HomeStop(event, transport)
    del transport.http_session_expires_at, transport._session_deadline_monotonic
    assert not stop.is_set()
    now[clock] += 5
    assert stop.is_set() and event.is_set() and stop.reason == "session_expired"


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
        queue.merge(torch.ones((4, 14)), torch.ones((4, 14)), 0)
    queue.invalidate()
    queue.merge(chunk, chunk, 0)
    assert queue.get() is None and queue.qsize() == 0


def test_expired_queue_rejected():
    from yamkit.remote_rollout import InvalidatableActionQueue

    queue = InvalidatableActionQueue(max_steps=3, max_age_s=1, observation_time=lambda: time.monotonic() - 2)
    chunk = torch.ones((2, 14))
    with pytest.raises(RemoteFault, match="expired"):
        queue.merge(chunk, chunk, 0)


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


@pytest.mark.parametrize("threshold", [None, 0, 15, 30])
def test_prefetch_threshold_does_not_enlarge_queue_capacity(transport, threshold):
    from lerobot.policies.factory import make_pre_post_processors

    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy
    from yamkit.remote_rollout import UnguidedRemoteInferenceEngine

    policy = YamkitRemotePolicy(YamkitRemoteConfig(modal_app="fake-app", prediction_queue_threshold=threshold))
    pre, post = make_pre_post_processors(policy.config)
    engine = UnguidedRemoteInferenceEngine(policy=policy, preprocessor=pre, postprocessor=post,
                                          robot_wrapper=SimpleNamespace(), hw_features={}, task="task", fps=30,
                                          shutdown_event=threading.Event())
    engine._action_queue = engine._new_queue()
    assert engine.max_steps == engine.action_queue.max_steps == 45
    oversized = torch.ones((46, 14))
    with pytest.raises(RemoteFault, match="capacity"):
        engine.action_queue.merge(oversized, oversized, 0)
    assert engine.failed and engine.action_queue.qsize() == 0


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


def test_stop_during_overlapping_prediction_releases_and_final_metrics_include_late_reply(
        transport, rollout_config, fake_connect):
    from yamkit.remote_rollout import run_remote_rollout

    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    results = []
    rollout_config.duration = 5

    def wait_second_response():
        if len(transport.requests) == 2:
            entered.set()
            assert release.wait(3)

    transport.hook = wait_second_response
    thread = threading.Thread(target=lambda: results.append(run_remote_rollout(rollout_config, shutdown_event=stop)))
    thread.start()
    try:
        assert entered.wait(2)
        assert all(r.commands for r in fake_connect.values())
        stop.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not all(r.closed for r in fake_connect.values()):
            time.sleep(0.01)
        assert all(r.closed for r in fake_connect.values())
        commands_at_release = [len(r.commands) for r in fake_connect.values()]
    finally:
        stop.set()
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert [len(r.commands) for r in fake_connect.values()] == commands_at_release
    assert results[0]["queue_depth"] == 0
    assert results[0]["last_queue_depth_before_stop"] > 0
    assert results[0]["failed_request_count"] == 1
    assert results[0]["prediction_samples"][-1]["error"] == "invalidated"
    assert 0 <= results[0]["stop_to_robot_release_s"] < 1


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
    # An untrained random head can exceed A's verified raw joint bounds. Keep a
    # genuine ACT forward pass, with deterministic valid robot-unit predictions.
    torch.nn.init.zeros_(policy.model.action_head.weight)
    torch.nn.init.constant_(policy.model.action_head.bias, 0.2)
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
