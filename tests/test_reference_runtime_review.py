"""Independent deadline/Stop regression checks; fake HTTP only, no hardware or model."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_http_transport import (  # noqa: F401 — fake HTTP factory and bounded thread helpers
    background,
    factory,
    reply,
    retired,
    transport,
)
from tests.test_remote_policy import FakeTransport
from yamkit.inference.client import InvalidatedRequest, RemoteFault, RemoteSession
from yamkit.inference.http_wire import decode_message
from yamkit.inference.profiles import get_profile


def inputs(observation_time=None):
    return {"state": [0.0] * 14,
            "images": {name: {"encoding": "rgb8", "height": 1, "width": 1, "data": b"\0\0\0"}
                       for name in get_profile("molmoact2").image_keys},
            "task": "offline deadline test", "observation_time": time.monotonic()
            if observation_time is None else observation_time}


def response(payload):
    result = {key: payload[key] for key in (
        "protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time")}
    result.update(action_units="robot", instance_id="fake", action_names=list(get_profile("molmoact2").action_names),
                  mode=payload.get("mode", "robot"), execution_mode=payload.get("execution_mode", "eager"),
                  chunk=[[0.0] * 14] * 30,
                  timing=dict.fromkeys(("preprocess_s", "inference_s", "postprocess_s", "total_s"), 0.0))
    return reply(result)


@pytest.mark.parametrize("stop", [False, True])
def test_blocked_rpc_ends_at_absolute_deadline_or_operator_stop(factory, stop):  # noqa: F811
    entered, release, shutdown = threading.Event(), threading.Event(), threading.Event()
    requests = []

    def handler(request):
        payload = decode_message(request.content)["payload"]
        requests.append(payload)
        entered.set()
        assert release.wait(2)
        return response(payload)

    factory(handler)
    value = transport(shutdown_event=shutdown)
    session = RemoteSession(value, get_profile("molmoact2"), timeout_s=10)
    deadline = time.monotonic() + (1 if stop else .15)
    result, done, thread = background(lambda: session.predict(**inputs(), deadline_monotonic_s=deadline))
    try:
        assert entered.wait(1)
        if stop:
            shutdown.set()
        assert done.wait(.5), "RPC ignored the phase deadline or Stop and kept its ten-second timeout"
        assert isinstance(result.get("error"), InvalidatedRequest if stop else RemoteFault)
        assert not session.samples
        assert len(requests) == 1
        # Even a subsequently arriving valid response cannot become executable evidence.
        release.set()
        thread.join(1)
        retired(value)
        assert not session.samples and "value" not in result
    finally:
        release.set()
        thread.join(1)
        retired(value)
        value.close()


def test_absolute_deadline_budget_includes_request_validation(monkeypatch):
    from yamkit.inference import client, protocol

    clock = [100.0]
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = protocol.validate_request

    def slow_validation(*args, **kwargs):
        original(*args, **kwargs)
        clock[0] += .06

    monkeypatch.setattr(protocol, "validate_request", slow_validation)
    value = FakeTransport()
    budgets = []
    original_predict = value.predict_chunk

    def predict(request, timeout_s):
        budgets.append(timeout_s)
        return original_predict(request, timeout_s)

    value.predict_chunk = predict
    session = RemoteSession(value, get_profile("molmoact2"), timeout_s=10)
    session.predict(**inputs(observation_time=100.0), deadline_monotonic_s=100.1)
    assert budgets == pytest.approx([.04])
    assert session.timeout_s == 10  # This request's short deadline does not mutate the session configuration.


def test_already_elapsed_phase_deadline_never_invokes_transport(monkeypatch):
    from yamkit.inference import client

    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: 100.0))
    value = FakeTransport()
    session = RemoteSession(value, get_profile("molmoact2"), timeout_s=10)
    with pytest.raises(RemoteFault):
        session.predict(**inputs(observation_time=100.0), deadline_monotonic_s=100.0)
    assert not value.requests and not session.samples


def test_response_validation_cannot_extend_absolute_deadline(monkeypatch):
    from yamkit.inference import client, protocol

    clock = [100.0]
    monkeypatch.setattr(client, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    original = protocol.validate_response

    def delayed_validation(*args, **kwargs):
        original(*args, **kwargs)
        clock[0] = 100.2

    monkeypatch.setattr(protocol, "validate_response", delayed_validation)
    value = FakeTransport()
    session = RemoteSession(value, get_profile("molmoact2"), timeout_s=10)
    with pytest.raises(RemoteFault, match="expired|deadline"):
        session.predict(**inputs(observation_time=100.0), deadline_monotonic_s=100.1)
    assert len(value.requests) == 1 and not session.samples


@pytest.mark.parametrize("duration", [3, 10])
def test_next_chunk_uses_committed_state_or_stops_at_short_phase_boundary(monkeypatch, duration):
    from yamkit import reference_rollout
    from yamkit.inference.command_shaping import ACTION_NAMES, JOINT_NAMES
    from yamkit.inference.reference import ReferenceCommandGuard

    clock = [100.0]
    monkeypatch.setattr(reference_rollout, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    initial = dict.fromkeys(ACTION_NAMES, 0.0)
    limits = {name: {"lower": -2.0, "upper": 2.0, "max_step": .1} for name in JOINT_NAMES}
    grips = {name: .1 for name in ACTION_NAMES if "gripper" in name}
    guard = ReferenceCommandGuard(initial, limits, grips)
    measured_reads, batches = [], []

    def measured_state():
        measured_reads.append(True)
        return dict(initial)

    def predict(batch):
        batches.append(batch["observation.state"].clone())
        return torch.full((1, 30, 14), .02)

    policy = SimpleNamespace(transport=SimpleNamespace(), config=SimpleNamespace(
        request_timeout_s=10, max_observation_age_s=2), _last_prediction_timing={},
        predict_action_chunk=predict, reset=lambda: None, close=lambda: None)
    robot = SimpleNamespace(command_shaper=guard, robot_type="bi_yam_follower",
                            invalidate_shaping=guard.invalidate,
                            inner=SimpleNamespace(get_joint_state=measured_state,
                                                  validate_action_target=lambda target: None,
                                                  joint_command_limits=lambda: limits))
    engine = reference_rollout.ReferenceRemoteInferenceEngine(
        policy=policy, preprocessor=lambda value: value, postprocessor=lambda value: value,
        robot_wrapper=robot, task="put the red cube into the black container", fps=30,
        shutdown_event=threading.Event(), duration=duration, gripper_max_step=grips)
    engine.start()
    engine.resume()
    observation = {"observation.state": np.zeros(14, dtype=np.float32),
                   **{f"observation.images.{name}": np.zeros((2, 2, 3), dtype=np.uint8)
                      for name in get_profile("molmoact2").image_keys}}
    deadline = None
    for _ in range(200):
        clock[0] += 1 / 30
        engine.notify_observation(observation)
        action = engine.get_action(observation)
        if deadline is None:
            deadline = engine._plan_deadline
        assert engine._plan_deadline == deadline  # Consumption never extends the lease.
        assert len(batches) == 1
        assert engine.completed_chunks == 0
        step = guard.prepare(dict(zip(ACTION_NAMES, action.tolist(), strict=True)), now=clock[0])
        guard.commit(step, step.shaped, deadline_monotonic_s=engine.action_deadline)
        engine.record_execution()
        engine.record_commit()
        if engine.completed_chunks:
            break
    assert engine.completed_chunks == 1 and engine.completed_steps == 30
    assert torch.equal(batches[0], torch.zeros((1, 14)))
    observation["observation.state"] = np.full(14, .5, dtype=np.float32)
    clock[0] += 1 / 30
    engine.notify_observation(observation)
    action = engine.get_action(observation)
    if duration == 3:
        # A full chunk finished, but no full request freshness budget remains.
        # Monitoring can continue until the unchanged phase ends, without another
        # RPC or command and without converting normal completion into a fault.
        assert 0 < engine.phase_deadline - clock[0] < 2
        assert action is None and len(batches) == 1
        dispatched = engine.dequeued_actions
        clock[0] = engine.phase_deadline
        engine.notify_observation(observation)
        assert engine.get_action(observation) is None
        assert engine.dequeued_actions == dispatched
        assert not engine.failed and guard.valid and not engine._shutdown_event.is_set()
        return
    assert len(batches) == 2
    assert torch.allclose(batches[1], torch.full((1, 14), .02))
    assert np.array_equal(observation["observation.state"], np.full(14, .5, dtype=np.float32))
    assert measured_reads == []  # ReferenceStrategy captures the initial raw seed before the RPC.
