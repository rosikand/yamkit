"""Action-time deadlines and real upstream overlap, without any paid/hardware resources."""

import threading
from types import SimpleNamespace

import pytest
import torch

from yamkit.inference.client import RemoteFault
from yamkit.remote_rollout import InvalidatableActionQueue, _StoppableRobot


def test_queue_drops_elapsed_and_overlapping_prefix_without_shifting_time(monkeypatch):
    clock = [10.0]
    observation = [10.0]
    monkeypatch.setattr("yamkit.remote_rollout.time", SimpleNamespace(monotonic=lambda: clock[0]))
    events = []
    queue = InvalidatableActionQueue(max_steps=45, max_age_s=2, fps=30,
                                    observation_time=lambda: observation[0], on_merge=events.append)
    actions = torch.arange(30).reshape(30, 1).float()
    clock[0] = 10.1
    queue.merge(actions, actions, 3)
    assert queue.get().item() == 3
    # The first RPC spent three timesteps; no stale 0/1/2 prefix was executed.
    assert queue.expired_prefix_dropped == 3
    for _ in range(11):
        queue.get()
    assert queue.qsize() == 15
    observation[0] = 10.5
    # Three more old commands execute while a new independent chunk is predicted.
    for step in range(3):
        clock[0] = 10.5 + step / 30
        queue.get()
    clock[0] = 10.6
    queue.merge(actions + 100, actions + 100, 3)
    assert events[-1]["queue_depth_at_merge"] == 12
    assert events[-1]["expired_prefix_dropped"] == 3
    assert events[-1]["overlap_prefix_dropped"] == 12
    assert queue.qsize() == 27
    assert queue.queue[-15, 0].item() == 115
    assert queue._deadlines[-15] == pytest.approx(10.5 + 16 / 30)


def test_entire_one_second_chunk_expired_after_recorded_molmo_latency(monkeypatch):
    monkeypatch.setattr("yamkit.remote_rollout.time", SimpleNamespace(monotonic=lambda: 11.48))
    queue = InvalidatableActionQueue(max_steps=45, max_age_s=2, fps=30, observation_time=lambda: 10.0)
    chunk = torch.ones(30, 14)
    with pytest.raises(RemoteFault, match="no valid future actions"):
        queue.merge(chunk, chunk, 45)
    assert queue.expired_prefix_dropped == 30 and queue.expired_chunks == 1
    assert queue.qsize() == 0


def test_action_expiration_checked_again_immediately_before_dispatch(monkeypatch):
    events = []
    wrapper = _StoppableRobot(SimpleNamespace(send_action=lambda action: events.append(action)), threading.Event())
    wrapper.action_deadline = lambda: 10.0
    monkeypatch.setattr("yamkit.remote_rollout.time", SimpleNamespace(monotonic=lambda: 10.01))
    with pytest.raises(RemoteFault, match="before hardware dispatch"):
        wrapper.send_action({"joint_1.pos": 0.2})
    assert events == []


@pytest.mark.parametrize("at_deadline", [True, False])
def test_dequeued_action_keeps_original_deadline_across_background_merge(monkeypatch, at_deadline):
    clock = [10.0]
    observation = [10.0]
    monkeypatch.setattr("yamkit.remote_rollout.time", SimpleNamespace(monotonic=lambda: clock[0]))
    queue = InvalidatableActionQueue(max_steps=45, max_age_s=2, fps=30,
                                    observation_time=lambda: observation[0])
    actions = torch.ones(30, 14) * 0.2
    queue.merge(actions, actions, 0)
    queue.get()
    original_deadline = queue.last_action_deadline
    assert original_deadline == pytest.approx(10 + 1 / 30)
    # A newer chunk arrives while the main thread is processing the popped action.
    observation[0] = clock[0] = 10.01
    queue.merge(actions, actions, 0)
    assert queue.last_action_deadline == original_deadline
    events = []
    wrapper = _StoppableRobot(SimpleNamespace(send_action=events.append), threading.Event())
    wrapper.action_deadline = lambda: queue.last_action_deadline
    clock[0] = original_deadline + (0.0 if at_deadline else 0.01)
    with pytest.raises(RemoteFault, match="before hardware dispatch"):
        wrapper.send_action({"joint_1.pos": 0.2})
    assert events == []


def test_queue_rejects_action_at_exact_deadline(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr("yamkit.remote_rollout.time", SimpleNamespace(monotonic=lambda: clock[0]))
    queue = InvalidatableActionQueue(max_steps=30, max_age_s=2, fps=30, observation_time=lambda: 10.0)
    actions = torch.ones(30, 14) * 0.2
    queue.merge(actions, actions, 0)
    clock[0] = queue._deadlines[0]
    with pytest.raises(RemoteFault, match="Queued remote actions expired"):
        queue.get()


def test_actual_upstream_prediction_overlaps_canonical_fake_robot_execution():
    from scripts.benchmark_remote import run_scenario

    result = run_scenario("test_overlap", [0.06, 0.09, 0.07], duration=1.8, image_hw=(8, 8))
    assert result["all_fake_robots_released"]
    assert result["error"] is None and result["underruns"] == 0
    assert result["sample_count"] >= 3
    assert result["expired_prefix_dropped"] > 0
    assert any(count > 0 for count in result["sdk_commands_during_completed_rpc"][1:])
    assert any(event["actions_executed_during_prediction"] > 0 for event in result["prediction_samples"])
    sample = result["samples"][0]
    assert sample["image_encoding"] == "rgb8" and sample["jpeg_encoding_s"] == 0
    assert sample["wire_payload_bytes"] is None and sample["camera_exposure_timestamp_s"] is None
    for key in ("observation_processing_s", "queue_depth_at_start", "queue_depth_at_return",
                "remaining_valid_action_horizon_s", "expired_prefix_dropped"):
        assert key in result["prediction_samples"][0]


def test_late_rpc_underrun_releases_without_replay():
    from scripts.benchmark_remote import run_scenario

    result = run_scenario("test_underrun", [0.04, 0.7], duration=3, image_hw=(8, 8))
    assert result["error"] is not None and result["underruns"] == 1
    assert result["sample_count"] == 1
    assert result["all_fake_robots_released"]
    assert result["failed"]


def test_transport_failure_never_reuses_previous_request_timing(monkeypatch):
    import sys

    from yamkit.inference.client import ModalTransport

    remote = SimpleNamespace(spawn=lambda: SimpleNamespace(get=lambda **kw: {}, cancel=lambda: None))
    modal = SimpleNamespace(Cls=SimpleNamespace(from_name=lambda *a: lambda: SimpleNamespace(ready=remote)))
    monkeypatch.setitem(sys.modules, "modal", modal)
    transport = ModalTransport("fake-app", "molmoact2")
    transport.ready(1)
    assert "dispatch_s" in transport.last_timing

    def fail():
        raise RuntimeError("synthetic SDK failure")

    remote.spawn = fail
    with pytest.raises(RemoteFault):
        transport.ready(1)
    assert "dispatch_s" not in transport.last_timing
    assert "handle_lookup_s" in transport.last_timing
    assert transport.last_timing["modal_queue_s"] is None
