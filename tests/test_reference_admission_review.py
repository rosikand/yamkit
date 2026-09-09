"""First-chunk freshness admission under a fake clock; no hardware or transport."""

import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from yamkit import reference_rollout
from yamkit.inference import reference
from yamkit.inference.client import RemoteFault
from yamkit.inference.command_shaping import ACTION_NAMES, JOINT_NAMES
from yamkit.inference.profiles import get_profile


@pytest.mark.parametrize("overrun_s", [0.0, 0.01])
def test_initial_plan_cannot_outlive_original_admission_deadline(monkeypatch, overrun_s):
    clock = [100.0]
    monkeypatch.setattr(reference_rollout, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    initial = dict.fromkeys(ACTION_NAMES, 0.0)
    limits = {name: {"lower": -2.0, "upper": 2.0, "max_step": .1} for name in JOINT_NAMES}
    grips = {name: .1 for name in ACTION_NAMES if "gripper" in name}
    guard = reference.ReferenceCommandGuard(initial, limits, grips)
    measured_reads, predictions, closes = [], [], []
    main_thread = threading.get_ident()

    def measured_state():
        assert threading.get_ident() == main_thread
        measured_reads.append(clock[0])
        return dict(initial)

    def predict(batch):
        predictions.append(batch["observation.state"].clone())
        return torch.full((1, 30, 14), .02)

    def forbidden_send(*args, **kwargs):
        raise AssertionError("An expired initial plan must never reach hardware")

    policy = SimpleNamespace(transport=SimpleNamespace(), config=SimpleNamespace(
        request_timeout_s=10, max_observation_age_s=2), _last_prediction_timing={},
        predict_action_chunk=predict, reset=lambda: None, close=lambda: closes.append(True))
    robot = SimpleNamespace(command_shaper=guard, robot_type="bi_yam_follower",
                            invalidate_shaping=guard.invalidate,
                            inner=SimpleNamespace(get_joint_state=measured_state,
                                                  validate_action_target=lambda target: None,
                                                  joint_command_limits=lambda: limits,
                                                  send_action=forbidden_send))
    stop = threading.Event()
    engine = reference_rollout.ReferenceRemoteInferenceEngine(
        policy=policy, preprocessor=lambda value: value, postprocessor=lambda value: value,
        robot_wrapper=robot, task="put the red cube into the black container", fps=30,
        shutdown_event=stop, duration=30, gripper_max_step=grips)
    observation = {"observation.state": np.zeros(14, dtype=np.float32),
                   **{f"observation.images.{name}": np.zeros((2, 2, 3), dtype=np.uint8)
                      for name in get_profile("molmoact2").image_keys}}
    engine.start()
    engine.resume()
    engine.notify_observation(observation)
    engine._begin_prediction(observation)
    slot = engine._rpc
    assert slot["done"].wait(2)
    assert slot["error"] is None and slot["returned"] < slot["deadline"]
    deadline = slot["deadline"]
    assert deadline == 102.0
    planned_rows = []
    original_plan = reference.plan_reference_row

    def delayed_plan(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        planned_rows.append(result)
        # The measured seed and wait-end check passed while fresh. Only pure
        # plan construction consumes the remaining admission budget.
        clock[0] = deadline + overrun_s
        return result

    monkeypatch.setattr(reference, "plan_reference_row", delayed_plan)
    with pytest.raises(RemoteFault, match="admission exceeded the original inference wait deadline"):
        engine.get_action(observation)

    assert measured_reads == [100.0] and len(planned_rows) == 30
    assert len(predictions) == 1 and slot["deadline"] == deadline
    assert guard.generation == 0 and guard.last_at is None
    assert engine._plans == [] and engine._plan_deadline is None and engine._pending is None
    assert engine.admitted_steps == engine.dequeued_actions == engine.executed_actions == 0
    assert slot["event"]["accepted_steps"] == 0 and slot["event"]["error"] == "RemoteFault"
    assert engine.failed and stop.is_set() and not guard.valid and closes
