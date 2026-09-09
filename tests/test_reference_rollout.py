"""Reference mode through real LeRobot factories/strategy, fake HTTP and YAM only."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by parameter name

import threading

import numpy as np
import pytest

from tests.test_http_policy_readiness import rollout_config, transport  # noqa: F401 -- shared fixtures
from yamkit.inference.client import InvalidatedRequest
from yamkit.remote_rollout import run_remote_rollout

TASK = "put the red cube into the black container"


@pytest.fixture
def reference_config(rollout_config):
    from yamkit.config import RigConfig

    rig = RigConfig.load(rollout_config.robot.rig)
    rig.control.home_speed = .5
    rig.save()
    rollout_config.task = rollout_config.policy.task = TASK
    rollout_config.policy.controller_mode = "reference"
    rollout_config.duration = 8
    return rollout_config


def test_full_chunk_finishes_before_second_request_and_stop_releases(
        transport, reference_config, fake_connect):
    stop = threading.Event()

    def returned(response):
        # First request is pre-hardware warmup. Stop during the second policy
        # request proves no speculative observation while chunk one executes.
        if len(transport.requests) == 3:
            stop.set()
            raise InvalidatedRequest("Fake HTTP rejects the reply after Stop, as the real transport does")

    transport.response_hook = returned
    metrics = run_remote_rollout(reference_config, shutdown_event=stop)
    reference = metrics["reference_execution"]
    assert metrics["failed"] is False and metrics["controller_mode"] == "reference"
    assert reference["predicted_steps"] == reference["completed_steps"] == 30
    assert reference["completed_chunks"] == 1 and reference["partial_chunk_at_stop"] is False
    assert reference["interpolation_dispatches"] > 30
    assert reference["coherence_violations"] == 0
    assert metrics["expired_prefix_dropped"] == metrics["overlap_prefix_dropped"] == 0
    assert all(event["actions_executed_during_prediction"] == 0 for event in metrics["prediction_samples"])
    assert metrics["prediction_samples"][1]["completed_steps_at_start"] == 30
    assert transport.requests[-1]["state"] == pytest.approx([.2] * 14)
    assert transport.requests[-1]["task"] == TASK
    assert all(robot.closed for robot in fake_connect.values())


def test_invalid_final_row_is_rejected_before_any_interpolated_command(
        transport, reference_config, fake_connect):
    def corrupt(response):
        response["chunk"][-1][0] = 100.0

    transport.response_hook = corrupt
    with pytest.raises(ValueError) as raised:
        run_remote_rollout(reference_config, shutdown_event=threading.Event())
    metrics = raised.value.metrics
    assert metrics["failed"] is True
    assert metrics["executed_actions"] == 0
    assert metrics["reference_execution"]["predicted_steps"] == 30
    assert metrics["reference_execution"]["admitted_steps"] == 0
    assert metrics["reference_execution"]["completed_steps"] == 0
    assert all(robot.closed for robot in fake_connect.values())


def test_duration_cuts_partial_chunk_without_replanning_and_releases(
        transport, reference_config, fake_connect, monkeypatch):
    from yamkit.arm import YamArm
    from yamkit.config import RigConfig

    rig = RigConfig.load(reference_config.robot.rig)
    rig.control.home_speed = .25
    rig.save()
    original_home = YamArm.go_home
    homes = {}

    def home(arm, *args, **kwargs):
        begin = len(arm.robot.commands)
        result = original_home(arm, *args, **kwargs)
        homes.setdefault(arm.name, []).append((begin, len(arm.robot.commands), kwargs["speed"]))
        return result

    monkeypatch.setattr(YamArm, "go_home", home)
    reference_config.duration = .3
    metrics = run_remote_rollout(reference_config, shutdown_event=threading.Event())
    assert metrics["failed"] is False and metrics["duration_completed"] is True
    assert metrics["home_attempted"] and metrics["home_completed"] and not metrics["home_aborted"]
    assert metrics["fault_stop_to_robot_release_s"] is None
    assert metrics["reference_execution"]["partial_chunk_at_stop"] is True
    assert metrics["reference_execution"]["completed_steps"] < 30
    assert len(transport.requests) == 2
    assert all(robot.closed for robot in fake_connect.values())
    for name, robot in fake_connect.items():
        assert len(homes[name]) == 2  # Initial home and the healthy duration-completion home.
        begin, end, speed = homes[name][-1]
        assert end > begin and speed == .25  # Actual fake-motor home commands were sent.
        np.testing.assert_allclose(robot.pos[:6], 0, atol=1e-9)


def test_postclamp_gripper_intervention_faults_instead_of_breaking_coordination(
        transport, reference_config, fake_connect, monkeypatch):
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    original = BiYamFollower.send_reference_action

    def changed(robot, action, **kwargs):
        sent = original(robot, action, **kwargs)
        sent["left_gripper.pos"] -= .001  # The startup point is now fully open; keep the injected mismatch in bounds.
        return sent

    monkeypatch.setattr(BiYamFollower, "send_reference_action", changed)
    with pytest.raises(ValueError, match="Postclamp reference") as raised:
        run_remote_rollout(reference_config, shutdown_event=threading.Event())
    metrics = raised.value.metrics
    assert metrics["executed_actions"] == 1
    assert metrics["reference_execution"]["coherence_violations"] == 1
    assert metrics["reference_execution"]["completed_steps"] == 0
    assert all(robot.closed for robot in fake_connect.values())
