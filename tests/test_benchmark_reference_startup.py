"""Qualification exercises production startup/home using fake cameras and motors."""
# ruff: noqa: F811 -- imported pytest fixtures are injected by parameter name

import pytest

from scripts.benchmark_remote import run_scenario
from tests.test_http_policy_readiness import transport  # noqa: F401
from yamkit.arm import YamArm
from yamkit.inference.client import InvalidatedRequest
from yamkit.inference.profiles import get_profile


def test_reference_benchmark_homes_open_before_first_request_and_counts_only_policy_sends(transport, monkeypatch):
    stop_event = None
    homes = []
    original_home = YamArm.go_home

    def home(arm, *args, **kwargs):
        before = len(arm.robot.commands)
        result = original_home(arm, *args, **kwargs)
        homes.append((arm.name, kwargs.get("gripper"), len(arm.robot.commands) - before))
        return result

    def factory(stop):
        nonlocal stop_event
        stop_event = stop
        return transport

    def response(_response):
        if len(transport.requests) == 3:
            stop_event.set()
            raise InvalidatedRequest("Fake HTTP cancels the second policy request")

    monkeypatch.setattr(YamArm, "go_home", home)
    transport.response_hook = response
    result = run_scenario("reference-startup", [0], duration=8, image_hw=(8, 8),
                          transport_factory=factory,
                          policy_options={"call_mode": "http", "execution_mode": "cuda_graph10",
                                          "controller_mode": "reference"})
    startup = result["reference_startup"]
    expected = {name: 1.0 if "gripper" in name else 0.0 for name in get_profile("molmoact2").action_names}
    assert sorted((name, grip) for name, grip, _ in homes) == [("left_follower", 1.0), ("right_follower", 1.0)]
    assert all(count > 0 for _, _, count in homes)
    assert startup["configured_start_action"] == startup["cached_start_action"] == startup["last_startup_sdk_action"] == expected
    assert startup["first_policy_state"] == pytest.approx(list(expected.values()))
    assert startup["phase_boundary"] == "ReferenceStrategy.run" and startup["home_speed"] == .5
    assert result["reference_execution"]["completed_chunks"] == 1
    assert result["executed_actions"] == result["reference_execution"]["interpolation_dispatches"]
    for side in ("left", "right"):
        assert startup["sdk_sends_by_phase"]["startup"][side] > 0
        assert startup["sdk_sends_by_phase"]["policy"][side] == result["executed_actions"]
        assert startup["sdk_sends_by_phase"]["cleanup"][side] == 0
        assert startup["last_startup_sdk_send_monotonic_s"][side] <= startup["policy_phase_started_monotonic_s"]
    assert startup["sdk_total_sends"] > 2 * result["executed_actions"]
    assert result["sdk_sends_during_completed_rpc"] == [0, 0]
    assert result["commands_after_stop"] == 0 and result["stop_requested_during_inflight_rpc"]
    assert result["all_fake_robots_released"] and not result["failed"]


def test_reference_benchmark_excludes_normal_completion_home_from_policy_count(transport):
    result = run_scenario("reference-duration-home", [0], duration=.3, image_hw=(8, 8),
                          transport_factory=lambda stop: transport,
                          policy_options={"call_mode": "http", "execution_mode": "cuda_graph10",
                                          "controller_mode": "reference"})
    startup = result["reference_startup"]
    assert result["duration_completed"] and result["home_completed"] and not result["failed"]
    assert result["all_fake_robots_released"] and result["executed_actions"] > 0
    for side in ("left", "right"):
        assert startup["sdk_sends_by_phase"]["startup"][side] > 0
        assert startup["sdk_sends_by_phase"]["policy"][side] == result["executed_actions"]
        assert startup["sdk_sends_by_phase"]["cleanup"][side] > 0
    assert startup["sdk_total_sends"] == sum(count for phase in startup["sdk_sends_by_phase"].values()
                                              for count in phase.values())
    assert result["sdk_sends_during_completed_rpc"] == [0, 0]
