"""Hardware-free, fake-clock qualification of the explicit frozen-base interface."""

from threading import Event

import numpy as np
import pytest

from yamkit.inference.mapping import YAM_NAMES
from yamkit.openpi.executor import OpenPiYamExecutor
from yamkit.openpi.interface import (
    ACTION_TRANSFORM,
    COMMITTED_ROWS,
    MAX_COMMAND_DT,
    MODEL_ROWS,
    OpenPiExecutionFault,
    linear_transition,
    prepare_chunk,
    speed_vector,
)


class Clock:
    def __init__(self):
        self.now = 100.0
        self.wait_hook = None

    def __call__(self):
        return self.now

    def wait(self, seconds):
        assert 0 < seconds <= 0.01
        self.now += max(seconds, 1e-12)
        if self.wait_hook:
            self.wait_hook()


def vector(joint=0.0, grip=0.5):
    value = np.full(14, joint)
    value[[6, 13]] = grip
    return value


def chunk(joint=0.0, grip=0.5):
    return np.repeat(vector(joint, grip)[None], MODEL_ROWS, axis=0)


class Rig:
    def __init__(self):
        self.clock = Clock()
        self.stop = Event()
        self.state = vector()
        self.sent = []
        self.events = []
        self.predict_calls = []
        self.in_rpc = False
        self.rpc_s = 0.1
        self.result = chunk()
        self.send_hook = self.predict_hook = self.observe_hook = None
        self.last_sdk_at = None
        self.last_sdk_target = None

    def validate(self, target):
        value = np.array([target[name] for name in YAM_NAMES])
        if np.any(np.abs(value) > 3.0):
            raise ValueError("test joint bounds")
        if np.any(value[[6, 13]] < 0) or np.any(value[[6, 13]] > 1):
            raise ValueError("test gripper bounds")

    def observe(self):
        assert not self.in_rpc
        if self.observe_hook:
            self.observe_hook()
        return {"state": self.state.copy()}

    def predict(self, observation, timeout):
        assert not self.in_rpc
        self.in_rpc = True
        self.predict_calls.append((self.clock(), observation["state"].copy(), timeout))
        try:
            self.clock.now += self.rpc_s
            if self.predict_hook:
                return self.predict_hook(observation, timeout)
            return self.result
        finally:
            self.in_rpc = False

    def send(self, target, check):
        assert not self.in_rpc
        check()
        value = np.array([target[name] for name in YAM_NAMES])
        # Emulate ordinary YamArm.command, including its capped elapsed time
        # and stale-measured reset. Never emulate limit_speed=False.
        age = None if self.last_sdk_at is None else self.clock() - self.last_sdk_at
        stale = age is None or age > 0.5
        previous = self.state if stale else self.last_sdk_target
        dt = MAX_COMMAND_DT if stale else min(age, MAX_COMMAND_DT)
        actual = previous + np.clip(value - previous, -3 * dt, 3 * dt)
        self.last_sdk_at, self.last_sdk_target = self.clock(), actual.copy()
        self.state = actual.copy()
        self.sent.append((self.clock(), actual.copy(), value.copy()))
        result = dict(zip(YAM_NAMES, actual.tolist(), strict=True))
        return self.send_hook(result, check) if self.send_hook else result

    def engine(self, **kwargs):
        return OpenPiYamExecutor(predict=self.predict, observe=self.observe, send=self.send,
                                 validate_target=self.validate, stop=self.stop, clock=self.clock,
                                 wait=self.clock.wait,
                                 event=lambda kind, **data: self.events.append((kind, data)), **kwargs)


def test_endpoint_mapping_is_explicit_not_a_fitted_percentage():
    raw = chunk()
    raw[0, 6] = 1.0333
    raw[1, 13] = -0.25
    raw[49, 13] = 7.0
    result = prepare_chunk(raw)
    np.testing.assert_array_equal(raw, result["raw_rows"])
    assert result["rows"][0, 6] == 1
    assert result["rows"][1, 13] == 0
    assert result["rows"][49, 13] == 1
    assert [value["committed_prefix"] for value in result["gripper_conversions"]] == [True, True, False]
    assert ACTION_TRANSFORM["raw_anomaly_percentage"] is None
    assert ACTION_TRANSFORM["native_openpi_transform"] is False
    joint_columns = [index for index in range(14) if index not in (6, 13)]
    np.testing.assert_array_equal(result["raw_rows"][:, joint_columns], result["rows"][:, joint_columns])


@pytest.mark.parametrize("bad", [np.zeros((25, 14)), np.zeros((50, 32)), [[True] * 14] * 50,
                                np.full((50, 14), np.nan), np.full((50, 14), np.inf)])
def test_invalid_full_chunk_rejected(bad):
    with pytest.raises(OpenPiExecutionFault):
        prepare_chunk(bad)


@pytest.mark.parametrize("delta", [0.0, 0.001, 0.03, 0.030000000001, 0.367, 2.0, -2.0])
def test_transition_reaches_exact_target_and_preserves_caps(delta):
    origin, target = vector(), vector(delta, 0.99)
    speeds = speed_vector(3.0, 3.0)
    points = linear_transition(origin, target, speeds)
    np.testing.assert_array_equal(points[-1], target)
    assert np.max(np.abs(np.diff(np.vstack((origin, points)), axis=0))) <= 0.03
    assert len(points) >= 17  # .49 gripper travel also receives its ordinary speed bound.


def test_transition_has_no_extra_substeps_for_small_target():
    points = linear_transition(vector(), vector(0.001), speed_vector(3, 3))
    assert points.shape == (1, 14)


@pytest.mark.parametrize("speed", [0, -1, 3.01, True, float("nan"), float("inf")])
def test_speed_guards_not_weakened(speed):
    with pytest.raises(ValueError):
        speed_vector(speed, 3)


def test_exact_25_prefix_of_50_and_native_maximum_row_rate():
    rig = Rig()
    rig.result[:, 0] = np.linspace(0.001, 0.049, 50)
    engine = rig.engine()
    report = engine.run(duration_s=5, max_chunks=1)
    assert report["predicted_rows"] == 50
    assert report["completed_rows"] == 25
    assert report["intended_unused_rows"] == 25
    assert report["uncompleted_committed_rows"] == 0
    assert report["inserted_transition_points"] == 0
    assert report["modified_commands"] == report["faults"] == 0
    assert len(rig.sent) == 25
    np.testing.assert_allclose(np.array([value[1] for value in rig.sent]), rig.result[:25], atol=1e-16)
    assert np.min(np.diff(engine.endpoint_times)) >= 0.02 - 1e-12
    assert report["observations"] == 26


def test_initial_and_cross_chunk_transitions_keep_every_endpoint():
    rig = Rig()
    first, second = chunk(0.367), chunk(-0.211)
    result = iter([first, second])
    rig.predict_hook = lambda *_: next(result)
    report = rig.engine().run(duration_s=10, max_chunks=2)
    endpoints = [data for kind, data in rig.events if kind == "dispatch" and data["endpoint"]]
    requested = np.array([[value["requested"][name] for name in YAM_NAMES] for value in endpoints])
    np.testing.assert_array_equal(requested, np.concatenate([first[:25], second[:25]]))
    assert report["completed_rows"] == 50
    assert report["inserted_transition_points"] >= 30
    assert report["maximum_row_time_dilation_s"] > 0.1
    assert report["modified_commands"] == 0
    assert np.max(np.abs(np.diff(np.vstack((vector(), *[value[1] for value in rig.sent])), axis=0))) <= 0.03
    assert rig.predict_calls[1][0] >= endpoints[24]["monotonic_s"]


def test_all_committed_rows_validate_before_any_dispatch():
    rig = Rig()
    rig.result[24, 0] = 4
    engine = rig.engine()
    with pytest.raises(ValueError, match="joint bounds"):
        engine.run(duration_s=5)
    assert not rig.sent
    assert engine.metrics()["invalid_chunks"] == 1


def test_unused_tail_finite_but_not_dispatched_or_rig_bound_checked():
    rig = Rig()
    rig.result[25:, 0] = 400
    report = rig.engine().run(duration_s=5, max_chunks=1)
    assert report["completed_rows"] == 25
    assert report["faults"] == 0


def test_unused_tail_nan_invalidates_before_send():
    rig = Rig()
    rig.result[49, 0] = np.nan
    with pytest.raises(OpenPiExecutionFault):
        rig.engine().run(duration_s=5)
    assert not rig.sent


def test_gripper_conversion_is_counted_separately_for_unused_tail():
    rig = Rig()
    rig.result[:, 6] = 1.05
    report = rig.engine().run(duration_s=5, max_chunks=1)
    assert report["projected_gripper_values"] == 50
    assert report["executed_projected_gripper_values"] == 25
    assert report["maximum_gripper_projection"] == pytest.approx(0.05)
    assert report["completed_rows"] == 25
    assert report["inserted_transition_points"] >= 16


def test_stop_during_rpc_never_dispatches_returned_actions():
    rig = Rig()
    def predict(*_):
        rig.stop.set()
        return rig.result
    rig.predict_hook = predict
    report = rig.engine().run(duration_s=5)
    assert not rig.sent
    assert report["stop_requested"] is True
    assert report["faults"] == 0
    assert report["late_response_rows_discarded"] == 50


def test_stop_invalidated_rpc_exception_is_expected_not_an_automatic_retry():
    rig = Rig()
    def predict(*_):
        rig.stop.set()
        raise RuntimeError("fake invalidated RPC")
    rig.predict_hook = predict
    report = rig.engine().run(duration_s=5)
    assert report["stopped_rpc_exceptions"] == 1
    assert report["inference_calls"] == 1
    assert not rig.sent


def test_late_rpc_response_faults_before_send():
    rig = Rig()
    rig.rpc_s = 2.1
    engine = rig.engine()
    with pytest.raises(OpenPiExecutionFault, match="deadline"):
        engine.run(duration_s=5)
    assert not rig.sent


def test_stop_during_transition_wait_has_no_further_commands():
    rig = Rig()
    rig.result = chunk(0.367)
    rig.clock.wait_hook = lambda: rig.stop.set() if len(rig.sent) >= 2 else None
    engine = rig.engine()
    report = engine.run(duration_s=5)
    assert len(rig.sent) == 2
    assert report["completed_rows"] == 0
    assert report["uncompleted_committed_rows"] == 25
    assert report["rows_discarded_on_stop_or_deadline"] == 25


def test_dispatch_guard_between_two_fake_arms_records_unknown_partial():
    rig = Rig()
    def send_hook(value, check):
        rig.stop.set()
        check()
        return value
    rig.send_hook = send_hook
    engine = rig.engine()
    with pytest.raises(OpenPiExecutionFault, match="cancelled"):
        engine.run(duration_s=5)
    assert engine.metrics()["unknown_partial_dispatches"] == 1
    assert engine.metrics()["completed_points"] == 0


def test_unplanned_sdk_clamp_faults_instead_of_skipping_endpoint():
    rig = Rig()
    def send_hook(value, _):
        value[YAM_NAMES[0]] += 0.001
        return value
    rig.send_hook = send_hook
    engine = rig.engine()
    with pytest.raises(OpenPiExecutionFault, match="SDK altered"):
        engine.run(duration_s=5)
    assert engine.metrics()["modified_commands"] == 1
    assert engine.metrics()["coherence_violations"] == 1
    assert len(rig.sent) == 1


def test_fresh_measurement_after_long_rpc_defines_transition_origin_not_model_anchor():
    rig = Rig()
    rig.rpc_s = 0.6
    def predict(*_):
        rig.state = vector(0.2)
        return chunk(0.21)
    rig.predict_hook = predict
    report = rig.engine().run(duration_s=5, max_chunks=1)
    np.testing.assert_array_equal(rig.predict_calls[0][1], vector())
    first = report["rows"][0]
    assert first["transition_origin"] == "fresh_measured"
    np.testing.assert_array_equal(first["origin"], vector(0.2))
    assert first["planned_points"] == 1


def test_session_expiry_during_wait_prevents_next_dispatch():
    rig = Rig()
    def session():
        if len(rig.sent) >= 1:
            raise RuntimeError("fake expired session")
    engine = rig.engine(session_check=session)
    with pytest.raises(RuntimeError, match="expired"):
        engine.run(duration_s=5)
    assert len(rig.sent) == 1
    assert engine.metrics()["faults"] == 1
    assert engine.metrics()["completed_rows"] == 1
    assert engine.metrics()["chunks"][0]["completed_rows"] == 1


def test_scheduler_stall_does_not_trigger_a_large_sdk_stale_reset_step():
    rig = Rig()
    rig.result = chunk(0.367)
    jumped = False
    def late_wait():
        nonlocal jumped
        if not jumped and len(rig.sent) == 1:
            rig.clock.now += 0.6
            jumped = True
    rig.clock.wait_hook = late_wait
    engine = rig.engine()
    with pytest.raises(OpenPiExecutionFault, match="stalled"):
        engine.run(duration_s=5)
    assert len(rig.sent) == 1
    assert engine.metrics()["completed_rows"] == 0


def test_invalid_measured_gripper_is_not_endpoint_projected():
    rig = Rig()
    rig.state[6] = 1.001
    with pytest.raises(OpenPiExecutionFault, match="Measured gripper"):
        rig.engine().run(duration_s=5)
    assert not rig.sent
    assert not rig.predict_calls


def test_stop_before_start_does_not_read_or_request():
    rig = Rig()
    rig.stop.set()
    report = rig.engine().run(duration_s=5)
    assert report["observations"] == report["inference_calls"] == report["completed_points"] == 0
    assert report["stop_requested"] is True


def test_fault_does_not_auto_retry_predictor():
    rig = Rig()
    def predict(*_):
        raise RuntimeError("fake connection failure")
    rig.predict_hook = predict
    engine = rig.engine()
    with pytest.raises(RuntimeError, match="connection failure"):
        engine.run(duration_s=5)
    assert len(rig.predict_calls) == 1
    assert not rig.sent
    assert engine.metrics()["faults"] == 1


def test_deadline_during_transition_does_not_complete_or_retry_row():
    rig = Rig()
    rig.result = chunk(2.5)
    engine = rig.engine()
    report = engine.run(duration_s=0.15)
    assert report["completed_rows"] == 0
    assert 0 < report["completed_points"] < 84
    assert rig.clock() <= 100.15 + 1e-12
    with pytest.raises(OpenPiExecutionFault, match="single use"):
        engine.run(duration_s=1)


def test_no_rpc_within_phase_tail_after_complete_chunk():
    rig = Rig()
    report = rig.engine().run(duration_s=1.0)
    assert report["completed_chunks"] == 1
    assert report["inference_calls"] == 1
    assert report["phase_tail_requests_avoided"] == 1
    assert report["faults"] == 0


def test_model_response_audit_is_preserved_in_admission_event():
    rig = Rig()
    rig.result = {"chunk": chunk(), "audit": {"raw_model_shape": [50, 32], "model_unchanged": True}}
    rig.engine().run(duration_s=5, max_chunks=1)
    admission = next(data for kind, data in rig.events if kind == "chunk_admitted")
    assert admission["audit"] == rig.result["audit"]
    assert len(admission["raw_rows"]) == MODEL_ROWS
    assert admission["committed_prefix_rows"] == COMMITTED_ROWS


def test_policy_input_observation_seam_adds_no_observations_or_reads():
    rig = Rig()
    anchors = []
    def policy_observe():
        anchors.append(rig.clock())
        return rig.observe()
    report = rig.engine(observe_policy_input=policy_observe).run(duration_s=5, max_chunks=2)
    assert len(anchors) == report["inference_calls"] == report["policy_input_observations"] == 2
    assert report["observations"] == 52  # Two request anchors + the existing fifty row observations.
    assert [row["policy_observation_index"] for row in report["chunks"]] == [0, 26]


def test_phase_tail_uses_monitor_seam_not_policy_input_seam():
    rig = Rig()
    anchors = []
    def policy_observe():
        anchors.append(rig.clock())
        return rig.observe()
    report = rig.engine(observe_policy_input=policy_observe).run(duration_s=1)
    assert report["phase_tail_requests_avoided"] == 1
    assert len(anchors) == report["inference_calls"] == 1
    assert report["observations"] > 26


def test_policy_observation_acquisition_consumes_rpc_deadline_budget():
    rig = Rig()
    def policy_observe():
        rig.clock.now += 0.08
        return rig.observe()
    report = rig.engine(observe_policy_input=policy_observe).run(duration_s=0.05)
    assert report["policy_input_observations"] == 1
    assert not rig.predict_calls and not rig.sent
