"""Reference evidence must prove serial full chunks, separately from async."""

import copy
import json
import time
from pathlib import Path

import pytest

from tests import test_http_qualification, test_http_qualification_collection
from yamkit.inference import qualification as q
from yamkit.inference.profiles import get_profile
from yamkit.modal_qualification import collect_qualification

sdk_evidence = test_http_qualification.sdk_evidence
http_evidence = test_http_qualification.http_evidence
collection = test_http_qualification_collection.collection


@pytest.fixture
def reference_evidence(http_evidence):
    settings, direct, integrated = copy.deepcopy(http_evidence)
    settings["controller_mode"] = "reference"
    integrated["policy_options"]["controller_mode"] = "reference"
    integrated.update(minimum_execution_queue_depth=0, expired_prefix_dropped=0,
                      overlap_prefix_dropped=0, executed_actions=51 * 60 + 51 * 4)
    integrated["prediction_samples"] = [
        {"error": None, "accepted_steps": 30, "completed_chunks_at_start": index,
         "completed_steps_at_start": index * 30, "actions_executed_during_prediction": 0,
         "observation_age_at_return_s": 0.31, "prediction_started_monotonic_s": 100 + index * 3,
         "prediction_s": 0.31, "plan_dispatches": 60, "planned_duration_s": 2.0,
         "plan_deadline_monotonic_s": 102.6 + index * 3} for index in range(51)]
    integrated["reference_execution"] = {
        "controller_mode": "reference", "predicted_steps": 1530, "admitted_steps": 1530, "completed_steps": 1530,
        "completed_chunks": 51, "interpolation_dispatches": 3060,
        "inference_hold_dispatches": 204, "inference_hold_mismatches": 0,
        "uncompleted_steps_at_stop": 0, "expired_prefix_dropped": 0, "overlap_prefix_dropped": 0,
        "prefix_drop": 0, "expired_plans": 0, "coherence_violations": 0,
        "partial_chunk_at_stop": False, "next_observation_after_full_chunk": True,
    }
    integrated["prediction_samples"].append({
        "error": "invalidated", "accepted_steps": 0, "actions_executed_during_prediction": 0,
        "prediction_started_monotonic_s": 253, "prediction_s": 0.35})
    for index, event in enumerate(integrated["prediction_samples"]):
        started = event["prediction_started_monotonic_s"]
        anchor = dict.fromkeys(get_profile("molmoact2").action_names, 0.0) if index else None
        event.update(observation_timestamp_monotonic_s=started,
                     inference_wait_deadline_monotonic_s=started + 2,
                     maintenance_hold_anchor=anchor, maintenance_holds_during_prediction=4 if index else 0,
                     maintenance_hold_samples_dropped=0,
                     maintenance_hold_samples=[{
                         "dispatch_index": index * 60 + (index - 1) * 4 + step,
                         "monotonic_s": started + (step + 1) * 0.08,
                         "deadline_monotonic_s": started + (step + 1) * 0.08 + 0.09,
                         "sent": dict(anchor)} for step in range(4)] if index else [])
    integrated["command_shaping"] = {"inference_hold_count": 204, "postclamp_modified_count": 0,
                                      "sample_count": integrated["executed_actions"]}
    integrated["transport_predictions"] = [
        {"mode": "native_fixture", "started": 90.0, "returned": 90.3},
        *[{"mode": "robot", "started": 100 + index * 3 + 0.02, "returned": 100 + index * 3 + 0.30}
          for index in range(51)], {"mode": "robot", "started": 253.02}]
    integrated["sdk_commands_during_completed_rpc"] = [0, 0] + [3] * 50
    return settings, direct, integrated


def assessment(evidence):
    settings, direct, integrated = evidence
    return q.build_qualification(settings, direct=direct, integrated=integrated)["assessment"]


def test_reference_full_consumption_qualifies_without_an_async_queue(reference_evidence, tmp_path):
    settings, direct, integrated = reference_evidence
    record = q.build_qualification(settings, direct=direct, integrated=integrated)
    result = record["assessment"]
    assert result["qualified"], result["reasons"]
    assert result["controller_mode"] == "reference"
    assert result["completed_integrated_warm_samples"] == 50
    assert result["latency_budget_basis"] == "synchronous RPC admission"
    assert result["maximum_qualifying_rpc_p95_s"] == pytest.approx(1.6)
    path = q.save_qualification(record, tmp_path / "reference.json")
    assert q.validate_qualification(settings, path=path)["assessment"]["qualified"]


def test_inference_holds_are_additional_commands_not_policy_steps(reference_evidence):
    _, _, integrated = reference_evidence
    proof = integrated["reference_execution"]
    assert integrated["executed_actions"] == proof["interpolation_dispatches"] + proof["inference_hold_dispatches"]
    assert proof["completed_steps"] == 1530 and proof["inference_hold_dispatches"] == 204
    assert assessment(reference_evidence)["qualified"]
    integrated["executed_actions"] = proof["interpolation_dispatches"]
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("field,value", [
    ("maintenance_holds_during_prediction", 3), ("maintenance_holds_during_prediction", None),
    ("maintenance_hold_samples_dropped", 1), ("maintenance_hold_samples", []),
    ("maintenance_hold_anchor", None), ("maintenance_hold_anchor", {"left_joint_1.pos": 0.0}),
    ("inference_wait_deadline_monotonic_s", 105.1),
    ("observation_timestamp_monotonic_s", 103.1), ("actions_executed_during_prediction", 1),
])
def test_reference_hold_evidence_must_be_complete_and_fresh(reference_evidence, field, value):
    reference_evidence[2]["prediction_samples"][1][field] = value
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("field,value", [
    ("monotonic_s", 102.9), ("monotonic_s", 105.1),
    ("deadline_monotonic_s", 105.1), ("deadline_monotonic_s", 103.4),
    ("dispatch_index", 3264), ("dispatch_index", True), ("sent", None),
])
def test_bad_individual_hold_cannot_qualify(reference_evidence, field, value):
    reference_evidence[2]["prediction_samples"][1]["maintenance_hold_samples"][0][field] = value
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("value", [0.001, True, float("nan")])
def test_hold_must_preserve_exact_fourteen_dimensional_anchor(reference_evidence, value):
    hold = reference_evidence[2]["prediction_samples"][1]["maintenance_hold_samples"][0]
    hold["sent"]["left_joint_1.pos"] = value
    assert not assessment(reference_evidence)["qualified"]


def test_duplicate_hold_or_stalled_cadence_is_rejected(reference_evidence):
    holds = reference_evidence[2]["prediction_samples"][1]["maintenance_hold_samples"]
    holds[1]["dispatch_index"] = holds[0]["dispatch_index"]
    assert not assessment(reference_evidence)["qualified"]
    holds[1]["dispatch_index"] += 1
    holds[1]["monotonic_s"] += 0.021
    holds[1]["deadline_monotonic_s"] += 0.021
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("container,field,value", [
    ("reference_execution", "inference_hold_dispatches", 203),
    ("reference_execution", "inference_hold_mismatches", 1),
    ("command_shaping", "inference_hold_count", 203),
    ("command_shaping", "sample_count", 3060),
    ("command_shaping", "postclamp_modified_count", 1),
])
def test_hold_totals_and_guard_proof_are_required(reference_evidence, container, field, value):
    reference_evidence[2][container][field] = value
    assert not assessment(reference_evidence)["qualified"]


def test_completed_http_overlap_must_fit_proven_holds(reference_evidence):
    evidence = reference_evidence[2]
    evidence["sdk_commands_during_completed_rpc"][2] = 5
    assert not assessment(reference_evidence)["qualified"]
    evidence["sdk_commands_during_completed_rpc"][2] = 3
    evidence["transport_predictions"][2]["returned"] += 0.1
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("value", [None, [], [0]])
def test_missing_sdk_overlap_measurement_cannot_qualify(reference_evidence, value):
    reference_evidence[2]["sdk_commands_during_completed_rpc"] = value
    assert not assessment(reference_evidence)["qualified"]


def test_worker_return_boundary_and_cancelled_probe_holds_are_accounted(reference_evidence):
    _, _, integrated = reference_evidence
    event = integrated["prediction_samples"][1]
    last = event["maintenance_hold_samples"][-1]
    assert last["monotonic_s"] > event["prediction_started_monotonic_s"] + event["prediction_s"]
    assert integrated["sdk_commands_during_completed_rpc"][2] < event["maintenance_holds_during_prediction"]
    assert assessment(reference_evidence)["qualified"]
    # Removing a final cancelled request's holds would conceal successful SDK
    # sends even though that request supplied no admitted policy rows.
    integrated["prediction_samples"].pop()
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("field,value", [
    ("completed_chunks", 50), ("predicted_steps", 1500), ("completed_steps", 1529),
    ("interpolation_dispatches", 3059), ("uncompleted_steps_at_stop", 1),
    ("expired_prefix_dropped", 1), ("overlap_prefix_dropped", 1), ("prefix_drop", 1),
    ("coherence_violations", 1), ("expired_plans", 1), ("partial_chunk_at_stop", True),
    ("next_observation_after_full_chunk", False), ("controller_mode", "async"),
])
def test_partial_replaced_or_modified_reference_execution_is_rejected(reference_evidence, field, value):
    reference_evidence[2]["reference_execution"][field] = value
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("field,value", [
    ("accepted_steps", 29), ("completed_chunks_at_start", 1), ("completed_steps_at_start", 1),
    ("actions_executed_during_prediction", 1), ("observation_age_at_return_s", 2.01),
    ("plan_dispatches", 29), ("planned_duration_s", 1.0), ("plan_deadline_monotonic_s", 103.0),
    ("plan_deadline_monotonic_s", 100.1), ("plan_deadline_monotonic_s", float("inf")),
])
def test_individual_request_must_preserve_full_chunk_order_and_fixed_lease(reference_evidence, field, value):
    reference_evidence[2]["prediction_samples"][0][field] = value
    assert not assessment(reference_evidence)["qualified"]


@pytest.mark.parametrize("field,value", [
    ("expired_prefix_dropped", 1), ("overlap_prefix_dropped", 1), ("commands_after_stop", 1),
    ("stop_requested_during_inflight_rpc", False), ("all_fake_robots_released", False),
    ("expired_before_dispatch", 1), ("failed", True),
])
def test_reference_keeps_stop_release_and_expiry_guards(reference_evidence, field, value):
    reference_evidence[2][field] = value
    assert not assessment(reference_evidence)["qualified"]


def test_reference_needs_fifty_warm_completed_chunks_not_fifty_rpc_returns(reference_evidence):
    proof = reference_evidence[2]["reference_execution"]
    reference_evidence[2]["prediction_samples"].pop(-2)
    proof.update(predicted_steps=1500, completed_steps=1500, completed_chunks=50, interpolation_dispatches=3000)
    reference_evidence[2]["executed_actions"] = 3000
    result = assessment(reference_evidence)
    assert not result["qualified"]
    assert result["completed_integrated_warm_samples"] == 49
    assert any("Insufficient completed warm" in reason for reason in result["reasons"])


def test_reference_rpc_and_observation_admission_keep_twenty_percent_margin(reference_evidence):
    for event in reference_evidence[2]["prediction_samples"]:
        event["observation_age_at_return_s"] = 1.61
    assert not assessment(reference_evidence)["qualified"]
    for event in reference_evidence[2]["prediction_samples"]:
        event["observation_age_at_return_s"] = 0.31
    for sample in reference_evidence[1]["samples"]:
        sample["round_trip_s"] = 1.61
    reference_evidence[1]["warm_round_trip_s"].update(p50=1.61, p95=1.61, p99=1.61)
    assert not assessment(reference_evidence)["qualified"]


def test_async_evidence_cannot_be_relabelled_as_reference(http_evidence):
    http_evidence[0]["controller_mode"] = "reference"
    assert not assessment(http_evidence)["qualified"]
    http_evidence[2]["policy_options"]["controller_mode"] = "reference"
    assert not assessment(http_evidence)["qualified"]


def test_modes_use_distinct_records_and_exact_settings(reference_evidence, monkeypatch, tmp_path):
    monkeypatch.setattr(q, "DATA_DIR", tmp_path)
    settings, direct, integrated = reference_evidence
    reference_path = q.save_qualification(q.build_qualification(settings, direct=direct, integrated=integrated))
    assert reference_path.name == "modal-molmoact2-reference.json"
    assert q._settings_path({**settings, "controller_mode": "async"}).name == "modal-molmoact2.json"
    external = {**settings, "backend": "external", "external_service_name": "lambda-georgia"}
    assert q._settings_path(external).name == "external-lambda-georgia-molmoact2-reference.json"
    with pytest.raises(q.QualificationError, match="settings changed"):
        q.validate_qualification({**settings, "controller_mode": "async"}, path=reference_path)


def test_collector_runs_all_warm_requests_with_bounded_reference_window(collection):
    c = collection
    result = collect_qualification(rig_path=c.path, call_mode="http", execution_mode="cuda_graph10",
                                   controller_mode="reference")
    assert c.calls["profile"]["warm_samples"] == c.calls["scenario"]["target_warm_samples"] == 50
    assert c.calls["scenario"]["duration"] == 1800
    assert c.calls["scenario"]["policy_options"]["controller_mode"] == "reference"
    assert c.calls["settings"]["controller_mode"] == "reference"
    assert c.calls["build"]["requested_warm_samples"] == 50
    assert Path(result["qualification_path"]).name == "modal-molmoact2-reference.json"
    assert json.loads(Path(result["qualification_path"]).read_text())["settings"]["controller_mode"] == "reference"


def test_failed_reference_collection_does_not_overwrite_async_record(collection):
    c = collection
    prior = q.save_qualification({"settings": {"profile": "molmoact2", "controller_mode": "async"},
                                 "assessment": {"qualified": True}, "sentinel": "async"})
    original = prior.read_bytes()
    c.direct.update(terminated="failure", warm_sample_count=0)
    result = collect_qualification(rig_path=c.path, call_mode="http", execution_mode="cuda_graph10",
                                   controller_mode="reference")
    assert not result["assessment"]["qualified"]
    assert prior.read_bytes() == original
    assert Path(result["qualification_path"]).name == "modal-molmoact2-reference.json"
    assert "scenario" not in c.calls


@pytest.mark.parametrize("options", [
    {"controller_mode": "unknown"}, {"controller_mode": "reference"},
    {"controller_mode": "reference", "call_mode": "http", "execution_mode": "cuda_graph10", "center_crop": True},
])
def test_bad_reference_selection_fails_before_service_access(monkeypatch, options):
    monkeypatch.setattr("yamkit.modal_ops.owned_service", lambda: pytest.fail("Service must not be accessed"))
    with pytest.raises(ValueError, match="controller|Reference"):
        collect_qualification(**options)


@pytest.mark.parametrize("controller_mode", ["async", "reference"])
def test_benchmark_waits_for_next_inflight_request_after_completed_sample_limit(monkeypatch, controller_mode):
    from scripts.benchmark_remote import run_scenario
    from yamkit.inference.client import InvalidatedRequest

    calls = []
    captured = {}

    class Transport:
        call_mode = "http"

        def __init__(self, stop):
            self.stop = stop

        def ready(self, timeout_s):
            return {"instance_id": "software-only-fixture"}

        def predict_chunk(self, request, timeout_s):
            calls.append(request["mode"])
            if calls.count("robot") == 3:
                assert self.stop.wait(1), "The target monitor did not stop the next in-flight request"
                raise InvalidatedRequest("software-only fixture stopped")
            return {"instance_id": "software-only-fixture"}

        def cancel(self):
            pass

    def rollout(cfg, shutdown_event):
        from yamkit.remote_policy.modeling_yamkit_remote import make_transport

        captured["cfg"] = cfg
        transport = make_transport(cfg.policy)
        transport.ready(1)
        request = {"task": cfg.task, "execution_mode": "cuda_graph10"}
        transport.predict_chunk({**request, "mode": "native_fixture"}, 1)
        transport.predict_chunk({**request, "mode": "robot"}, 1)
        transport.predict_chunk({**request, "mode": "robot"}, 1)
        # Finishing the last response cannot itself trigger Stop: reference mode
        # still needs to execute that entire response before asking again.
        time.sleep(0.12)
        assert not shutdown_event.is_set()
        with pytest.raises(InvalidatedRequest):
            transport.predict_chunk({**request, "mode": "robot"}, 1)
        return {"sample_count": 2}

    monkeypatch.setattr("yamkit.remote_rollout.run_remote_rollout", rollout)
    report = run_scenario("reference-stop-boundary", [0], duration=1, image_hw=(8, 8),
                          target_warm_samples=1, transport_factory=Transport,
                          policy_options={"call_mode": "http", "execution_mode": "cuda_graph10",
                                          "controller_mode": controller_mode})
    assert captured["cfg"].policy.controller_mode == controller_mode
    assert report["policy_options"]["controller_mode"] == controller_mode
    assert calls == ["native_fixture", "robot", "robot", "robot"]
    assert report["stop_requested_during_inflight_rpc"]
    assert report["commands_after_stop"] == 0


@pytest.mark.parametrize("duration", [0, -1, 1800.1, float("inf"), float("nan")])
def test_integrated_diagnostic_window_is_finite_and_bounded(duration):
    from scripts.benchmark_remote import run_scenario

    with pytest.raises(ValueError, match="duration"):
        run_scenario("invalid-window", [0], duration=duration, image_hw=(8, 8))
