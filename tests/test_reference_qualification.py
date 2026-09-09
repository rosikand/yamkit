"""Reference evidence must prove serial full chunks, separately from async."""

import copy
import json
import time
from pathlib import Path

import pytest

from tests import test_http_qualification, test_http_qualification_collection
from yamkit.inference import qualification as q
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
                      overlap_prefix_dropped=0, executed_actions=51 * 60)
    integrated["prediction_samples"] = [
        {"error": None, "accepted_steps": 30, "completed_chunks_at_start": index,
         "completed_steps_at_start": index * 30, "actions_executed_during_prediction": 0,
         "observation_age_at_return_s": 0.31, "prediction_started_monotonic_s": 100 + index * 3,
         "prediction_s": 0.31, "plan_dispatches": 60, "planned_duration_s": 2.0,
         "plan_deadline_monotonic_s": 102.6 + index * 3} for index in range(51)]
    integrated["reference_execution"] = {
        "controller_mode": "reference", "predicted_steps": 1530, "completed_steps": 1530,
        "completed_chunks": 51, "interpolation_dispatches": 3060,
        "uncompleted_steps_at_stop": 0, "expired_prefix_dropped": 0, "overlap_prefix_dropped": 0,
        "prefix_drop": 0, "expired_plans": 0, "coherence_violations": 0,
        "partial_chunk_at_stop": False, "next_observation_after_full_chunk": True,
    }
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
    reference_evidence[2]["prediction_samples"].pop()
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
