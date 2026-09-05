"""Qualification evidence cannot bypass host, timing, Stop or mapping checks."""

import copy
import json

import pytest

from yamkit.inference import qualification as q


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr(q, "host_identity", lambda: {"hostname": "lenovo", "machine_fingerprint": "test-host"})
    monkeypatch.setattr(q, "is_cloud_host", lambda: False)
    settings = q.qualification_settings("molmoact2", modal_app="yamkit-vla-test", observed_region="us-west-1",
                                        image_encoding="jpeg")
    metadata = {"profile": settings["profile"], "model_revision": settings["model_revision"],
                "requested_compute_region": "us-west", "compute_region": "us-west-1",
                "routing_region": "us-west", "instance_id": "same-container"}
    direct = {"measurement": "real Modal RPC; generated fixtures", "readiness": metadata, "measurement_host": q.host_identity(),
              "image_hw": [480, 640], "image_encoding": "jpeg", "jpeg_quality": 85,
              "call_mode": "remote", "crop": "none", "warm_sample_count": 50,
              "terminated": "request_limit", "container_instance_count": 1,
              "warm_round_trip_s": {"p50": 0.3, "p95": 0.3, "p99": 0.3},
              "samples": [{"sequence_id": index, "instance_id": "same-container", "round_trip_s": 0.3}
                          for index in range(51)]}
    integrated = {"source": "final LeRobot worker; real Modal RPC", "readiness": metadata, "fps": 30, "chunk_steps": 30,
                  "measurement_host": q.host_identity(),
                  "image_hw": [480, 640], "policy_options": {"image_encoding": "jpeg", "jpeg_quality": 85,
                  "call_mode": "remote", "center_crop": False, "prediction_queue_threshold": 30},
                  "prediction_samples": [{"error": None, "accepted_steps": 10,
                                          "observation_age_at_return_s": 0.31,
                                          "remaining_valid_action_horizon_s": 0.69} for _ in range(51)],
                  "failed": False, "executed_actions": 500, "all_fake_robots_released": True,
                  "underruns": 0, "expired_chunks": 0, "expired_queued_actions": 0,
                  "expired_before_dispatch": 0, "minimum_execution_queue_depth": 8,
                  "stop_requested_during_inflight_rpc": True, "commands_after_stop": 0,
                  "failures": [{"reason": "InvalidatedRequest"}]}
    return settings, direct, integrated


def record(evidence):
    settings, direct, integrated = evidence
    return q.build_qualification(settings, direct=direct, integrated=integrated)


def test_host_bound_record_requires_recent_exact_settings(evidence, tmp_path, monkeypatch):
    value = record(evidence)
    path = q.save_qualification(value, tmp_path / "record.json")
    assert q.validate_qualification(evidence[0], path=path)["assessment"]["qualified"]
    with pytest.raises(q.QualificationError, match="expired"):
        q.validate_qualification(evidence[0], path=path, now=value["created_unix_s"] + q.MAX_AGE_S + 1)
    with pytest.raises(q.QualificationError, match="future"):
        q.validate_qualification(evidence[0], path=path, now=value["created_unix_s"] - 1)
    for key, changed in (("image_encoding", "rgb8"), ("jpeg_quality", 90), ("image_hw", [360, 640]),
                         ("call_mode", "spawn"), ("crop", "center_16_9"), ("model_revision", "new"),
                         ("observed_region", "us-east-1"), ("prediction_queue_threshold", 15)):
        with pytest.raises(q.QualificationError, match="settings changed"):
            q.validate_qualification({**evidence[0], key: changed}, path=path)
    monkeypatch.setattr(q, "host_identity", lambda: {"hostname": "cloud", "machine_fingerprint": "another"})
    with pytest.raises(q.QualificationError, match="another host"):
        q.validate_qualification(evidence[0], path=path)


@pytest.mark.parametrize("field,value", [
    ("underruns", 1), ("expired_chunks", 1), ("expired_queued_actions", 1),
    ("expired_before_dispatch", 1), ("commands_after_stop", 1),
    ("stop_requested_during_inflight_rpc", False), ("minimum_execution_queue_depth", 0),
    ("failed", True), ("executed_actions", 0), ("all_fake_robots_released", False),
])
def test_failed_queue_or_stop_cannot_be_qualified(evidence, tmp_path, field, value):
    evidence[2][field] = value
    result = record(evidence)
    assert not result["assessment"]["qualified"]
    result["assessment"]["qualified"] = True  # Never trust the cached assessment flag.
    path = q.save_qualification(result, tmp_path / "record.json")
    with pytest.raises(q.QualificationError):
        q.validate_qualification(evidence[0], path=path)


def test_qualification_needs_effective_horizon_margin_and_enough_actual_merges(evidence):
    for sample in evidence[1]["samples"]:
        sample["round_trip_s"] = 0.56
    evidence[1]["warm_round_trip_s"].update(p50=0.56, p99=0.56)
    evidence[1]["warm_round_trip_s"]["p95"] = 0.56
    result = record(evidence)
    assert result["assessment"]["effective_usable_action_horizon_s"] == pytest.approx(0.69)
    assert not result["assessment"]["qualified"]
    for sample in evidence[1]["samples"]:
        sample["round_trip_s"] = 0.3
    evidence[1]["warm_round_trip_s"].update(p50=0.3, p95=0.3, p99=0.3)
    evidence[2]["prediction_samples"].pop()
    assert not record(evidence)["assessment"]["qualified"]


def test_missing_warm_evidence_can_be_saved_but_cannot_qualify(evidence, tmp_path):
    evidence[1].update(warm_round_trip_s={}, warm_sample_count=0, readiness=None)
    evidence[2]["prediction_samples"] = []
    result = record(evidence)
    assert not result["assessment"]["qualified"]
    assert q.save_qualification(result, tmp_path / "failed.json").is_file()


def test_cloud_evidence_never_authorizes_local_motion_even_on_same_host(evidence, monkeypatch):
    from yamkit.inference import performance

    monkeypatch.setattr(q, "is_cloud_host", lambda: True)
    result = record(evidence)
    assert result["assessment"]["qualified"]
    assert result["status"] == "READY_FOR_LENOVO_QUALIFICATION"
    monkeypatch.setattr(performance, "QUALIFICATION_GATE_ENABLED", True)
    with pytest.raises(q.QualificationError, match="cloud"):
        performance.require_physical_modal_rollout(evidence[0], supervised_confirmed=True, mapping_accepted=True)


def test_mapping_and_supervised_confirmation_are_separate_required_guards(evidence, monkeypatch):
    from yamkit.inference import performance

    monkeypatch.setattr(performance, "QUALIFICATION_GATE_ENABLED", True)
    validated = []
    monkeypatch.setattr(q, "validate_qualification", lambda settings: validated.append(settings))
    for kwargs in ({}, {"supervised_confirmed": True}, {"mapping_accepted": True}):
        with pytest.raises(q.QualificationError, match="confirmation"):
            performance.require_physical_modal_rollout(evidence[0], **kwargs)
    assert not validated
    performance.require_physical_modal_rollout(evidence[0], supervised_confirmed=True, mapping_accepted=True)
    assert validated == [evidence[0]]


def test_malformed_record_does_not_authorize_motion(evidence, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(q.QualificationError):
        q.validate_qualification(evidence[0], path=path)
    altered = copy.deepcopy(evidence)
    altered[1]["readiness"]["compute_region"] = "us-east-1"
    assert not record(altered)["assessment"]["qualified"]


@pytest.mark.parametrize("malformation", ["missing_samples", "wrong_percentile", "missing_sequence",
                                         "duplicate_sequence", "another_instance"])
def test_raw_request_evidence_is_required_and_recomputed(evidence, malformation):
    direct = evidence[1]
    if malformation == "missing_samples":
        direct.pop("samples")
    elif malformation == "wrong_percentile":
        direct["warm_round_trip_s"]["p95"] = 0.1
    elif malformation == "missing_sequence":
        direct["samples"][3].pop("sequence_id")
    elif malformation == "duplicate_sequence":
        direct["samples"][3]["sequence_id"] = 2
    else:
        direct["samples"][3]["instance_id"] = "another"
    assert not record(evidence)["assessment"]["qualified"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 1.0, "3"])
@pytest.mark.parametrize("field", ["executed_actions", "minimum_execution_queue_depth", "commands_after_stop"])
def test_counters_require_finite_exact_integers(evidence, field, value):
    evidence[2][field] = value
    assert not record(evidence)["assessment"]["qualified"]


def test_short_returned_chunks_use_their_measured_usable_horizon(evidence):
    for event in evidence[2]["prediction_samples"]:
        event["remaining_valid_action_horizon_s"] = 0.35
    assessment = record(evidence)["assessment"]
    assert assessment["effective_usable_action_horizon_s"] == pytest.approx(0.35)
    assert not assessment["qualified"]


def test_nonfinite_json_is_rejected_before_assessment(evidence, tmp_path):
    value = record(evidence)
    value["integrated"]["minimum_execution_queue_depth"] = float("nan")
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(value))
    with pytest.raises(q.QualificationError, match="nonfinite"):
        q.validate_qualification(evidence[0], path=path)


def test_direct_upstream_proxy_construction_requires_validated_runner_scope(monkeypatch):
    from yamkit.inference import performance

    monkeypatch.setattr(performance, "QUALIFICATION_GATE_ENABLED", True)
    with pytest.raises(q.QualificationError, match="validated hardware"):
        q.require_runner_context()
    with q.validated_runner_context():
        q.require_runner_context()
    with pytest.raises(q.QualificationError):
        q.require_runner_context()


def test_imported_cloud_measurements_cannot_be_reissued_as_local_qualification(evidence):
    evidence[1]["measurement_host"] = {"hostname": "cloud", "machine_fingerprint": "cloud-host"}
    value = record(evidence)
    assert value["host"]["hostname"] == "lenovo"
    assert not value["assessment"]["qualified"]


@pytest.mark.parametrize("location", ["direct", "sample", "integrated"])
@pytest.mark.parametrize("experiment", [
    {"diagnostic_num_inference_steps": 5},
    {"diagnostic_num_inference_steps": 10},
    {"diagnostic_cuda_graph": True},
    {"experiment_only": True},
    {"model_execution": {"default_num_inference_steps": 10, "effective_num_inference_steps": 5}},
    {"model_execution": {"default_num_inference_steps": 5, "effective_num_inference_steps": 5}},
    {"model_execution": {"cuda_graph_used": True}},
])
def test_inference_experiments_cannot_qualify_default_policy(evidence, tmp_path, location, experiment):
    target = {"direct": evidence[1], "sample": evidence[1]["samples"][4],
              "integrated": evidence[2]["prediction_samples"][4]}[location]
    target.update(experiment)
    value = record(evidence)
    assert not value["assessment"]["qualified"]
    assert any("inference experiments" in reason for reason in value["assessment"]["reasons"])
    value["assessment"]["qualified"] = True
    path = q.save_qualification(value, tmp_path / "experiment.json")
    with pytest.raises(q.QualificationError, match="inference experiments"):
        q.validate_qualification(evidence[0], path=path)


def test_default_execution_metadata_remains_qualifiable(evidence):
    evidence[1].update(diagnostic_num_inference_steps=None, diagnostic_cuda_graph=None, experiment_only=False)
    for sample in evidence[1]["samples"]:
        sample["model_execution"] = {"default_num_inference_steps": 10, "effective_num_inference_steps": 10,
                                     "cuda_graph_enabled": False, "cuda_graph_used": False,
                                     "cuda_graph_supported": True, "cuda_graph_cache_populated_before": True}
    assert record(evidence)["assessment"]["qualified"]
