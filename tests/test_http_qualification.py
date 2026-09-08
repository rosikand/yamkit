"""Reassess HTTP evidence independently of the saved successful verdict."""

import copy
import json

import pytest

from tests import test_qualification
from yamkit.inference import identity
from yamkit.inference import qualification as q
from yamkit.inference.execution import request_execution_signature, signature_digest
from yamkit.inference.http_wire import MAX_MESSAGE_BYTES, WIRE_CODEC, WIRE_VERSION
from yamkit.inference.profiles import get_profile

TASK = "put the blue block in the black bin"
BUILD = "b" * 64
CACHE = "c" * 64
sdk_evidence = test_qualification.evidence


@pytest.fixture
def http_evidence(sdk_evidence, monkeypatch):
    monkeypatch.setattr(identity, "inference_build_id", lambda: BUILD)
    _, direct, integrated = copy.deepcopy(sdk_evidence)
    profile = get_profile("molmoact2")
    execution_identity = {
        "version": 1, "execution_mode": "cuda_graph10", "profile": profile.id,
        "model_revision": profile.revision, "model_dtype": "bfloat16", "num_inference_steps": 10,
        "cuda_graph": True, "chunk_size": 30, "action_width": 14,
        "parameter_dtype_numel": {"torch.bfloat16": 5442196208},
    }
    execution = {
        "configured_model_dtype": "bfloat16", "parameter_dtype_numel": {"torch.bfloat16": 5442196208},
        "production_cuda_graph_configured": True, "cuda_graph_enabled": True,
        "default_num_inference_steps": 10, "effective_num_inference_steps": 10,
        "cuda_graph_used": True, "execution_mode": "cuda_graph10", "execution_identity": execution_identity,
        "graph_cache_key_sha256": CACHE, "graph_capture_required": False,
    }
    signature = request_execution_signature({
        "mode": "native_fixture", "task": TASK, "state": [0.0] * 14,
        "images": {name: {"height": 480, "width": 640, "encoding": "rgb8"}
                   for name in profile.native_image_keys},
    }, profile)
    warmup = {"ready": True, "signature": signature, "signature_sha256": signature_digest(signature),
              "cache_key_sha256": CACHE, "generation": 1}
    metadata = {**direct["readiness"], "ready": True, "transport": "http", "execution_mode": "cuda_graph10",
                "inference_build_id": BUILD, "http_wire_version": WIRE_VERSION, "http_wire_codec": WIRE_CODEC,
                "execution_identity": execution_identity, "model_execution": execution, "graph_warmup": warmup}
    settings = q.qualification_settings(
        profile, modal_app="yamkit-vla-test", observed_region="us-west-1", call_mode="http",
        execution_mode="cuda_graph10", task=TASK, metadata=metadata,
        endpoint_url="https://yamkit-vla-test.modal.run",
    )
    raw_size = 3 * 480 * 640 * 3
    timing = {"call_mode": "http", "wire_version": WIRE_VERSION, "wire_codec": WIRE_CODEC,
              "wire_compression": "none", "wire_request_bytes": raw_size + 2048, "wire_response_bytes": 10000}
    row = {"instance_id": metadata["instance_id"], "task": TASK, "execution_mode": "cuda_graph10",
           "execution_identity": execution_identity, "model_execution": execution,
           "graph_warmup": warmup, "image_encoding": "rgb8", "image_hw": [480, 640],
           "payload_bytes": raw_size, "wire_payload_bytes": timing["wire_request_bytes"], "transport_timing": timing}
    for sample in direct["samples"]:
        sample.update(copy.deepcopy(row))
    direct["samples"][0]["model_execution"]["graph_capture_required"] = True
    direct.update(readiness=copy.deepcopy(metadata), execution_mode="cuda_graph10", task=TASK,
                  call_mode="http", image_encoding="rgb8", jpeg_quality=None)
    integrated["readiness"] = copy.deepcopy(metadata)
    integrated["samples"] = [copy.deepcopy(row) for _ in range(51)]
    integrated["policy_options"].update(call_mode="http", image_encoding="rgb8", jpeg_quality=None,
                                         execution_mode="cuda_graph10", task=TASK)
    return settings, direct, integrated


def build(evidence):
    settings, direct, integrated = evidence
    return q.build_qualification(settings, direct=direct, integrated=integrated)


def assert_rejected(evidence, tmp_path):
    record = build(evidence)
    assert record["assessment"]["qualified"] is False
    assert record["assessment"]["reasons"]
    # A file edited to claim success must still fail independent reassessment.
    record["assessment"]["qualified"] = True
    record["assessment"]["reasons"] = []
    record["status"] = "QUALIFIED_FOR_THIS_HOST"
    path = q.save_qualification(record, tmp_path / "qualification.json")
    with pytest.raises(q.QualificationError):
        q.validate_qualification(evidence[0], path=path)


def test_complete_http_graph_evidence_passes_after_json_round_trip(http_evidence, tmp_path):
    record = build(http_evidence)
    assert record["assessment"]["qualified"], record["assessment"]["reasons"]
    assert record["assessment"]["completed_integrated_warm_samples"] == 50
    assert record["hardware_tested"] is False
    path = q.save_qualification(record, tmp_path / "valid.json")
    assert q.validate_qualification(http_evidence[0], path=path)["assessment"]["qualified"]


def test_existing_sdk_evidence_still_passes_without_http_fields(sdk_evidence, tmp_path):
    record = build(sdk_evidence)
    assert record["assessment"]["qualified"]
    path = q.save_qualification(record, tmp_path / "sdk.json")
    assert q.validate_qualification(sdk_evidence[0], path=path)["assessment"]["qualified"]


def test_http_eager_requires_the_same_wire_evidence_without_graph_warmup(http_evidence, tmp_path):
    _, direct, integrated = http_evidence
    for report in (direct, integrated):
        for row in [report["readiness"], *report["samples"]]:
            row["execution_mode"] = "eager"
            row["execution_identity"].update(execution_mode="eager", cuda_graph=False)
            row["graph_warmup"] = {"ready": False}
            row["model_execution"].update(execution_mode="eager", production_cuda_graph_configured=False,
                                           cuda_graph_enabled=False, cuda_graph_used=False)
        report["execution_mode"] = "eager"
    integrated["policy_options"]["execution_mode"] = "eager"
    settings = q.qualification_settings(
        "molmoact2", modal_app="yamkit-vla-test", observed_region="us-west-1", call_mode="http",
        execution_mode="eager", task=TASK, metadata=direct["readiness"],
        endpoint_url="https://yamkit-vla-test.modal.run",
    )
    record = build((settings, direct, integrated))
    assert record["assessment"]["qualified"], record["assessment"]["reasons"]
    path = q.save_qualification(record, tmp_path / "eager.json")
    assert q.validate_qualification(settings, path=path)["assessment"]["qualified"]


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
@pytest.mark.parametrize("field,value", [
    ("inference_build_id", "old-build"), ("transport", "remote"),
    ("http_wire_version", 2), ("http_wire_version", True), ("http_wire_codec", "pickle"),
    ("execution_mode", "eager"), ("instance_id", "replacement-container"),
    ("ready", False), ("ready", 1),
    ("execution_identity", []), ("model_execution", []), ("graph_warmup", []),
])
def test_readiness_identity_cannot_be_replaced(http_evidence, tmp_path, report_index, field, value):
    http_evidence[report_index]["readiness"][field] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
@pytest.mark.parametrize("field,value", [
    ("default_num_inference_steps", 10.0), ("production_cuda_graph_configured", 1),
    ("parameter_dtype_numel", {"torch.bfloat16": 5442196208.0}),
])
def test_readiness_execution_types_must_be_exact(http_evidence, tmp_path, report_index, field, value):
    http_evidence[report_index]["readiness"]["model_execution"][field] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
@pytest.mark.parametrize("field,value", [
    ("ready", False), ("ready", 1), ("signature_sha256", "a" * 64),
    ("cache_key_sha256", "d" * 64), ("signature", {"task": "another task"}),
])
def test_readiness_warmup_must_match_current_task_shape_and_cache(http_evidence, tmp_path,
                                                                report_index, field, value):
    http_evidence[report_index]["readiness"]["graph_warmup"][field] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
@pytest.mark.parametrize("field,value", [("task", "a different instruction"), ("execution_mode", "eager")])
def test_report_options_cannot_claim_another_task_or_mode(http_evidence, tmp_path, report_index, field, value):
    report = http_evidence[report_index]
    target = report if report_index == 1 else report["policy_options"]
    target[field] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
@pytest.mark.parametrize("path,value", [
    (("instance_id",), "replacement"), (("task",), "another task"), (("execution_mode",), "eager"),
    (("execution_identity", "num_inference_steps"), 5), (("execution_identity", "version"), True),
    (("execution_identity", "parameter_dtype_numel", "torch.bfloat16"), 5442196208.0),
    (("model_execution", "cuda_graph_used"), False), (("model_execution", "cuda_graph_enabled"), False),
    (("model_execution", "effective_num_inference_steps"), 10.0),
    (("model_execution", "production_cuda_graph_configured"), False),
    (("model_execution", "configured_model_dtype"), "float16"),
    (("model_execution", "execution_identity", "num_inference_steps"), 5),
    (("model_execution", "graph_capture_required"), True),
    (("model_execution", "graph_cache_key_sha256"), "a" * 64),
    (("graph_warmup", "ready"), False), (("graph_warmup", "signature", "task"), "different task"),
    (("graph_warmup", "signature_sha256"), "a" * 64), (("graph_warmup", "cache_key_sha256"), "a" * 64),
    (("image_encoding",), "jpeg"), (("payload_bytes",), 600000),
    (("wire_payload_bytes",), 1), (("wire_payload_bytes",), True),
    (("wire_payload_bytes",), MAX_MESSAGE_BYTES + 1),
    (("transport_timing", "wire_request_bytes"), 42), (("transport_timing", "wire_response_bytes"), 0),
    (("transport_timing", "wire_response_bytes"), MAX_MESSAGE_BYTES + 1),
    (("transport_timing", "call_mode"), "remote"), (("transport_timing", "wire_codec"), "json-base64"),
    (("transport_timing", "wire_version"), True), (("transport_timing", "wire_compression"), "zlib"),
    (("diagnostic_cuda_graph",), True), (("diagnostic_num_inference_steps",), 10),
])
def test_every_raw_response_is_reassessed(http_evidence, tmp_path, report_index, path, value):
    target = http_evidence[report_index]["samples"][27]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
@pytest.mark.parametrize("field", ["samples", "readiness"])
@pytest.mark.parametrize("value", [None, 3, "malformed", []])
def test_malformed_http_collections_fail_cleanly(http_evidence, tmp_path, report_index, field, value):
    http_evidence[report_index][field] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
@pytest.mark.parametrize("field", [None, "graph_warmup", "model_execution", "transport_timing"])
@pytest.mark.parametrize("value", [None, 3, "malformed", []])
def test_malformed_http_rows_fail_cleanly(http_evidence, tmp_path, report_index, field, value):
    if field is None:
        http_evidence[report_index]["samples"][27] = value
    else:
        http_evidence[report_index]["samples"][27][field] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("field", ["prediction_samples", "failures", "policy_options"])
@pytest.mark.parametrize("value", [None, 3, "malformed", [None]])
def test_malformed_integrated_execution_evidence_fails_cleanly(http_evidence, tmp_path, field, value):
    http_evidence[2][field] = value
    assert_rejected(http_evidence, tmp_path)


@pytest.mark.parametrize("report_index", [1, 2], ids=["direct", "integrated"])
def test_short_execution_sample_lists_do_not_qualify(http_evidence, tmp_path, report_index):
    http_evidence[report_index]["samples"].pop()
    assert_rejected(http_evidence, tmp_path)


def test_requested_budget_needs_matching_raw_integrated_evidence(http_evidence, tmp_path):
    settings, direct, integrated = http_evidence
    direct["samples"].append({**copy.deepcopy(direct["samples"][-1]), "sequence_id": 51})
    direct["warm_sample_count"] = 51
    integrated["prediction_samples"].append(copy.deepcopy(integrated["prediction_samples"][-1]))
    record = q.build_qualification(settings, direct=direct, integrated=integrated, requested_warm_samples=51)
    assert not record["assessment"]["qualified"]
    assert record["assessment"]["completed_integrated_warm_samples"] == 51
    record["assessment"]["qualified"] = True
    path = q.save_qualification(record, tmp_path / "short.json")
    with pytest.raises(q.QualificationError, match="raw integrated HTTP"):
        q.validate_qualification(settings, path=path)


@pytest.mark.parametrize("value", [None, [], 10 ** 400])
def test_malformed_summary_fails_cleanly(http_evidence, tmp_path, value):
    http_evidence[1]["warm_round_trip_s"] = value
    assert_rejected(http_evidence, tmp_path)


def test_source_update_invalidates_saved_record_without_trusting_cached_pass(http_evidence, tmp_path, monkeypatch):
    path = q.save_qualification(build(http_evidence), tmp_path / "old-build.json")
    monkeypatch.setattr(identity, "inference_build_id", lambda: "new-build")
    with pytest.raises(q.QualificationError, match="HTTP readiness"):
        q.validate_qualification(http_evidence[0], path=path)


@pytest.mark.parametrize("field,value", [
    ("task", "another task"), ("execution_mode", "eager"), ("inference_build_id", "old-build"),
    ("instance_id", "replacement"), ("graph_signature_sha256", "a" * 64),
    ("graph_cache_key_sha256", "a" * 64), ("http_endpoint", "https://replacement.modal.run"),
])
def test_saved_settings_cannot_be_relabelled(http_evidence, tmp_path, field, value):
    record = build(http_evidence)
    record["settings"] = {**record["settings"], field: value}
    path = q.save_qualification(record, tmp_path / "relabelled.json")
    with pytest.raises(q.QualificationError, match="settings changed"):
        q.validate_qualification(http_evidence[0], path=path)


def test_serialized_malformed_row_does_not_hide_behind_saved_pass(http_evidence, tmp_path):
    record = build(http_evidence)
    record["integrated"]["samples"][27] = "not an execution record"
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(record))
    with pytest.raises(q.QualificationError, match="Malformed"):
        q.validate_qualification(http_evidence[0], path=path)
