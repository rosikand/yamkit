"""HTTP qualification binds current code, actual execution and the warmed task."""

import copy

import pytest

from yamkit.inference import identity, qualification
from yamkit.inference.execution import request_execution_signature, signature_digest
from yamkit.inference.http_wire import WIRE_CODEC, WIRE_VERSION
from yamkit.inference.profiles import get_profile

TASK = "put the blue block into the black bin"
IMAGE_HW = (8, 12)


def runtime_metadata(*, mode="cuda_graph10", task=TASK, image_hw=IMAGE_HW, warmed=True):
    profile = get_profile("molmoact2")
    graph = mode == "cuda_graph10"
    counts = {"torch.bfloat16": 5442196208}
    signature = request_execution_signature({
        "mode": "native_fixture", "task": task, "crop": "none", "state": [0.0] * 14,
        "images": {name: {"height": image_hw[0], "width": image_hw[1], "encoding": "rgb8"}
                   for name in profile.native_image_keys}}, profile)
    return {**profile.metadata(), "ready": True, "fresh_chunk": True, "saved_processors": True,
            "instance_id": "same-container", "prediction_count": 57, "image_encoding": "rgb8",
            "supported_image_encodings": ["rgb8", "jpeg"], "transport": "http", "execution_mode": mode,
            "inference_build_id": identity.inference_build_id(), "http_wire_version": WIRE_VERSION,
            "http_wire_codec": WIRE_CODEC, "requested_compute_region": "us-west",
            "compute_region": "us-west-1", "routing_region": "us-west",
            "execution_identity": {"version": 1, "execution_mode": mode, "profile": profile.id,
                                   "model_revision": profile.revision, "model_dtype": "bfloat16",
                                   "num_inference_steps": 10, "cuda_graph": graph,
                                   "chunk_size": 30, "action_width": 14, "parameter_dtype_numel": counts},
            "model_execution": {"configured_model_dtype": "bfloat16", "default_num_inference_steps": 10,
                                "parameter_dtype_numel": counts, "production_cuda_graph_configured": graph,
                                "cuda_graph_enabled": graph},
            "graph_warmup": {"ready": warmed, "signature": signature, "signature_sha256": signature_digest(signature),
                             "cache_key_sha256": "c" * 64, "generation": 1}}


def binding(metadata, **kwargs):
    options = {"execution_mode": "cuda_graph10", "task": TASK, "image_hw": IMAGE_HW}
    options.update(kwargs)
    return identity.http_runtime_binding("molmoact2", metadata, **options)


@pytest.fixture(autouse=True)
def stable_build(monkeypatch):
    # Tests can run alongside source edits in other Conductor agents.
    monkeypatch.setattr(identity, "inference_build_id", lambda: "b" * 64)


def test_binding_accepts_exact_graph_contract_and_returns_detached_identity():
    metadata = runtime_metadata()
    result = binding(metadata)
    assert result["graph_signature_sha256"] == metadata["graph_warmup"]["signature_sha256"]
    assert result["graph_cache_key_sha256"] == "c" * 64
    assert result["instance_id"] == "same-container" and result["task"] == TASK
    result["execution_identity"]["parameter_dtype_numel"].clear()
    assert metadata["execution_identity"]["parameter_dtype_numel"]


@pytest.mark.parametrize("path,value", [
    (("inference_build_id",), "old-build"), (("transport",), "remote"),
    (("http_wire_version",), True), (("http_wire_version",), WIRE_VERSION + 1),
    (("http_wire_codec",), "other-codec"), (("execution_mode",), "eager"),
    (("execution_identity", "model_revision"), "other-weights"),
    (("execution_identity", "profile"), "smolvla"), (("execution_identity", "model_dtype"), "float32"),
    (("execution_identity", "num_inference_steps"), 5), (("execution_identity", "num_inference_steps"), 10.0),
    (("execution_identity", "cuda_graph"), False), (("execution_identity", "chunk_size"), 15),
    (("execution_identity", "action_width"), 7), (("execution_identity", "parameter_dtype_numel"), {"torch.float32": 5442196208}),
    (("model_execution", "default_num_inference_steps"), 5),
    (("model_execution", "configured_model_dtype"), "float32"),
    (("model_execution", "cuda_graph_enabled"), False),
    (("model_execution", "production_cuda_graph_configured"), False),
    (("instance_id",), ""), (("graph_warmup", "ready"), False),
    (("graph_warmup", "signature", "task"), "another task"),
    (("graph_warmup", "signature_sha256"), "d" * 64), (("graph_warmup", "cache_key_sha256"), "z" * 64),
])
def test_binding_rejects_stale_or_changed_runtime_metadata(path, value):
    metadata = runtime_metadata()
    target = metadata
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        binding(metadata)


@pytest.mark.parametrize("options", [{"task": "different task"}, {"image_hw": (12, 12)},
                                      {"crop": "center_16_9"}, {"image_encoding": "jpeg"}])
def test_binding_requires_the_actual_task_and_image_contract(options):
    with pytest.raises(ValueError):
        binding(runtime_metadata(), **options)


def test_only_initial_readiness_may_omit_warmup_and_eager_has_no_graph_proof():
    initial = runtime_metadata(warmed=False)
    assert binding(initial, require_warmup=False)["graph_signature_sha256"]
    with pytest.raises(ValueError, match="warmed"):
        binding(initial)
    eager = runtime_metadata(mode="eager", warmed=False)
    assert "graph_signature_sha256" not in binding(eager, execution_mode="eager")


@pytest.fixture
def http_evidence(monkeypatch):
    from tests.test_qualification import evidence

    _, direct, integrated = evidence.__wrapped__(monkeypatch)
    metadata = runtime_metadata(image_hw=(480, 640))
    settings = qualification.qualification_settings(
        "molmoact2", modal_app="yamkit-http-test", call_mode="http", observed_region="us-west-1",
        task=TASK, execution_mode="cuda_graph10", metadata=metadata, endpoint_url="https://yamkit-http.modal.run")
    direct.update(readiness=copy.deepcopy(metadata), task=TASK, execution_mode="cuda_graph10",
                  image_encoding="rgb8", call_mode="http", measurement="real Modal HTTP; synthetic native fixtures")
    integrated.update(readiness=copy.deepcopy(metadata), source="final LeRobot worker; real Modal HTTP")
    integrated["policy_options"].update(task=TASK, execution_mode="cuda_graph10", image_encoding="rgb8", call_mode="http")
    sample_fields = {"task": TASK, "instance_id": "same-container", "execution_mode": "cuda_graph10",
                     "execution_identity": metadata["execution_identity"], "graph_warmup": metadata["graph_warmup"],
                     "model_execution": {**metadata["model_execution"], "cuda_graph_used": True,
                                         "effective_num_inference_steps": 10, "execution_mode": "cuda_graph10",
                                         "execution_identity": metadata["execution_identity"],
                                         "graph_capture_required": False, "graph_cache_key_sha256": "c" * 64},
                     "wire_payload_bytes": 2765950, "payload_bytes": 3 * 480 * 640 * 3,
                     "image_encoding": "rgb8", "image_hw": [480, 640],
                     "transport_timing": {"call_mode": "http", "wire_codec": WIRE_CODEC,
                                          "wire_version": WIRE_VERSION, "wire_compression": "none",
                                          "wire_request_bytes": 2765950, "wire_response_bytes": 10000}}
    for row in direct["samples"]:
        row.update(copy.deepcopy(sample_fields))
    integrated["samples"] = copy.deepcopy(direct["samples"])
    return settings, direct, integrated


def assess(evidence):
    settings, direct, integrated = evidence
    return qualification.build_qualification(settings, direct=direct, integrated=integrated)["assessment"]


def test_current_http_graph_evidence_can_qualify_without_diagnostic_overrides(http_evidence):
    result = assess(http_evidence)
    assert result["qualified"], result["reasons"]


@pytest.mark.parametrize("report", [1, 2])
@pytest.mark.parametrize("path,value", [
    (("execution_mode",), "eager"), (("task",), "another task"), (("instance_id",), "new-container"),
    (("execution_identity", "num_inference_steps"), 5), (("model_execution", "cuda_graph_used"), False),
    (("model_execution", "effective_num_inference_steps"), 5),
    (("graph_warmup", "signature_sha256"), "e" * 64), (("graph_warmup", "cache_key_sha256"), "d" * 64),
    (("wire_payload_bytes",), None), (("diagnostic_cuda_graph",), True),
])
def test_every_direct_and_integrated_response_must_match_bound_execution(http_evidence, report, path, value):
    target = http_evidence[report]["samples"][25]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert not assess(http_evidence)["qualified"]


@pytest.mark.parametrize("changed", ["inference_build_id", "instance_id", "task", "cache_key_sha256"])
def test_evidence_readiness_cannot_switch_code_container_or_warmed_inputs(http_evidence, changed):
    metadata = http_evidence[2]["readiness"]
    if changed == "task":
        metadata["graph_warmup"]["signature"]["task"] = "different task"
    elif changed == "cache_key_sha256":
        metadata["graph_warmup"][changed] = "d" * 64
    else:
        metadata[changed] = "changed"
    assert not assess(http_evidence)["qualified"]
