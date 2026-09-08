"""HTTP qualification collection propagates measured identities, never hardware."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_remote
from tests.test_benchmark_remote import fake_profile_transport
from yamkit.config import RigConfig
from yamkit.inference import qualification
from yamkit.inference.profiles import get_profile
from yamkit.modal_qualification import collect_qualification

TASK = "put the blue block in the black bin"


@pytest.mark.parametrize("ingress", ["asgi", "tunnel"])
def test_http_factory_uses_only_matching_endpoint_credential(monkeypatch, ingress):
    captured = {}
    token = "test-only-token-that-never-enters-a-public-report"
    endpoint = "https://owned-http.modal.run" if ingress == "asgi" else "https://owned-http.r5.modal.host"
    expires = None if ingress == "asgi" else 1234567890.0

    def credentials(app_name):
        assert app_name == "owned-http-app"
        return {"endpoint_url": endpoint, "token": token,
                **({"http_ingress": ingress, "http_session_expires_at": expires} if ingress == "tunnel" else {})}

    def transport(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return "http-transport"

    monkeypatch.setattr("yamkit.modal_ops.http_credentials", credentials)
    monkeypatch.setattr("yamkit.inference.http_transport.HttpTransport", transport)
    stop = object()
    assert benchmark_remote.make_benchmark_transport("owned-http-app", shutdown_event=stop,
                                                      call_mode="http") == "http-transport"
    assert captured == {"args": ("owned-http-app", "molmoact2"), "kwargs": {
        "endpoint_url": endpoint, "token": token, "shutdown_event": stop,
        "http_ingress": ingress, "http_session_expires_at": expires}}
    for kwargs in ({"uncached_handles": True}, {"sdk_metrics": {}}):
        with pytest.raises(ValueError, match="SDK"):
            benchmark_remote.make_benchmark_transport("owned-http-app", call_mode="http", **kwargs)


def graph_profile_transport(monkeypatch, *, changed_refresh=False):
    transport = fake_profile_transport(monkeypatch)
    ready, predict = transport.ready, transport.predict_chunk
    transport.call_mode = "http"
    transport.ready_count = 0

    def graph_ready(timeout_s):
        transport.ready_count += 1
        return {**ready(timeout_s), "transport": "http", "execution_mode": "cuda_graph10",
                "instance_id": "changed" if changed_refresh and transport.ready_count > 1 else "same",
                "execution_identity": {"mode": "cuda_graph10"},
                "graph_warmup": {"ready": transport.ready_count > 1, "signature_sha256": "shape-and-task-hash"}}

    def graph_predict(request, timeout_s):
        response = predict(request, timeout_s)
        transport.last_timing = {"wire_request_bytes": 12345, "wire_response_bytes": 2345}
        return {**response, "execution_mode": "cuda_graph10", "execution_identity": {"mode": "cuda_graph10"},
                "graph_warmup": {"ready": True, "signature_sha256": "shape-and-task-hash"}}

    transport.ready, transport.predict_chunk = graph_ready, graph_predict
    return transport


def test_direct_http_uses_production_mode_and_refreshes_same_instance_graph_readiness(monkeypatch):
    transport = graph_profile_transport(monkeypatch)
    report = benchmark_remote.profile_modal(transport, warm_samples=50, image_hw=(8, 8),
                                             execution_mode="cuda_graph10", task=TASK)
    assert transport.ready_count == 2
    assert report["terminated"] == "request_limit" and report["warm_sample_count"] == 50
    assert report["readiness"]["graph_warmup"]["ready"] is True
    assert report["readiness_s"] == report["post_warmup_readiness_s"] == 5
    assert report["first_request"]["round_trip_s"] == 1.5
    assert "real Modal HTTP" in report["measurement"] and not report["experiment_only"]
    assert report["task"] == TASK and report["execution_mode"] == "cuda_graph10"
    assert all(request["task"] == TASK and request["execution_mode"] == "cuda_graph10"
               and "diagnostic_cuda_graph" not in request and "diagnostic_num_inference_steps" not in request
               for request in transport.requests)
    for sample in report["samples"]:
        assert sample["wire_payload_bytes"] == 12345
        assert sample["task"] == TASK and sample["execution_mode"] == "cuda_graph10"
        assert sample["graph_warmup"]["ready"] is True and sample["execution_identity"]
        assert sample["model_execution"]["effective_num_inference_steps"] == 10
    assert transport.cancelled


def test_refresh_rejects_container_change_before_collecting_warm_samples(monkeypatch):
    transport = graph_profile_transport(monkeypatch, changed_refresh=True)
    report = benchmark_remote.profile_modal(transport, warm_samples=50, image_hw=(8, 8),
                                             execution_mode="cuda_graph10", task=TASK)
    assert report["terminated"] == "failure" and report["warm_sample_count"] == 0
    assert len(transport.requests) == 1 and transport.cancelled


@pytest.mark.parametrize("kwargs", [
    {"execution_mode": "unknown"}, {"execution_mode": "cuda_graph10", "image_encoding": "jpeg"},
    {"execution_mode": "cuda_graph10", "diagnostic_cuda_graph": True}, {"task": " "},
])
def test_bad_production_profile_options_fail_before_remote_readiness(monkeypatch, kwargs):
    transport = graph_profile_transport(monkeypatch)
    with pytest.raises(ValueError):
        benchmark_remote.profile_modal(transport, **kwargs)
    assert transport.ready_count == 0


@pytest.fixture
def collection(monkeypatch, tmp_path):
    rig_path = tmp_path / "rig.yaml"
    RigConfig(cameras={name: {"type": "opencv", "index_or_path": index, "height": 480,
                              "width": 640, "fps": 30}
                       for index, name in enumerate(("top", "left_wrist", "right_wrist"))}).save(rig_path)
    profile = get_profile("molmoact2")
    metadata = {**profile.metadata(), "instance_id": "same-runtime", "transport": "http",
                "execution_mode": "cuda_graph10", "inference_build_id": "current-build",
                "requested_compute_region": "us-west", "compute_region": "us-west4", "routing_region": "us-west",
                "graph_warmup": {"ready": True, "signature_sha256": "actual-shape-and-task"}}
    receipt = {"app_name": "owned-http-app", "status": "ready", "transport": "http",
               "http_endpoint": "https://owned-http.modal.run", "metadata": {"graph_warmup": {"ready": False}}}
    direct = {"terminated": "request_limit", "warm_sample_count": 50, "readiness": metadata}
    integrated = {"readiness": metadata, "warm_sample_count": 50}
    calls = {}

    def transport(*args, **kwargs):
        calls.setdefault("transport", []).append((args, kwargs))
        return object()

    def profile_modal(*args, **kwargs):
        calls["profile"] = kwargs
        return direct

    def scenario(*args, **kwargs):
        calls["scenario"] = kwargs
        kwargs["transport_factory"]("fake-stop-event")
        return integrated

    def settings(profile_arg, **kwargs):
        assert profile_arg == profile
        calls["settings"] = kwargs
        return {"profile": profile.id, **kwargs}

    def build(settings, **kwargs):
        # This isolated collector test substitutes assessment; production gates
        # have their own tests. No real qualification path is written here.
        calls["build"] = kwargs
        return {"settings": settings, **kwargs, "hardware_tested": False,
                "assessment": {"qualified": True}, "status": "QUALIFIED"}

    benchmark = SimpleNamespace(make_benchmark_transport=transport, profile_modal=profile_modal, run_scenario=scenario)
    monkeypatch.setattr("yamkit.modal_qualification._benchmark_module", lambda: benchmark)
    monkeypatch.setattr("yamkit.modal_ops.owned_service", lambda: receipt)
    monkeypatch.setattr("yamkit.modal_ops._ownership_lock", nullcontext)
    monkeypatch.setattr("yamkit.modal_ops._save", lambda value: calls.setdefault("saved_receipt", value))
    monkeypatch.setattr(qualification, "qualification_settings", settings)
    monkeypatch.setattr(qualification, "build_qualification", build)
    monkeypatch.setattr(qualification, "DATA_DIR", tmp_path)
    return SimpleNamespace(path=rig_path, calls=calls, receipt=receipt, metadata=metadata,
                           direct=direct, integrated=integrated)


def test_http_collection_forwards_actual_task_mode_app_and_warmed_identity(collection):
    c = collection
    result = collect_qualification(rig_path=c.path, call_mode="http", execution_mode="cuda_graph10", task=TASK)
    assert result["assessment"]["qualified"] and not result["hardware_tested"]
    assert c.calls["profile"]["execution_mode"] == c.calls["settings"]["execution_mode"] == "cuda_graph10"
    assert c.calls["profile"]["task"] == c.calls["scenario"]["task"] == c.calls["settings"]["task"] == TASK
    options = c.calls["scenario"]["policy_options"]
    assert options["modal_app"] == "owned-http-app" and options["call_mode"] == "http"
    assert options["task"] == TASK and options["execution_mode"] == "cuda_graph10"
    assert c.calls["saved_receipt"]["metadata"] == c.calls["settings"]["metadata"] == c.metadata
    assert c.calls["settings"]["endpoint_url"] == c.receipt["http_endpoint"]
    assert c.calls["build"]["direct"] is c.direct and c.calls["build"]["integrated"] is c.integrated
    assert c.calls["build"]["requested_warm_samples"] == 50
    assert len(c.calls["transport"]) == 2
    assert c.calls["transport"][1][1]["shutdown_event"] == "fake-stop-event"
    saved = json.loads(Path(result["qualification_path"]).read_text())
    assert saved["settings"]["task"] == TASK and saved["hardware_tested"] is False


def test_failed_direct_http_retires_prior_evidence_without_integrated_requests(collection):
    c = collection
    qualification.save_qualification({"settings": {"profile": "molmoact2"}, "assessment": {"qualified": True}})
    c.direct.update(terminated="failure", warm_sample_count=0)
    result = collect_qualification(rig_path=c.path, call_mode="http", execution_mode="cuda_graph10", task=TASK)
    assert result["status"] == "QUALIFICATION_FAILED" and not result["assessment"]["qualified"]
    assert result["settings"]["task"] == TASK and result["settings"]["execution_mode"] == "cuda_graph10"
    assert "scenario" not in c.calls and "saved_receipt" not in c.calls
    saved = json.loads(Path(result["qualification_path"]).read_text())
    assert saved["assessment"]["qualified"] is False


def test_service_changed_during_collection_does_not_overwrite_new_receipt(collection, monkeypatch):
    c = collection
    receipts = iter([c.receipt, {**c.receipt, "app_name": "another-app"}])
    monkeypatch.setattr("yamkit.modal_ops.owned_service", lambda: next(receipts))
    result = collect_qualification(rig_path=c.path, call_mode="http", execution_mode="cuda_graph10", task=TASK)
    assert not result["assessment"]["qualified"]
    assert "scenario" not in c.calls and "saved_receipt" not in c.calls


def test_integrated_observer_preserves_http_mode_task_app_and_last_readiness(monkeypatch):
    captured = {}

    class Transport:
        call_mode = "http"

        def __init__(self):
            self.last_timing = {"wire_request_bytes": 5678}
            self.ready_count = 0

        def ready(self, timeout_s):
            self.ready_count += 1
            return {"instance_id": "same", "graph_warmup": {"ready": self.ready_count > 1}}

        def predict_chunk(self, request, timeout_s):
            return {"instance_id": "same", "execution_mode": "cuda_graph10", "model_execution": {"steps": 10},
                    "execution_identity": {"mode": "cuda_graph10"}, "graph_warmup": {"ready": True}}

        def cancel(self):
            pass

    def rollout(cfg, shutdown_event):
        from yamkit.remote_policy.modeling_yamkit_remote import make_transport

        captured["cfg"] = cfg
        observed = make_transport(cfg.policy)
        assert observed.call_mode == "http"
        observed.ready(1)
        observed.predict_chunk({"mode": "native_fixture", "task": cfg.task, "execution_mode": "cuda_graph10"}, 1)
        observed.ready(1)
        return {"sample_count": 0}

    monkeypatch.setattr("yamkit.remote_rollout.run_remote_rollout", rollout)
    report = benchmark_remote.run_scenario("forwarding", [0], duration=1, task=TASK,
                                           transport_factory=lambda stop: Transport(),
                                           policy_options={"modal_app": "owned-http-app", "call_mode": "http",
                                                           "execution_mode": "cuda_graph10"})
    assert captured["cfg"].task == captured["cfg"].policy.task == TASK
    assert captured["cfg"].policy.modal_app == "owned-http-app"
    assert report["readiness"]["graph_warmup"]["ready"] is True
    assert report["transport_predictions"][0]["wire_payload_bytes"] == 5678
    assert report["transport_predictions"][0]["task"] == TASK
