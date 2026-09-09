"""External provenance adds no shortcuts to the real integrated qualification."""

import copy
import hashlib
from types import SimpleNamespace

import pytest

from scripts import benchmark_remote
from tests import test_http_qualification
from tests.test_external_ops import ENDPOINT, NAME, external_fields
from yamkit.config import RigConfig
from yamkit.inference import qualification as q
from yamkit.modal_qualification import collect_qualification

http_evidence = test_http_qualification.http_evidence
sdk_evidence = test_http_qualification.sdk_evidence


@pytest.fixture
def external_evidence(http_evidence):
    _, direct, integrated = copy.deepcopy(http_evidence)
    fields = external_fields()
    for report in (direct, integrated):
        for key in ("requested_compute_region", "compute_region", "routing_region"):
            report["readiness"].pop(key, None)
        report["readiness"].update(copy.deepcopy(fields))
        report["service_provenance"] = {"backend": "external", "transport": "http",
                                         "external_service": copy.deepcopy(fields["external_service"])}
        for row in report["samples"]:
            row["transport_timing"].update(http_ingress="ssh",
                                           http_session_expires_at=fields["http_session_expires_at"],
                                           http_endpoint_sha256=hashlib.sha256(ENDPOINT.encode()).hexdigest())
    direct["measurement"] = "real external HTTP; generated fixtures"
    integrated["source"] = "final LeRobot worker; real external HTTP"
    integrated["policy_options"].update(backend="external", external_service=NAME)
    settings = q.qualification_settings("molmoact2", backend="external", external_service=NAME,
                                        call_mode="http", execution_mode="cuda_graph10", task=direct["task"],
                                        metadata=direct["readiness"], endpoint_url=ENDPOINT)
    return settings, direct, integrated


def test_external_record_uses_own_path_and_honest_host_provenance(external_evidence, monkeypatch, tmp_path):
    settings, direct, integrated = external_evidence
    monkeypatch.setattr(q, "DATA_DIR", tmp_path)
    record = q.build_qualification(settings, direct=direct, integrated=integrated)
    assert record["assessment"]["qualified"], record["assessment"]["reasons"]
    path = q.save_qualification(record)
    assert path.name == "external-lambda-test-molmoact2.json"
    assert q.validate_qualification(settings)["assessment"]["qualified"]
    assert "requested_region" not in settings and "modal_app" not in settings
    assert settings["external_service"]["region_source"] == "operator_declared"
    monkeypatch.setattr(q, "host_identity", lambda: {"hostname": "lambda", "machine_fingerprint": "other"})
    with pytest.raises(q.QualificationError, match="another host"):
        q.validate_qualification(settings)


@pytest.mark.parametrize("change", ["missing_provenance", "other_host", "other_policy", "missing_expiry_timing",
                                      "underrun", "late_stop_command", "other_runtime"])
def test_external_provenance_route_and_existing_safety_guards_are_reassessed(external_evidence, change):
    settings, direct, integrated = external_evidence
    if change == "missing_provenance":
        direct.pop("service_provenance")
    elif change == "other_host":
        integrated["readiness"]["external_service"]["host_id"] = "e" * 64
    elif change == "other_policy":
        integrated["policy_options"]["external_service"] = "another"
    elif change == "missing_expiry_timing":
        direct["samples"][17]["transport_timing"].pop("http_session_expires_at")
    elif change == "underrun":
        integrated["underruns"] = 1
    elif change == "late_stop_command":
        integrated["commands_after_stop"] = 1
    else:
        integrated["readiness"]["runtime_provenance"]["gpu"]["name"] = "different GPU"
    assert not q.build_qualification(settings, direct=direct, integrated=integrated)["assessment"]["qualified"]


def test_external_ingress_cannot_be_relabelled_as_modal(external_evidence):
    _, direct, _ = external_evidence
    with pytest.raises(ValueError, match="provider"):
        q.qualification_settings("molmoact2", modal_app="yamkit-vla-test", observed_region="us-west-1",
                                 call_mode="http", execution_mode="cuda_graph10", task=direct["task"],
                                 metadata=direct["readiness"], endpoint_url=ENDPOINT)


@pytest.mark.parametrize("direct_fails", [False, True])
def test_collection_reuses_final_diagnostic_and_refreshes_exact_external_attachment(
        external_evidence, monkeypatch, tmp_path, direct_fails):
    settings, direct, integrated = external_evidence
    rig_path = tmp_path / "rig.yaml"
    RigConfig(cameras={name: {"type": "opencv", "index_or_path": index, "height": 480, "width": 640, "fps": 30}
                       for index, name in enumerate(("top", "left_wrist", "right_wrist"))}).save(rig_path)
    calls = {}
    receipt = {"status": "ready", "http_endpoint": ENDPOINT, "metadata": copy.deepcopy(direct["readiness"])}
    monkeypatch.setattr("yamkit.external_ops.owned_service", lambda name: receipt)
    monkeypatch.setattr("yamkit.external_ops.http_credentials", lambda name: {"token": "private"})
    monkeypatch.setattr("yamkit.external_ops.update_ready", lambda *args, **kwargs: calls.update(refresh=(args, kwargs)))
    monkeypatch.setattr(q, "DATA_DIR", tmp_path)

    def profile(*args, **kwargs):
        calls["direct"] = kwargs
        return direct

    def scenario(*args, **kwargs):
        calls["integrated"] = kwargs
        return integrated

    def transport(*args, **kwargs):
        calls["transport"] = (args, kwargs)
        return object()

    monkeypatch.setattr("yamkit.modal_qualification._benchmark_module", lambda: SimpleNamespace(
        make_benchmark_transport=transport, profile_modal=profile, run_scenario=scenario))
    if direct_fails:
        direct.update(terminated="failure", warm_sample_count=0)
    result = collect_qualification(backend="external", external_service=NAME, rig_path=rig_path,
                                   call_mode="http", execution_mode="cuda_graph10", task=settings["task"])
    if direct_fails:
        assert not result["assessment"]["qualified"]
        assert "integrated" not in calls and "refresh" not in calls
        assert result["qualification_path"].endswith("external-lambda-test-molmoact2.json")
        return
    assert result["assessment"]["qualified"], result["assessment"]["reasons"]
    assert calls["direct"]["backend"] == calls["transport"][1]["backend"] == "external"
    assert calls["integrated"]["policy_options"]["external_service"] == NAME
    assert calls["integrated"]["policy_options"]["backend"] == "external"
    assert calls["refresh"][1] == {"expected_instance_id": direct["readiness"]["instance_id"]}
    assert not result["hardware_tested"]


def test_external_transport_never_reads_modal_credentials(monkeypatch):
    captured = {}
    monkeypatch.setattr("yamkit.modal_ops.http_credentials", lambda name: pytest.fail("Modal credentials accessed"))
    monkeypatch.setattr("yamkit.external_ops.http_credentials", lambda name: {
        "endpoint_url": ENDPOINT, "token": "private", "http_ingress": "ssh", "http_session_expires_at": 12345})
    monkeypatch.setattr("yamkit.inference.http_transport.HttpTransport",
                        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    benchmark_remote.make_benchmark_transport(NAME, backend="external", call_mode="http")
    assert captured["args"] == (NAME, "molmoact2")
    assert captured["kwargs"]["http_ingress"] == "ssh"
    with pytest.raises(ValueError, match="HTTP"):
        benchmark_remote.make_benchmark_transport(NAME, backend="external", call_mode="remote")


def test_integrated_observer_preserves_external_backend_and_measured_provenance(monkeypatch):
    metadata = external_fields()
    captured = {}

    class Transport:
        call_mode = "http"

        def ensure_session_active(self):
            captured["session_guard_calls"] = captured.get("session_guard_calls", 0) + 1
            if captured.get("expired"):
                raise ValueError("expired session")

        def ready(self, timeout_s):
            return metadata

        def cancel(self):
            pass

    def rollout(cfg, shutdown_event):
        from yamkit.remote_policy.modeling_yamkit_remote import make_transport

        captured["cfg"] = cfg
        observed = make_transport(cfg.policy)
        observed.ready(1)
        observed.ensure_session_active()
        captured["expired"] = True
        with pytest.raises(ValueError, match="expired session"):
            observed.ensure_session_active()
        return {"sample_count": 0}

    monkeypatch.setattr("yamkit.remote_rollout.run_remote_rollout", rollout)
    result = benchmark_remote.run_scenario("external-forwarding", [0], duration=1,
                                           transport_factory=lambda stop: Transport(),
                                           policy_options={"backend": "external", "external_service": NAME,
                                                           "modal_app": "", "call_mode": "http",
                                                           "execution_mode": "cuda_graph10"})
    assert captured["cfg"].policy.backend == "external" and captured["cfg"].policy.external_service == NAME
    assert "real external HTTP" in result["source"] and "Modal" not in result["source"]
    assert result["service_provenance"]["external_service"] == metadata["external_service"]
    assert captured["session_guard_calls"] == 2
