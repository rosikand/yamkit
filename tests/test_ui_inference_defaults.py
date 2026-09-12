"""Inference form defaults discover local attachments without granting readiness."""

import copy
import json
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from yamkit import arm, external_ops, modal_ops
from yamkit.inference import identity, qualification
from yamkit.inference.profiles import get_profile
from yamkit.ui import camstream, server


@pytest.fixture
def defaults_ui(rig, tmp_path, monkeypatch):
    data = tmp_path / "data"
    monkeypatch.setattr(external_ops, "DATA_DIR", data)
    monkeypatch.setattr(qualification, "DATA_DIR", data)
    monkeypatch.setattr(identity, "inference_build_id", lambda: "b" * 64)
    monkeypatch.setattr(modal_ops, "owned_service", lambda: None)

    def prohibited(*args, **kwargs):
        pytest.fail("Default selection must not read credentials, start a session, or contact hardware/service")

    monkeypatch.setattr(external_ops, "http_credentials", prohibited)
    monkeypatch.setattr(external_ops, "_probe_ready", prohibited)
    monkeypatch.setattr(arm.YamArm, "connect", staticmethod(prohibited))
    monkeypatch.setattr(camstream._Camera, "ensure_running", prohibited)
    manager = server.SessionManager()
    monkeypatch.setattr(manager, "start", prohibited)
    app = server.create_app(rig.path, outputs_dir=tmp_path / "outputs", session_manager=manager)
    with TestClient(app) as client:
        yield SimpleNamespace(data=data, client=client)


def write_attachment(ui, name="retained-gpu", *, created=None, expiry=None):
    now = time.time()
    created = now - 60 if created is None else created
    expiry = now + 3600 if expiry is None else expiry
    profile = get_profile("molmoact2")
    metadata = {"instance_id": "current-instance", "inference_build_id": "b" * 64,
                "execution_identity": {"execution_mode": "cuda_graph10"},
                "external_service": {"service_id": name}, "runtime_provenance": {"python": "3.12.12"},
                "http_endpoint": "http://127.0.0.1:8765", "http_session_expires_at": expiry,
                "graph_warmup": {"signature_sha256": "d" * 64, "cache_key_sha256": "e" * 64}}
    receipt = {"schema_version": 1, "name": name, "backend": "external", "status": "ready",
               "profile_id": "molmoact2", "revision": profile.revision, "transport": "http",
               "execution_mode": "cuda_graph10", "attached_at": created - 60,
               "http_endpoint": metadata["http_endpoint"], "http_session_expires_at": expiry,
               "metadata": metadata, "token": "DO_NOT_EXPOSE_RECEIPT_EXTRA"}
    directory = ui.data / "inference" / "external" / name
    directory.mkdir(parents=True, mode=0o700)
    receipt_path = directory / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    receipt_path.chmod(0o600)
    # The endpoint must not inspect this file, even when the receipt is expired.
    (directory / "http-auth.json").write_text("THIS_IS_NOT_JSON_AND_MUST_NOT_BE_READ")
    settings = {"backend": "external", "external_service_name": name, "profile": profile.id,
                "model_revision": profile.revision, "controller_mode": "reference", "call_mode": "http",
                "execution_mode": "cuda_graph10", "image_encoding": "rgb8", "crop": "none",
                "image_hw": [480, 640], "fps": 30, "reference_contract": qualification.REFERENCE_CONTRACT,
                "graph_signature_sha256": "d" * 64, "graph_cache_key_sha256": "e" * 64,
                "task": "put the red cube into the green bowl", **metadata}
    record = {"schema_version": 1, "hardware_tested": False, "host": qualification.host_identity(),
              "created_unix_s": created, "settings": settings, "assessment": {"qualified": True},
              "token": "DO_NOT_EXPOSE_QUALIFICATION_EXTRA"}
    record_path = ui.data / "qualifications" / f"external-{name}-molmoact2-reference.json"
    record_path.parent.mkdir(exist_ok=True)
    record_path.write_text(json.dumps(record))
    return receipt_path, record_path, receipt, record


def test_no_attachment_keeps_local_selection_and_does_not_create_data(defaults_ui):
    response = defaults_ui.client.get("/api/inference/profiles")
    assert response.status_code == 200
    body = response.json()
    assert body["default_backend"] == body["defaults"]["backend"] == "local"
    assert body["defaults_source"] == "local" and body["external_services"] == []
    assert not defaults_ui.data.exists()


def test_current_reference_selection_is_sanitized_and_never_approves(defaults_ui):
    receipt_path, record_path, _, record = write_attachment(defaults_ui)
    before = [(path.stat().st_mtime_ns, path.read_bytes()) for path in (receipt_path, record_path)]
    response = defaults_ui.client.get("/api/inference/profiles")
    assert response.status_code == 200
    body = response.json()
    defaults = body["defaults"]
    assert body["defaults_source"] == "external_reference_qualification"
    assert body["default_backend"] == defaults["backend"] == "external"
    assert defaults["external_service"] == "retained-gpu"
    assert defaults["task"] == record["settings"]["task"]
    assert defaults["controller_mode"] == "reference"
    assert defaults["async_chunks"] is False
    assert defaults["call_mode"] == "http" and defaults["execution_mode"] == "cuda_graph10"
    assert defaults["image_encoding"] == "rgb8" and defaults["center_crop"] is False
    assert defaults["duration"] == 60 and defaults["fps"] == 30
    assert defaults["arms"] == ["left_follower", "right_follower"]
    assert defaults["mapping_accepted"] is defaults["supervised_confirmed"] is defaults["capture_trace"] is False
    assert defaults["upload_repo_id"] is None
    assert "confirm_motion" not in defaults
    assert "ready" not in body and "qualified" not in response.text
    assert "DO_NOT_EXPOSE" not in response.text and "THIS_IS_NOT_JSON" not in response.text
    assert body["external_services"][0]["reference_qualification"]["expired"] is False
    assert before == [(path.stat().st_mtime_ns, path.read_bytes()) for path in (receipt_path, record_path)]


@pytest.mark.parametrize("stale", ["session", "qualification", "future"])
def test_stale_records_are_only_form_suggestions_never_ready(defaults_ui, stale):
    now = time.time()
    write_attachment(defaults_ui, created=now + 100 if stale == "future" else (
        now - qualification.MAX_AGE_S - 1 if stale == "qualification" else now - 60),
        expiry=now - 1 if stale == "session" else now + 3600)
    body = defaults_ui.client.get("/api/inference/profiles").json()
    assert body["defaults"]["task"] == "put the red cube into the green bowl"
    assert body["external_services"][0]["reference_qualification"]["expired"] is True
    assert "ready" not in body and body["defaults"]["supervised_confirmed"] is False


@pytest.mark.parametrize("change", ["instance", "host", "controller", "endpoint", "build", "task_warmup",
                                    "malformed", "symlink"])
def test_unrelated_or_invalid_record_cannot_supply_task(defaults_ui, change):
    receipt_path, path, receipt, record = write_attachment(defaults_ui)
    if change == "instance":
        record["settings"]["instance_id"] = "old-instance"
    elif change == "host":
        record["host"] = {"hostname": "another-host"}
    elif change == "controller":
        record["settings"]["controller_mode"] = "async"
    elif change == "endpoint":
        record["settings"]["http_endpoint"] = "http://127.0.0.1:9999"
    elif change == "build":
        record["settings"]["inference_build_id"] = "c" * 64
        receipt["metadata"]["inference_build_id"] = "c" * 64
        receipt_path.write_text(json.dumps(receipt))
    elif change == "task_warmup":
        record["settings"]["graph_signature_sha256"] = "f" * 64
    path.write_text("not json" if change == "malformed" else json.dumps(record))
    if change == "symlink":
        actual = path.with_suffix(".original")
        path.rename(actual)
        path.symlink_to(actual)
    body = defaults_ui.client.get("/api/inference/profiles").json()
    assert body["defaults_source"] == "external_attachment"
    assert body["defaults"]["external_service"] == "retained-gpu"
    assert body["defaults"]["task"] == ""
    assert body["external_services"][0]["reference_qualification"] is None


def test_current_reference_preferred_over_newer_unqualified_attachment(defaults_ui):
    write_attachment(defaults_ui, "current-reference", created=time.time() - 600)
    _, path, _, record = write_attachment(defaults_ui, "new-but-unqualified")
    other = copy.deepcopy(record)
    other["settings"]["instance_id"] = "old-instance"
    path.write_text(json.dumps(other))
    body = defaults_ui.client.get("/api/inference/profiles").json()
    assert body["defaults"]["external_service"] == "current-reference"
    assert len(body["external_services"]) == 2


@pytest.mark.parametrize("invalid", ["detached", "public", "symlink"])
def test_invalid_attachments_are_not_offered(defaults_ui, invalid):
    path, _, receipt, _ = write_attachment(defaults_ui)
    if invalid == "detached":
        receipt["status"] = "detached"
        path.write_text(json.dumps(receipt))
    elif invalid == "public":
        path.chmod(0o644)
    else:
        actual = path.with_suffix(".original")
        path.rename(actual)
        path.symlink_to(actual)
    body = defaults_ui.client.get("/api/inference/profiles").json()
    assert body["defaults_source"] == "local" and body["external_services"] == []
