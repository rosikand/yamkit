"""External attachments prove identity using readiness only and keep secrets private."""

import copy
import json
import stat
import time
from types import SimpleNamespace

import pytest

from tests.test_http_runtime_binding import runtime_metadata
from yamkit import external_ops as ops
from yamkit.inference import identity, qualification

NAME = "lambda-test"
ENDPOINT = "http://127.0.0.1:8765"
TOKEN = "test_only_private_bearer_" + "a" * 32


def external_fields():
    return {"http_ingress": "ssh", "http_endpoint": ENDPOINT,
            "http_session_expires_at": time.time() + 3600, "supported_call_modes": ["http"],
            "external_service": {"provider": "lambda", "service_id": NAME, "host_id": "d" * 64,
                                 "region": "Georgia", "region_source": "operator_declared"},
            "runtime_provenance": {"packages": {"lerobot": "0.6.1", "torch": "2.11.0+cu128",
                                                 "transformers": "5.5.4"},
                                   "python": "3.12.12", "torch_cuda": "12.8",
                                   "gpu": {"name": "NVIDIA H100 80GB HBM3"}}}


@pytest.fixture
def attachment(monkeypatch, tmp_path):
    monkeypatch.setattr(ops, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(identity, "inference_build_id", lambda: "b" * 64)
    metadata = json.loads(json.dumps({**runtime_metadata(), **external_fields()}))
    for key in ("requested_compute_region", "compute_region", "routing_region"):
        metadata.pop(key)
    token_file = tmp_path / "token"
    token_file.write_text(TOKEN + "\n")
    token_file.chmod(0o600)
    calls = []

    def probe(*args):
        calls.append(args)
        return copy.deepcopy(metadata)

    monkeypatch.setattr(ops, "_probe_ready", probe)
    return SimpleNamespace(metadata=metadata, token_file=token_file, calls=calls)


def attach(fixture):
    return ops.attach_service(NAME, ENDPOINT, fixture.token_file)


def test_attach_binds_identity_and_keeps_token_out_of_private_receipt(attachment):
    assert ops.owned_service(NAME) is None
    receipt = attach(attachment)
    assert attachment.calls == [(NAME, ENDPOINT, TOKEN)]
    assert receipt == ops.owned_service(NAME)
    assert TOKEN not in json.dumps(receipt)
    assert receipt["metadata"]["external_service"]["region_source"] == "operator_declared"
    for path in ops._directory(NAME).iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ops.http_credentials(NAME) == {"endpoint_url": ENDPOINT, "token": TOKEN, "http_ingress": "ssh",
                                          "http_session_expires_at": attachment.metadata["http_session_expires_at"]}
    assert len(attachment.calls) == 1  # status and credentials do not contact the server


@pytest.mark.parametrize("field,value", [
    ("inference_build_id", "other-source"), ("instance_id", ""), ("revision", "wrong-model"),
    ("http_session_expires_at", 1), ("http_endpoint", "http://127.0.0.1:8766"),
    ("runtime_provenance", {}), ("http_ingress", "asgi"),
])
def test_changed_source_model_route_or_runtime_cannot_attach(attachment, field, value):
    attachment.metadata[field] = value
    with pytest.raises(ValueError):
        attach(attachment)
    assert ops.owned_service(NAME) is None


@pytest.mark.parametrize("field,value", [("service_id", "other"), ("provider", "modal"),
                                          ("host_id", "unknown"), ("region_source", "provider_verified")])
def test_external_identity_must_match_explicit_attachment(attachment, field, value):
    attachment.metadata["external_service"][field] = value
    with pytest.raises(ValueError):
        attach(attachment)


@pytest.mark.parametrize("bad", ["http://localhost:8765", "https://remote.example", "http://127.0.0.1:8765/path"])
def test_noncanonical_endpoint_fails_before_readiness(attachment, bad):
    with pytest.raises(ValueError):
        ops.attach_service(NAME, bad, attachment.token_file)
    assert not attachment.calls


def test_permissions_symlinks_and_secret_echo_are_rejected(attachment, tmp_path):
    attachment.token_file.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        attach(attachment)
    attachment.token_file.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(attachment.token_file)
    with pytest.raises(ValueError):
        ops.attach_service(NAME, ENDPOINT, link)
    attachment.metadata["runtime_provenance"]["unexpected"] = TOKEN
    with pytest.raises(ValueError, match="private credential"):
        attach(attachment)
    assert ops.owned_service(NAME) is None


def test_detach_is_local_and_retires_credentials_without_stopping_server(attachment):
    attach(attachment)
    result = ops.detach_service(NAME)
    assert result["status"] == ops.owned_service(NAME)["status"] == "detached"
    assert len(attachment.calls) == 1
    assert not (ops._directory(NAME) / "http-auth.json").exists()
    with pytest.raises(ValueError, match="Attach"):
        ops.http_credentials(NAME)


def test_warmup_refresh_rejects_restart_or_host_change_and_preserves_receipt(attachment):
    receipt = attach(attachment)
    refreshed = copy.deepcopy(attachment.metadata)
    refreshed["prediction_count"] += 1
    assert ops.update_ready(NAME, refreshed, expected_instance_id="same-container")["metadata"] == refreshed
    for field, value in (("instance_id", "restarted"),
                         ("external_service", {**refreshed["external_service"], "host_id": "e" * 64})):
        bad = {**refreshed, field: value}
        with pytest.raises(ValueError, match="changed"):
            ops.update_ready(NAME, bad, expected_instance_id="same-container")
    assert ops.owned_service(NAME)["attachment_id"] == receipt["attachment_id"]
    assert ops.owned_service(NAME)["metadata"] == refreshed


def test_warmup_refresh_cannot_archive_private_credential(attachment):
    attach(attachment)
    bad = copy.deepcopy(attachment.metadata)
    bad["runtime_provenance"]["unexpected"] = TOKEN
    with pytest.raises(ValueError, match="private credential"):
        ops.update_ready(NAME, bad, expected_instance_id="same-container")
    assert TOKEN not in json.dumps(ops.owned_service(NAME))


def test_replaced_credentials_and_expired_receipts_cannot_be_used(attachment):
    receipt = attach(attachment)
    auth_path = ops._directory(NAME) / "http-auth.json"
    auth = json.loads(auth_path.read_text())
    ops._save(auth_path, {**auth, "attachment_id": "another"})
    with pytest.raises(ValueError, match="differ"):
        ops.http_credentials(NAME)
    ops._save(auth_path, auth)
    receipt["metadata"]["http_session_expires_at"] = 1
    ops._save(ops._directory(NAME) / "receipt.json", receipt)
    with pytest.raises(ValueError, match="unexpired"):
        ops.http_credentials(NAME)


def test_current_settings_binds_attachment_host_and_runtime(attachment):
    attach(attachment)
    options = SimpleNamespace(backend="external", external_service=NAME, profile="molmoact2", modal_app="",
                              call_mode="http", execution_mode="cuda_graph10", image_encoding="rgb8",
                              jpeg_quality=85, center_crop=False, prediction_queue_threshold=None,
                              task="put the blue block into the black bin")
    settings = qualification.current_settings(options, image_hw=(8, 12))
    assert settings["external_service_name"] == NAME
    assert "modal_app" not in settings and "observed_region" not in settings
    changed = copy.deepcopy(attachment.metadata)
    changed["external_service"]["host_id"] = "f" * 64
    with pytest.raises(ValueError, match="host or runtime"):
        qualification.current_settings(options, image_hw=(8, 12), metadata=changed)


def test_bootstrap_reads_only_ready_and_always_closes(monkeypatch):
    calls = []

    class Transport:
        def __init__(self, *args, **kwargs):
            assert kwargs["http_ingress"] == "ssh"
            assert 0 < kwargs["http_session_expires_at"] - time.time() <= 120

        def _invoke(self, method, payload, timeout):
            calls.append((method, payload, timeout))
            return {"ready": True}

        def close(self):
            calls.append("closed")

    monkeypatch.setattr("yamkit.inference.http_transport.HttpTransport", Transport)
    assert ops._probe_ready(NAME, ENDPOINT, TOKEN) == {"ready": True}
    assert calls == [("ready", None, 120), "closed"]
