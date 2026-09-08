"""Private HTTP credentials and owned lifecycle; no tests contact Modal."""

import json
import os
import traceback
from types import SimpleNamespace

import pytest

from yamkit import modal_ops
from yamkit.inference.http_wire import WIRE_CODEC, WIRE_VERSION
from yamkit.inference.profiles import get_profile

TOKEN = "test-only-independent-http-token-0123456789"
ENDPOINT = "https://owned-http.modal.run"
APP = "yamkit-vla-owned"


@pytest.fixture
def paths(monkeypatch, tmp_path):
    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned-service.json")
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected cloud command"))
    return tmp_path


def owned(**kwargs):
    profile = get_profile("molmoact2")
    return {"app_name": APP, "app_id": "ap-owned", "status": "ready", "transport": "http",
            "http_endpoint": ENDPOINT, "execution_mode": "cuda_graph10", "profile_id": profile.id,
            "revision": profile.revision, "gpu": "H100!", "region": "us-west", "routing_region": "us-west",
            "memory_mib": 65536, "cache_volume_name": "yamkit-policy-weights", "deployment_started": True,
            **kwargs}


def save_owned(**kwargs):
    receipt = owned(**kwargs)
    modal_ops._save(receipt)
    modal_ops._save_http_auth(APP, ENDPOINT, TOKEN)
    return receipt


def test_private_atomic_credential_file_is_separate_from_public_receipt(paths):
    save_owned()
    path = modal_ops.http_auth_path()
    assert path.stat().st_mode & 0o777 == 0o600
    assert modal_ops.http_credentials(APP) == {"app_name": APP, "endpoint_url": ENDPOINT, "token": TOKEN}
    assert TOKEN not in modal_ops.receipt_path().read_text()
    assert not list(paths.glob(".http-auth-*.tmp"))


@pytest.mark.parametrize("alter", ["world_readable", "symlink", "directory", "fifo", "oversize", "invalid_json",
                                  "invalid_utf8", "wrong_owner", "bad_token", "bad_url", "extra_field"])
def test_invalid_or_nonprivate_credentials_are_rejected_without_data_leak(paths, monkeypatch, alter):
    save_owned()
    path = modal_ops.http_auth_path()
    if alter == "world_readable":
        path.chmod(0o644)
    elif alter in ("symlink", "directory", "fifo"):
        retained = paths / "retained-auth.json"
        path.rename(retained)
        if alter == "symlink":
            path.symlink_to(retained)
        elif alter == "directory":
            path.mkdir()
        else:
            os.mkfifo(path, 0o600)
    elif alter == "oversize":
        path.write_text("x" * 4097)
    elif alter == "invalid_json":
        path.write_text('{"token": "' + TOKEN)
    elif alter == "invalid_utf8":
        path.write_bytes(b"\xff\xff")
    elif alter == "wrong_owner":
        monkeypatch.setattr(modal_ops.os, "geteuid", lambda: path.stat().st_uid + 1)
    else:
        value = json.loads(path.read_text())
        value.update({"bad_token": {"token": "private-invalid"}, "bad_url": {"endpoint_url": "http://private.invalid"},
                      "extra_field": {"unexpected": "private-extra"}}[alter])
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError) as error:
        modal_ops.http_credentials(APP)
    rendered = "".join(traceback.format_exception(error.value))
    assert TOKEN not in rendered and "private-invalid" not in rendered and "private-extra" not in rendered


@pytest.mark.parametrize("receipt_change,credential_change,requested", [
    ({"status": "preparing"}, {}, APP), ({"transport": "sdk"}, {}, APP), ({}, {}, "yamkit-vla-other"),
    ({"http_endpoint": "https://different.modal.run"}, {}, APP),
    ({}, {"app_name": "yamkit-vla-other"}, APP), ({}, {"endpoint_url": "https://different.modal.run"}, APP),
])
def test_credentials_bind_exact_ready_app_and_endpoint(paths, receipt_change, credential_change, requested):
    save_owned(**receipt_change)
    if credential_change:
        path = modal_ops.http_auth_path()
        path.write_text(json.dumps({**json.loads(path.read_text()), **credential_change}))
    with pytest.raises(ValueError):
        modal_ops.http_credentials(requested)


def readiness(**changes):
    return {**get_profile("molmoact2").metadata(), "ready": True, "fresh_chunk": True, "saved_processors": True,
            "transport": "http", "execution_mode": "cuda_graph10", "inference_build_id": "current-build",
            "http_wire_version": WIRE_VERSION, "http_wire_codec": WIRE_CODEC,
            "supported_call_modes": ["remote", "http"], "graph_warmup": {"ready": False}, **changes}


def mock_service(monkeypatch, metadata):
    from yamkit.inference import identity, modal_service

    captured = {}
    monkeypatch.setattr(identity, "inference_build_id", lambda: "current-build")
    monkeypatch.setattr(modal_ops.secrets, "token_urlsafe", lambda _: TOKEN)
    monkeypatch.setattr(modal_ops, "call", lambda method, **kwargs: method())
    monkeypatch.setattr(modal_ops, "service_handle", lambda *args: SimpleNamespace(
        ready=lambda: metadata, http=SimpleNamespace(get_web_url=lambda: ENDPOINT)))

    def create_app(**kwargs):
        captured["factory"] = kwargs

        async def deploy(**kwargs):
            captured["deploy"] = kwargs

        return SimpleNamespace(app_id="ap-created", deploy=SimpleNamespace(aio=deploy))

    monkeypatch.setattr(modal_service, "create_app", create_app)
    monkeypatch.setattr("yamkit.inference.http_transport.HttpTransport", lambda *a, **k:
                        pytest.fail("Endpoint validation must not construct a client"))
    return captured


def test_prepare_stores_only_dedicated_token_privately_and_does_not_require_graph_warmup(paths, monkeypatch):
    captured = mock_service(monkeypatch, readiness())
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-account-secret-never-forward")
    result = modal_ops.prepare("molmoact2", gpu="H100!", transport="http", execution_mode="cuda_graph10")
    assert captured["factory"]["http_token"] == TOKEN
    assert captured["factory"]["transport"] == "http" and captured["factory"]["execution_mode"] == "cuda_graph10"
    assert "MODAL_TOKEN_SECRET" not in captured["factory"]
    assert result["status"] == "ready" and result["http_endpoint"] == ENDPOINT
    assert result["metadata"]["graph_warmup"]["ready"] is False
    assert modal_ops.http_credentials(result["app_name"])["token"] == TOKEN
    assert TOKEN not in json.dumps(result) and TOKEN not in modal_ops.receipt_path().read_text()
    assert "test-account-secret-never-forward" not in json.dumps(result)


def test_reuse_verifies_current_build_wire_and_execution_without_second_pool(paths, monkeypatch):
    save_owned()
    captured = mock_service(monkeypatch, readiness())
    result = modal_ops.prepare("molmoact2", gpu="H100!", transport="http", execution_mode="cuda_graph10")
    assert result["app_name"] == APP and "factory" not in captured
    assert result["metadata"] == readiness()
    assert modal_ops.http_credentials(APP)["token"] == TOKEN


@pytest.mark.parametrize("changes", [
    {"transport": "sdk"}, {"execution_mode": "eager"}, {"inference_build_id": "stale-build"},
    {"http_wire_version": 2}, {"http_wire_version": True}, {"http_wire_codec": "unknown"},
    {"supported_call_modes": ["remote"]},
])
def test_reuse_rejects_stale_runtime_and_preserves_owned_service_and_auth(paths, monkeypatch, changes):
    prior = save_owned()
    captured = mock_service(monkeypatch, readiness(**changes))
    with pytest.raises(ValueError, match="HTTP runtime"):
        modal_ops.prepare("molmoact2", gpu="H100!", transport="http", execution_mode="cuda_graph10")
    assert modal_ops.owned_service() == prior and "factory" not in captured
    assert modal_ops.http_credentials(APP)["token"] == TOKEN


@pytest.mark.parametrize("credential_app", [APP, "yamkit-vla-other"])
def test_successful_shutdown_retires_only_exact_owned_app_credential(paths, monkeypatch, credential_app):
    save_owned()
    modal_ops._save_http_auth(credential_app, ENDPOINT, TOKEN)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv[3:])
        assert "YAMKIT_HTTP_TOKEN" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="[]")

    monkeypatch.setenv("YAMKIT_HTTP_TOKEN", "private-endpoint-env-token")
    monkeypatch.setattr(modal_ops.subprocess, "run", run)
    result = modal_ops.shutdown()
    assert result["status"] == "stopped" and result["remaining_containers"] == 0
    assert calls == [["app", "stop", "--yes", "ap-owned"],
                     ["container", "list", "--app-id", "ap-owned", "--json"]]
    assert modal_ops.http_auth_path().exists() is (credential_app != APP)


def test_unverified_shutdown_keeps_credential_for_retry(paths, monkeypatch):
    save_owned()
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="[]"))
    with pytest.raises(RuntimeError, match="unverified"):
        modal_ops.shutdown()
    assert modal_ops.owned_service()["status"] == "shutdown_unverified"
    assert modal_ops.http_auth_path().exists()


@pytest.mark.parametrize("status", ["stopped", "preparing"])
def test_cleanup_after_confirmed_or_never_invoked_deployment_needs_no_cloud(paths, status):
    save_owned(status=status, app_id=None, deployment_started=False)
    assert modal_ops.shutdown()["status"] == "stopped"
    assert not modal_ops.http_auth_path().exists()


def test_bad_fresh_http_readiness_stops_its_app_without_erasing_other_credentials(paths, monkeypatch):
    modal_ops._save_http_auth("yamkit-vla-other", ENDPOINT, TOKEN)
    mock_service(monkeypatch, readiness(http_wire_codec="wrong"))
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="[]"))
    with pytest.raises(ValueError, match="HTTP runtime"):
        modal_ops.prepare("molmoact2", gpu="H100!", transport="http", execution_mode="cuda_graph10")
    assert modal_ops.owned_service()["status"] == "stopped"
    assert json.loads(modal_ops.http_auth_path().read_text())["app_name"] == "yamkit-vla-other"


def test_sdk_prepare_keeps_existing_factory_contract_and_no_http_token(paths, monkeypatch):
    metadata = {**get_profile("molmoact2").metadata(), "ready": True, "fresh_chunk": True, "saved_processors": True}
    captured = mock_service(monkeypatch, metadata)
    result = modal_ops.prepare("molmoact2")
    assert result["status"] == "ready" and result["transport"] == "sdk"
    assert not {"http_token", "transport", "execution_mode"} & captured["factory"].keys()
    assert not modal_ops.http_auth_path().exists()
