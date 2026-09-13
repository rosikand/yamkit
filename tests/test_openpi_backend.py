"""Official base startup remains isolated and preserves the other runtime argv."""

import errno
import json
import shlex
import socket
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import test_backend_lifecycle as shared
from yamkit import backend_workflow as backend
from yamkit.backend_workflow import WorkflowError

helpers, bootstrap, target = shared.helpers, shared.bootstrap, shared.target


@pytest.mark.parametrize("policy,module,interpreter", [
    ("molmoact2", "yamkit.inference.standalone_service", ".venv-inference/bin/python"),
    ("pi05-yam", "yamkit.pi05.service", ".venv-inference/bin/python"),
    ("pi05-base", "yamkit.openpi.service", "data/openpi/venv/bin/python"),
])
def test_bootstrap_selects_exact_isolated_runtime_without_substitution(bootstrap, monkeypatch, policy, module, interpreter):
    bootstrap.values["policy"] = policy
    monkeypatch.setattr(sys, "argv", ["bootstrap", json.dumps(bootstrap.values)])
    assert bootstrap.run()["status"] == "started"
    argv, kwargs = bootstrap.calls[0]
    assert argv[:3] == [str(bootstrap.root / interpreter), "-m", module]
    if policy == "pi05-base":
        assert argv[3:] == ["--root", str(bootstrap.root), "--service", "lambda-test", "--port",
                           str(bootstrap.values["port"]), "--token-file", "data/inference/token", "--statistics",
                           "data/openpi/yam/normalization.json", "--session-seconds", "28800"]
        assert kwargs["env"]["PYTHONPATH"] == str(bootstrap.root / "src")
        assert "--task" not in argv  # Actual task warms through authenticated native HTTP.
    else:
        expected = ["--service-id", "lambda-test", "--region", "Georgia", "--port",
                    str(bootstrap.values["port"]), "--token-file", "data/inference/token",
                    "--session-seconds", "28800", "--task", "task"]
        if policy == "molmoact2":
            expected += ["--provider", "lambda"]
        assert argv[3:] == expected


@pytest.mark.parametrize("policy,interpreter,uses_inference_env", [
    ("molmoact2", ".venv-inference/bin/python", True),
    ("pi05-yam", ".venv-inference/bin/python", True),
    ("pi05-base", "data/openpi/venv/bin/python", False),
])
def test_remote_start_wrapper_preserves_protected_environment(target, monkeypatch, policy, interpreter, uses_inference_env):
    target = replace(target, policy=policy)
    seen = []
    monkeypatch.setattr(backend, "_ssh", lambda _target, argv, **_kw: seen.append(argv) or '{"status":"started"}')
    assert backend._start_remote(target, "task") == "started"
    words = shlex.split(seen[0][1])
    assert interpreter in words
    assert ("data/inference/env.sh" in words) is uses_inference_env
    payload = json.loads(words[-1])
    assert payload["policy"] == policy
    assert payload["minimum_free_mib"] == backend.GPU_STARTUP_RESERVE_MIB[policy] + backend.GPU_HEADROOM_MIB
    assert "opaque" not in seen[0][1]


def test_openpi_authenticated_warm_reuse_never_starts_or_inspects_remote(target, monkeypatch):
    target = replace(target, policy="pi05-base")
    monkeypatch.setattr(backend, "assert_ui_idle", lambda: None)
    for name in ("_ssh", "_start_remote", "_check_remote_startup", "_listening"):
        monkeypatch.setattr(backend, name, lambda *_a, **_kw: pytest.fail("warm reuse must be non-mutating"))
    metadata = {"instance_id": "explicit fake native instance"}
    assert backend.connect_configured_runtime(target, "task", lambda: metadata) is metadata


def test_openpi_cold_start_once_and_original_forward_is_retained(target, monkeypatch):
    target = replace(target, policy="pi05-base")
    monkeypatch.setattr(backend, "assert_ui_idle", lambda: None)
    monkeypatch.setattr(backend, "_listening", lambda _: True)
    monkeypatch.setattr(backend, "_managed_forward_present", lambda _: True)
    starts, calls = [], []
    monkeypatch.setattr(backend, "_start_remote", lambda *_: starts.append(True) or "started")
    monkeypatch.setattr(backend, "_ssh", lambda *_a, **_kw: pytest.fail("do not duplicate a retained forward"))

    def probe():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("untrusted private fixture details")
        return {"instance_id": "new fake official base instance"}

    result = backend.connect_configured_runtime(target, "task", probe)
    assert result["instance_id"].startswith("new fake") and len(starts) == 1 and len(calls) == 2


def test_identity_failure_does_not_start_or_echo_private_remote_error(target, monkeypatch):
    target = replace(target, policy="pi05-base")
    monkeypatch.setattr(backend, "assert_ui_idle", lambda: None)
    monkeypatch.setattr(backend, "_start_remote", lambda *_: pytest.fail("rejected identity cannot trigger restart"))

    def probe():
        raise ValueError("fake_private_credential_never_logged")

    with pytest.raises(WorkflowError, match="identity") as caught:
        backend.connect_configured_runtime(target, "task", probe)
    assert "fake_private_credential" not in str(caught.value)


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    path = tmp_path / backend.CONFIG_RELATIVE
    path.parent.mkdir(parents=True)
    entry = {"service": "lambda-openpi", "endpoint": "http://127.0.0.1:8767",
             "token_file": "data/inference/openpi.token", "saved_observations": ["data/saved.npz"],
             "remote": {"repo": "/home/example/yamkit", "token_file": "data/inference/openpi.token", "region": "Georgia"}}
    value = {"version": 1, "backends": {"lambda": {"ssh": {"host": "ubuntu@gpu.example"},
                                                    "policies": {"pi05-base": entry}}}}

    def save():
        path.write_text(json.dumps(value))
        path.chmod(0o600)

    save()
    return SimpleNamespace(root=tmp_path, path=path, entry=entry, value=value, save=save)


def test_official_alias_loads_separate_config_with_paths_only(configured):
    target = backend.load_target("lambda", "pi05_base")
    assert target.policy == "pi05-base" and target.service == "lambda-openpi"
    assert target.saved_observations == (configured.root / "data/saved.npz",)
    assert target.token_file == configured.root / "data/inference/openpi.token"


@pytest.mark.parametrize("field,value", [("token_file", "../private"), ("saved_observations", ["../private"]),
                                      ("saved_observations", ["data/saved.npz"] * 51),
                                      ("endpoint", "http://public.example:8767"),
                                      ("bearer", "fake_secret_must_not_echo")])
def test_invalid_official_config_is_rejected_without_credentials_or_network(configured, field, value):
    configured.entry[field] = value
    configured.save()
    with pytest.raises(WorkflowError) as caught:
        backend.load_target("lambda", "pi05-base")
    assert "fake_secret" not in str(caught.value)


def test_policies_cannot_share_official_service_or_port(configured):
    configured.value["backends"]["lambda"]["policies"]["molmoact2"] = {
        "service": "lambda-ma2", "endpoint": configured.entry["endpoint"]}
    configured.save()
    with pytest.raises(WorkflowError, match="own service name and loopback port"):
        backend.load_target("lambda", "pi05-base")


def test_missing_explicit_official_configuration_cannot_fall_back(configured, monkeypatch):
    from yamkit import external_ops

    monkeypatch.setattr(external_ops, "owned_service", lambda *_a: pytest.fail("explicit missing config forbids fallback"))
    with pytest.raises(WorkflowError, match="Explicit backend configuration"):
        backend.load_target("lambda", "pi05-base", config=configured.root / "missing.json")


@pytest.fixture
def retired_http_port():
    """Real local HTTP-style active close, leaving only a TCP TIME_WAIT tuple."""
    with socket.socket() as listener:
        # Match ThreadingHTTPServer's normal server_bind behavior.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(1)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen()
        with socket.create_connection(("127.0.0.1", port), timeout=1) as client:
            accepted, _ = listener.accept()
            accepted.close()  # Server sends FIN first, so the server owns TIME_WAIT.
            assert client.recv(1) == b""
    address = "0100007F:" + format(port, "04X")
    rows = [row.split() for row in Path("/proc/net/tcp").read_text().splitlines()[1:]]
    assert any(row[1] == address and row[3] == "06" for row in rows)
    assert not any(row[1] == address and row[3] == "0A" for row in rows)
    with socket.socket() as plain:
        with pytest.raises(OSError) as occupied:
            plain.bind(("127.0.0.1", port))
        assert occupied.value.errno == errno.EADDRINUSE
    with socket.socket() as reusable:
        reusable.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        reusable.bind(("127.0.0.1", port))
    return port


@pytest.mark.parametrize("policy,expected", [("pi05-base", "started"), ("molmoact2", "listener_present"),
                                          ("pi05-yam", "listener_present")])
def test_only_official_base_prebind_ignores_retired_http_time_wait(bootstrap, retired_http_port, monkeypatch, policy, expected):
    bootstrap.values.update(policy=policy, port=retired_http_port)
    monkeypatch.setattr(sys, "argv", ["bootstrap", json.dumps(bootstrap.values)])
    assert bootstrap.run()["status"] == expected
    assert len(bootstrap.calls) == (1 if policy == "pi05-base" else 0)


@pytest.mark.parametrize("reuse_address", [False, True])
def test_official_base_prebind_never_claims_a_real_listener_even_if_reusable(bootstrap, monkeypatch, reuse_address):
    bootstrap.values["policy"] = "pi05-base"
    monkeypatch.setattr(sys, "argv", ["bootstrap", json.dumps(bootstrap.values)])
    monkeypatch.setitem(bootstrap.namespace, "memory_free_mib", lambda _: pytest.fail("live listener must block before GPU query"))
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse_address))
        listener.bind(("127.0.0.1", bootstrap.values["port"]))
        listener.listen()
        assert bootstrap.run()["status"] == "listener_present"
        assert not bootstrap.calls
