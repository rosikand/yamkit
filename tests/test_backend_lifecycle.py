"""Fake bootstrap and kernel metadata checks; no real GPU service or robot may start."""

import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from yamkit import backend_workflow as backend


@pytest.fixture
def target(tmp_path):
    return backend.BackendTarget("lambda", "molmoact2", "lambda-test", "http://127.0.0.1:8765",
                                 tmp_path / "token", {"host": "gpu.example"},
                                 {"repo": str(tmp_path), "token_file": "data/inference/token",
                                  "region": "Georgia", "gpu": 0, "session_seconds": 28800})


@pytest.fixture
def helpers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {"root": tmp_path}
    exec(compile(backend._REMOTE_LIFECYCLE_HELPERS, "<fake-remote-helpers>", "exec"), namespace)  # noqa: S102 — fixed reviewed source
    return namespace


@pytest.fixture
def bootstrap(tmp_path, monkeypatch, capsys, helpers):
    inference = tmp_path / "data/inference"
    inference.mkdir(parents=True)
    (inference / "token").write_text("opaque-fixture-not-read")
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    values = {"service": "lambda-test", "policy": "molmoact2", "port": port,
              "token_file": "data/inference/token", "gpu": 0, "region": "Georgia",
              "session_seconds": 28800, "task": "task", "minimum_free_mib": 26624}
    calls = []

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        # Explicit fake: no subprocess is launched. This test owns its own lock fd.
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setitem(helpers, "memory_free_mib", lambda _: 60000)
    monkeypatch.setattr(sys, "argv", ["bootstrap", json.dumps(values)])

    def run():
        try:
            try:
                exec(compile(backend._REMOTE_BOOTSTRAP[len(backend._REMOTE_LIFECYCLE_HELPERS):],  # noqa: S102 — fixed source, fake Popen
                             "<fake-remote-bootstrap>", "exec"), helpers)
            except SystemExit as exited:
                assert exited.code == 0
            return json.loads(capsys.readouterr().out)
        finally:
            for name in ("fd", "gpufd", "logfd"):
                descriptor = helpers.pop(name, None)
                if descriptor is not None:
                    os.close(descriptor)

    return SimpleNamespace(run=run, calls=calls, values=values, namespace=helpers, root=tmp_path)


@pytest.mark.parametrize("raw,exit_code,expected", [("60000\n", 0, 60000), ("0\n", 0, 0),
                                                   ("N/A", 0, None), ("60000\n60000", 0, None),
                                                   ("-1", 0, None), ("60000", 1, None),
                                                   ("secret-shaped-fixture", 0, None)])
def test_gpu_query_is_single_device_bounded_and_only_returns_integer(helpers, monkeypatch, raw, exit_code, expected):
    seen = []

    def run(argv, **kwargs):
        seen.append((argv, kwargs))
        return SimpleNamespace(stdout=raw, stderr="unrelated-credential-shaped-fixture", returncode=exit_code)

    monkeypatch.setattr(subprocess, "run", run)
    assert helpers["memory_free_mib"](2) == expected
    assert seen[0][0] == ["nvidia-smi", "--id=2", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]
    assert seen[0][1]["timeout"] == 5


@pytest.mark.parametrize("error", [FileNotFoundError(), subprocess.TimeoutExpired("nvidia-smi", 5)])
def test_gpu_query_errors_fail_closed_without_echo(helpers, monkeypatch, error):
    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", fail)
    assert helpers["memory_free_mib"](0) is None


@pytest.mark.parametrize("free,status", [(26623, "gpu_memory_insufficient"), (None, "gpu_memory_unavailable")])
def test_low_or_unavailable_memory_starts_nothing(bootstrap, monkeypatch, free, status):
    monkeypatch.setitem(bootstrap.namespace, "memory_free_mib", lambda _: free)
    result = bootstrap.run()
    assert result["status"] == status
    assert not bootstrap.calls
    assert not (bootstrap.root / "data/inference/managed/lambda-test/startup.json").exists()


def test_cold_start_preserves_model_command_and_records_exact_owned_identity(bootstrap):
    result = bootstrap.run()
    assert result["status"] == "started"
    assert result["required_mib"] == 26624
    assert len(bootstrap.calls) == 1
    argv, kwargs = bootstrap.calls[0]
    assert argv[1:3] == ["-m", "yamkit.inference.standalone_service"]
    assert argv[-2:] == ["--provider", "lambda"]
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["start_new_session"] is True
    assert len(kwargs["pass_fds"]) == 1  # Only service lock; never the per-GPU startup lock.
    path = bootstrap.root / "data/inference/managed/lambda-test/startup.json"
    record = json.loads(path.read_text())
    assert record["pid"] == os.getpid() and record["start"] == result["start"]
    assert record["uid"] == os.geteuid()
    assert set(record) == {"version", "uid", "service", "policy", "gpu", "port", "pid", "start",
                           "lock_fd", "lock_inode", "lock_device"}
    assert path.stat().st_mode & 0o777 == 0o600
    assert "opaque-fixture" not in path.read_text()


def test_owned_cold_start_in_progress_blocks_other_policy_without_touching_receipt(bootstrap, monkeypatch):
    managed = bootstrap.root / "data/inference/managed"
    managed.mkdir()
    pending = managed / ".gpu-0-startup.json"
    saved = {"gpu": 0, "service": "lambda-pi05"}
    pending.write_text(json.dumps(saved))
    pending.chmod(0o600)
    monkeypatch.setitem(bootstrap.namespace, "record_state", lambda _: "alive")
    monkeypatch.setitem(bootstrap.namespace, "owns_listener", lambda _: False)
    result = bootstrap.run()
    assert result == {"status": "gpu_startup_busy", "service": "lambda-pi05"}
    assert json.loads(pending.read_text()) == saved
    assert not bootstrap.calls


@pytest.mark.parametrize("state,listener,status", [("alive", True, "started"), ("exited", False, "started"),
                                                 ("unverified", False, "gpu_reservation_unverified")])
def test_only_verified_warmed_or_exited_startup_releases_pending_admission(bootstrap, monkeypatch,
                                                                         state, listener, status):
    managed = bootstrap.root / "data/inference/managed"
    managed.mkdir()
    pending = managed / ".gpu-0-startup.json"
    pending.write_text('{"gpu":0,"service":"lambda-pi05"}')
    pending.chmod(0o600)
    original = pending.read_bytes()
    monkeypatch.setitem(bootstrap.namespace, "record_state", lambda _: state)
    monkeypatch.setitem(bootstrap.namespace, "owns_listener", lambda _: listener)
    result = bootstrap.run()
    assert result["status"] == status
    assert bool(bootstrap.calls) == (status == "started")
    if state == "unverified":
        assert pending.read_bytes() == original


def test_cross_gpu_metadata_cannot_release_wrong_gpu_reservation(bootstrap, monkeypatch):
    managed = bootstrap.root / "data/inference/managed"
    managed.mkdir()
    pending = managed / ".gpu-0-startup.json"
    pending.write_text('{"gpu":1,"service":"lambda-pi05"}')
    pending.chmod(0o600)
    monkeypatch.setitem(bootstrap.namespace, "record_state", lambda _: pytest.fail("wrong GPU is not inspected"))
    assert bootstrap.run()["status"] == "gpu_reservation_unverified"
    assert not bootstrap.calls


def test_child_already_exited_is_not_reported_as_successful_start(bootstrap, monkeypatch):
    monkeypatch.setitem(bootstrap.namespace, "process_stamp", lambda _: None)
    assert bootstrap.run()["status"] == "startup_exited"
    assert len(bootstrap.calls) == 1
    assert not (bootstrap.root / "data/inference/managed/lambda-test/startup.json").exists()


@pytest.mark.parametrize("kind,status", [("service", "starting_or_running"), ("gpu", "gpu_startup_busy")])
def test_parallel_bootstrap_locks_never_start_duplicate_models(bootstrap, monkeypatch, kind, status):
    managed = bootstrap.root / "data/inference/managed"
    (managed / "lambda-test").mkdir(parents=True)
    lock = managed / ("lambda-test/service.lock" if kind == "service" else ".gpu-0.lock")
    descriptor = bootstrap.namespace["private_fd"](lock, os.O_CREAT | os.O_RDWR)
    bootstrap.namespace["fcntl"].flock(descriptor, bootstrap.namespace["fcntl"].LOCK_EX)
    monkeypatch.setitem(bootstrap.namespace, "memory_free_mib", lambda _: pytest.fail("no admission query on lock conflict"))
    try:
        assert bootstrap.run()["status"] == status
        assert not bootstrap.calls
    finally:
        os.close(descriptor)


def test_existing_remote_listener_is_never_claimed_or_replaced(bootstrap, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", bootstrap.values["port"]))
        listener.listen()
        monkeypatch.setitem(bootstrap.namespace, "memory_free_mib", lambda _: pytest.fail("existing listener untouched"))
        assert bootstrap.run()["status"] == "listener_present"
        assert not bootstrap.calls


def test_real_exact_owned_lock_identity_and_listener_kernel_metadata(helpers, tmp_path):
    directory = tmp_path / "data/inference/managed/lambda-test"
    directory.mkdir(parents=True)
    lock = directory / "service.lock"
    descriptor = helpers["private_fd"](lock, os.O_CREAT | os.O_RDWR)
    helpers["fcntl"].flock(descriptor, helpers["fcntl"].LOCK_EX)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        stamp = helpers["process_stamp"](os.getpid())
        details = os.fstat(descriptor)
        record = {"version": 1, "uid": os.geteuid(), "service": "lambda-test", "policy": "molmoact2", "gpu": 0,
                  "port": listener.getsockname()[1], "pid": os.getpid(), "start": stamp["start"],
                  "lock_fd": descriptor, "lock_inode": details.st_ino, "lock_device": details.st_dev}
        try:
            assert helpers["record_state"](record) == "alive"
            assert helpers["owns_listener"](record)
            assert helpers["record_state"]({**record, "start": "0"}) == "exited"
            assert helpers["record_state"]({**record, "lock_inode": details.st_ino + 1}) == "unverified"
            assert helpers["record_state"]({**record, "uid": os.geteuid() + 1}) == "unverified"
            helpers["fcntl"].flock(descriptor, helpers["fcntl"].LOCK_UN)
            assert helpers["record_state"](record) == "unverified"
        finally:
            os.close(descriptor)


def test_zombie_original_identity_is_classified_exited(helpers, monkeypatch):
    record = {"version": 1, "uid": os.geteuid(), "service": "lambda-test", "policy": "molmoact2", "gpu": 0,
              "port": 8765, "pid": os.getpid(), "start": "1", "lock_fd": 3, "lock_inode": 1, "lock_device": 1}
    monkeypatch.setitem(helpers, "process_stamp", lambda _: {"start": "1", "state": "Z", "cwd": ""})
    assert helpers["record_state"](record) == "exited"


@pytest.mark.parametrize("status,match", [("gpu_memory_unavailable", "nvidia-smi"),
                                         ("gpu_memory_insufficient", "26624 MiB"),
                                         ("gpu_startup_busy", "still starting"),
                                         ("gpu_reservation_unverified", "ownership metadata"),
                                         ("startup_exited", "supervisor exited")])
def test_start_failures_are_actionable_and_no_remote_payload_is_echoed(target, monkeypatch, status, match):
    payload = {"status": status, "service": "secret-shaped://fixture", "free_mib": "secret-shaped://fixture",
               "error": "credential-must-not-leak"}
    monkeypatch.setattr(backend, "_ssh", lambda *_a, **_kw: json.dumps(payload))
    with pytest.raises(backend.WorkflowError, match=match) as failed:
        backend._start_remote(target, "task")
    assert "secret-shaped" not in str(failed.value) and "credential-must-not-leak" not in str(failed.value)


@pytest.mark.parametrize("status,raises", [("alive", False), ("unknown", False), ("exited", True),
                                        ("unverified", True), ("wrong", True)])
def test_readiness_poll_checks_only_sanitized_lifecycle_state(target, monkeypatch, status, raises):
    calls = []
    monkeypatch.setattr(backend, "_ssh", lambda *_a, **kw: calls.append(kw) or json.dumps({"status": status}))
    if raises:
        with pytest.raises(backend.WorkflowError):
            backend._check_remote_startup(target)
    else:
        backend._check_remote_startup(target)
    assert calls == [{"timeout": 15}]


def test_warm_authenticated_reuse_never_starts_queries_memory_or_checks_remote(target, monkeypatch):
    monkeypatch.setattr(backend, "assert_ui_idle", lambda **_: None)
    monkeypatch.setattr(backend, "_start_remote", lambda *_: pytest.fail("no cold startup"))
    monkeypatch.setattr(backend, "_check_remote_startup", lambda *_: pytest.fail("no extra warm SSH"))
    monkeypatch.setattr(backend, "_ssh", lambda *_a, **_kw: pytest.fail("no warm remote writes"))
    ready = {"instance": "authenticated-correct-model"}
    assert backend.connect_configured_runtime(target, "task", lambda: ready) is ready


def test_failed_owned_start_stops_early_without_900_second_poll_or_retry(target, monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "assert_ui_idle", lambda **_: None)
    monkeypatch.setattr(backend, "_listening", lambda _: True)
    monkeypatch.setattr(backend, "_managed_forward_present", lambda _: True)
    monkeypatch.setattr(backend, "_start_remote", lambda *_: calls.append("start") or "started")

    def failed_start(_target):
        calls.append("status")
        raise backend.WorkflowError("The owned model supervisor exited before readiness")

    monkeypatch.setattr(backend, "_check_remote_startup", failed_start)
    monkeypatch.setattr(backend.time, "sleep", lambda _: pytest.fail("no repeated polling after owned exit"))

    def unavailable():
        raise ConnectionRefusedError("transport details not echoed")

    with pytest.raises(backend.WorkflowError, match="supervisor exited"):
        backend.connect_configured_runtime(target, "task", unavailable)
    assert calls == ["start", "status"]


def test_unrecognized_listener_still_prevents_any_remote_start(target, monkeypatch):
    monkeypatch.setattr(backend, "assert_ui_idle", lambda **_: None)
    monkeypatch.setattr(backend, "_listening", lambda _: True)
    monkeypatch.setattr(backend, "_managed_forward_present", lambda _: False)
    monkeypatch.setattr(backend, "_start_remote", lambda *_: pytest.fail("no startup for unknown listener"))

    def unavailable():
        raise ConnectionRefusedError

    with pytest.raises(backend.WorkflowError, match="unrecognized listener"):
        backend.connect_configured_runtime(target, "task", unavailable)


def test_existing_zero_timeout_does_not_add_remote_poll_budget(target, monkeypatch):
    monkeypatch.setattr(backend, "assert_ui_idle", lambda **_: None)
    monkeypatch.setattr(backend, "_listening", lambda _: True)
    monkeypatch.setattr(backend, "_managed_forward_present", lambda _: True)
    monkeypatch.setattr(backend, "_start_remote", lambda *_: "started")
    monkeypatch.setattr(backend, "_check_remote_startup", lambda *_: pytest.fail("deadline already exhausted"))

    def unavailable():
        raise ConnectionRefusedError

    with pytest.raises(backend.WorkflowError, match="Model readiness failed"):
        backend.connect_configured_runtime(target, "task", unavailable, startup_timeout=0)
