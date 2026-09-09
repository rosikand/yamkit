"""Standalone lifecycle and API checks use fake runtime/processes, never GPU or hardware."""

import asyncio
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_http_runtime_binding import runtime_metadata
from tests.test_http_service import TOKEN, rpc
from tests.test_ssh_inference_binding import provenance
from yamkit import paths
from yamkit.inference import identity
from yamkit.inference import standalone_service as service
from yamkit.inference.profiles import get_profile
from yamkit.inference.protocol import decode_image


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(identity, "inference_build_id", lambda: "b" * 64)
    token = tmp_path / "token"
    token.write_text(TOKEN + "\n")
    token.chmod(0o600)
    return service.ServiceConfig(service_id="lambda-georgia", provider="lambda", region="Georgia", port=8765,
                                 token_file=str(token), task="put the orange lid in the black container",
                                 image_height=8, image_width=12)


def test_standalone_import_never_imports_model_driver_or_modal():
    code = ("import sys; import yamkit.inference.standalone_service; "
            "assert not any(n == p or n.startswith(p + '.') for n in sys.modules "
            "for p in ['modal', 'i2rt', 'lerobot_robot_yamkit', 'yamkit.arm', 'yamkit.inference.service'])")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr


def test_server_defaults_keep_loopback_one_day_maximum_and_native_rgb_shape(config):
    config.validate()
    assert config.endpoint == "http://127.0.0.1:8765"
    assert config.session_seconds == 28800
    assert service.ServiceConfig.__dataclass_fields__["image_height"].default == 480
    assert service.ServiceConfig.__dataclass_fields__["image_width"].default == 640
    replace(config, session_seconds=86400).validate()


@pytest.mark.parametrize("changes", [
    {"port": 0}, {"port": 65536}, {"port": True}, {"provider": "modal"}, {"service_id": "a/b"},
    {"region": "Georgia\n"}, {"task": " "}, {"task": "x" * 2049}, {"session_seconds": 0},
    {"session_seconds": 86401}, {"session_seconds": float("inf")}, {"session_seconds": True},
    {"image_width": 0}, {"image_height": True}, {"request_timeout_s": 121},
])
def test_invalid_service_configuration_fails_without_starting_a_process(config, changes):
    with pytest.raises(ValueError):
        replace(config, **changes).validate()


def test_token_reader_requires_private_regular_repo_local_file(config, tmp_path, monkeypatch):
    assert service.read_token_file(config.token_file) == TOKEN
    token = Path(config.token_file)
    token.chmod(0o644)
    with pytest.raises(ValueError) as exc:
        service.read_token_file(token)
    assert TOKEN not in str(exc.value)
    token.chmod(0o600)
    link = tmp_path / "token-link"
    link.symlink_to(token)
    with pytest.raises(ValueError):
        service.read_token_file(link)
    token.write_text("x" * 258)
    with pytest.raises(ValueError):
        service.read_token_file(token)
    token.write_text(TOKEN)
    inside = tmp_path / "checkout"
    inside.mkdir()
    monkeypatch.setattr(paths, "ROOT", inside)
    with pytest.raises(ValueError):
        service.read_token_file(token)


def test_ready_preserves_runtime_fields_and_reports_declared_provider_without_modal_fields(config):
    runtime = SimpleNamespace(ready=lambda: runtime_metadata(task=config.task, image_hw=(8, 12)))
    base = runtime.ready()
    for key in ("requested_compute_region", "compute_region", "routing_region"):
        base.pop(key)
    runtime.ready = lambda: dict(base)
    value = service.readiness_metadata(runtime, config, expires_at=time.time() + 28800, host_id="a" * 64,
                                       provenance=provenance(), build_id="b" * 64)
    assert value["instance_id"] == base["instance_id"]
    assert value["execution_identity"] == base["execution_identity"]
    assert value["graph_warmup"] == base["graph_warmup"]
    assert value["external_service"] == {"provider": "lambda", "service_id": "lambda-georgia",
                                         "host_id": "a" * 64, "region": "Georgia",
                                         "region_source": "operator_declared"}
    assert value["http_ingress"] == "ssh" and value["runtime_provenance"] == provenance()
    assert not any(key in value for key in ("modal_app", "compute_region", "routing_region", "scaledown_window_s"))
    assert TOKEN not in json.dumps(value)


def test_standalone_application_reuses_auth_raw_binary_runtime_and_finite_session(config):
    calls = []
    runtime = SimpleNamespace(_lock=threading.Lock(), ready=lambda: {"instance_id": "real-fake-instance"},
                              predict_chunk=lambda request: calls.append(request) or {"pixels": request["pixels"]},
                              reset=lambda session: calls.append(session))
    app = service.make_application(runtime, config, expires_at=time.time() + 30,
                                   host_id="a" * 64, provenance=provenance(), build_id="b" * 64)
    status, _, reads = asyncio.run(rpc(app, headers=[]))
    assert status == 401 and not reads and not calls
    status, body, _ = asyncio.run(rpc(app))
    assert status == 200 and body["result"]["external_service"]["service_id"] == config.service_id
    image_bytes = bytes(range(256)) * 4
    status, body, _ = asyncio.run(rpc(app, {"method": "predict_chunk", "payload": {"pixels": image_bytes}}))
    assert status == 200 and body["result"]["pixels"] == image_bytes
    expired = service.make_application(runtime, config, expires_at=time.time() - 1,
                                       host_id="a" * 64, provenance=provenance(), build_id="b" * 64)
    status, _, reads = asyncio.run(rpc(expired, {"method": "predict_chunk", "payload": {"pixels": image_bytes}}))
    assert status == 410 and not reads and len(calls) == 1


def test_warmup_uses_configured_task_shape_raw_rgb_and_nonexecutable_fixture(config):
    requests = []
    runtime = SimpleNamespace(profile=get_profile("molmoact2"), instance_id="fake-instance")

    def predict(request):
        requests.append(request)
        return {**{key: request[key] for key in (
            "protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time")},
            "action_names": list(runtime.profile.action_names), "action_units": "checkpoint_native",
            "instance_id": runtime.instance_id, "execution_mode": "cuda_graph10",
            "chunk": [[0.2] * 14 for _ in range(30)],
            "timing": {key: 0.0 for key in ("preprocess_s", "inference_s", "postprocess_s", "total_s")}}

    runtime.predict_chunk = predict
    service.warm_runtime(runtime, config, expires_at=time.time() + 30)
    assert len(requests) == 1
    request = requests[0]
    assert request["mode"] == "native_fixture" and request["task"] == config.task
    assert request["execution_mode"] == "cuda_graph10" and request["crop"] == "none"
    assert set(request["images"]) == set(runtime.profile.native_image_keys)
    for image in request["images"].values():
        assert image["encoding"] == "rgb8" and decode_image(image).shape == (8, 12, 3)
    with pytest.raises(TimeoutError):
        service.warm_runtime(runtime, config, expires_at=time.time() - 1)
    assert len(requests) == 1


def test_supervisor_expiry_terminates_only_its_stuck_child_and_escalates_boundedly(config, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0], time=lambda: 1000 + clock[0]))
    signals, constructed = [], []

    class Process:
        pid = 43210
        exitcode = None
        alive = False

        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def start(self):
            self.alive = True

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            if self.alive:
                clock[0] += timeout

    process = Process()
    monkeypatch.setattr(service.os, "getpgid", lambda pid: pid)

    def kill_group(pid, number):
        signals.append((pid, number))
        if number == signal.SIGKILL:
            process.alive = False
            process.exitcode = -number

    monkeypatch.setattr(service.os, "killpg", kill_group)
    monkeypatch.setattr(service.os, "kill", lambda *_: pytest.fail("Must target only the owned child group"))

    def factory(**kwargs):
        constructed.append(kwargs)
        return process

    assert service.supervise(replace(config, session_seconds=1), process_factory=factory) == 0
    assert signals == [(43210, signal.SIGTERM), (43210, signal.SIGKILL)]
    assert not process.alive and clock[0] <= 101 + 2 * service.STOP_GRACE_S
    args = constructed[-1]["args"]
    assert args[1] == 1101.0 and args[2] == service.os.getpid()
    assert args[0]["token_file"] == config.token_file and TOKEN not in repr(constructed)
    assert constructed[-1]["target"] is service._worker


def test_preexisting_stop_never_starts_inference_child(config):
    stop = threading.Event()
    stop.set()
    process = SimpleNamespace(start=lambda: pytest.fail("Cannot start after Stop"))
    assert service.supervise(config, process_factory=lambda **kw: process, stop_event=stop) == 0


def test_linux_parent_death_guard_is_bound_before_model_loading(monkeypatch):
    import ctypes

    calls = []
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=lambda *args: calls.append(args) or 0))
    monkeypatch.setattr(service.os, "getppid", lambda: 12345)
    service._guard_supervisor(12345)
    assert calls == [(1, signal.SIGKILL, 0, 0, 0)]
    with pytest.raises(RuntimeError, match="supervisor exited"):
        service._guard_supervisor(11111)


def test_worker_failure_keeps_diagnostic_context_but_redacts_actual_bearer_and_other_secrets(
        config, monkeypatch, capsys):
    monkeypatch.setattr(service.os, "setsid", lambda: None)
    monkeypatch.setattr(service, "_guard_supervisor", lambda pid: None)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace())

    def fail_load(*args, **kwargs):
        raise RuntimeError("CUDA allocation failed; " + TOKEN
                           + " https://user:private@example.test/path?token=hidden"
                           + " hf_ABCDEFGH12345678 password=example-private-value")

    monkeypatch.setitem(sys.modules, "yamkit.inference.service",
                        SimpleNamespace(ModelRuntime=SimpleNamespace(load=fail_load)))
    with pytest.raises(SystemExit) as exc:
        service._worker(asdict(config), time.time() + 30, 12345)
    assert exc.value.code == 1
    emitted = capsys.readouterr().out
    value = json.loads(emitted)
    assert value["event"] == "service_failed" and value["error_type"] == "RuntimeError"
    assert "CUDA allocation failed" in value["error"] and "fail_load" in value["traceback"]
    for private in (TOKEN, "https://user", "hidden", "hf_ABCDEFGH12345678", "example-private-value"):
        assert private not in emitted


def test_failure_diagnostics_are_bounded_after_sanitization():
    exc = ValueError("Useful error; " + "x" * 18000 + TOKEN)
    value = service._failure_details("service_failed", exc, token=TOKEN)
    assert len(value["error"]) == 2048 and len(value["traceback"]) == 12000
    assert TOKEN not in json.dumps(value)
