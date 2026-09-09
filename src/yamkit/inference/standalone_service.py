"""One bounded, authenticated inference process on an existing Lambda host.

The supervisor owns only its child process. It never provisions, stops or removes
a VM. Importing this module does not load a model, contact a service or open hardware.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import signal
import socket
import stat
import threading
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from .http_service import create_http_app, validate_http_token
from .http_transport import validate_endpoint_url
from .identity import (
    SSH_DEFAULT_SESSION_S,
    SSH_MAX_SESSION_S,
    external_service_binding,
    http_runtime_binding,
    inference_build_id,
)

STOP_GRACE_S = 3.0
PACKAGES = ("lerobot", "torch", "torchvision", "transformers", "numpy", "huggingface-hub", "tokenizers",
            "accelerate", "safetensors", "uvicorn", "h11")


@dataclass(frozen=True)
class ServiceConfig:
    service_id: str
    provider: str
    region: str
    port: int
    token_file: str
    task: str
    session_seconds: float = SSH_DEFAULT_SESSION_S
    image_height: int = 480
    image_width: int = 640
    request_timeout_s: float = 120.0

    @property
    def endpoint(self) -> str:
        return validate_endpoint_url(f"http://127.0.0.1:{self.port}", http_ingress="ssh")

    def validate(self) -> None:
        from .protocol import MAX_IMAGE_HEIGHT, MAX_IMAGE_WIDTH

        if type(self.port) is not int:
            raise ValueError("An explicit integer listener port is required")
        validate_endpoint_url(self.endpoint, http_ingress="ssh")
        external_service_binding({"external_service": {
            "provider": self.provider, "service_id": self.service_id, "host_id": "0" * 64,
            "region": self.region, "region_source": "operator_declared",
        }})
        if (type(self.session_seconds) not in (int, float) or not math.isfinite(self.session_seconds)
                or not 0 < self.session_seconds <= SSH_MAX_SESSION_S):
            raise ValueError("Inference session must be positive and at most 86400 seconds")
        if (type(self.request_timeout_s) not in (int, float) or not math.isfinite(self.request_timeout_s)
                or not 0 < self.request_timeout_s <= 120):
            raise ValueError("Inference requests require a timeout of at most 120 seconds")
        if not isinstance(self.task, str) or not self.task.strip() or len(self.task) > 2048:
            raise ValueError("An explicit task of 1–2048 characters is required")
        if (type(self.image_height) is not int or not 1 <= self.image_height <= MAX_IMAGE_HEIGHT
                or type(self.image_width) is not int or not 1 <= self.image_width <= MAX_IMAGE_WIDTH):
            raise ValueError("Image dimensions exceed the reviewed protocol boundary")


def read_token_file(path: str | Path) -> str:
    """Read one private, bounded regular file inside this checkout, without symlinks."""
    from yamkit.paths import ROOT

    try:
        original = Path(path).absolute()
        resolved = original.resolve(strict=True)
        resolved.relative_to(Path(ROOT).resolve())
        if resolved != original:
            raise ValueError("Credential path must not contain symlinks")
        fd = os.open(original, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "r", encoding="ascii") as stream:
            details = os.fstat(stream.fileno())
            if (not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid()
                    or details.st_mode & 0o077 or not 32 <= details.st_size <= 257):
                raise ValueError("Credential file is not private, regular and bounded")
            token = stream.read(258).strip()
        validate_http_token(token)
        return token
    except (OSError, ValueError, UnicodeError):
        raise ValueError("Use a private regular bearer-token file inside this checkout") from None


def stable_host_id() -> str:
    """A stable machine binding; neither the raw machine-id nor hostname is published."""
    hostname = socket.gethostname()
    machine = Path("/etc/machine-id")
    identity = machine.read_text().strip() if machine.is_file() else hostname
    return hashlib.sha256(f"{hostname}:{identity}".encode()).hexdigest()


def runtime_provenance(runtime) -> dict:
    import torch

    return {"python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
            "torch_cuda": torch.version.cuda,
            "gpu": {"device": runtime.device, "name": torch.cuda.get_device_name(runtime.device),
                    "compute_capability": list(torch.cuda.get_device_capability(runtime.device))},
            "python_gc": {"strategy": "automatic", "enabled": gc.isenabled(),
                          "thresholds": list(gc.get_threshold())}}


def readiness_metadata(runtime, config: ServiceConfig, *, expires_at: float, host_id: str,
                       provenance: dict, build_id: str) -> dict:
    from .http_wire import WIRE_CODEC, WIRE_VERSION

    return {**runtime.ready(), "transport": "http", "execution_mode": "cuda_graph10",
            "supported_call_modes": ["http"], "preferred_call_mode": "http",
            "http_ingress": "ssh", "http_endpoint": config.endpoint,
            "http_session_expires_at": expires_at, "request_timeout_s": config.request_timeout_s,
            "http_wire_codec": WIRE_CODEC, "http_wire_version": WIRE_VERSION,
            "inference_build_id": build_id, "runtime_provenance": provenance,
            "external_service": {"provider": config.provider, "service_id": config.service_id,
                                 "host_id": host_id, "region": config.region,
                                 "region_source": "operator_declared"},
            "prepared_task": config.task,
            "payload_routing": "Loopback binary HTTP through the operator-owned SSH forward; "
                               "network route and provider placement are not independently observed."}


def warm_runtime(runtime, config: ServiceConfig, *, expires_at: float) -> None:
    """Warm the actual task/shape using a non-executable fixture, never a robot observation."""
    import numpy as np

    from .protocol import encode_image, native_fixture_request, validate_response

    remaining = expires_at - time.time()
    if remaining <= 0:
        raise TimeoutError("Inference session expired while loading the model")
    request = native_fixture_request(runtime.profile, encoding="rgb8", crop="none")
    request.update(task=config.task, execution_mode="cuda_graph10",
                   timeout_s=min(config.request_timeout_s, remaining))
    image = encode_image(np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8), encoding="rgb8")
    request["images"] = {name: dict(image) for name in runtime.profile.native_image_keys}
    response = runtime.predict_chunk(request)
    validate_response(response, request, runtime.profile)
    if response.get("instance_id") != runtime.instance_id or time.time() >= expires_at:
        raise RuntimeError("Inference warmup changed runtime or exceeded the session")


def make_application(runtime, config: ServiceConfig, *, expires_at: float, host_id: str,
                     provenance: dict, build_id: str):
    """Pure server factory around an already loaded runtime; model loading belongs to the child."""
    config.validate()
    token = read_token_file(config.token_file)

    def ready():
        if not runtime._lock.acquire(blocking=False):
            raise RuntimeError("Inference runtime is busy")
        try:
            return readiness_metadata(runtime, config, expires_at=expires_at, host_id=host_id,
                                      provenance=provenance, build_id=build_id)
        finally:
            runtime._lock.release()

    return create_http_app(runtime, token=token, ready=ready, request_timeout_s=config.request_timeout_s,
                           session_expires_at=expires_at)


def _guard_supervisor(supervisor_pid: int) -> None:
    """Linux kills this model child if its watchdog disappears unexpectedly."""
    import ctypes

    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise RuntimeError("Cannot establish inference supervisor ownership")
    if os.getppid() != supervisor_pid:
        raise RuntimeError("Inference supervisor exited before its child was guarded")


def _failure_details(event: str, exc: BaseException, *, token: str | None = None) -> dict:
    """Retain bounded startup diagnostics without credentials, URLs or local variables."""
    from yamkit.rollout_artifacts import sanitize_text

    secrets = (token,) if token else ()
    return {"event": event, "error_type": type(exc).__name__,
            "error": sanitize_text(str(exc), secrets=secrets)[:2048],
            "traceback": sanitize_text("".join(traceback.format_exception(exc)), secrets=secrets)[-12000:]}


def _worker(config_values: dict, expires_at: float, supervisor_pid: int) -> None:
    """Dedicated one-model child. Only its supervisor may terminate this process group."""
    token = None
    try:
        os.setsid()
        _guard_supervisor(supervisor_pid)
        config = ServiceConfig(**config_values)
        config.validate()
        token = read_token_file(config.token_file)  # Validate before imports or allocating a GPU model.
        import uvicorn

        from .service import ModelRuntime

        runtime = ModelRuntime.load("molmoact2", device="cuda:0", execution_mode="cuda_graph10")
        provenance = runtime_provenance(runtime)
        build_id, host_id = inference_build_id(), stable_host_id()
        warm_runtime(runtime, config, expires_at=expires_at)
        metadata = readiness_metadata(runtime, config, expires_at=expires_at, host_id=host_id,
                                      provenance=provenance, build_id=build_id)
        http_runtime_binding(runtime.profile, metadata, execution_mode="cuda_graph10", task=config.task,
                             image_hw=(config.image_height, config.image_width), endpoint_url=config.endpoint)
        application = make_application(runtime, config, expires_at=expires_at, host_id=host_id,
                                       provenance=provenance, build_id=build_id)
        print(json.dumps({"event": "model_ready", "service_id": config.service_id,
                          "instance_id": runtime.instance_id, "expires_at": expires_at}), flush=True)
        uvicorn.run(application, host="127.0.0.1", port=config.port, workers=1, loop="asyncio", http="h11",
                    access_log=False, log_config=None, log_level="critical", proxy_headers=False,
                    timeout_keep_alive=60, timeout_graceful_shutdown=2)
    except BaseException as exc:  # noqa: BLE001 — bounded diagnostics; no request payloads or local variables
        print(json.dumps(_failure_details("service_failed", exc, token=token)), flush=True)
        raise SystemExit(1) from None


def _signal_owned_child(process, signal_number: int) -> None:
    """Signal only the child we created, including its group after setsid()."""
    if process.pid is None or not process.is_alive():
        return
    try:
        if os.getpgid(process.pid) == process.pid:
            os.killpg(process.pid, signal_number)
        else:
            os.kill(process.pid, signal_number)
    except ProcessLookupError:
        pass


def stop_owned_child(process) -> None:
    _signal_owned_child(process, signal.SIGTERM)
    process.join(STOP_GRACE_S)
    if process.is_alive():
        _signal_owned_child(process, signal.SIGKILL)
        process.join(STOP_GRACE_S)
    if process.is_alive():
        raise RuntimeError("Inference child shutdown could not be verified")


def supervise(config: ServiceConfig, *, process_factory=None, stop_event=None) -> int:
    """A finite deadline covers model loading, warmup and serving on an existing VM."""
    config.validate()
    read_token_file(config.token_file)
    stop = stop_event if stop_event is not None else threading.Event()
    deadline = time.monotonic() + config.session_seconds
    expires_at = time.time() + config.session_seconds
    factory = process_factory or multiprocessing.get_context("spawn").Process
    process = factory(target=_worker, args=(asdict(config), expires_at, os.getpid()),
                      daemon=True, name="yamkit-owned-inference")
    started = False
    handlers = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: stop.set())
        if stop.is_set():
            return 0
        process.start()
        started = True
        while process.is_alive() and not stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            process.join(min(0.2, remaining))
        return 0 if stop.is_set() or time.monotonic() >= deadline or process.exitcode == 0 else 1
    finally:
        try:
            if started:
                stop_owned_child(process)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--provider", choices=["lambda"], default="lambda")
    parser.add_argument("--region", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--session-seconds", dest="session_seconds", type=float, default=SSH_DEFAULT_SESSION_S)
    parser.add_argument("--task", required=True)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    args = vars(parser.parse_args(argv))
    try:
        return supervise(ServiceConfig(**args))
    except (ValueError, OSError, RuntimeError) as exc:
        print(json.dumps(_failure_details("service_start_failed", exc)), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
