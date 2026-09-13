"""Bounded native π0.5 service on an existing GPU; no provisioning or robot I/O."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import signal
import threading
import time
from dataclasses import asdict

from yamkit.inference.http_service import create_http_app
from yamkit.inference.identity import inference_build_id
from yamkit.inference.standalone_service import (
    ServiceConfig,
    _failure_details,
    _guard_supervisor,
    read_token_file,
    readiness_metadata,
    runtime_provenance,
    stable_host_id,
    stop_owned_child,
)


def make_application(runtime, config: ServiceConfig, *, expires_at: float):
    """PI identity around existing authenticated/bounded HTTP session machinery."""
    config.validate()
    token = read_token_file(config.token_file)
    provenance, host_id = runtime_provenance(runtime), stable_host_id()
    source = inference_build_id()

    def ready():
        if not runtime._lock.acquire(blocking=False):
            raise RuntimeError("π0.5 inference runtime is busy")
        try:
            result = readiness_metadata(runtime, config, expires_at=expires_at, host_id=host_id,
                                        provenance=provenance, build_id=source)
            result["execution_mode"] = "eager"
            return result
        finally:
            runtime._lock.release()

    return create_http_app(runtime, token=token, ready=ready, request_timeout_s=config.request_timeout_s,
                           session_expires_at=expires_at)


def warm_runtime(runtime, config: ServiceConfig, expires_at: float) -> dict:
    import numpy as np

    from yamkit.inference.protocol import encode_image, native_fixture_request

    request = native_fixture_request(runtime.profile)
    remaining = expires_at - time.time()
    if remaining <= 0:
        raise TimeoutError("π0.5 session expired during model load")
    request.update(task=config.task, execution_mode="eager", timeout_s=min(remaining, config.request_timeout_s))
    image = encode_image(np.zeros((config.image_height, config.image_width, 3), dtype=np.uint8))
    request["images"] = {name: dict(image) for name in runtime.profile.native_image_keys}
    return runtime.predict_chunk(request)


def _worker(values: dict, expires_at: float, supervisor_pid: int, device: str) -> None:
    token = None
    try:
        os.setsid()
        _guard_supervisor(supervisor_pid)
        config = ServiceConfig(**values)
        config.validate()
        token = read_token_file(config.token_file)
        import uvicorn

        from .runtime import Pi05Runtime

        runtime = Pi05Runtime.load(device=device)
        warm_runtime(runtime, config, expires_at)
        app = make_application(runtime, config, expires_at=expires_at)
        print(json.dumps({"event": "pi05_model_ready", "service_id": config.service_id,
                          "instance_id": runtime.instance_id, "expires_at": expires_at}), flush=True)
        uvicorn.run(app, host="127.0.0.1", port=config.port, workers=1, loop="asyncio", http="h11",
                    access_log=False, log_config=None, log_level="critical", proxy_headers=False,
                    timeout_keep_alive=60, timeout_graceful_shutdown=2)
    except BaseException as exc:  # noqa: BLE001 — sanitized bounded service diagnostics
        print(json.dumps(_failure_details("pi05_service_failed", exc, token=token)), flush=True)
        raise SystemExit(1) from None


def supervise(config: ServiceConfig, *, device: str = "cuda:0") -> int:
    """Own one finite model child only; reuse existing host, never create a VM."""
    config.validate()
    read_token_file(config.token_file)
    stop = threading.Event()
    deadline, expires_at = time.monotonic() + config.session_seconds, time.time() + config.session_seconds
    process = multiprocessing.get_context("spawn").Process(
        target=_worker, args=(asdict(config), expires_at, os.getpid(), device), daemon=True,
        name="yamkit-owned-pi05-inference")
    handlers = {}
    started = False
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: stop.set())
        process.start()
        started = True
        while process.is_alive() and not stop.is_set() and time.monotonic() < deadline:
            process.join(min(0.2, max(0.0, deadline - time.monotonic())))
        return 0 if stop.is_set() or time.monotonic() >= deadline or process.exitcode == 0 else 1
    finally:
        try:
            if started:
                stop_owned_child(process)
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-id", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--session-seconds", type=float, default=28800)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    args = vars(parser.parse_args())
    device = args.pop("device")
    return supervise(ServiceConfig(provider="lambda", **args), device=device)


if __name__ == "__main__":
    raise SystemExit(main())
