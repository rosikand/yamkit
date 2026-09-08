"""Lazy Modal app factory. Constructing it neither deploys nor starts a GPU.

One class, one immutable model, one pool. There are deliberately no parameterized
class constructors (each parameter set would otherwise have its own GPU pool).
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

from .profiles import get_profile

DEFAULT_GPU = "L40S"
DEFAULT_SCALEDOWN_S = 300
DEV_SCALEDOWN_S = 15
REMOTE_ROOT = "/opt/yamkit"
MEMORY_MIB = 65536
CPU_CORES = 4


def _prepare_http_graph_gc(runtime) -> dict:
    """Exclude the immutable container's loaded heap from future cyclic scans.

    GC freezing is process-wide, so this must never run in a local policy, UI,
    factory definition or test process. New observations, responses and graph
    warm-up objects remain subject to ordinary automatic GC. Frozen cycles live
    until this one-model container retires; models are never replaced in place.
    """
    import modal

    if (getattr(modal, "is_local", lambda: True)() is not False
            or os.environ.get("YAMKIT_ROOT") != REMOTE_ROOT
            or getattr(getattr(runtime, "profile", None), "id", None) != "molmoact2"
            or getattr(runtime, "execution_mode", None) != "cuda_graph10"
            or not str(getattr(runtime, "device", "")).startswith("cuda")
            or getattr(runtime, "_prediction_count", None) != 0):
        raise RuntimeError("Model GC preparation requires the dedicated Modal HTTP graph container before its first prediction")
    if not gc.isenabled():
        raise RuntimeError("Model GC preparation requires automatic collection to remain enabled")
    thresholds = gc.get_threshold()
    started = time.monotonic()
    collected = gc.collect(2)
    collected_at = time.monotonic()
    gc.freeze()
    frozen_at = time.monotonic()
    frozen = gc.get_freeze_count()
    if not gc.isenabled() or gc.get_threshold() != thresholds or type(frozen) is not int or frozen <= 0:
        raise RuntimeError("Model GC preparation failed to preserve automatic collection")
    return {"strategy": "freeze_after_model_load_v1", "automatic_gc_enabled": True,
            "thresholds": list(thresholds), "collected_objects": collected, "frozen_objects": frozen,
            "collection_s": collected_at - started, "freeze_s": frozen_at - collected_at,
            "scope": "Loaded container heap only; later request allocations use automatic GC"}


def create_app(profile_id: str = "smolvla", *, gpu: str = DEFAULT_GPU, development: bool = False,
               app_name: str | None = None, scaledown_window: int = DEFAULT_SCALEDOWN_S,
               timeout: int = 120, startup_timeout: int = 600, region: str | None = "us-west",
               routing_region: str = "us-west", cache_volume_name: str = "yamkit-policy-weights",
               memory_mib: int = MEMORY_MIB, transport: str = "sdk", execution_mode: str = "eager",
               http_token: str | None = None, min_containers: int = 0):
    """Build an App definition; the caller owns deployment, budget and shutdown.

    Only ``HF_TOKEN`` and, for opt-in HTTP, a dedicated bearer secret reach the
    container through Modal Secrets. Modal account credentials stay in the SDK.
    HTTP and SDK methods share the same unparameterized class and GPU pool.
    ``min_containers=1`` is intended for an owned, time-bounded ``app.run()``
    context whose coordinator guarantees shutdown. Persistent preparation keeps
    the default zero; either setting retains the one-container maximum.
    """
    profile = get_profile(profile_id)
    if transport not in ("sdk", "http"):
        raise ValueError("Inference transport must be sdk or http")
    if execution_mode not in ("eager", "cuda_graph10"):
        raise ValueError("Unsupported model execution mode")
    if execution_mode == "cuda_graph10" and profile.id != "molmoact2":
        raise ValueError("CUDA graph execution requires the reviewed MolmoAct2 profile")
    if transport == "http":
        from .http_service import validate_http_token

        validate_http_token(http_token)
    elif http_token is not None:
        raise ValueError("A dedicated HTTP token requires the HTTP transport")
    if gpu not in (DEFAULT_GPU, "H100!"):
        raise ValueError("Use one L40S or an exact H100! for explicit diagnostics")
    if type(min_containers) is not int or min_containers not in (0, 1):
        raise ValueError("min_containers must be exactly 0 or 1")
    if type(memory_mib) is not int or not 49152 <= memory_mib <= MEMORY_MIB:
        raise ValueError("Host memory must be 49152–65536 MiB; lower loading peaks are unmeasured")
    if not 1 <= timeout <= 120 or not 1 <= startup_timeout <= 900:
        raise ValueError("Finite request/startup timeouts are required")
    if development:
        scaledown_window = min(scaledown_window, DEV_SCALEDOWN_S)
        timeout = min(timeout, 90)
        startup_timeout = min(startup_timeout, 240)
    elif not 300 <= scaledown_window <= 600:
        raise ValueError("Production scaledown_window must be 300–600 seconds")
    if routing_region not in ("us-east", "us-west"):
        raise ValueError("Unsupported Modal routing region")
    if region not in (None, "us", "us-east", "us-west", "us-central", "eu", "eu-west", "ap"):
        raise ValueError("Unsupported compute region")
    import modal

    from yamkit.paths import ROOT

    root = Path(ROOT)
    image = modal.Image.debian_slim(python_version="3.12").pip_install_from_requirements(
        str(root / "configs" / "modal-requirements.txt"),
    ).env({
        "PYTHONPATH": f"{REMOTE_ROOT}/src", "YAMKIT_ROOT": REMOTE_ROOT,
        "HF_HOME": f"{REMOTE_ROOT}/data/hf", "TORCH_HOME": f"{REMOTE_ROOT}/data/torch",
        "XDG_CACHE_HOME": f"{REMOTE_ROOT}/data/cache", "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }).add_local_dir(str(root / "src" / "yamkit"), f"{REMOTE_ROOT}/src/yamkit",
                    ignore=lambda path: Path(path).suffix != ".py")
    if transport == "http":
        image = image.add_local_file(str(root / "configs" / "modal-requirements.txt"),
                                     f"{REMOTE_ROOT}/configs/modal-requirements.txt")
    volume = modal.Volume.from_name(cache_volume_name, create_if_missing=True)
    secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []
    if transport == "http":
        from .http_service import HTTP_TOKEN_ENV

        secrets.append(modal.Secret.from_dict({HTTP_TOKEN_ENV: http_token}))
    app = modal.App(app_name or f"yamkit-policy-{profile.id}")
    fixed_profile_id = profile.id

    @app.cls(image=image, gpu=gpu, cpu=(CPU_CORES, CPU_CORES), memory=(memory_mib, memory_mib),
             min_containers=min_containers, max_containers=1, buffer_containers=0,
             scaledown_window=scaledown_window, timeout=timeout, startup_timeout=startup_timeout,
             retries=0, region=region, routing_region=routing_region,
             volumes={f"{REMOTE_ROOT}/data/hf": volume}, secrets=secrets, serialized=True, include_source=False)
    class PolicyService:
        @modal.enter()
        def load(self):
            entered = time.monotonic()
            from yamkit.inference.service import ModelRuntime

            self.runtime = ModelRuntime.load(fixed_profile_id, device="cuda", execution_mode=execution_mode)
            loaded = time.monotonic()
            volume.commit()
            self.volume_commit_s = time.monotonic() - loaded
            if transport == "http":
                from yamkit.inference.identity import inference_build_id

                self.inference_build_id = inference_build_id()
            self.python_gc = {"strategy": "automatic"}
            if transport == "http" and execution_mode == "cuda_graph10":
                # The process keeps exactly this loaded model for its lifetime.
                # Freeze before requests exist, never on task changes or resets.
                self.python_gc = _prepare_http_graph_gc(self.runtime)
            self.container_init_s = time.monotonic() - entered

        def _readiness(self) -> dict:
            call_modes = ["remote", "spawn"] if routing_region == "us-east" else ["remote"]
            metadata = {**self.runtime.ready(), "gpu": gpu,
                    "compute_region": os.environ.get("MODAL_REGION", "unknown"),
                    "requested_compute_region": region, "routing_region": routing_region,
                    "scaledown_window_s": scaledown_window, "min_containers": min_containers, "max_containers": 1,
                    "request_timeout_s": timeout, "startup_timeout_s": startup_timeout,
                    "requested_memory_mib": memory_mib,
                    "memory_scope": "Requested memory is host RAM; CUDA telemetry measures GPU memory separately",
                    "container_init_s": self.container_init_s, "volume_commit_s": self.volume_commit_s,
                    "python_gc": dict(self.python_gc),
                    "transport": transport, "execution_mode": execution_mode,
                    "supported_call_modes": call_modes + (["http"] if transport == "http" else []),
                    "preferred_call_mode": "http" if transport == "http" else "remote",
                    "observed_compute_region": os.environ.get("MODAL_REGION"),
                    "requested_routing_region": routing_region, "observed_routing_region": None,
                    "payload_routing": "Requested SDK routing region; actual routing is not independently observable. "
                    "Spawned SDK payloads use US storage; use remote for non-default routing."}
            if transport == "http":
                from yamkit.inference.http_wire import WIRE_CODEC, WIRE_VERSION

                metadata.update(inference_build_id=self.inference_build_id, http_wire_version=WIRE_VERSION,
                                http_wire_codec=WIRE_CODEC,
                                payload_routing="Persistent binary HTTPS to this class's Modal endpoint; "
                                "actual network routing is not independently observable.")
            return metadata

        @modal.method()
        def ready(self) -> dict:
            return self._readiness()

        @modal.method()
        def predict_chunk(self, request: dict) -> dict:
            return self.runtime.predict_chunk(request)

        @modal.method()
        def reset(self, session_id: str) -> None:
            self.runtime.reset(session_id)

        if transport == "http":
            @modal.asgi_app()
            def http(self):
                from yamkit.inference.http_service import HTTP_TOKEN_ENV, create_http_app

                return create_http_app(self.runtime, token=os.environ[HTTP_TOKEN_ENV],
                                       ready=self._readiness, request_timeout_s=timeout)

    return app
