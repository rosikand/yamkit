"""Lazy Modal app factory. Constructing it neither deploys nor starts a GPU.

One class, one immutable model, one pool. There are deliberately no parameterized
class constructors (each parameter set would otherwise have its own GPU pool).
"""

from __future__ import annotations

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


def create_app(profile_id: str = "smolvla", *, gpu: str = DEFAULT_GPU, development: bool = False,
               app_name: str | None = None, scaledown_window: int = DEFAULT_SCALEDOWN_S,
               timeout: int = 120, startup_timeout: int = 600, region: str | None = "us-west",
               routing_region: str = "us-west", cache_volume_name: str = "yamkit-policy-weights",
               memory_mib: int = MEMORY_MIB):
    """Build an App definition; the caller owns deployment, budget and shutdown.

    ``HF_TOKEN`` is the only environment credential forwarded, through Modal Secrets.
    Modal authentication stays in the calling SDK; there are no web endpoints.
    """
    profile = get_profile(profile_id)
    if gpu not in (DEFAULT_GPU, "H100!"):
        raise ValueError("Use one L40S or an exact H100! for explicit diagnostics")
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
    volume = modal.Volume.from_name(cache_volume_name, create_if_missing=True)
    secrets = [modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})] if os.environ.get("HF_TOKEN") else []
    app = modal.App(app_name or f"yamkit-policy-{profile.id}")
    fixed_profile_id = profile.id

    @app.cls(image=image, gpu=gpu, cpu=CPU_CORES, memory=memory_mib,
             min_containers=0, max_containers=1, buffer_containers=0,
             scaledown_window=scaledown_window, timeout=timeout, startup_timeout=startup_timeout,
             retries=0, region=region, routing_region=routing_region,
             volumes={f"{REMOTE_ROOT}/data/hf": volume}, secrets=secrets, serialized=True)
    class PolicyService:
        @modal.enter()
        def load(self):
            entered = time.monotonic()
            from yamkit.inference.service import ModelRuntime

            self.runtime = ModelRuntime.load(fixed_profile_id, device="cuda")
            loaded = time.monotonic()
            volume.commit()
            self.container_init_s = time.monotonic() - entered
            self.volume_commit_s = self.container_init_s - (loaded - entered)

        @modal.method()
        def ready(self) -> dict:
            return {**self.runtime.ready(), "gpu": gpu, "compute_region": os.environ.get("MODAL_REGION", "unknown"),
                    "requested_compute_region": region, "routing_region": routing_region,
                    "scaledown_window_s": scaledown_window, "min_containers": 0, "max_containers": 1,
                    "request_timeout_s": timeout, "startup_timeout_s": startup_timeout,
                    "requested_memory_mib": memory_mib,
                    "memory_scope": "Requested memory is host RAM; CUDA telemetry measures GPU memory separately",
                    "container_init_s": self.container_init_s, "volume_commit_s": self.volume_commit_s,
                    "supported_call_modes": ["remote", "spawn"] if routing_region == "us-east" else ["remote"],
                    "preferred_call_mode": "remote",
                    "observed_compute_region": os.environ.get("MODAL_REGION"),
                    "requested_routing_region": routing_region, "observed_routing_region": None,
                    "payload_routing": "Requested SDK routing region; actual routing is not independently observable. "
                    "Spawned SDK payloads use US storage; use remote for non-default routing."}

        @modal.method()
        def predict_chunk(self, request: dict) -> dict:
            return self.runtime.predict_chunk(request)

        @modal.method()
        def reset(self, session_id: str) -> None:
            self.runtime.reset(session_id)

    return app
