"""Unmodified official JAX model in normalized space, explicitly not a YAM policy.

Heavy imports occur only in ``load``. There is no robot or camera dependency,
unnormalization, joint mapper, 14D projection, or physical execution entrypoint.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .assets import checkpoint_path, local_path, verify_assets
from .contract import IMAGE_KEYS, MODEL_CONFIG, UPSTREAM_REVISION, identity


def configure_environment(root: Path) -> None:
    """Keep all explicit official-runtime caches below the GPU checkout."""
    paths = {
        "OPENPI_DATA_HOME": "data/openpi/cache", "HF_HOME": "data/openpi/hf",
        "XDG_CACHE_HOME": "data/openpi/xdg", "XDG_DATA_HOME": "data/openpi/xdg-data",
        "XDG_CONFIG_HOME": "data/openpi/xdg-config", "TMPDIR": "data/openpi/tmp",
        "JAX_COMPILATION_CACHE_DIR": "data/openpi/jax-cache",
        "CUDA_CACHE_PATH": "data/openpi/cuda-cache", "TORCH_HOME": "data/openpi/torch",
        "TRITON_CACHE_DIR": "data/openpi/triton-cache", "WANDB_DIR": "data/openpi/wandb",
    }
    for name, relative in paths.items():
        path = local_path(root, Path(relative))
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    # Prevent the default JAX 75% preallocation from taking another policy's memory.
    # This changes allocation strategy only, not the official model computation.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def verify_upstream(root: Path) -> Path:
    upstream = local_path(root, Path("data/openpi/upstream"))
    revision = subprocess.run(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    status = subprocess.run(["git", "-C", str(upstream), "status", "--porcelain"],
                            check=True, capture_output=True, text=True, timeout=10).stdout
    if revision != UPSTREAM_REVISION or status.strip():
        raise ValueError("Official OpenPI checkout must match its clean pinned revision")
    return upstream


def normalized_observation(images: dict[str, np.ndarray], task: str) -> dict:
    """Input-shape diagnostic only. Measured YAM joints are deliberately NOT consumed."""
    if set(images) != set(IMAGE_KEYS):
        raise ValueError("Official diagnostic requires all three explicit native image roles")
    if not isinstance(task, str) or not task.strip() or len(task) > 512:
        raise ValueError("Provide a nonempty diagnostic task of at most 512 characters")
    for image in images.values():
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.ndim != 3 or image.shape[2] != 3
                or not (1 <= image.shape[0] <= 2160 and 1 <= image.shape[1] <= 3840)):
            raise ValueError("Diagnostic images must be bounded HWC uint8 RGB arrays")
    return {
        "image": dict(images), "image_mask": dict.fromkeys(IMAGE_KEYS, np.True_),
        "state": np.zeros(32, dtype=np.float32), "prompt": task,
    }


def validate_normalized_chunk(value) -> np.ndarray:
    """Preserve all 50x32 dimensions, including values outside normalized [-1,1]."""
    result = np.asarray(value)
    if result.shape != (50, 32) or not np.issubdtype(result.dtype, np.floating):
        raise ValueError("Official diagnostic returned a different native 50x32 floating chunk")
    if not np.isfinite(result).all():
        raise ValueError("Official diagnostic returned non-finite native outputs")
    return result


class OfficialPi05Diagnostic:
    """Owns only official inference; it cannot send commands or become physical-ready."""

    def __init__(self, policy, provenance: dict):
        self.policy = policy
        self.provenance = provenance

    @classmethod
    def load(cls, root: Path) -> OfficialPi05Diagnostic:
        root = root.resolve()
        configure_environment(root)
        upstream = verify_upstream(root)
        assets = verify_assets(root)
        started = time.monotonic()
        import jax
        import jax.numpy as jnp
        from openpi.models import model, pi0_config
        from openpi.policies import policy
        from openpi.training.config import ModelTransformFactory

        if not Path(pi0_config.__file__).resolve().is_relative_to(upstream / "src/openpi"):
            raise ValueError("Imported OpenPI runtime is not the pinned isolated checkout")
        if not any(device.platform == "gpu" for device in jax.devices()):
            raise ValueError("The official software diagnostic requires the configured existing GPU")
        config = pi0_config.Pi0Config(pi05=True)
        if any(getattr(config, key) != value for key, value in MODEL_CONFIG.items()):
            raise ValueError("Official Pi0Config(pi05=True) no longer matches the pinned native defaults")
        loaded = config.load(model.restore_params(checkpoint_path(root) / "params", dtype=jnp.bfloat16))
        transforms = ModelTransformFactory()(config)
        native_policy = policy.Policy(loaded, transforms=transforms.inputs,
                                      output_transforms=transforms.outputs)
        packages = {name: importlib.metadata.version(name) for name in
                    ("openpi", "jax", "jaxlib", "flax", "orbax-checkpoint", "numpy", "sentencepiece")}
        return cls(native_policy, {
            **identity(), "asset_objects_verified": len(assets["objects"]),
            "packages": packages, "load_s": time.monotonic() - started,
            "normalization_assets_used": [], "output_space": "raw normalized model space, NOT YAM commands",
            "state_source": "synthetic normalized zero vector, 32D; not measured YAM state",
            "input_transforms": [type(transform).__name__ for transform in transforms.inputs],
            "output_transforms": [type(transform).__name__ for transform in transforms.outputs],
            "model_native_defaults_verified": True, "jax_preallocate": False,
        })

    def predict(self, images: dict[str, np.ndarray], task: str, *, noise=None) -> np.ndarray:
        if noise is not None:
            noise = validate_normalized_chunk(noise)
        result = self.policy.infer(normalized_observation(images, task), noise=noise)
        return validate_normalized_chunk(result["actions"])
