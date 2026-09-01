"""Redirect every cache / output directory into this repository.

Imported automatically at interpreter start (``yamkit_env.pth`` in site-packages), so
``lerobot-*`` commands, ``huggingface_hub`` downloads and ``wandb`` all stay inside
``<repo>/data`` and ``<repo>/outputs`` instead of ``~/.cache``. Every value is a
``setdefault``: an explicit environment variable always wins.
"""

from __future__ import annotations

import os
from pathlib import Path


def find_root() -> Path | None:
    env = os.environ.get("YAMKIT_ROOT")
    if env:
        return Path(env).expanduser()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    return None


def apply() -> None:
    root = find_root()
    if root is None:
        return
    data = root / "data"
    os.environ.setdefault("YAMKIT_ROOT", str(root))
    os.environ.setdefault("HF_HOME", str(data / "hf"))  # hub models/datasets cache
    os.environ.setdefault("HF_LEROBOT_HOME", str(data / "lerobot"))  # datasets + calibration
    os.environ.setdefault("TORCH_HOME", str(data / "torch"))
    os.environ.setdefault("WANDB_DIR", str(root / "outputs" / "wandb"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


apply()
