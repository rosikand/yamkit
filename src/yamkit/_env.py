"""Redirect every cache / output directory into this repository.

Imported automatically at interpreter start (``yamkit_env.pth`` in site-packages), so
``lerobot-*`` commands, ``huggingface_hub`` downloads and ``wandb`` all stay inside
``<repo>/data`` and ``<repo>/outputs`` instead of ``~/.cache``. Every value is a
``setdefault``: an explicit environment variable always wins.
"""

from __future__ import annotations

import os
from pathlib import Path


def _checkout_of(path: Path) -> Path | None:
    """The yamkit checkout containing `path` (has pyproject.toml + configs/), or None."""
    for parent in (path, *path.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    return None


def find_root() -> Path | None:
    """The checkout this code runs from wins; `YAMKIT_ROOT` only counts when the code is not inside
    a checkout (e.g. installed elsewhere). A stale variable from another clone's `scripts/env.sh`
    must never make a fresh clone read or write the other clone's rig and data."""
    here = _checkout_of(Path(__file__).resolve())
    if here is not None:
        return here
    env = os.environ.get("YAMKIT_ROOT")
    return Path(env).expanduser() if env else None


_DERIVED = ("HF_HOME", "HF_LEROBOT_HOME", "TORCH_HOME", "WANDB_DIR")


def apply() -> None:
    root = find_root()
    if root is None:
        return
    data = root / "data"
    stale = os.environ.get("YAMKIT_ROOT")
    if stale and Path(stale).expanduser().resolve() != root.resolve():
        # variables exported by another checkout's env.sh: drop the ones that point into that checkout
        for var in _DERIVED:
            if os.environ.get(var, "").startswith(stale.rstrip("/")):
                del os.environ[var]
        os.environ["YAMKIT_ROOT"] = str(root)
    os.environ.setdefault("YAMKIT_ROOT", str(root))
    os.environ.setdefault("HF_HOME", str(data / "hf"))  # hub models/datasets cache
    os.environ.setdefault("HF_LEROBOT_HOME", str(data / "lerobot"))  # datasets + calibration
    os.environ.setdefault("TORCH_HOME", str(data / "torch"))
    os.environ.setdefault("WANDB_DIR", str(root / "outputs" / "wandb"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


apply()
