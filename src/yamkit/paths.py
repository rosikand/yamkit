"""Repository-relative paths (everything lives inside the repo)."""

from __future__ import annotations

import os
from pathlib import Path

from ._env import find_root

ROOT: Path = find_root() or Path.cwd()
CONFIG_DIR = ROOT / "configs"
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"  # models/checkpoints pulled from cloud storage
STAGING_DIR = DATA_DIR / ".staging"  # temporary local staging for cloud-only artifacts
OUTPUT_DIR = ROOT / "outputs"
DEFAULT_RIG = CONFIG_DIR / "rig.yaml"
DEFAULT_SETTINGS = CONFIG_DIR / "yamkit.yaml"


def resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve a path relative to the repo root unless it is absolute."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p
