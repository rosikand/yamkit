"""Cloud backend interface. Hugging Face is the only implementation for now; a new backend
registers itself in ``BACKENDS`` (lazy import path) and implements ``CloudBackend``."""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from pathlib import Path

from .settings import ArtifactKind, StorageSettings


class StorageError(RuntimeError):
    """Push/pull failed or could not be verified. The local copy is always kept when raised."""


class CloudBackend(ABC):
    name: str

    def __init__(self, settings: StorageSettings):
        self.settings = settings

    @abstractmethod
    def default_namespace(self) -> str:
        """Account namespace to prefix bare artifact names with (needs authentication)."""

    @abstractmethod
    def exists(self, repo_id: str, kind: ArtifactKind) -> bool: ...

    @abstractmethod
    def push_dir(self, local_dir: Path, repo_id: str, kind: ArtifactKind, private: bool) -> str:
        """Upload a directory tree; create the repo if needed. Returns a commit/revision id."""

    @abstractmethod
    def pull_dir(self, repo_id: str, dest: Path, kind: ArtifactKind) -> Path:
        """Download the artifact into ``dest`` and return it."""

    @abstractmethod
    def list_files(self, repo_id: str, kind: ArtifactKind) -> set[str]:
        """Relative file paths present in the cloud repo (used to verify uploads)."""


# backend name → "module:Class", imported lazily so the CLI never pays for unused backends
BACKENDS: dict[str, str] = {
    "huggingface": "yamkit.storage.huggingface:HuggingFaceBackend",
}


def get_backend(settings: StorageSettings) -> CloudBackend:
    try:
        target = BACKENDS[settings.backend]
    except KeyError:
        raise StorageError(f"unknown storage backend {settings.backend!r} (have: {sorted(BACKENDS)})") from None
    mod_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(settings)
