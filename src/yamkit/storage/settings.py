"""Storage settings: cloud backend, account namespace, per-artifact save/push policy.

Loaded from the ``storage:`` section of ``configs/yamkit.yaml`` (settings that are not
rig/hardware related). A missing file or section means local-only defaults, so nothing
changes for setups that never configured cloud storage.

Policy semantics per artifact kind (datasets / models):
    save_local: true,  auto_push: false  → local only (the default)
    save_local: true,  auto_push: true   → local + cloud
    save_local: false, auto_push: true   → cloud-only (staged locally, deleted after a
                                           verified upload; kept on upload failure)
    save_local: false, auto_push: false  → rejected: the artifact would persist nowhere

Credentials are never stored here: the Hugging Face backend uses the standard
``HF_TOKEN`` environment variable or ``hf auth login`` token.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from ..paths import DEFAULT_SETTINGS, resolve

ArtifactKind = Literal["dataset", "model"]


@dataclass
class ArtifactPolicy:
    save_local: bool = True
    auto_push: bool = False


@dataclass
class StorageSettings:
    backend: str = "huggingface"
    namespace: str | None = None  # cloud user/org; None → backend account default (HF whoami)
    private: bool = True  # newly created cloud repos are private unless overridden
    datasets: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    models: ArtifactPolicy = field(default_factory=ArtifactPolicy)
    path: Path | None = None

    def policy(self, kind: ArtifactKind) -> ArtifactPolicy:
        return self.datasets if kind == "dataset" else self.models

    def validate(self) -> list[str]:
        problems = []
        for kind in ("datasets", "models"):
            p: ArtifactPolicy = getattr(self, kind)
            if not p.save_local and not p.auto_push:
                problems.append(f"storage.{kind}: save_local=false with auto_push=false would persist nothing")
        return problems

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: Path | None = None) -> StorageSettings:
        s = cls(
            backend=d.get("backend", "huggingface"),
            namespace=d.get("namespace") or None,
            private=bool(d.get("private", True)),
            datasets=ArtifactPolicy(**(d.get("datasets") or {})),
            models=ArtifactPolicy(**(d.get("models") or {})),
            path=path,
        )
        problems = s.validate()
        if problems:
            raise ValueError("; ".join(problems))
        return s

    @classmethod
    def load(cls, path: str | Path | None = None) -> StorageSettings:
        p = resolve(path) if path else DEFAULT_SETTINGS
        if not p.is_file():
            return cls(path=p)
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data.get("storage") or {}, path=p)
