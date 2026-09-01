"""High-level dataset/model storage operations: push, pull, resolve, finalize.

The rules that matter for data safety:
  * A local copy is only ever deleted after the upload has been *verified* (every local
    file listed in the cloud repo). Any push or verification failure keeps the local copy.
  * ``finalize`` (called after a successful recording/training) never deletes a local
    artifact unless it was pushed in the same call.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from ..paths import DATASETS_DIR, MODELS_DIR
from ..paths import resolve as resolve_path
from .base import CloudBackend, StorageError, get_backend
from .settings import ArtifactKind, StorageSettings

_KIND_DIR: dict[ArtifactKind, Path] = {"dataset": DATASETS_DIR, "model": MODELS_DIR}
# path components that make a poor cloud repo name (checkpoint layout boilerplate)
_GENERIC_NAMES = {"pretrained_model", "checkpoints", "last", "model", "train", "outputs"}


@dataclass
class PushResult:
    repo_id: str
    revision: str
    n_files: int
    deleted_local: bool
    local_dir: Path


def _backend(settings: StorageSettings | None, backend: CloudBackend | None) -> tuple[StorageSettings, CloudBackend]:
    settings = settings or StorageSettings.load()
    return settings, (backend or get_backend(settings))


def _local_files(local_dir: Path) -> set[str]:
    return {p.relative_to(local_dir).as_posix() for p in local_dir.rglob("*") if p.is_file()}


def repo_name_for(source: Path) -> str:
    """Default cloud repo name for a local artifact dir (skips checkpoint-layout boilerplate)."""
    for part in [source.name, *[p.name for p in source.parents]]:
        if part and part not in _GENERIC_NAMES:
            return part
    return source.name


def full_repo_id(name_or_id: str, kind: ArtifactKind, settings: StorageSettings | None = None, backend: CloudBackend | None = None) -> str:
    """``pick_cube`` → ``<namespace>/pick_cube``; ids that already have a namespace pass through."""
    if "/" in name_or_id:
        return name_or_id
    settings, backend = _backend(settings, backend)
    namespace = settings.namespace or backend.default_namespace()
    return f"{namespace}/{name_or_id}"


def _source_dir(source: str | Path, kind: ArtifactKind) -> Path:
    p = resolve_path(source)
    if p.is_dir():
        return p
    candidate = _KIND_DIR[kind] / str(source)
    if candidate.is_dir():
        return candidate
    raise StorageError(f"no local {kind} at {p} or {candidate}")


def push(
    source: str | Path,
    kind: ArtifactKind,
    *,
    repo_id: str | None = None,
    private: bool | None = None,
    keep_local: bool = True,
    settings: StorageSettings | None = None,
    backend: CloudBackend | None = None,
) -> PushResult:
    """Upload a local artifact dir; with ``keep_local=False`` delete it after a verified upload."""
    settings, backend = _backend(settings, backend)
    local_dir = _source_dir(source, kind)
    repo_id = full_repo_id(repo_id or repo_name_for(local_dir), kind, settings, backend)
    files = _local_files(local_dir)
    if not files:
        raise StorageError(f"{local_dir} is empty; nothing to push")
    revision = backend.push_dir(local_dir, repo_id, kind, settings.private if private is None else private)
    missing = files - backend.list_files(repo_id, kind)
    if missing:
        raise StorageError(f"upload to {repo_id} could not be verified; missing remotely: {sorted(missing)[:5]} — local copy kept at {local_dir}")
    deleted = False
    if not keep_local:
        shutil.rmtree(local_dir)
        deleted = True
    return PushResult(repo_id=repo_id, revision=revision, n_files=len(files), deleted_local=deleted, local_dir=local_dir)


def pull(
    name_or_id: str,
    kind: ArtifactKind,
    *,
    dest: str | Path | None = None,
    settings: StorageSettings | None = None,
    backend: CloudBackend | None = None,
) -> Path:
    """Download ``user/name`` (or a bare name in your namespace) into data/{datasets,models}/<name>."""
    settings, backend = _backend(settings, backend)
    repo_id = full_repo_id(name_or_id, kind, settings, backend)
    target = resolve_path(dest) if dest else _KIND_DIR[kind] / repo_id.rsplit("/", 1)[1]
    if not backend.exists(repo_id, kind):
        raise StorageError(f"{kind} {repo_id!r} not found on {backend.name}")
    return backend.pull_dir(repo_id, target, kind)


def resolve(
    name_or_id: str | Path,
    kind: ArtifactKind,
    *,
    settings: StorageSettings | None = None,
    backend: CloudBackend | None = None,
) -> Path:
    """Local path if it exists (also under data/{datasets,models}); otherwise pull from the cloud."""
    p = resolve_path(name_or_id)
    if p.is_dir():
        return p
    local = _KIND_DIR[kind] / str(name_or_id)
    if local.is_dir():
        return local
    return pull(str(name_or_id), kind, settings=settings, backend=backend)


def finalize(
    source: str | Path,
    kind: ArtifactKind,
    *,
    repo_id: str | None = None,
    push_override: bool | None = None,
    save_local_override: bool | None = None,
    settings: StorageSettings | None = None,
    backend: CloudBackend | None = None,
) -> PushResult | None:
    """Apply the configured storage policy to a freshly produced artifact.

    Returns the push result, or None if the policy is local-only. Refuses to delete a local
    copy that was not pushed (so a ``--no-save-local --no-push`` combination cannot lose data).
    """
    settings = settings or StorageSettings.load()
    policy = settings.policy(kind)
    do_push = policy.auto_push if push_override is None else push_override
    save_local = policy.save_local if save_local_override is None else save_local_override
    if not do_push:
        if not save_local:
            raise StorageError(f"refusing to discard {kind} {source}: no push requested and save_local is false")
        return None
    return push(source, kind, repo_id=repo_id, keep_local=save_local, settings=settings, backend=backend)


# ---- convenience wrappers (the public Python API) ------------------------------------------------
def push_dataset(source: str | Path, **kw) -> PushResult:
    return push(source, "dataset", **kw)


def pull_dataset(name_or_id: str, **kw) -> Path:
    return pull(name_or_id, "dataset", **kw)


def resolve_dataset(name_or_id: str | Path, **kw) -> Path:
    return resolve(name_or_id, "dataset", **kw)


def push_model(source: str | Path, **kw) -> PushResult:
    return push(source, "model", **kw)


def pull_model(name_or_id: str, **kw) -> Path:
    return pull(name_or_id, "model", **kw)


def resolve_model(name_or_id: str | Path, **kw) -> Path:
    return resolve(name_or_id, "model", **kw)


def load_lerobot_dataset(name_or_id: str | Path, **kw):
    """Resolve (pulling if needed) and open as a ``lerobot`` ``LeRobotDataset``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = resolve_dataset(name_or_id, **kw)
    return LeRobotDataset(repo_id=f"yamkit/{root.name}", root=root)
