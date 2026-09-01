"""Cloud dataset/model storage (Hugging Face backend; local storage stays the default).

Python API::

    from yamkit import storage

    storage.push_dataset("pick_cube")                  # data/datasets/pick_cube → <namespace>/pick_cube
    storage.pull_dataset("user/pick_cube")             # → data/datasets/pick_cube
    storage.resolve_dataset("pick_cube")               # local path, pulled from the cloud if absent
    storage.push_model("outputs/train/model")
    storage.pull_model("user/model")                   # → data/models/model
    storage.load_lerobot_dataset("user/pick_cube")     # resolve + open as LeRobotDataset

Behaviour (local-only, local+cloud, cloud-only) is configured in ``configs/yamkit.yaml``
under ``storage:`` — see ``yamkit.storage.settings``.
"""

from .artifacts import (
    PushResult,
    finalize,
    full_repo_id,
    load_lerobot_dataset,
    pull,
    pull_dataset,
    pull_model,
    push,
    push_dataset,
    push_model,
    resolve,
    resolve_dataset,
    resolve_model,
)
from .base import BACKENDS, CloudBackend, StorageError, get_backend
from .settings import ArtifactPolicy, StorageSettings

__all__ = [
    "BACKENDS",
    "ArtifactPolicy",
    "CloudBackend",
    "PushResult",
    "StorageError",
    "StorageSettings",
    "finalize",
    "full_repo_id",
    "get_backend",
    "load_lerobot_dataset",
    "pull",
    "pull_dataset",
    "pull_model",
    "push",
    "push_dataset",
    "push_model",
    "resolve",
    "resolve_dataset",
    "resolve_model",
]
