"""Hugging Face Hub backend. Authentication is the standard huggingface_hub mechanism
(``HF_TOKEN`` env var or the token stored by ``hf auth login``) — never the yamkit config."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

from .base import CloudBackend, StorageError
from .settings import ArtifactKind

_REPO_TYPE: dict[ArtifactKind, str] = {"dataset": "dataset", "model": "model"}


class HuggingFaceBackend(CloudBackend):
    name = "huggingface"

    def __init__(self, settings):
        super().__init__(settings)
        self.api = HfApi()

    def default_namespace(self) -> str:
        try:
            return self.api.whoami()["name"]
        except Exception as e:
            raise StorageError(
                "not logged in to Hugging Face and no storage.namespace configured — "
                "set HF_TOKEN or run `hf auth login`, or set storage.namespace in configs/yamkit.yaml"
            ) from e

    def exists(self, repo_id: str, kind: ArtifactKind) -> bool:
        return self.api.repo_exists(repo_id, repo_type=_REPO_TYPE[kind])

    def push_dir(self, local_dir: Path, repo_id: str, kind: ArtifactKind, private: bool) -> str:
        repo_type = _REPO_TYPE[kind]
        self.api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)
        info = self.api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=f"yamkit {kind} push",
        )
        return getattr(info, "oid", "") or str(info)

    def pull_dir(self, repo_id: str, dest: Path, kind: ArtifactKind) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.api.snapshot_download(repo_id=repo_id, repo_type=_REPO_TYPE[kind], local_dir=str(dest))
        return dest

    def list_files(self, repo_id: str, kind: ArtifactKind) -> set[str]:
        return set(self.api.list_repo_files(repo_id, repo_type=_REPO_TYPE[kind]))
