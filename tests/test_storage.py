"""Cloud storage layer: settings, push/pull/resolve semantics, CLI — all against a fake
in-memory backend (or a mocked HfApi); no network access."""

import pytest

from yamkit.storage import (
    StorageError,
    StorageSettings,
    finalize,
    full_repo_id,
    pull,
    push,
    resolve,
)
from yamkit.storage.artifacts import repo_name_for
from yamkit.storage.base import CloudBackend, get_backend


class FakeBackend(CloudBackend):
    """In-memory cloud: {(kind, repo_id): {relpath: bytes}}."""

    name = "fake"

    def __init__(self, settings):
        super().__init__(settings)
        self.repos = {}
        self.fail_push = False
        self.drop_on_push = set()  # simulate a partial upload

    def default_namespace(self):
        return "tester"

    def exists(self, repo_id, kind):
        return (kind, repo_id) in self.repos

    def push_dir(self, local_dir, repo_id, kind, private):
        if self.fail_push:
            raise StorageError("network down")
        files = {p.relative_to(local_dir).as_posix(): p.read_bytes() for p in local_dir.rglob("*") if p.is_file()}
        for rel in self.drop_on_push:
            files.pop(rel, None)
        self.repos[(kind, repo_id)] = files
        self.last_private = private
        return "rev1"

    def pull_dir(self, repo_id, dest, kind):
        for rel, data in self.repos[(kind, repo_id)].items():
            f = dest / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        return dest

    def list_files(self, repo_id, kind):
        return set(self.repos.get((kind, repo_id), {}))


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Fake backend + data dirs and settings redirected into tmp_path."""
    from yamkit.storage import artifacts
    from yamkit.storage import settings as settings_mod

    dirs = {"dataset": tmp_path / "datasets", "model": tmp_path / "models"}
    monkeypatch.setattr(artifacts, "_KIND_DIR", dirs)
    monkeypatch.setattr(settings_mod, "DEFAULT_SETTINGS", tmp_path / "yamkit.yaml")
    s = StorageSettings()
    backend = FakeBackend(s)
    monkeypatch.setattr(artifacts, "get_backend", lambda _s: backend)
    return s, backend, dirs


def make_dataset(dirs, name="pick_cube"):
    d = dirs["dataset"] / name
    (d / "meta").mkdir(parents=True)
    (d / "meta" / "info.json").write_text("{}")
    (d / "data.parquet").write_bytes(b"x" * 32)
    return d


# ---- settings -----------------------------------------------------------------------------------
def test_settings_defaults_when_missing(tmp_path):
    s = StorageSettings.load(tmp_path / "nope.yaml")
    assert s.backend == "huggingface" and s.datasets.save_local and not s.datasets.auto_push


def test_settings_load_and_validate(tmp_path):
    p = tmp_path / "yamkit.yaml"
    p.write_text("storage:\n  namespace: alice\n  private: false\n  datasets: {save_local: false, auto_push: true}\n")
    s = StorageSettings.load(p)
    assert s.namespace == "alice" and not s.private
    assert not s.datasets.save_local and s.datasets.auto_push and s.models.save_local
    p.write_text("storage:\n  models: {save_local: false, auto_push: false}\n")
    with pytest.raises(ValueError, match="persist nothing"):
        StorageSettings.load(p)


def test_unknown_backend():
    with pytest.raises(StorageError, match="unknown storage backend"):
        get_backend(StorageSettings(backend="s3"))


# ---- repo id / name resolution ------------------------------------------------------------------
def test_full_repo_id(store):
    s, backend, _ = store
    assert full_repo_id("user/pick_cube", "dataset", s, backend) == "user/pick_cube"
    assert full_repo_id("pick_cube", "dataset", s, backend) == "tester/pick_cube"  # backend whoami
    s.namespace = "org"
    assert full_repo_id("pick_cube", "dataset", s, backend) == "org/pick_cube"


def test_repo_name_skips_checkpoint_boilerplate(tmp_path):
    p = tmp_path / "outputs" / "train" / "smolvla_pick" / "checkpoints" / "last" / "pretrained_model"
    assert repo_name_for(p) == "smolvla_pick"
    assert repo_name_for(tmp_path / "my_model_dir") == "my_model_dir"


# ---- push / pull / resolve ----------------------------------------------------------------------
def test_push_pull_dataset_roundtrip(store, tmp_path):
    s, backend, dirs = store
    local = make_dataset(dirs)
    res = push("pick_cube", "dataset", settings=s, backend=backend)
    assert res.repo_id == "tester/pick_cube" and res.n_files == 2 and not res.deleted_local
    assert local.is_dir()  # local + cloud: local copy kept
    assert backend.last_private is True  # default private
    out = pull("tester/pick_cube", "dataset", dest=tmp_path / "out", settings=s, backend=backend)
    assert (out / "meta" / "info.json").read_text() == "{}"


def test_push_cloud_only_deletes_after_verify(store):
    s, backend, dirs = store
    local = make_dataset(dirs)
    res = push("pick_cube", "dataset", keep_local=False, settings=s, backend=backend)
    assert res.deleted_local and not local.exists()
    assert backend.list_files("tester/pick_cube", "dataset") == {"meta/info.json", "data.parquet"}


def test_failed_or_partial_push_keeps_local(store):
    s, backend, dirs = store
    local = make_dataset(dirs)
    backend.fail_push = True
    with pytest.raises(StorageError, match="network down"):
        push("pick_cube", "dataset", keep_local=False, settings=s, backend=backend)
    assert local.is_dir()
    backend.fail_push = False
    backend.drop_on_push = {"data.parquet"}  # upload "succeeds" but a file is missing remotely
    with pytest.raises(StorageError, match="could not be verified"):
        push("pick_cube", "dataset", keep_local=False, settings=s, backend=backend)
    assert local.is_dir() and (local / "data.parquet").exists()


def test_push_missing_or_empty(store, tmp_path):
    s, backend, dirs = store
    with pytest.raises(StorageError, match="no local dataset"):
        push("nope", "dataset", settings=s, backend=backend)
    (dirs["dataset"] / "empty").mkdir(parents=True)
    with pytest.raises(StorageError, match="nothing to push"):
        push("empty", "dataset", settings=s, backend=backend)


def test_pull_missing_repo(store):
    s, backend, _ = store
    with pytest.raises(StorageError, match="not found"):
        pull("tester/nope", "dataset", settings=s, backend=backend)


def test_resolve_local_first_then_cloud(store, tmp_path):
    s, backend, dirs = store
    local = make_dataset(dirs, "here")
    assert resolve("here", "dataset", settings=s, backend=backend) == local
    assert resolve(local, "dataset", settings=s, backend=backend) == local
    backend.repos[("model", "tester/vla")] = {"config.json": b"{}"}
    got = resolve("tester/vla", "model", settings=s, backend=backend)
    assert got == dirs["model"] / "vla" and (got / "config.json").exists()


# ---- finalize (auto-sync policy after record/train) ---------------------------------------------
def test_finalize_policies(store):
    s, backend, dirs = store
    make_dataset(dirs)
    assert finalize("pick_cube", "dataset", settings=s, backend=backend) is None  # local-only default
    with pytest.raises(StorageError, match="refusing to discard"):
        finalize("pick_cube", "dataset", save_local_override=False, settings=s, backend=backend)
    s.datasets.auto_push = True
    res = finalize("pick_cube", "dataset", settings=s, backend=backend)
    assert res.repo_id == "tester/pick_cube" and not res.deleted_local
    make_dataset(dirs, "cloud_only")
    s.datasets.save_local = False
    res = finalize("cloud_only", "dataset", settings=s, backend=backend)
    assert res.deleted_local and not (dirs["dataset"] / "cloud_only").exists()


# ---- Hugging Face backend (mocked HfApi) --------------------------------------------------------
def test_hf_backend_calls(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from yamkit.storage import huggingface as hf_mod

    api = MagicMock()
    api.whoami.return_value = {"name": "alice"}
    api.list_repo_files.return_value = ["a.txt", ".gitattributes"]
    monkeypatch.setattr(hf_mod, "HfApi", lambda: api)
    b = hf_mod.HuggingFaceBackend(StorageSettings())
    assert b.default_namespace() == "alice"
    src = tmp_path / "ds"
    src.mkdir()
    (src / "a.txt").write_text("hi")
    b.push_dir(src, "alice/ds", "dataset", private=True)
    api.create_repo.assert_called_once_with("alice/ds", repo_type="dataset", private=True, exist_ok=True)
    assert api.upload_folder.call_args.kwargs["repo_type"] == "dataset"
    assert b.list_files("alice/ds", "dataset") == {"a.txt", ".gitattributes"}
    b.pull_dir("alice/m", tmp_path / "m", "model")
    assert api.snapshot_download.call_args.kwargs["repo_type"] == "model"

    api.whoami.side_effect = RuntimeError("401")
    with pytest.raises(StorageError, match="not logged in"):
        b.default_namespace()


# ---- CLI ----------------------------------------------------------------------------------------
@pytest.fixture
def cli(store):
    from typer.testing import CliRunner

    from yamkit.cli import app

    return CliRunner(), app


def test_cli_dataset_push_pull(cli, store):
    runner, app = cli
    _, _, dirs = store
    make_dataset(dirs)
    res = runner.invoke(app, ["dataset", "push", "pick_cube"])
    assert res.exit_code == 0 and "tester/pick_cube" in res.output
    res = runner.invoke(app, ["dataset", "pull", "tester/pick_cube", "--dest", str(dirs["dataset"] / "copy")])
    assert res.exit_code == 0 and (dirs["dataset"] / "copy" / "data.parquet").exists()
    res = runner.invoke(app, ["model", "pull", "tester/nope"])
    assert res.exit_code == 1 and "not found" in res.output


def test_cli_push_delete_local(cli, store):
    runner, app = cli
    _, backend, dirs = store
    local = make_dataset(dirs)
    res = runner.invoke(app, ["dataset", "push", "pick_cube", "--delete-local"])
    assert res.exit_code == 0 and not local.exists()
    backend.fail_push = True
    local = make_dataset(dirs)
    res = runner.invoke(app, ["dataset", "push", "pick_cube", "--delete-local"])
    assert res.exit_code == 1 and local.is_dir()


def test_cli_storage_status(cli):
    runner, app = cli
    res = runner.invoke(app, ["storage"])
    assert res.exit_code == 0 and "huggingface" in res.output and "local only" in res.output


def test_record_dry_run_push_and_staging(cli, store, rig, tmp_path):
    runner, app = cli
    base = ["record", "--name", "d1", "--task", "t", "--rig", str(tmp_path / "rig.yaml"), "--dry-run"]
    res = runner.invoke(app, base)
    assert res.exit_code == 0, res.output
    assert "--dataset.push_to_hub=false" in res.output and "cloud" not in res.output
    res = runner.invoke(app, [*base, "--push"])
    assert res.exit_code == 0 and "cloud storage sync" in res.output
    res = runner.invoke(app, [*base, "--push", "--no-save-local"])
    assert res.exit_code == 0 and "data/.staging/datasets/d1" in res.output.replace("\n", "")
    res = runner.invoke(app, [*base, "--no-push", "--no-save-local"])
    assert res.exit_code != 0  # would persist nowhere


def test_train_dry_run_push_flags(cli, store, tmp_path):
    runner, app = cli
    base = ["train", "--dataset", "d1", "--dry-run"]
    res = runner.invoke(app, base)
    assert res.exit_code == 0 and "outputs/train/smolvla_d1" in res.output.replace("\n", "")
    res = runner.invoke(app, [*base, "--push", "--no-save-local"])
    assert res.exit_code == 0 and "data/.staging/train/smolvla_d1" in res.output.replace("\n", "")
