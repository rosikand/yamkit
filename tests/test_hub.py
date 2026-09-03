"""Hugging Face Hub helpers with a fake `huggingface_hub` — no network."""

import json
import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from yamkit import hub


class _Sibling:
    def __init__(self, name, size):
        self.rfilename, self.size = name, size


class _Info:
    def __init__(self, files, private=False):
        self.siblings = [_Sibling(n, s) for n, s in files.items()]
        self.private = private
        self.last_modified = datetime(2026, 9, 3, 12, 0, 0, tzinfo=UTC)


class FakeApi:
    """Two datasets (one LeRobot, one plain) and two models (one LeRobot policy, one not)."""

    def __init__(self, token=None):
        self.token = token

    def whoami(self):
        if self.token == "bad":
            raise RuntimeError("Invalid user token")
        return {"name": "tester"}

    def list_datasets(self, author=None, limit=None):
        return [types.SimpleNamespace(id=f"{author}/pick_cube"), types.SimpleNamespace(id=f"{author}/notes")]

    def dataset_info(self, rid, files_metadata=False):
        if rid.endswith("pick_cube"):
            return _Info({"meta/info.json": 500, "meta/tasks.parquet": 100, "videos/a.mp4": 5_000_000}, private=True)
        return _Info({"README.md": 10})

    def list_models(self, author=None, limit=None):
        return [types.SimpleNamespace(id=f"{author}/act_pick_cube"), types.SimpleNamespace(id=f"{author}/paper")]

    def model_info(self, rid, files_metadata=False):
        if rid.endswith("act_pick_cube"):
            return _Info({"config.json": 300, "model.safetensors": 200_000_000, "train_config.json": 900})
        return _Info({"README.md": 10})

    def create_repo(self, rid, private=True, exist_ok=False):
        self.created = (rid, private)

    def upload_folder(self, folder_path=None, repo_id=None, commit_message=None):
        self.uploaded = (folder_path, repo_id)


@pytest.fixture
def fake_hf(monkeypatch, tmp_path):
    import huggingface_hub

    files = {
        "meta/info.json": json.dumps({"total_episodes": 12, "total_frames": 3000, "fps": 30, "robot_type": "bi_yam_follower",
                                      "features": {"observation.images.top": {}, "observation.state": {}}}),
        "meta/tasks.parquet": "",  # parsing is stubbed below
        "config.json": json.dumps({"type": "act"}),
        "train_config.json": json.dumps({"steps": 4000, "dataset": {"repo_id": "tester/pick_cube"}}),
    }

    def download(rid, filename, repo_type=None):
        if filename not in files:
            raise FileNotFoundError(filename)
        p = tmp_path / rid.replace("/", "_") / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(files[filename])
        return str(p)

    api = FakeApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: FakeApi(token))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(hub, "get_token", lambda: "hf_test")
    monkeypatch.setattr(hub, "_tasks_from_parquet", lambda path: ["pick up the cube"])
    hub.clear_cache()
    yield api
    hub.clear_cache()


def test_status_and_repo_id(fake_hf, monkeypatch):
    st = hub.status()
    assert st["logged_in"] and st["online"] and st["username"] == "tester"
    assert hub.repo_id("pick_cube") == "tester/pick_cube"
    assert hub.repo_id("pick_cube", "rigger") == "rigger/pick_cube"  # the rig's username wins
    assert hub.repo_id("someone/else") == "someone/else"
    monkeypatch.setattr(hub, "get_token", lambda: None)
    hub.clear_cache()
    assert hub.status() == {"logged_in": False, "username": None, "token_path": hub.status()["token_path"], "online": None, "error": None}
    with pytest.raises(RuntimeError, match="sign in"):
        hub.repo_id("x")
    assert hub.list_datasets() == [] and hub.list_models() == []


def test_status_offline_is_not_an_error(monkeypatch):
    import huggingface_hub

    class Down:
        def __init__(self, token=None):
            pass

        def whoami(self):
            raise ConnectionError("no route to host")

    monkeypatch.setattr(huggingface_hub, "HfApi", Down)
    monkeypatch.setattr(hub, "get_token", lambda: "hf_test")
    hub.clear_cache()
    st = hub.status()
    assert st["logged_in"] and st["online"] is False and "no route" in st["error"]


def test_list_datasets_keeps_only_lerobot_datasets(fake_hf):
    rows = hub.list_datasets()
    assert [r["name"] for r in rows] == ["pick_cube"]
    r = rows[0]
    assert r["repo_id"] == "tester/pick_cube" and r["private"] is True
    assert r["episodes"] == 12 and r["fps"] == 30 and r["cameras"] == ["observation.images.top"]
    assert r["tasks"] == ["pick up the cube"] and r["size_bytes"] == 5_000_600
    assert r["url"].endswith("/datasets/tester/pick_cube")
    assert hub.list_datasets() is rows  # cached


def test_list_models_and_detail(fake_hf):
    rows = hub.list_models()
    assert [r["name"] for r in rows] == ["act_pick_cube"]
    assert rows[0]["policy_type"] == "act" and rows[0]["steps"] == 4000 and rows[0]["dataset"] == "tester/pick_cube"
    d = hub.model_detail("tester/act_pick_cube")
    assert d["where"] == "cloud" and d["config"] == {"type": "act"} and {f["name"] for f in d["files"]} == {"config.json", "model.safetensors", "train_config.json"}
    assert hub.model_detail("tester/paper") is None


def test_login_validates_then_stores(fake_hf, monkeypatch):
    import huggingface_hub

    stored = {}
    monkeypatch.setattr(huggingface_hub, "login", lambda token, add_to_git_credential: stored.update(token=token))
    assert hub.login(" hf_good ") == "tester" and stored == {"token": "hf_good"}
    with pytest.raises(RuntimeError):
        hub.login("bad")
    with pytest.raises(ValueError):
        hub.login("   ")


def test_push_model_names_from_job_dir(fake_hf, tmp_path):
    import huggingface_hub

    ckpt = tmp_path / "outputs" / "train" / "act_pick_cube" / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    calls = {}

    class Api(FakeApi):
        def create_repo(self, rid, private=True, exist_ok=False):
            calls["repo"] = (rid, private)

        def upload_folder(self, folder_path=None, repo_id=None, commit_message=None):
            calls["upload"] = (Path(folder_path), repo_id)

    huggingface_hub.HfApi = lambda token=None: Api(token)
    url = hub.push_model(ckpt, private=False)
    assert calls["repo"] == ("tester/act_pick_cube", False) and calls["upload"] == (ckpt, "tester/act_pick_cube")
    assert url.endswith("/tester/act_pick_cube")
    with pytest.raises(FileNotFoundError):
        hub.push_model(tmp_path)


def test_push_dataset_requires_a_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        hub.push_dataset("nope", root=tmp_path / "nope")
