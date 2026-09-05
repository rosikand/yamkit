"""Hardware-free tests for the web UI backend (`yamkit ui`).

Sessions are exercised with harmless `python -c` child processes; the API is exercised with
FastAPI's TestClient. Nothing here (and nothing in the UI's read-only endpoints) may ever call
YamArm.connect — the last test enforces that.
"""

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from yamkit.ui import catalog
from yamkit.ui.server import create_app
from yamkit.ui.sessions import DeploymentLog, SessionManager, parse_line

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "data" / "datasets" / "smoke"


# ----------------------------------------------------------------------------- line parsing --
def test_parse_read_line():
    parsed = {}
    parse_line("   left_follower q=[+0.001 -0.512 +0.300 +0.000 -0.100 +0.200] grip=0.98", parsed)
    parse_line("     left_leader q=[+0.000 +0.000 +0.000 +0.000 +0.000 +0.000] grip=- btn=10", parsed)
    assert parsed["arms"]["left_follower"]["q"][1] == -0.512
    assert parsed["arms"]["left_follower"]["gripper"] == 0.98
    assert parsed["arms"]["left_leader"]["gripper"] is None
    assert parsed["arms"]["left_leader"]["buttons"] == "10"


def test_parse_teleop_line():
    parsed = {}
    parse_line(
        "[ 99.8Hz] left_leader->left_follower: ENGAGED err=0.012rad grip=0.98 | "
        "right_leader->right_follower: idle    err=nanrad grip=-",
        parsed,
    )
    assert parsed["rate_hz"] == 99.8
    assert parsed["pairs"]["left_leader->left_follower"]["engaged"] is True
    assert parsed["pairs"]["left_leader->left_follower"]["error_rad"] == 0.012
    assert parsed["pairs"]["right_leader->right_follower"]["engaged"] is False


def test_parse_record_and_policy_lines():
    parsed = {}
    parse_line("INFO 2026-01-01 Recording episode 3", parsed)
    assert parsed["episode"] == 3 and parsed["phase"] == "record" and time.time() - parsed["phase_since"] < 5
    parse_line("INFO Reset the environment", parsed)
    assert parsed["phase"] == "reset"
    parse_line("[yamkit] recording finished — uploading x to the Hub", parsed)
    assert parsed["phase"] == "upload"
    parse_line("│ first call (new chunk) │ 834 ms │", parsed)
    assert parsed["first_call_ms"] == 834.0


# --------------------------------------------------------------------------------- sessions --
def test_session_lifecycle_and_exclusivity():
    mgr = SessionManager()
    child = "print('[ 50.0Hz] a->b: ENGAGED err=0.001rad grip=0.50'); import time; time.sleep(30)"
    argv = [sys.executable, "-u", "-c", child]
    mgr.start("teleop", argv, meta={"x": 1})
    try:
        with pytest.raises(RuntimeError, match="already running"):
            mgr.start("read", argv)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and "pairs" not in mgr.parsed:
            time.sleep(0.02)
        st = mgr.status()
        assert st["active"] and st["mode"] == "teleop" and st["meta"] == {"x": 1}
        assert st["parsed"]["pairs"]["a->b"]["engaged"] is True
    finally:
        mgr.stop()
    assert mgr.wait(timeout=5) is not None
    assert not mgr.active


def test_session_exit_callback_and_deployment_log(tmp_path):
    done = {}
    mgr = SessionManager(on_exit=lambda st: done.update(st))
    dlog = DeploymentLog(tmp_path / "deployments")
    mgr.start("policy-check", [sys.executable, "-c", "print('first call (new chunk)  834 ms')"],
              meta={"policy": "p", "task": "t"})
    run_dir = dlog.create(mgr.status())
    assert mgr.wait(timeout=10) == 0
    dlog.finalize(run_dir, mgr.status())
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "success" and meta["policy"] == "p"
    assert meta["first_call_ms"] == 834.0
    assert (run_dir / "log.txt").is_file()
    assert done["returncode"] == 0
    assert catalog.list_deployments(tmp_path / "deployments")[0]["id"] == run_dir.name


# ---------------------------------------------------------------------------------- catalog --
def test_dataset_catalog_reads_committed_smoke_dataset():
    ds = catalog.dataset_summary(SMOKE)
    assert ds["episodes"] == 1 and ds["frames"] == 120 and ds["fps"] == 30
    assert ds["tasks"] == ["smoke test"] and ds["cameras"] == []
    detail = catalog.dataset_detail(SMOKE)
    assert detail["episode_list"][0]["length"] == 120
    series = catalog.episode_series(SMOKE, 0)
    assert series["n_frames"] == 120
    assert len(series["observation.state"][0]) == 14 and len(series["names"]) == 14
    assert len(series["timestamp"]) == len(series["action"])


def test_models_catalog(tmp_path):
    ckpt = tmp_path / "train" / "job" / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"type": "smolvla"}')
    (ckpt / "model.safetensors").write_bytes(b"\0" * 8)
    models = catalog.list_models(tmp_path)
    assert len(models) == 1
    assert models[0]["policy_type"] == "smolvla"
    assert models[0]["path"] == "train/job/checkpoints/last/pretrained_model"


# -------------------------------------------------------------------------------------- api --
@pytest.fixture
def client(rig, tmp_path, monkeypatch):
    """App wired to the fixture rig and a tmp outputs dir; arm connections are booby-trapped."""
    from yamkit import arm as arm_mod

    def boom(*a, **kw):
        raise AssertionError("the UI backend must never connect to an arm in-process")

    monkeypatch.setattr(arm_mod.YamArm, "connect", staticmethod(boom))
    app = create_app(rig.path, datasets_dir=ROOT / "data" / "datasets", outputs_dir=tmp_path,
                     frontend_dir=ROOT / "ui")
    with TestClient(app) as c:
        yield c


def test_overview_and_pages_are_hardware_free(client):
    r = client.get("/api/overview")
    assert r.status_code == 200
    body = r.json()
    assert body["rig"]["found"] is True
    assert set(body["rig"]["arms"]) == {"left_leader", "left_follower", "right_leader", "right_follower"}
    assert body["session"] == {"active": False, "mode": None}
    assert client.get("/api/session").json()["active"] is False
    assert client.get("/api/cameras").status_code == 200
    assert client.get("/api/models").json() == []
    assert client.get("/api/deployments").json() == []
    # frontend is served
    assert b"yamkit" in client.get("/").content


def test_dataset_endpoints(client):
    names = [d["name"] for d in client.get("/api/datasets").json()]
    assert "smoke" in names
    d = client.get("/api/datasets/smoke").json()
    assert d["episodes"] == 1 and d["features"]["action"]["shape"] == [14]
    ep = client.get("/api/datasets/smoke/episodes/0").json()
    assert ep["n_frames"] == 120
    assert client.get("/api/datasets/nope").status_code == 404
    assert client.get("/api/datasets/../secrets").status_code == 404
    assert client.get("/api/datasets/smoke/video/top/0").status_code == 404  # no videos in smoke


def test_session_endpoints_spawn_and_stop(client, monkeypatch):
    # Replace the yamkit CLI child with a harmless stand-in that just idles.
    monkeypatch.setattr(SessionManager, "yamkit_argv",
                        lambda self, *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    r = client.post("/api/session/teleop", json={"auto_engage": False})
    assert r.status_code == 200 and r.json()["active"] is True
    assert client.post("/api/session/record",
                       json={"name": "x", "task": "y"}).status_code == 409  # busy
    r = client.post("/api/session/stop")
    assert r.status_code == 200
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and client.get("/api/session").json()["active"]:
        time.sleep(0.05)
    assert client.get("/api/session").json()["active"] is False


def test_record_requires_name_and_task(client):
    assert client.post("/api/session/record", json={}).status_code == 422


def test_cli_teleop_print_state_emits_read_format_lines(rig, fake_connect):
    """`yamkit teleop --print-state` interleaves per-arm state lines the UI parser understands."""
    from typer.testing import CliRunner

    from yamkit.cli import app as cli_app

    res = CliRunner().invoke(
        cli_app, ["teleop", "--rig", str(rig.path), "--pair", "left_follower",
                  "--duration", "0.6", "--print-state"])
    assert res.exit_code == 0, res.output
    parsed = {}
    for line in res.output.splitlines():
        parse_line(line, parsed)
    assert "left_leader->left_follower" in parsed["pairs"]
    assert len(parsed["arms"]["left_follower"]["q"]) == 6
    assert parsed["arms"]["left_leader"]["buttons"] is not None


def test_teleop_session_requests_state_lines(client, monkeypatch):
    seen = {}

    def argv(self, *args):
        seen["args"] = args
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(SessionManager, "yamkit_argv", argv)
    assert client.post("/api/session/teleop", json={}).status_code == 200
    assert "--print-state" in seen["args"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and client.get("/api/session").json()["active"]:
        time.sleep(0.05)


def test_config_get_and_structured_save(client, rig):
    c = client.get("/api/config").json()
    assert c["found"] and c["control"]["teleop_hz"] == 100.0
    assert set(c["arms"]) == {"left_leader", "left_follower", "right_leader", "right_follower"}
    r = client.post("/api/config", json={"control": {"max_joint_speed": 2.5}})
    assert r.status_code == 200 and r.json()["control"]["max_joint_speed"] == 2.5
    assert "max_joint_speed: 2.5" in rig.path.read_text()
    assert client.post("/api/config", json={"control": {"nope": 1}}).status_code == 422


def test_config_yaml_save_validates_before_writing(client, rig):
    before = rig.path.read_text()
    # invalid: broken syntax, bad role, and pair referencing a missing arm — none may be written
    for bad in ("{{not yaml", "arms:\n  a:\n    role: sideways\n",
                before.replace("left_follower:", "other_follower:", 1)):
        assert client.post("/api/config", json={"yaml_text": bad}).status_code == 422
        assert rig.path.read_text() == before
    good = before + "\n# tuned on 2026-09-01\n"
    r = client.post("/api/config", json={"yaml_text": good})
    assert r.status_code == 200
    assert rig.path.read_text() == good  # verbatim write keeps comments


def test_config_save_rejected_while_session_runs(client, monkeypatch):
    monkeypatch.setattr(SessionManager, "yamkit_argv",
                        lambda self, *args: [sys.executable, "-c", "import time; time.sleep(30)"])
    client.post("/api/session/teleop", json={})
    try:
        assert client.post("/api/config", json={"control": {"teleop_hz": 50}}).status_code == 409
    finally:
        client.post("/api/session/stop")


def test_model_detail_endpoint(client, tmp_path):
    ckpt = tmp_path / "train" / "job" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"type": "act", "n_action_steps": 100}')
    (ckpt / "model.safetensors").write_bytes(b"\0" * 16)
    d = client.get("/api/models/train/job/pretrained_model").json()
    assert d["policy_type"] == "act" and d["config"]["n_action_steps"] == 100
    assert {f["name"] for f in d["files"]} == {"config.json", "model.safetensors"}
    assert client.get("/api/models/nope").status_code == 404
    assert client.get("/api/models/../secrets").status_code == 404


def test_config_save_reloads_camera_list(client, rig):
    assert client.get("/api/cameras").json() == []
    text = rig.path.read_text().replace("cameras: {}", "cameras:\n  top: {type: opencv, index_or_path: /dev/video99, width: 640, height: 480, fps: 30}\n")
    assert client.post("/api/config", json={"yaml_text": text}).status_code == 200
    cams = client.get("/api/cameras").json()
    assert [c["name"] for c in cams] == ["top"] and cams[0]["device"] == "/dev/video99"
    assert client.get("/api/overview").json()["cameras"][0]["name"] == "top"
    assert client.post("/api/config", json={"control": {"teleop_hz": 50}}).status_code == 200
    assert [c["name"] for c in client.get("/api/cameras").json()] == ["top"]  # unchanged entry kept


def test_park_endpoint_runs_rest(client, monkeypatch):
    seen = {}

    def argv(self, *args):
        seen["args"] = args
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(SessionManager, "yamkit_argv", argv)
    r = client.post("/api/session/rest", json={"arms": ["left_follower"]})
    assert r.status_code == 200 and r.json()["mode"] == "rest"
    assert seen["args"][:2] == ("rest", "left_follower")
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and client.get("/api/session").json()["active"]:
        time.sleep(0.05)


@pytest.fixture
def fake_hub(monkeypatch):
    """The UI's hub calls, without network: one cloud dataset that also exists locally ("smoke") and one cloud-only."""
    from yamkit import hub

    monkeypatch.setattr(hub, "get_token", lambda: "hf_test")
    monkeypatch.setattr(hub, "status", lambda: {"logged_in": True, "username": "tester", "token_path": "/x/token", "online": True, "error": None})
    monkeypatch.setattr(hub, "list_datasets", lambda user=None: [
        {"name": "smoke", "repo_id": "tester/smoke", "private": True, "episodes": 1, "frames": 120, "fps": 30, "robot_type": "bi", "tasks": [], "cameras": [], "size_bytes": 5, "modified": 1.0, "url": "https://huggingface.co/datasets/tester/smoke"},
        {"name": "cloud_only", "repo_id": "tester/cloud_only", "private": False, "episodes": 30, "frames": 9000, "fps": 30, "robot_type": "bi", "tasks": ["t"], "cameras": ["observation.images.top"], "size_bytes": 99, "modified": 2.0, "url": "https://huggingface.co/datasets/tester/cloud_only"},
    ])
    monkeypatch.setattr(hub, "list_models", lambda user=None: [
        {"name": "job", "repo_id": "tester/job", "path": "tester/job", "private": True, "policy_type": "smolvla", "files": [], "size_bytes": 8, "modified": 1.0, "steps": 100, "dataset": None, "url": "https://huggingface.co/tester/job"},
        {"name": "cloud_model", "repo_id": "tester/cloud_model", "path": "tester/cloud_model", "private": True, "policy_type": "act", "files": [], "size_bytes": 8, "modified": 1.0, "steps": 100, "dataset": None, "url": "https://huggingface.co/tester/cloud_model"},
    ])
    monkeypatch.setattr(hub, "login", lambda token: "tester" if token == "hf_good" else (_ for _ in ()).throw(RuntimeError("Invalid user token")))
    monkeypatch.setattr(hub, "logout", lambda: None)
    monkeypatch.setattr(hub, "model_detail", lambda repo: {"path": repo, "repo_id": repo, "where": "cloud", "url": "u", "policy_type": "act", "files": [], "size_bytes": 0, "modified": None, "config": {"type": "act"}, "train_config": {}} if repo == "tester/cloud_model" else None)


def test_datasets_and_models_merge_local_and_hub(client, fake_hub, tmp_path):
    rows = {d["name"]: d for d in client.get("/api/datasets").json()}
    assert rows["smoke"]["where"] == "both" and rows["smoke"]["repo_id"] == "tester/smoke" and rows["smoke"]["frames"] == 120  # local numbers kept
    assert rows["cloud_only"]["where"] == "cloud" and rows["cloud_only"]["episodes"] == 30
    ckpt = tmp_path / "train" / "job" / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text('{"type": "smolvla"}')
    (ckpt / "model.safetensors").write_bytes(b"\0" * 8)
    models = {m["name"]: m for m in client.get("/api/models").json()}
    assert models["job"]["where"] == "both" and models["job"]["path"].startswith("train/job/")
    assert models["cloud_model"]["where"] == "cloud" and models["cloud_model"]["repo_id"] == "tester/cloud_model"
    assert client.get("/api/hub/models/tester/cloud_model").json()["policy_type"] == "act"
    assert client.get("/api/hub/models/tester/nope").status_code == 404


def test_datasets_without_token_are_local_only(client, monkeypatch):
    from yamkit import hub

    monkeypatch.setattr(hub, "get_token", lambda: None)
    rows = client.get("/api/datasets").json()
    assert rows and all(d["where"] == "local" for d in rows)
    assert client.get("/api/overview").json()["hub"]["logged_in"] is False


def test_hub_login_status_and_settings(client, fake_hub, rig):
    st = client.get("/api/hub").json()
    assert st["logged_in"] and st["username"] == "tester" and st["settings"]["datasets"] == "local"
    assert client.post("/api/hub/login", json={"token": "bad"}).status_code == 400
    assert client.post("/api/hub/login", json={"token": "hf_good"}).json() == {"username": "tester"}
    r = client.post("/api/config", json={"hub": {"username": "rigger", "datasets": "hub", "private": False}})
    assert r.status_code == 200 and r.json()["hub"] == {"username": "rigger", "private": False, "datasets": "hub"}
    assert "datasets: hub" in rig.path.read_text()
    assert client.post("/api/config", json={"hub": {"datasets": "moon"}}).status_code == 422
    assert client.post("/api/config", json={"hub": {"nope": 1}}).status_code == 422
    assert client.get("/api/overview").json()["hub"] == {"logged_in": True, "username": "rigger", "private": False, "datasets": "hub"}


def test_record_and_transfers_pass_destination(client, fake_hub, monkeypatch, tmp_path):
    seen = []

    def argv(self, *args):
        seen.append(args)
        return [sys.executable, "-c", "pass"]

    monkeypatch.setattr(SessionManager, "yamkit_argv", argv)

    def wait():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.get("/api/session").json()["active"]:
            time.sleep(0.05)

    assert client.post("/api/session/record", json={"name": "x", "task": "y", "to": "hub"}).status_code == 200
    wait()
    assert "--to" in seen[-1] and seen[-1][seen[-1].index("--to") + 1] == "hub"
    assert client.post("/api/session/record", json={"name": "x", "task": "y", "to": "moon"}).status_code == 422
    assert client.post("/api/hub/push-dataset", json={"name": "smoke", "remove_local": True}).status_code == 200
    wait()
    assert seen[-1][:2] == ("push-dataset", "smoke") and "--remove-local" in seen[-1]
    assert client.post("/api/hub/push-dataset", json={"name": "nope"}).status_code == 404
    assert client.post("/api/hub/pull-dataset", json={"name": "tester/cloud_only"}).status_code == 200
    wait()
    assert seen[-1][:2] == ("pull-dataset", "tester/cloud_only")
    ckpt = tmp_path / "train" / "job" / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "config.json").write_text("{}")
    assert client.post("/api/hub/push-model", json={"name": "train/job/checkpoints/last/pretrained_model"}).status_code == 200
    wait()
    assert seen[-1][0] == "push-model" and seen[-1][1].endswith("pretrained_model")
    assert client.post("/api/hub/push-model", json={"name": "../etc"}).status_code == 404


def test_cameras_come_back_when_the_upload_phase_starts(rig, tmp_path, monkeypatch):
    """A record session owns the cameras; once the recorder has exited (upload phase) the feeds return."""
    from yamkit import arm as arm_mod

    monkeypatch.setattr(arm_mod.YamArm, "connect", staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("no arms in tests"))))
    rig.cameras = {"top": {"type": "opencv", "index_or_path": "/dev/video99", "width": 640, "height": 480, "fps": 30}}
    rig.save()
    app = create_app(rig.path, datasets_dir=ROOT / "data" / "datasets", outputs_dir=tmp_path, frontend_dir=ROOT / "ui")
    child = "from yamkit.camera_ownership import claim_from_env; lease = claim_from_env(['top']); print('Recording episode 0'); lease.release(); print('[yamkit] recording finished — uploading x to the Hub'); import time; time.sleep(30)"
    monkeypatch.setattr(SessionManager, "yamkit_argv", lambda self, *args: [sys.executable, "-u", "-c", child])
    with TestClient(app) as client:
        assert client.post("/api/session/record", json={"name": "x", "task": "y", "to": "hub"}).status_code == 200
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.get("/api/session").json().get("parsed", {}).get("phase") != "upload":
            time.sleep(0.05)
        st = client.get("/api/session").json()
        assert st["active"] and st["parsed"]["phase"] == "upload"
        assert client.get("/api/cameras").json()[0]["suspended_by"] is None  # feeds released during the upload
        client.post("/api/session/stop")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.get("/api/session").json()["active"]:
            time.sleep(0.05)


def test_ui_children_get_no_display(tmp_path, monkeypatch):
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    mgr = SessionManager()
    probe = "import os, json; print(json.dumps({'d': os.environ.get('DISPLAY'), 'w': os.environ.get('WAYLAND_DISPLAY')}))"
    mgr.start("record", [sys.executable, "-c", probe])
    assert mgr.wait(timeout=10) == 0
    out = json.loads(next(ln for ln in mgr.log if ln.startswith("{")))
    assert out["d"] is None and out["w"] is None  # no system-wide keyboard hook in the recorder


def test_phase_timer_is_reported(rig, tmp_path, monkeypatch):
    from yamkit import arm as arm_mod

    monkeypatch.setattr(arm_mod.YamArm, "connect", staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("no arms"))))
    app = create_app(rig.path, datasets_dir=ROOT / "data" / "datasets", outputs_dir=tmp_path, frontend_dir=ROOT / "ui")
    child = "print('Recording episode 0'); import time; time.sleep(30)"
    monkeypatch.setattr(SessionManager, "yamkit_argv", lambda self, *args: [sys.executable, "-u", "-c", child])
    with TestClient(app) as client:
        client.post("/api/session/record", json={"name": "x", "task": "y"})
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.get("/api/session").json().get("parsed", {}).get("phase") != "record":
            time.sleep(0.05)
        time.sleep(0.3)
        st = client.get("/api/session").json()
        assert st["parsed"]["episode"] == 0 and st["phase_elapsed_s"] is not None and 0.2 <= st["phase_elapsed_s"] < 5
        client.post("/api/session/stop")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.get("/api/session").json()["active"]:
            time.sleep(0.05)


def test_camera_stream_is_refused_while_a_session_owns_the_cameras(rig, tmp_path, monkeypatch):
    """A camera owner without a publisher cannot fall back to direct capture."""
    from yamkit import arm as arm_mod

    monkeypatch.setattr(arm_mod.YamArm, "connect", staticmethod(lambda *a, **k: (_ for _ in ()).throw(AssertionError("no arms"))))
    rig.cameras = {"top": {"type": "opencv", "index_or_path": "/dev/video99", "width": 640, "height": 480, "fps": 30}}
    rig.save()
    app = create_app(rig.path, datasets_dir=ROOT / "data" / "datasets", outputs_dir=tmp_path, frontend_dir=ROOT / "ui")
    child = "from yamkit.camera_ownership import claim_from_env; claim_from_env(['top']); import time; time.sleep(30)"
    monkeypatch.setattr(SessionManager, "yamkit_argv", lambda self, *args: [sys.executable, "-u", "-c", child])
    with TestClient(app) as client:
        assert client.get("/api/cameras/nope/stream").status_code == 404
        assert client.post("/api/session/record", json={"name": "x", "task": "y"}).status_code == 200
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not client.get("/api/session").json()["cameras_owned"]:
            time.sleep(0.02)
        assert client.get("/api/cameras").json()[0]["suspended_by"] == "record"
        assert client.get("/api/cameras").json()[0]["preview_state"] == "waiting"
        monkeypatch.setattr("yamkit.ui.sessions.PREVIEW_START_TIMEOUT_S", 0)
        assert client.get("/api/cameras").json()[0]["preview_state"] == "unavailable"
        r = client.get("/api/cameras/top/stream")
        assert r.status_code == 409 and "preview" in r.json()["detail"]
        assert not [r for r in app.routes if getattr(r, "path", "").endswith("/frame")]  # no snapshot-polling endpoint
        client.post("/api/session/stop")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.get("/api/session").json()["active"]:
            time.sleep(0.05)
        time.sleep(0.2)
        assert client.get("/api/cameras").json()[0]["suspended_by"] is None  # ownership released → streams allowed again


def test_page_files_are_never_cached_stale(client):
    for path in ("/", "/app.js", "/style.css"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers.get("cache-control") == "no-cache", path
    assert "cache-control" not in {k.lower() for k in client.get("/api/session").headers}  # API untouched


def test_index_references_versioned_page_files(client):
    html = client.get("/").text
    assert 'src="app.js?v=' in html and 'href="style.css?v=' in html
    v = html.split('src="app.js?v=')[1].split('"')[0]
    assert v.isdigit() and client.get(f"/app.js?v={v}").status_code == 200
