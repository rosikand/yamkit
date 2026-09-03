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
    assert parsed == {"episode": 3, "phase": "record"}
    parse_line("INFO Reset the environment", parsed)
    assert parsed["phase"] == "reset"
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
