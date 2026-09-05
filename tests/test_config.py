import pytest

from yamkit.config import ArmSpec, RigConfig


def test_roundtrip(rig, tmp_path):
    loaded = RigConfig.load(tmp_path / "rig.yaml")
    assert set(loaded.arms) == set(rig.arms)
    assert loaded.arm("left_follower").can_serial == "BBB"
    assert loaded.arm("left_leader").has_handle and not loaded.arm("left_leader").has_motor_gripper
    assert loaded.arm("left_follower").n_dofs == 7 and loaded.arm("left_leader").n_dofs == 6
    assert [p.follower for p in loaded.pairs] == ["left_follower", "right_follower"]
    assert loaded.validate() == []


def test_validation_errors():
    with pytest.raises(ValueError):
        ArmSpec(name="x", role="boss")
    with pytest.raises(ValueError):
        ArmSpec(name="x", role="leader", gripper="claw")
    r = RigConfig(arms={"a": ArmSpec(name="a", role="leader", can_serial="1")})
    r.pairs = [type("P", (), {"leader": "a", "follower": "missing"})()]
    assert any("unknown arm" in p for p in r.validate())


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        RigConfig.load(tmp_path / "nope.yaml")


def test_swap_command(rig, tmp_path):
    from typer.testing import CliRunner

    from yamkit.cli import app

    res = CliRunner().invoke(app, ["swap", "left_leader", "right_leader", "--rig", str(tmp_path / "rig.yaml")])
    assert res.exit_code == 0, res.output
    loaded = RigConfig.load(tmp_path / "rig.yaml")
    assert loaded.arm("left_leader").can_serial == "CCC" and loaded.arm("right_leader").can_serial == "AAA"
    assert [(p.leader, p.follower) for p in loaded.pairs] == [("left_leader", "left_follower"), ("right_leader", "right_follower")]
    bad = CliRunner().invoke(app, ["swap", "left_leader", "left_follower", "--rig", str(tmp_path / "rig.yaml")])
    assert bad.exit_code == 1


def test_saved_rig_is_commented_and_readable(rig, tmp_path):
    rig.arm("left_follower").gripper_limits = [6.47, 1.16]
    rig.arm("left_leader").rest_pose = [0.0, -0.5, 0.3, 0.0, 0.1, 0.0]
    rig.cameras = {"top": {"type": "opencv", "index_or_path": "/dev/video10", "width": 640, "height": 480, "fps": 30, "notes": "RealSense D435"}}
    rig.save()
    text = rig.path.read_text()
    for section in ("# ---- Arms", "# ---- Pairs", "# ---- Cameras", "# ---- Control", "yamkit swap", "yamkit discover --write"):
        assert section in text
    assert "gripper_limits: [6.47, 1.16]" in text  # short numeric lists inline
    assert "rest_pose: [0.0, -0.5, 0.3, 0.0, 0.1, 0.0]" in text
    assert text.index("arms:") < text.index("pairs:") < text.index("cameras:") < text.index("control:")
    loaded = RigConfig.load(rig.path)
    assert loaded.to_dict() == rig.to_dict()
    assert rig.to_yaml() == text


def test_empty_sections_render_and_load(tmp_path):
    r = RigConfig()
    r.save(tmp_path / "empty.yaml")
    text = (tmp_path / "empty.yaml").read_text()
    assert "arms: {}" in text and "pairs: []" in text and "cameras: {}" in text
    assert RigConfig.load(tmp_path / "empty.yaml").arms == {}


def test_example_rig_loads_and_matches_writer():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "configs" / "rig.example.yaml"
    cfg = RigConfig.load(example)
    assert cfg.validate() == []
    assert set(cfg.arms) == {"left_leader", "left_follower", "right_leader", "right_follower"}
    assert set(cfg.cameras) == {"top", "left_wrist", "right_wrist"}
    # the example body is exactly what the writer produces (only the banner is hand-written)
    assert example.read_text().endswith(cfg.to_yaml().split("\n", 12)[-1])


def test_home_pose_offsets_and_home_speed(rig):
    a = rig.arm("left_follower")
    assert a.home_pose == [0.0] * 6
    a.rest_pose = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    assert a.home_pose == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    rig.arm("left_leader").joint_offsets = [0.05, 0, 0, 0, 0, 0]
    assert rig.control.home_speed == 0.25 and rig.control.leader_home_speed == 0.25
    rig.save()
    text = rig.path.read_text()
    assert "joint_offsets: [0.05, 0, 0, 0, 0, 0]" in text and "home_speed: 0.25" in text and "leader_home_speed: 0.25" in text
    assert "yamkit align" in text and "home" in text
    loaded = RigConfig.load(rig.path)
    assert loaded.arm("left_leader").joint_offsets == [0.05, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        ArmSpec(name="x", role="leader", can_serial="1", joint_offsets=[0.1, 0.2])


def test_train_adds_cpu_defaults_without_cuda(monkeypatch):
    import torch
    from typer.testing import CliRunner

    from yamkit.cli import app

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    res = CliRunner().invoke(app, ["train", "--dataset", "x", "--policy-type", "act", "--pretrained", "", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "--policy.device=cpu" in res.output and "--num_workers=0" in res.output
    res = CliRunner().invoke(app, ["train", "--dataset", "x", "--dry-run", "--num_workers=2"])
    assert "--num_workers=0" not in res.output and "--num_workers=2" in res.output and "--policy.device=cpu" in res.output
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    res = CliRunner().invoke(app, ["train", "--dataset", "x", "--dry-run"])
    assert "--policy.device=cpu" not in res.output and "--num_workers" not in res.output


def test_hub_settings_roundtrip_and_validation(rig):
    from yamkit.config import HubSpec

    assert rig.hub == HubSpec() and rig.hub.datasets == "local" and rig.hub.private is True
    rig.hub.username = "tester"
    rig.hub.datasets = "hub"
    rig.save()
    text = rig.path.read_text()
    assert "# ---- Hugging Face Hub" in text and "username: tester" in text and "datasets: hub" in text
    assert not [ln for ln in text.splitlines() if "token" in ln.lower() and not ln.lstrip().startswith("#")]  # comments explain where the token lives; the data holds none
    loaded = RigConfig.load(rig.path)
    assert loaded.hub.username == "tester" and loaded.hub.datasets == "hub"
    with pytest.raises(ValueError):
        HubSpec(datasets="moon")
    old = RigConfig.from_dict({"arms": {}, "pairs": [], "control": {}})  # rigs written before this field
    assert old.hub == HubSpec()


def test_record_and_train_hub_flags(rig, monkeypatch):
    from typer.testing import CliRunner

    from yamkit import hub
    from yamkit.cli import app

    rig.hub.username = "tester"
    rig.save()
    monkeypatch.setattr(hub, "get_token", lambda: None)  # no sign-in needed when hub.username is set
    res = CliRunner().invoke(app, ["record", "--name", "cube", "--task", "t", "--rig", str(rig.path), "--to", "hub", "--dry-run"])
    assert res.exit_code == 0, res.output
    # the recorder itself is started exactly as before: no Hub repo id, no push, nothing Hub-related until afterwards
    assert "--dataset.repo_id=yamkit/cube" in res.output and "--dataset.push_to_hub=false" in res.output
    assert "then: upload" in res.output and "remove the local copy" in res.output
    res = CliRunner().invoke(app, ["record", "--name", "cube", "--task", "t", "--rig", str(rig.path), "--dry-run"])  # rig default: local
    assert "--dataset.repo_id=yamkit/cube" in res.output and "upload" not in res.output
    rig.hub.datasets = "both"
    rig.save()
    res = CliRunner().invoke(app, ["record", "--name", "cube", "--task", "t", "--rig", str(rig.path), "--dry-run"])  # rig default: both
    assert "then: upload" in res.output and "remove the local copy" not in res.output
    res = CliRunner().invoke(app, ["record", "--name", "cube", "--task", "t", "--rig", str(rig.path), "--to", "local", "--dry-run"])
    assert "--dataset.repo_id=yamkit/cube" in res.output and "upload" not in res.output
    assert CliRunner().invoke(app, ["record", "--name", "cube", "--task", "t", "--rig", str(rig.path), "--to", "moon", "--dry-run"]).exit_code != 0

    res = CliRunner().invoke(app, ["train", "--dataset", "tester/cube", "--policy-type", "act", "--pretrained", "", "--rig", str(rig.path), "--push", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "--dataset.repo_id=tester/cube" in res.output and "--dataset.root" not in res.output  # pulled from the Hub
    assert "--policy.push_to_hub=true" in res.output and "--policy.repo_id=tester/act_cube" in res.output and "--policy.private=true" in res.output
    res = CliRunner().invoke(app, ["train", "--dataset", "cube", "--dry-run"])
    assert "--policy.push_to_hub=false" in res.output and "--dataset.root=" in res.output


def test_find_root_prefers_the_checkout_the_code_runs_from(monkeypatch, tmp_path):
    """A stale YAMKIT_ROOT from another clone's env.sh must not redirect this clone's rig and data."""
    from yamkit import _env

    here = _env.find_root()
    assert here is not None and (here / "pyproject.toml").is_file()
    other = tmp_path / "other-clone"
    (other / "configs").mkdir(parents=True)
    (other / "pyproject.toml").write_text("")
    monkeypatch.setenv("YAMKIT_ROOT", str(other))
    monkeypatch.setenv("HF_HOME", str(other / "data" / "hf"))  # exported by the other clone's env.sh
    monkeypatch.setenv("TMPDIR", str(other / "data" / "tmp"))
    monkeypatch.setenv("TORCH_HOME", "/somewhere/shared/torch")  # the user's own choice: untouched
    assert _env.find_root() == here
    _env.apply()
    import os

    assert os.environ["YAMKIT_ROOT"] == str(here)
    assert os.environ["HF_HOME"] == str(here / "data" / "hf")
    assert os.environ["TMPDIR"] == str(here / "data" / "tmp")
    assert (here / "data" / "tmp").is_dir()
    assert os.environ["TORCH_HOME"] == "/somewhere/shared/torch"


def test_discover_write_keeps_a_backup_of_the_previous_rig(rig, monkeypatch):
    from typer.testing import CliRunner

    from yamkit import cli
    from yamkit.can import CanIface
    from yamkit.discovery import ChannelProbe, MotorProbe

    arm = [MotorProbe(i, 40.0 if i <= 3 else 10.0) for i in range(1, 7)]
    ifaces = [CanIface(f"can{i}", True, 1000000, s, "CANable", "x", "3-1", 0, 0, 0) for i, s in enumerate(["AAA", "BBB", "CCC", "DDD"])]
    probes = [ChannelProbe("can0", arm, ["dev1:v2.4.0"]), ChannelProbe("can1", arm + [MotorProbe(7, 10.0)]),
              ChannelProbe("can2", arm, ["dev1:v2.4.0"]), ChannelProbe("can3", arm + [MotorProbe(7, 10.0)])]
    monkeypatch.setattr("yamkit.can.list_can_interfaces", lambda: ifaces)
    monkeypatch.setattr("yamkit.discovery.probe_all", lambda ifs: probes)
    before = rig.path.read_text()
    res = CliRunner().invoke(cli.app, ["discover", "--write", "--no-cameras", "--rig", str(rig.path)])
    assert res.exit_code == 0, res.output
    assert rig.path.with_suffix(".yaml.bak").read_text() == before
    assert RigConfig.load(rig.path).arm("left_leader").can_serial == "AAA"  # names kept by serial
