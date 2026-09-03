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
    assert rig.control.home_speed == 0.5
    rig.save()
    text = rig.path.read_text()
    assert "joint_offsets: [0.05, 0, 0, 0, 0, 0]" in text and "home_speed: 0.5" in text
    assert "yamkit align" in text and "home" in text
    loaded = RigConfig.load(rig.path)
    assert loaded.arm("left_leader").joint_offsets == [0.05, 0, 0, 0, 0, 0]
    with pytest.raises(ValueError):
        ArmSpec(name="x", role="leader", can_serial="1", joint_offsets=[0.1, 0.2])
