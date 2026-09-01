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
