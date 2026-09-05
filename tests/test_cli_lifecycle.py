import numpy as np
import pytest
from typer.testing import CliRunner

from yamkit import cli
from yamkit.arm import YamArm


@pytest.mark.parametrize("args", [
    ["read", "--hz", "0"], ["read", "--hz", "nan"],
    ["read", "--duration", "inf"], ["read", "--duration", "-1"],
    ["teleop", "--duration", "nan"], ["teleop", "--duration", "-1"],
    ["teleop", "--hz", "0"], ["teleop", "--bilateral-kp", "nan"],
    ["rest", "--speed", "nan"], ["rest", "--speed", "-1"],
    ["read", "left_leader", "missing"], ["rest", "left_leader", "missing"],
])
def test_cli_invalid_options_open_no_arms(rig, fake_connect, args):
    result = CliRunner().invoke(cli.app, [*args, "--rig", str(rig.path)])
    assert result.exit_code != 0
    assert not fake_connect


@pytest.mark.parametrize("command", ["read", "rest", "align"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_cli_partial_open_cancellation_closes_arms(rig, fake_connect, monkeypatch, command, error):
    connect = YamArm.connect

    def fail_connect(spec, channel, **kw):
        if spec.name == "left_follower":
            raise error("open failed")
        return connect(spec, channel, **kw)

    monkeypatch.setattr(YamArm, "connect", staticmethod(fail_connect))
    monkeypatch.setattr(cli, "_joint_stops", lambda spec: np.zeros((6, 2)))
    args = [command, "--rig", str(rig.path)]
    if command == "align":
        args += ["left_follower", "--yes"]
    CliRunner().invoke(cli.app, args)
    assert fake_connect and all(robot.closed for robot in fake_connect.values())


@pytest.mark.parametrize("command", ["read", "rest", "align"])
def test_cli_close_failure_does_not_skip_remaining_arms(rig, fake_connect, monkeypatch, command):
    close = YamArm.close

    def fail_close(arm, **kw):
        close(arm, **kw)
        if arm.name == "left_leader":
            raise RuntimeError("close failed")

    monkeypatch.setattr(YamArm, "close", fail_close)
    monkeypatch.setattr(cli, "_joint_stops", lambda spec: np.zeros((6, 2)))
    args = [command, "--rig", str(rig.path)]
    if command == "read":
        args += ["--duration", "0"]
    elif command == "align":
        args += ["left_follower", "--yes"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code != 0
    assert all(robot.closed for robot in fake_connect.values())


def test_teleop_cancellation_before_run_still_closes_session(rig, fake_connect, monkeypatch):
    def fail_print(*args, **kw):
        raise KeyboardInterrupt("cancel before run")

    monkeypatch.setattr(cli.console, "print", fail_print)
    result = CliRunner().invoke(cli.app, ["teleop", "--rig", str(rig.path)])
    assert result.exit_code != 0
    assert len(fake_connect) == 4
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())


def test_set_rest_preserves_precision_at_joint_bound(rig, fake_connect):
    from yamkit.arm import joint_limits
    from yamkit.config import RigConfig

    pose = joint_limits(rig.arm("left_follower"))[:, 1]
    fake_connect.presets["left_follower"] = pose
    result = CliRunner().invoke(cli.app, ["set-rest", "left_follower", "--rig", str(rig.path)])
    assert result.exit_code == 0, result.output
    saved = RigConfig.load(rig.path).arm("left_follower")
    np.testing.assert_array_equal(saved.rest_pose, pose)
    assert fake_connect["left_follower"].closed and not fake_connect["left_follower"].commands


def test_align_rejects_offsets_that_invalidate_home_without_saving(rig, fake_connect, monkeypatch):
    monkeypatch.setattr(cli, "_joint_stops", lambda spec: np.column_stack((np.zeros(6), np.ones(6))))
    fake_connect.presets["left_follower"] = np.array([0, 0.2, 0, 0, 0, 0])
    before = rig.path.read_text()
    result = CliRunner().invoke(cli.app, ["align", "left_follower", "--yes", "--rig", str(rig.path)])
    assert result.exit_code == 1 and "home pose" in str(result.exception)
    assert rig.path.read_text() == before
    assert all(robot.closed for robot in fake_connect.values())


@pytest.mark.parametrize("tolerance", [0, -1, float("nan"), float("inf")])
def test_alignment_invalid_tolerance_does_not_open_arms(rig, fake_connect, monkeypatch, tolerance):
    monkeypatch.setattr(cli, "STOP_TOL_DEG", tolerance)
    result = CliRunner().invoke(cli.app, ["align", "left_follower", "--yes", "--rig", str(rig.path)])
    assert result.exit_code != 0 and not fake_connect
