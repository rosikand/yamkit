import numpy as np
import pytest

from yamkit.teleop import TeleopSession


def test_teleop_engage_and_track(rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, auto_engage=True, home_speed=0.0)
    leader = fake_connect["left_leader"]
    follower = fake_connect["left_follower"]
    leader.pos = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    leader.encoder[0].position = 0.5  # trigger half squeezed -> gripper 0.5
    stats = session.run(duration=0.1)
    assert stats.ticks > 5
    assert np.allclose(follower.pos[:6], leader.pos, atol=0.05)
    assert follower.pos[6] == np.isclose(follower.pos[6], 0.5) or abs(follower.pos[6] - 0.5) < 0.1
    assert leader.closed and follower.closed


def test_button_toggles_engage(rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["right_follower"], hz=500.0, sync_seconds=0.01)
    pair = session.pairs[0]
    leader = fake_connect["right_leader"]
    assert not pair.engaged
    session.step()
    leader.encoder[0].io_inputs = [1, 0]  # press
    session.step()
    assert pair.engaged
    session.step()  # held: no toggle
    assert pair.engaged
    leader.encoder[0].io_inputs = [0, 0]
    session.step()
    leader.encoder[0].io_inputs = [1, 0]  # press again
    session.step()
    assert not pair.engaged
    session.shutdown()


def test_session_homes_every_arm_on_start_and_stop(rig, fake_connect):
    fake_connect.presets["left_leader"] = np.full(6, 0.4)
    fake_connect.presets["left_follower"] = np.array([0.2] * 6 + [0.5])
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, home_speed=50.0, leader_home_speed=50.0)
    leader, follower = fake_connect["left_leader"], fake_connect["left_follower"]
    stats = session.run(duration=0.05)  # start: home, idle loop (no engage), stop: home + close
    assert stats.ticks > 0
    assert np.allclose(follower.pos[:6], 0) and follower.pos[6] == pytest.approx(0.5)  # gripper left alone
    assert np.allclose(leader.pos, 0) and leader.idle_calls >= 1  # leader parked compliantly and released
    assert leader.closed and follower.closed


def test_stop_returns_home_after_teleop(rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=500.0, sync_seconds=0.01, auto_engage=True, home_speed=50.0, leader_home_speed=50.0)
    leader, follower = fake_connect["left_leader"], fake_connect["left_follower"]
    leader.pos = np.full(6, 0.3)
    session.engage(session.pairs[0])  # follower syncs to the leader pose
    session.step()
    assert np.allclose(follower.pos[:6], 0.3, atol=0.05)  # tracking
    session.shutdown()
    assert np.allclose(follower.pos[:6], 0) and np.allclose(leader.pos, 0)
    assert leader.closed and follower.closed


def test_home_speed_zero_keeps_arms_where_they_are(rig, fake_connect):
    fake_connect.presets["left_follower"] = np.array([0.3] * 6 + [1.0])
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, home_speed=0.0)
    follower = fake_connect["left_follower"]
    session.run(duration=0.05)
    assert follower.commands == [] and np.allclose(follower.pos[:6], 0.3)


def test_second_interrupt_during_return_home_releases_immediately(rig, fake_connect, monkeypatch):
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, home_speed=50.0, leader_home_speed=50.0)
    leader, follower = fake_connect["left_leader"], fake_connect["left_follower"]
    leader.pos = np.full(6, 0.3)
    calls = []

    def interrupted(*a, **k):
        calls.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(session.pairs[0].follower, "move_to", interrupted)
    session.shutdown()  # must not raise
    assert calls == [1]  # the leader move was skipped
    assert np.allclose(leader.pos, 0.3)  # leader never moved
    assert follower.idle_calls >= 1 and leader.closed and follower.closed


def test_align_command_stores_leader_offsets(rig, fake_connect):
    from typer.testing import CliRunner

    from yamkit.cli import app
    from yamkit.config import RigConfig

    fake_connect.presets["left_leader"] = np.array([0.00, 0.10, 0.00, 0.00, 0.00, 0.00])
    fake_connect.presets["left_follower"] = np.array([0.05, 0.00, 0.00, 0.00, 0.00, 0.00])
    res = CliRunner().invoke(app, ["align", "left_follower", "--rig", str(rig.path), "--yes"])
    assert res.exit_code == 0, res.output
    loaded = RigConfig.load(rig.path)
    assert np.allclose(loaded.arm("left_leader").joint_offsets, [0.05, -0.10, 0, 0, 0, 0], atol=1e-6)
    assert loaded.arm("left_follower").joint_offsets is None
    assert "joint_offsets" in rig.path.read_text()
    bad = CliRunner().invoke(app, ["align", "nope", "--rig", str(rig.path), "--yes"])
    assert bad.exit_code != 0


def test_rest_command_parks_all_arms(rig, fake_connect):
    from typer.testing import CliRunner

    from yamkit.cli import app

    for n in ("left_leader", "left_follower", "right_leader", "right_follower"):
        fake_connect.presets[n] = np.full(6, 0.2)
    res = CliRunner().invoke(app, ["rest", "--rig", str(rig.path), "--speed", "50"])
    assert res.exit_code == 0, res.output
    for n, r in fake_connect.items():
        assert np.allclose(r.pos[:6], 0) and r.closed and r.idle_calls >= 1, n


def test_leaders_home_at_their_own_gentler_speed(rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02)
    assert session.home_speed == 0.5 and session.leader_home_speed == 0.25  # rig defaults
    session.shutdown()
    assert TeleopSession(session.pairs, home_speed=1.0).leader_home_speed == 0.5  # falls back to half
