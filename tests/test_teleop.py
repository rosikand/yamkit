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


def test_second_interrupt_during_return_home_releases_every_arm(rig, fake_connect):
    import _thread
    import threading

    # very slow homing (0.3 rad at 0.01 rad/s = 30 s); a "second Ctrl-C" arrives after 0.15 s
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, home_speed=0.01, leader_home_speed=0.01)
    leader, follower = fake_connect["left_leader"], fake_connect["left_follower"]
    leader.pos = np.full(6, 0.3)
    follower.pos[:6] = 0.3
    threading.Timer(0.15, _thread.interrupt_main).start()
    session.shutdown()  # must not raise
    assert follower.pos[0] > 0.25 and leader.pos[0] > 0.25  # both stopped early, well short of home
    assert follower.idle_calls >= 1 and leader.idle_calls >= 1  # released where they are
    assert leader.closed and follower.closed


def test_all_arms_home_at_the_same_time(rig, fake_connect):
    import time

    for n in ("left_leader", "left_follower", "right_leader", "right_follower"):
        fake_connect.presets[n] = np.full(6, 0.3)
    session = TeleopSession.from_rig(rig, hz=200.0, sync_seconds=0.02, home_speed=1.0, leader_home_speed=1.0)  # 0.3 s per arm
    t0 = time.monotonic()
    session.home_all("test")
    elapsed = time.monotonic() - t0
    assert elapsed < 0.9  # four 0.3 s moves in parallel, not 1.2 s in sequence
    for n, r in fake_connect.items():
        assert np.allclose(r.pos[:6], 0), n
    session.shutdown()


def test_align_measures_only_joints_at_a_stop(rig, fake_connect, monkeypatch):
    from typer.testing import CliRunner

    from yamkit import cli
    from yamkit.config import RigConfig

    # YAM v1 stops from the vendor URDF (rad): base [-2.618, 3.142], shoulder [0, 3.665], elbow [0, 3.142], wrists...
    stops = np.array([[-2.618, 3.1416], [0.0, 3.665], [0.0, 3.1416], [-1.693, 1.5708], [-1.5708, 1.5708], [-2.094, 2.094]])
    monkeypatch.setattr(cli, "_joint_stops", lambda spec: stops)
    rig.control.home_speed = rig.control.leader_home_speed = 50.0  # the parking move after measuring, fast for the fake arms
    rig.save()
    # base + shoulder + elbow + wrists 1,2 against stops on both arms (with small zero errors); wrist roll mid-range
    fake_connect.presets["left_leader"] = np.array([-2.60, 0.10, 0.02, -1.69, -1.55, 0.30])
    fake_connect.presets["left_follower"] = np.array([-2.55, 0.00, 0.02, -1.69, -1.57, 0.90])
    res = CliRunner().invoke(cli.app, ["align", "left_follower", "--rig", str(rig.path), "--yes"])
    assert res.exit_code == 0, res.output
    got = RigConfig.load(rig.path).arm("left_leader").joint_offsets
    assert np.allclose(got, [0.05, -0.10, 0.0, 0.0, -0.02, 0.0], atol=1e-6)  # wrist roll untouched (not at a stop)
    assert "wrist roll" in res.output and "unchanged" in res.output
    assert RigConfig.load(rig.path).arm("left_follower").joint_offsets is None

    assert np.allclose(fake_connect["left_follower"].pos[:6], 0) and np.allclose(fake_connect["left_leader"].pos, 0)  # parked afterwards
    assert fake_connect["left_leader"].closed and fake_connect["left_follower"].closed

    # a base at the +180° stop can read -177° on one arm (encoder wrap): still "at the upper stop", difference wrap-safe
    fake_connect.presets["left_leader"] = np.array([3.07, 0.0, 0.0, 0.0, 0.0, 0.0])
    fake_connect.presets["left_follower"] = np.array([-3.09, 0.0, 0.0, 0.0, 0.0, 0.0])  # = +3.193 rad
    res = CliRunner().invoke(cli.app, ["align", "left_follower", "--rig", str(rig.path), "--yes", "--reset"])
    assert res.exit_code == 0, res.output
    assert RigConfig.load(rig.path).arm("left_leader").joint_offsets[0] == pytest.approx(3.193 - 3.07, abs=2e-3)
    fake_connect.presets["left_leader"] = np.array([-2.60, 0.10, 0.02, -1.69, -1.55, 0.30])
    fake_connect.presets["left_follower"] = np.array([-2.55, 0.00, 0.02, -1.69, -1.57, 0.90])
    assert CliRunner().invoke(cli.app, ["align", "left_follower", "--rig", str(rig.path), "--yes", "--reset"]).exit_code == 0

    # a joint not at a stop keeps its previous value; --reset forgets it
    fake_connect.presets["left_leader"][0] = 0.5  # base now mid-range on the leader only
    res = CliRunner().invoke(cli.app, ["align", "left_follower", "--rig", str(rig.path), "--yes"])
    assert res.exit_code == 0, res.output
    assert RigConfig.load(rig.path).arm("left_leader").joint_offsets[0] == pytest.approx(0.05)
    res = CliRunner().invoke(cli.app, ["align", "left_follower", "--rig", str(rig.path), "--yes", "--reset"])
    assert res.exit_code == 0, res.output
    assert RigConfig.load(rig.path).arm("left_leader").joint_offsets[0] == 0.0

    # nothing at a stop → refused, rig untouched
    fake_connect.presets["left_leader"] = np.full(6, 0.5)
    fake_connect.presets["left_follower"] = np.full(6, 0.5)
    before = rig.path.read_text()
    res = CliRunner().invoke(cli.app, ["align", "left_follower", "--rig", str(rig.path), "--yes"])
    assert res.exit_code == 1 and rig.path.read_text() == before
    assert CliRunner().invoke(cli.app, ["align", "nope", "--rig", str(rig.path), "--yes"]).exit_code != 0


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
    assert session.home_speed == 0.25 and session.leader_home_speed == 0.25  # rig defaults
    session.shutdown()
    assert TeleopSession(session.pairs, home_speed=1.0).leader_home_speed == 0.5  # falls back to half
