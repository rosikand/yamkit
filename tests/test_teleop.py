import numpy as np
import pytest

from yamkit.teleop import TeleopSession


def test_teleop_engage_and_track(rig, fake_connect):
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, auto_engage=True, home_speed=0.0)
    leader = fake_connect["left_leader"]
    follower = fake_connect["left_follower"]
    leader.pos = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    leader.encoder[0].position = 0.5  # trigger half squeezed -> gripper 0.5
    stats = session.run(duration=0.3)  # sync duration extends to respect the gripper/joint speed caps
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


def test_stop_returns_home_after_teleop(rig, fake_connect, monkeypatch):
    import time

    session = TeleopSession.from_rig(rig, ["left_follower"], hz=500.0, sync_seconds=0.01, auto_engage=True, home_speed=50.0, leader_home_speed=50.0)
    leader, follower = fake_connect["left_leader"], fake_connect["left_follower"]
    leader.pos = np.full(6, 0.3)
    realtime = time.monotonic
    clock = [realtime()]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    session.engage(session.pairs[0])  # synchronization advances without blocking the outer loop
    for _ in range(180):
        clock[0] += 1 / session.hz
        session.step()
    assert np.allclose(follower.pos[:6], 0.3, atol=0.05)  # tracking
    offset = max(0, clock[0] - realtime())
    monkeypatch.setattr(time, "monotonic", lambda: realtime() + offset)  # real pacing, continuous clock
    session.shutdown()
    assert np.allclose(follower.pos[:6], 0) and np.allclose(leader.pos, 0)
    assert leader.closed and follower.closed


def test_home_speed_zero_keeps_arms_where_they_are(rig, fake_connect):
    fake_connect.presets["left_follower"] = np.array([0.3] * 6 + [1.0])
    session = TeleopSession.from_rig(rig, ["left_follower"], hz=200.0, sync_seconds=0.02, home_speed=0.0)
    follower = fake_connect["left_follower"]
    session.run(duration=0.05)
    assert follower.commands and all(np.allclose(command, [0.3] * 6 + [1.0]) for command in follower.commands)


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

    # A wrapped encoder outside raw motor limits requires operator recovery before automatic homing.
    fake_connect.presets["left_leader"] = np.array([3.07, 0.0, 0.0, 0.0, 0.0, 0.0])
    fake_connect.presets["left_follower"] = np.array([-3.09, 0.0, 0.0, 0.0, 0.0, 0.0])  # = +3.193 rad
    before = rig.path.read_text()
    res = CliRunner().invoke(cli.app, ["align", "left_follower", "--rig", str(rig.path), "--yes", "--reset"])
    assert res.exit_code == 1 and "operator recovery" in str(res.exception)
    assert rig.path.read_text() == before
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())
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


@pytest.mark.parametrize("options", [
    {"hz": 0}, {"hz": float("nan")}, {"hz": float("inf")},
    {"sync_seconds": -1}, {"sync_seconds": float("nan")},
    {"bilateral_kp": -1}, {"bilateral_kp": float("inf")},
    {"engage_button": -1}, {"engage_button": 0.5},
    {"home_speed": -1}, {"leader_home_speed": float("nan")},
    {"on_tick": 42},
])
def test_session_options_rejected_before_connect(rig, fake_connect, options):
    with pytest.raises(ValueError):
        TeleopSession.from_rig(rig, **options)
    assert not fake_connect


def test_later_invalid_arm_rejected_before_connect(rig, fake_connect):
    rig.arm("right_follower").rest_pose = [0] * 7
    with pytest.raises(ValueError, match="rest_pose"):
        TeleopSession.from_rig(rig)
    assert not fake_connect


@pytest.mark.parametrize("failure_name", ["left_follower", "right_leader", "right_follower"])
@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_partial_session_startup_closes_every_opened_arm(rig, fake_connect, monkeypatch, failure_name, error):
    from yamkit.arm import YamArm

    connect = YamArm.connect

    def fail_connect(spec, channel, **kw):
        if spec.name == failure_name:
            raise error("connect failed")
        return connect(spec, channel, **kw)

    monkeypatch.setattr(YamArm, "connect", staticmethod(fail_connect))
    with pytest.raises(error, match="connect failed"):
        TeleopSession.from_rig(rig)
    assert fake_connect and all(robot.closed for robot in fake_connect.values())


def test_startup_cleanup_failure_preserves_cancellation_and_closes_remaining(rig, fake_connect, monkeypatch):
    from yamkit.arm import YamArm

    connect, close = YamArm.connect, YamArm.close

    def fail_connect(spec, channel, **kw):
        if spec.name == "right_follower":
            raise KeyboardInterrupt("cancel startup")
        return connect(spec, channel, **kw)

    def fail_close(arm, **kw):
        close(arm, **kw)
        if arm.name == "left_leader":
            raise RuntimeError("close failed")

    monkeypatch.setattr(YamArm, "connect", staticmethod(fail_connect))
    monkeypatch.setattr(YamArm, "close", fail_close)
    with pytest.raises(KeyboardInterrupt, match="cancel startup"):
        TeleopSession.from_rig(rig)
    assert all(robot.closed for robot in fake_connect.values())


def test_shutdown_without_home_skips_motion_and_repeated_shutdown(rig, fake_connect, monkeypatch):
    session = TeleopSession.from_rig(rig)
    for pair in session.pairs:
        pair.engaged = True
    monkeypatch.setattr(session, "home_all", lambda why: pytest.fail("unexpected home"))
    monkeypatch.setattr(session, "disengage", lambda pair: pytest.fail("unexpected hold"))
    session.shutdown(home=False)
    idle_counts = [robot.idle_calls for robot in fake_connect.values()]
    session.shutdown()  # defaults cannot revive an already closed session
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())
    assert [robot.idle_calls for robot in fake_connect.values()] == idle_counts
    assert all(not pair.engaged for pair in session.pairs)


@pytest.mark.parametrize("stage", ["disengage", "home", "close"])
def test_shutdown_failure_closes_remaining_arms(rig, fake_connect, monkeypatch, stage):
    session = TeleopSession.from_rig(rig)

    def fail(*args, **kw):
        raise RuntimeError(f"{stage} failed")

    if stage == "disengage":
        session.pairs[0].engaged = True
        monkeypatch.setattr(session.pairs[0].follower, "hold", fail)
    elif stage == "home":
        monkeypatch.setattr(session, "home_all", fail)
    else:
        first = session.pairs[0].leader
        close = first.close

        def fail_close():
            close()
            fail()

        monkeypatch.setattr(first, "close", fail_close)
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        session.shutdown()
    assert all(robot.closed for robot in fake_connect.values())


def test_shutdown_retries_failed_close_without_repeating_home(rig, fake_connect, monkeypatch):
    session = TeleopSession.from_rig(rig)
    robot = fake_connect["left_leader"]
    close = robot.close
    failures = []
    homes = []

    def fail_once():
        if not failures:
            failures.append(True)
            raise RuntimeError("SDK close failed")
        close()

    monkeypatch.setattr(robot, "close", fail_once)
    monkeypatch.setattr(session, "home_all", lambda why: homes.append(why))
    with pytest.raises(RuntimeError, match="SDK close failed"):
        session.shutdown()
    assert homes == ["stop"] and not robot.closed
    assert all(other.closed for name, other in fake_connect.items() if name != "left_leader")
    session.shutdown()
    assert homes == ["stop"] and robot.closed


def test_run_failure_skips_return_home(rig, fake_connect, monkeypatch):
    session = TeleopSession.from_rig(rig, home_speed=0)

    def fail_step():
        raise RuntimeError("bad observation")

    monkeypatch.setattr(session, "step", fail_step)
    with pytest.raises(RuntimeError, match="bad observation"):
        session.run(duration=0.1)
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_run_invalid_duration_releases_arms_without_motion(rig, fake_connect, duration):
    session = TeleopSession.from_rig(rig)
    with pytest.raises(ValueError, match="duration"):
        session.run(duration=duration)
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())


@pytest.mark.parametrize("bad_arm", ["right_leader", "right_follower"])
def test_complete_tick_prevalidated_before_any_command_or_gains(rig, fake_connect, bad_arm):
    session = TeleopSession.from_rig(rig, bilateral_kp=0.1, home_speed=0)
    for pair in session.pairs:
        pair.engaged = True
        pair.follower.zero_torque()
    fake_connect[bad_arm].pos[0] = float("nan")
    gains = {name: robot.kp.copy() for name, robot in fake_connect.items()}
    try:
        with pytest.raises(ValueError):
            session.step()
        assert all(not robot.commands for robot in fake_connect.values())
        for name, robot in fake_connect.items():
            np.testing.assert_array_equal(robot.kp, gains[name])
    finally:
        session.shutdown(home=False)


def test_bilateral_target_prevalidated_before_follower_command(rig, fake_connect):
    # A valid follower pose can be outside the aligned leader's smaller shifted interval.
    rig.arm("left_leader").joint_offsets = [0.5, 0, 0, 0, 0, 0]
    session = TeleopSession.from_rig(rig, ["left_follower"], bilateral_kp=0.1, home_speed=0)
    pair = session.pairs[0]
    pair.engaged = True
    pair.follower.zero_torque()
    fake_connect["left_follower"].pos[0] = -2.4
    gains = fake_connect["left_follower"].kp.copy()
    try:
        with pytest.raises(ValueError):
            session.step()
        assert not fake_connect["left_follower"].commands
        np.testing.assert_array_equal(fake_connect["left_follower"].kp, gains)
    finally:
        session.shutdown(home=False)


def test_stop_during_sync_does_not_engage_or_change_leader_gains(rig, fake_connect, monkeypatch):
    session = TeleopSession.from_rig(rig, ["left_follower"], bilateral_kp=0.1, home_speed=0)
    pair = session.pairs[0]
    leader = fake_connect["left_leader"]
    gains = leader.kp.copy()

    try:
        session.engage(pair)
        assert pair._gate.syncing and pair.engaged
        session.stop_event.set()
        session.step()
        assert not fake_connect["left_follower"].commands
        np.testing.assert_array_equal(leader.kp, gains)
    finally:
        session.shutdown(home=False)


def test_stop_during_auto_engage_skips_later_pairs(rig, fake_connect, monkeypatch):
    session = TeleopSession.from_rig(rig, auto_engage=True, home_speed=0)
    calls = []

    def cancel_engage(pair):
        calls.append(pair.name)
        session.stop_event.set()

    monkeypatch.setattr(session, "engage", cancel_engage)
    session.run(duration=0.1)
    assert calls == [session.pairs[0].name]
    assert all(robot.closed and not robot.commands for robot in fake_connect.values())


def test_stop_during_button_sync_skips_later_pair_commands(rig, fake_connect, monkeypatch):
    session = TeleopSession.from_rig(rig, home_speed=0)
    session.pairs[1].engaged = True
    fake_connect["left_leader"].encoder[0].io_inputs = [1, 0]

    def stop_sync(q, gripper, **kwargs):
        session.stop_event.set()

    monkeypatch.setattr(session.pairs[0].follower, "command", stop_sync)
    try:
        session.step()
        assert all(not robot.commands for robot in fake_connect.values())
    finally:
        session.shutdown(home=False)
