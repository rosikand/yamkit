"""Dashboard-style auto engagement through installed LeRobot loops and fake arms."""

import numpy as np
import pytest
from lerobot.processor import make_default_processors
from lerobot.scripts import lerobot_record as recorder
from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig
from lerobot_robot_yamkit.yam_follower import BiYamFollower, YamFollower
from lerobot_teleoperator_yamkit import BiYamLeaderConfig, YamLeaderConfig
from lerobot_teleoperator_yamkit.yam_leader import BiYamLeader, YamLeader

from yamkit.lerobot_teleop import make_teleop_processor


def plugins(rig, bimanual, auto_engage):
    if bimanual:
        return (BiYamFollower(BiYamFollowerConfig(rig=str(rig.path))),
                BiYamLeader(BiYamLeaderConfig(rig=str(rig.path), auto_engage=auto_engage)))
    return (YamFollower(YamFollowerConfig(rig=str(rig.path), arm="left_follower")),
            YamLeader(YamLeaderConfig(rig=str(rig.path), arm="left_leader", auto_engage=auto_engage)))


@pytest.mark.parametrize("bimanual", [False, True], ids=["single", "bimanual"])
@pytest.mark.parametrize("auto_engage", [False, True], ids=["manual", "automatic"])
@pytest.mark.parametrize("button_held", [False, True], ids=["released", "held-at-start"])
def test_actual_record_and_reset_track_and_label_without_button_edge(
    rig, fake_connect, monkeypatch, bimanual, auto_engage, button_held,
):
    from types import SimpleNamespace

    rig.control.home_speed = 0
    rig.control.sync_seconds = 0.08
    rig.save()
    clock = [10.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    monkeypatch.setattr("time.perf_counter", lambda: clock[0])
    robot, leader = plugins(rig, bimanual, auto_engage)
    leader.connect()
    robot.connect()
    sides = ("left", "right") if bimanual else ("left",)
    for side in sides:
        fake_connect[f"{side}_follower"].pos[:] = [0.1] * 6 + [0.5]
    cursor = [0]
    observed = robot.get_observation

    def observation():
        clock[0] = 10.0 + cursor[0] / 30
        for side in sides:
            handle = fake_connect[f"{side}_leader"]
            handle.pos[:] = 0.3 + cursor[0] * 0.002
            handle.encoder[0].position = 0.4 - cursor[0] * 0.002
            handle.encoder[0].io_inputs = [button_held, False]
        cursor[0] += 1
        return observed()

    monkeypatch.setattr(robot, "get_observation", observation)
    events = {"exit_early": False}
    stop_at = [0]
    monkeypatch.setattr(recorder, "precise_sleep", lambda _: events.update(exit_early=cursor[0] >= stop_at[0]))
    names = list(robot.action_features)
    frames = []
    dataset = SimpleNamespace(fps=30, add_frame=frames.append, features={
        key: {"dtype": "float32", "shape": (len(names),), "names": names}
        for key in ("action", "observation.state")
    })
    processor = make_teleop_processor(robot.config, leader.config, 30)
    _, action_processor, observation_processor = make_default_processors()
    try:
        for frame_count, recording in ((12, True), (8, False), (12, True)):
            stop_at[0] += frame_count
            recorder.record_loop(
                robot=robot, teleop=leader, fps=30, events=events,
                dataset=dataset if recording else None, control_time_s=100,
                single_task="automatic start fixture", teleop_action_processor=processor,
                robot_action_processor=action_processor, robot_observation_processor=observation_processor,
            )
        sent = np.stack([
            np.concatenate([fake_connect[f"{side}_follower"].commands[index] for side in sides])
            for index in range(32)
        ])
        np.testing.assert_allclose([frame["action"] for frame in frames], sent[list(range(12)) + list(range(20, 32))],
                                   atol=1e-7)
        assert len(frames) == 24  # reset teleoperates without contributing dataset frames
        np.testing.assert_allclose(sent[0], ([0.1] * 6 + [0.5]) * len(sides))
        assert np.max(np.abs(np.diff(sent, axis=0))) <= rig.control.max_joint_speed * 0.01 + 1e-9
        if auto_engage or button_held:
            for index, side in enumerate(sides):
                assert sent[-1, index * 7] > 0.35
                assert sent[-1, index * 7 + 6] > 0.65
            assert all(gate.engaged and not gate.syncing for gate in processor.steps[0].gates.values())
        else:
            np.testing.assert_allclose(sent, np.repeat(sent[0][None], 32, axis=0))
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


@pytest.mark.parametrize("bimanual", [False, True], ids=["single", "bimanual"])
def test_auto_start_phases_acknowledge_sync_then_optional_button_hold(
    rig, fake_connect, monkeypatch, caplog, bimanual,
):
    rig.control.home_speed = 0
    rig.control.sync_seconds = 0.05
    rig.save()
    clock = [10.0]
    monkeypatch.setattr("time.monotonic", lambda: clock[0])
    robot, leader = plugins(rig, bimanual, True)
    leader.connect()
    robot.connect()
    processor = make_teleop_processor(robot.config, leader.config, 30)
    sides = ("left", "right") if bimanual else ("left",)
    caplog.set_level("INFO", logger="yamkit.lerobot_teleop")

    def messages():
        return [record.message for record in caplog.records if record.name == "yamkit.lerobot_teleop"]

    def phases():
        return [message for message in messages() if message.startswith("[yamkit-operator]")]

    def pending():
        clock[0] += 1 / 30
        return processor((leader.get_action(), robot.get_observation()))

    try:
        action = pending()
        assert messages() == []
        sent = robot.send_action(action)
        assert phases() == ["[yamkit-operator] synchronizing"]
        assert len([message for message in messages() if "engaged automatically" in message]) == len(sides)
        action.acknowledge(sent)
        for _ in range(15):
            robot.send_action(pending())
        assert phases() == ["[yamkit-operator] synchronizing", "[yamkit-operator] ready"]
        # Optional pause must persist across future calls, including the next episode.
        fake_connect[f"{sides[-1]}_leader"].encoder[0].io_inputs[0] = True
        action = pending()
        assert phases()[-1] == "[yamkit-operator] ready"
        robot.send_action(action)
        assert phases()[-1] == "[yamkit-operator] holding"
        held = fake_connect[f"{sides[-1]}_follower"].pos.copy()
        for _ in range(15):
            fake_connect[f"{sides[-1]}_leader"].pos[:] = 0.5
            robot.send_action(pending())
        np.testing.assert_allclose(fake_connect[f"{sides[-1]}_follower"].pos, held)
        assert phases() == ["[yamkit-operator] synchronizing", "[yamkit-operator] ready", "[yamkit-operator] holding"]
        fake_connect[f"{sides[-1]}_leader"].encoder[0].io_inputs[0] = False
        robot.send_action(pending())
        fake_connect[f"{sides[-1]}_leader"].encoder[0].io_inputs[0] = True
        robot.send_action(pending())
        assert phases()[-1] == "[yamkit-operator] synchronizing"
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


def test_failed_auto_start_send_never_reports_engaged_or_ready(rig, fake_connect, monkeypatch, caplog):
    rig.control.home_speed = 0
    rig.save()
    robot, leader = plugins(rig, True, True)
    leader.connect()
    robot.connect()
    processor = make_teleop_processor(robot.config, leader.config, 30)
    caplog.set_level("INFO", logger="yamkit.lerobot_teleop")

    def fail_send(_):
        raise RuntimeError("follower rejected command")

    try:
        action = processor((leader.get_action(), robot.get_observation()))
        monkeypatch.setattr(fake_connect["right_follower"], "command_joint_pos", fail_send)
        with pytest.raises(RuntimeError, match="follower rejected command"):
            robot.send_action(action)
        assert not any(record.name == "yamkit.lerobot_teleop" for record in caplog.records)
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


@pytest.mark.parametrize("value", ["false", 0, None])
def test_invalid_auto_engage_rejected_before_hardware_connect(rig, fake_connect, value):
    robot, leader = plugins(rig, True, value)
    with pytest.raises(TypeError, match="auto_engage must be a boolean"):
        make_teleop_processor(robot.config, leader.config, 30)
    assert not fake_connect


@pytest.mark.parametrize("bimanual", [False, True], ids=["single", "bimanual"])
@pytest.mark.parametrize("auto_engage", [False, True])
def test_record_config_parses_auto_engage_flag_without_hardware(rig, fake_connect, bimanual, auto_engage):
    import draccus

    cfg = draccus.parse(recorder.RecordConfig, args=[
        f"--robot.type={'bi_yam_follower' if bimanual else 'yam_follower'}",
        f"--robot.rig={rig.path}",
        f"--teleop.type={'bi_yam_leader' if bimanual else 'yam_leader'}",
        f"--teleop.rig={rig.path}",
        f"--teleop.auto_engage={str(auto_engage).lower()}",
        "--dataset.repo_id=yamkit/fixture", "--dataset.single_task=fixture",
        "--dataset.push_to_hub=false",
    ])
    assert cfg.teleop.auto_engage is auto_engage
    assert make_teleop_processor(cfg.robot, cfg.teleop, cfg.dataset.fps).steps[0].auto_engage is auto_engage
    assert not fake_connect
