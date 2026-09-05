"""Equivalent native/installed-LeRobot traces; no arm, camera or service activation."""

import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from lerobot.processor import make_default_processors
from lerobot.scripts import lerobot_record as recorder
from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig
from lerobot_robot_yamkit.yam_follower import BiYamFollower, YamFollower
from lerobot_teleoperator_yamkit import BiYamLeaderConfig, YamLeaderConfig
from lerobot_teleoperator_yamkit.yam_leader import BiYamLeader, YamLeader

from yamkit.lerobot_teleop import make_teleop_processor
from yamkit.teleop import TeleopSession
from yamkit.teleop_control import GatedAction, LeaderAction


def plugins(rig, bimanual):
    if bimanual:
        return BiYamFollower(BiYamFollowerConfig(rig=str(rig.path))), BiYamLeader(BiYamLeaderConfig(rig=str(rig.path)))
    return (YamFollower(YamFollowerConfig(rig=str(rig.path), arm="left_follower")),
            YamLeader(YamLeaderConfig(rig=str(rig.path), arm="left_leader")))


@pytest.mark.parametrize("bimanual", [False, True], ids=["single", "bimanual"])
@pytest.mark.parametrize("fps", [30, 100])
def test_native_and_actual_record_loop_execute_and_label_equivalent_trace(rig, fake_connect, monkeypatch, bimanual, fps):
    rig.control.home_speed = 0
    rig.control.sync_seconds = 0.08
    rig.arm("left_leader").joint_offsets = [0.1, 0, 0, 0, 0, 0]
    rig.save()
    sides = ("left", "right") if bimanual else ("left",)
    count = 42
    clock = [100.0]
    times = [100 + index / fps + (2 if index >= 6 else 0) for index in range(count)]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "perf_counter", lambda: clock[0])

    def trajectory(index):
        for side in sides:
            leader = fake_connect[f"{side}_leader"]
            leader.pos[:] = 0.1 + index * 0.002
            leader.pos[0] = 0.35 if index < 9 else 0.8 - index * 0.003
            leader.encoder[0].position = 0.35 + index * 0.003  # normalized mapping is 1 - trigger
            button_index = index - (1 if side == "right" else 0)
            leader.encoder[0].io_inputs = [button_index in (2, 3, 8, 9, 14, 15, 36, 37), 0]

    def start_positions():
        for side in sides:
            follower = fake_connect[f"{side}_follower"]
            follower.pos[:6] = 0.1
            follower.pos[6] = 0.65

    native = TeleopSession.from_rig(rig, None if bimanual else ["left_follower"], hz=fps)
    start_positions()
    native_trace, sync_progress = [], []
    for index, tick in enumerate(times):
        clock[0] = tick
        trajectory(index)
        native.step()
        native_trace.append(np.concatenate([fake_connect[f"{side}_follower"].commands[-1] for side in sides]))
        sync_progress.append(native.pairs[0]._gate.elapsed)
    native.shutdown(home=False)

    clock[0] = times[0]
    robot, leader = plugins(rig, bimanual)
    robot.connect()
    leader.connect()
    start_positions()
    observed = robot.get_observation
    cursor = [0]

    def observation():
        clock[0] = times[cursor[0]]
        trajectory(cursor[0])
        cursor[0] += 1
        return observed()

    monkeypatch.setattr(robot, "get_observation", observation)
    events = {"exit_early": False}
    monkeypatch.setattr(recorder, "precise_sleep", lambda delay: events.update(exit_early=cursor[0] >= count))
    names = list(robot.action_features)
    frames = []
    dataset = SimpleNamespace(fps=fps, add_frame=frames.append, features={
        key: {"dtype": "float32", "shape": (len(names),), "names": names}
        for key in ("action", "observation.state")
    })
    _, action_processor, observation_processor = make_default_processors()
    try:
        recorder.record_loop(robot=robot, teleop=leader, fps=fps, events=events, dataset=dataset,
                             control_time_s=100, single_task="equivalent trace",
                             teleop_action_processor=make_teleop_processor(robot.config, leader.config, fps),
                             robot_action_processor=action_processor, robot_observation_processor=observation_processor)
        sent = np.stack([np.concatenate([fake_connect[f"{side}_follower"].commands[index] for side in sides])
                         for index in range(count)])
        np.testing.assert_allclose(sent, native_trace, atol=1e-12)
        np.testing.assert_allclose([frame["action"] for frame in frames], sent, atol=1e-7)
        assert len(frames) == count and all(frame["task"] == "equivalent trace" for frame in frames)
        np.testing.assert_allclose(sent[0], [0.1] * 6 + [0.65] if not bimanual else ([0.1] * 6 + [0.65]) * 2)
        for side_index, side in enumerate(sides):
            start = 8 + (side == "right")
            commands = sent[start:start + 6, side_index * 7:side_index * 7 + 7]
            np.testing.assert_allclose(commands, np.repeat(commands[0][None], 6, axis=0))
        assert not np.isclose(sent[10, 0], 0.8 - 10 * 0.003 + 0.1)  # freely moving raw leader is not a label
        assert np.max(np.abs(np.diff(sent, axis=0))) <= rig.control.max_joint_speed * 0.01 + 1e-9
        assert sync_progress[6] - sync_progress[5] <= 1 / fps + 1e-9  # two-second stall earns one tick
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


@pytest.mark.parametrize("bimanual", [False, True])
def test_capture_uses_fresh_measurement_and_latches_acknowledged_hold(rig, fake_connect, bimanual):
    rig.control.home_speed = 0
    rig.save()
    robot, leader = plugins(rig, bimanual)
    robot.connect()
    leader.connect()
    processor = make_teleop_processor(robot.config, leader.config, 30)
    sides = ("left", "right") if bimanual else ("left",)
    try:
        action = processor((leader.get_action(), robot.get_observation()))
        for side in sides:
            fake_connect[f"{side}_follower"].pos[:] = 0.2  # motion after obs, before send
        first = robot.send_action(action)
        assert all(value == pytest.approx(0.2) for value in first.values())
        assert action == first  # LeRobot's label object was acknowledged
        for side in sides:
            fake_connect[f"{side}_leader"].pos[:] = 0.8
        second = robot.send_action(processor((leader.get_action(), robot.get_observation())))
        assert second == first  # no drift toward the earlier zero observation
        far = {key: 1.0 for key in robot.action_features}
        forged = GatedAction(far, capture_hold=("left_", "right_") if bimanual else ("",))
        assert robot.send_action(forged) == first  # capture marker cannot bypass speed limits with caller targets
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


def test_bimanual_capture_prevalidates_second_arm_before_first_send(rig, fake_connect):
    rig.control.home_speed = 0
    rig.save()
    robot, leader = plugins(rig, True)
    robot.connect()
    leader.connect()
    action = make_teleop_processor(robot.config, leader.config, 30)((leader.get_action(), robot.get_observation()))
    fake_connect["right_follower"].pos[0] = float("nan")
    try:
        with pytest.raises(ValueError, match="finite"):
            robot.send_action(action)
        assert not fake_connect["left_follower"].commands
        assert not fake_connect["right_follower"].commands
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


def test_invalid_second_leader_target_rejects_entire_sync_before_consuming_buttons(rig, fake_connect):
    rig.control.home_speed = 0
    rig.save()
    robot, leader = plugins(rig, True)
    robot.connect()
    leader.connect()
    processor = make_teleop_processor(robot.config, leader.config, 30)
    for side in ("left", "right"):
        fake_connect[f"{side}_leader"].encoder[0].io_inputs = [1, 0]
    fake_connect["right_leader"].pos[0] = 9.0
    try:
        with pytest.raises(ValueError, match="bounds"):
            processor((leader.get_action(), robot.get_observation()))
        assert all(not gate.engaged and not gate.button_prev for gate in processor.steps[0].gates.values())
        assert all(not robot.commands for robot in fake_connect.values())
        fake_connect["right_leader"].pos[0] = 0.2
        robot.send_action(processor((leader.get_action(), robot.get_observation())))
        assert all(gate.engaged for gate in processor.steps[0].gates.values())
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


def test_button_metadata_is_not_a_dataset_feature_and_raw_bypass_is_rejected(rig, fake_connect):
    rig.control.home_speed = 0
    rig.save()
    robot, leader = plugins(rig, False)
    robot.connect()
    leader.connect()
    try:
        raw = leader.get_action()
        assert isinstance(raw, LeaderAction) and set(raw) == set(robot.action_features)
        assert raw.buttons == {"": (False, False)}
        teleop_processor, robot_processor, _ = make_default_processors()
        obs = robot.get_observation()
        unprocessed = robot_processor((teleop_processor((raw, obs)), obs))
        assert unprocessed is raw  # pin the upstream identity behavior used by acknowledgment
        with pytest.raises(TypeError, match="operator processor"):
            robot.send_action(unprocessed)
        assert not fake_connect["left_follower"].commands
    finally:
        robot.disconnect(home=False)
        leader.disconnect(home=False)


@pytest.mark.parametrize("bimanual", [False, True])
def test_bilateral_recording_rejected_before_connect(rig, fake_connect, bimanual):
    rig.control.bilateral_kp = 0.1
    rig.save()
    robot, leader = plugins(rig, bimanual)
    with pytest.raises(ValueError, match="does not support bilateral"):
        make_teleop_processor(robot.config, leader.config, 30)
    assert not fake_connect


def test_actual_teleoperate_entry_uses_gate_and_restores_processor_factory(rig, fake_connect):
    from lerobot.scripts import lerobot_teleoperate as upstream

    from yamkit.lerobot_teleop import teleoperate

    rig.control.home_speed = 0
    rig.save()
    fake_connect.presets["left_follower"] = np.array([0.2] * 6 + [0.5])
    fake_connect.presets["left_leader"] = np.full(6, 0.8)
    original = upstream.make_default_processors
    cfg = upstream.TeleoperateConfig(robot=YamFollowerConfig(rig=str(rig.path), arm="left_follower"),
                                     teleop=YamLeaderConfig(rig=str(rig.path), arm="left_leader"),
                                     fps=30, teleop_time_s=0.07)
    teleoperate(cfg)
    assert upstream.make_default_processors is original
    assert all(robot.closed for robot in fake_connect.values())
    follower = fake_connect["left_follower"]
    assert follower.commands and all(np.allclose(command, [0.2] * 6 + [0.5]) for command in follower.commands)


@pytest.mark.parametrize("command,options", [("record", ["--name", "fake", "--task", "fixture"]),
                                           ("teleoperate", [])])
def test_cli_wrappers_install_operator_adapter_without_activation(rig, fake_connect, command, options):
    from typer.testing import CliRunner

    from yamkit.cli import app

    result = CliRunner().invoke(app, [command, "--rig", str(rig.path), "--dry-run", *options])
    assert result.exit_code == 0, result.output
    assert f"yamkit.lerobot_teleop {command}" in " ".join(result.output.split())
    assert not fake_connect


@pytest.mark.parametrize("failure", ["normal", "interrupt", "bad_action", "finalize", "cancel"])
def test_actual_record_entry_stop_and_fault_lifecycle(rig, fake_connect, monkeypatch, failure):
    from lerobot.configs.dataset import DatasetRecordConfig

    from yamkit.lerobot_teleop import record

    rig.control.home_speed = 0
    rig.save()
    for name in rig.arms:
        fake_connect.presets[name] = np.array([0.2] * 6 + ([0.5] if name.endswith("follower") else []))
    robot_config = BiYamFollowerConfig(rig=str(rig.path))
    leader_config = BiYamLeaderConfig(rig=str(rig.path))
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    frames = []

    class Dataset:
        num_episodes = 0
        fps = 30

        def add_frame(self, frame):
            frames.append(frame)
            events["exit_early"] = events["stop_recording"] = True

        def save_episode(self):
            self.num_episodes += 1

        def finalize(self):
            if failure == "finalize":
                raise RuntimeError("finalizer failed")

    dataset = Dataset()

    def create(*args, **kwargs):
        dataset.features = kwargs["features"]
        return dataset

    monkeypatch.setattr(recorder.LeRobotDataset, "create", create)
    monkeypatch.setattr(recorder, "VideoEncodingManager", lambda dataset: nullcontext())
    monkeypatch.setattr(recorder, "init_keyboard_listener", lambda: (None, events))
    monkeypatch.setattr(recorder, "log_say", lambda *args, **kwargs: None)
    action = BiYamLeader.get_action

    def input_action(teleop):
        # Start had no motion; make normal stop homing observable if a fault accidentally requests it.
        for plugin in (robot_config._runtime_robot, leader_config._runtime_teleop):
            for handle in plugin._sides.values():
                handle.home_speed = 50
        if failure == "bad_action":
            fake_connect["right_leader"].pos[0] = float("nan")
        elif failure == "cancel":
            raise SystemExit("cancelled")
        elif failure == "interrupt":
            raise KeyboardInterrupt("operator Stop")
        return action(teleop)

    monkeypatch.setattr(BiYamLeader, "get_action", input_action)
    cfg = recorder.RecordConfig(robot=robot_config, teleop=leader_config, play_sounds=False,
                                dataset=DatasetRecordConfig(repo_id="yamkit/fake", single_task="fixture", fps=30,
                                                            video=False, push_to_hub=False, num_episodes=1,
                                                            episode_time_s=0.2, no_stamp=True))
    if failure == "normal":
        record(cfg)
    else:
        with pytest.raises((ValueError, RuntimeError, SystemExit, KeyboardInterrupt)):
            record(cfg)
    assert fake_connect and all(robot.closed for robot in fake_connect.values())
    for name, robot in fake_connect.items():
        if failure in ("normal", "interrupt"):
            assert robot.commands and np.allclose(robot.pos[:6], 0), name
        else:
            assert all(np.allclose(command[:6], 0.2) for command in robot.commands), name
            assert not robot.commands or (failure == "finalize" and name.endswith("follower") and len(robot.commands) == 1)
