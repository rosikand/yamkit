"""Episode timing through the pinned recorder, real dataset writer and fake YAMs."""

import signal
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.scripts import lerobot_record as recorder
from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig
from lerobot_robot_yamkit.yam_follower import BiYamFollower, YamFollower
from lerobot_teleoperator_yamkit import BiYamLeaderConfig, YamLeaderConfig

from yamkit import lerobot_teleop as operator_module


@pytest.fixture
def timed_recording(rig, fake_connect, monkeypatch, tmp_path):
    """Keep LeRobot's lifecycle/acquisition/writer; replace clocks and hardware."""
    rig.control.home_speed = rig.control.leader_home_speed = 0
    rig.control.sync_seconds = 0.5
    rig.save()
    for name in ("left_leader", "right_leader"):
        fake_connect.presets[name] = np.full(6, 0.2)
    state = SimpleNamespace(
        now=100.0, loops=[], frames=[], sends=[], sayings=[], created=[], homes=[],
        events={"exit_early": False, "stop_recording": False, "rerecord_episode": False},
        observation_hook=lambda run: None, frame_hook=lambda frame: None,
        say_hook=lambda message: None, save_hook=lambda dataset: None, processor=None, run=None,
    )
    cfg = recorder.RecordConfig(
        robot=BiYamFollowerConfig(rig=str(rig.path)),
        teleop=BiYamLeaderConfig(rig=str(rig.path), auto_engage=True),
        play_sounds=False,
        dataset=DatasetRecordConfig(
            repo_id="yamkit/readiness", root=tmp_path / "recording", single_task="fixture",
            fps=16, video=False, push_to_hub=False, num_episodes=1,
            episode_time_s=1, reset_time_s=0, no_stamp=True,
        ),
    )
    state.cfg = cfg
    monkeypatch.setattr("time.monotonic", lambda: state.now)
    monkeypatch.setattr("time.perf_counter", lambda: state.now)
    # A binary-exact period makes assertions independent of clock rounding.
    monkeypatch.setattr(recorder, "precise_sleep", lambda seconds: setattr(state, "now", state.now + seconds))
    monkeypatch.setattr(recorder, "init_keyboard_listener", lambda: (None, state.events))

    original_loop = recorder.record_loop
    original_create = recorder.LeRobotDataset.create
    original_add = recorder.LeRobotDataset.add_frame
    original_save = recorder.LeRobotDataset.save_episode
    original_factory = operator_module.make_teleop_processor

    def observe_loop(*args, **kwargs):
        run = {"dataset": kwargs.get("dataset", args[6] if len(args) > 6 else None),
               "duration": kwargs.get("control_time_s", args[8] if len(args) > 8 else None),
               "started": state.now, "observations": 0}
        state.loops.append(run)
        state.run = run
        try:
            return original_loop(*args, **kwargs)
        finally:
            run["ended"] = state.now

    def create(*args, **kwargs):
        dataset = original_create(*args, **kwargs)
        state.created.append(dataset)
        return dataset

    def make_processor(*args, **kwargs):
        state.processor = original_factory(*args, **kwargs)
        return state.processor

    def add_frame(dataset, frame):
        step = state.processor.steps[0]
        snapshot = {"time": state.now, "episode": dataset.num_episodes, "ready": step.ready,
                    "syncing": [gate.syncing for gate in step.gates.values()],
                    "engaged": [gate.engaged for gate in step.gates.values()],
                    "action": frame["action"].copy()}
        state.frames.append(snapshot)
        original_add(dataset, frame)
        state.frame_hook(snapshot)

    def say(message, *args, **kwargs):
        state.sayings.append((message, state.now, state.processor.steps[0].ready))
        state.say_hook(message)

    def save_episode(dataset, *args, **kwargs):
        result = original_save(dataset, *args, **kwargs)
        state.save_hook(dataset)
        return result

    monkeypatch.setattr(recorder, "record_loop", observe_loop)
    monkeypatch.setattr(recorder.LeRobotDataset, "create", create)
    monkeypatch.setattr(recorder.LeRobotDataset, "add_frame", add_frame)
    monkeypatch.setattr(recorder.LeRobotDataset, "save_episode", save_episode)
    monkeypatch.setattr(operator_module, "make_teleop_processor", make_processor)
    monkeypatch.setattr(recorder, "log_say", say)

    for plugin_type in (YamFollower, BiYamFollower):
        original_observation = plugin_type.get_observation
        original_send = plugin_type.send_action

        def observation(robot, original=original_observation):
            state.run["observations"] += 1
            state.observation_hook(state.run)
            return original(robot)

        def send(robot, action, original=original_send):
            result = original(robot, action)
            step = state.processor.steps[0]
            state.sends.append({"time": state.now, "ready": step.ready,
                                "syncing": [gate.syncing for gate in step.gates.values()],
                                "engaged": [gate.engaged for gate in step.gates.values()]})
            return result

        monkeypatch.setattr(plugin_type, "get_observation", observation)
        monkeypatch.setattr(plugin_type, "send_action", send)

    for module in ("lerobot_robot_yamkit.yam_follower", "lerobot_teleoperator_yamkit.yam_leader"):
        monkeypatch.setattr(module + ".go_home_all", lambda jobs, **kwargs: state.homes.extend(jobs))
    previous_interrupt = signal.getsignal(signal.SIGINT)
    yield cfg, state, fake_connect
    assert signal.getsignal(signal.SIGINT) is previous_interrupt
    assert recorder.record_loop is observe_loop
    assert recorder.log_say is say
    assert state.processor is None or state.processor.steps[0]._on_ready is None
    assert all(robot.closed for robot in fake_connect.values())


def episode_rows(cfg):
    paths = sorted(cfg.dataset.root.glob("data/**/*.parquet"))
    return [row for path in paths for row in pq.read_table(path).to_pylist()]


@pytest.mark.parametrize("bimanual", [False, True], ids=["single", "bimanual"])
def test_five_second_preparation_precedes_full_timed_episode(timed_recording, rig, bimanual):
    cfg, state, _ = timed_recording
    rig.control.sync_seconds = 5
    rig.save()
    if not bimanual:
        cfg.robot = YamFollowerConfig(rig=str(rig.path), arm="left_follower")
        cfg.teleop = YamLeaderConfig(rig=str(rig.path), arm="left_leader", auto_engage=True)

    dataset = operator_module.record(cfg)

    preparation, acquisition = state.loops
    assert preparation["dataset"] is None and preparation["duration"] == float("inf")
    assert preparation["ended"] - preparation["started"] >= 5
    assert preparation["observations"] >= 81
    assert acquisition["dataset"] is dataset
    assert acquisition["ended"] - acquisition["started"] == cfg.dataset.episode_time_s
    assert len(state.frames) == cfg.dataset.fps
    assert all(frame["ready"] and not any(frame["syncing"]) for frame in state.frames)
    assert state.frames[0]["time"] >= preparation["ended"]
    announcement = next(item for item in state.sayings if item[0] == "Recording episode 0")
    assert announcement[1] == acquisition["started"] and announcement[2]
    rows = episode_rows(cfg)
    assert len(rows) == 16 and rows[0]["timestamp"] == rows[0]["frame_index"] == 0
    np.testing.assert_allclose([row["timestamp"] for row in rows], np.arange(16) / 16)
    np.testing.assert_allclose([row["action"] for row in rows], [frame["action"] for frame in state.frames])
    assert dataset._is_finalized and dataset.num_episodes == 1


def test_bimanual_preparation_waits_for_paused_side_to_resynchronize(timed_recording):
    cfg, state, robots = timed_recording

    def pause_then_resume(run):
        count = run["observations"]
        robots["right_leader"].encoder[0].io_inputs[0] = run["dataset"] is None and count in (3, 12)

    state.observation_hook = pause_then_resume
    operator_module.record(cfg)

    assert any(send["engaged"] == [True, False] for send in state.sends)
    assert any(send["syncing"] == [False, True] for send in state.sends)
    assert state.loops[0]["observations"] >= 20  # resume at tick 12, then all eight sync ticks
    assert len(state.frames) == 16 and all(frame["ready"] for frame in state.frames)


def test_pause_in_recording_preserves_episode_clock_and_hold_frames(timed_recording):
    cfg, state, robots = timed_recording

    def pause_recording(run):
        if run["dataset"] is not None:
            robots["right_leader"].encoder[0].io_inputs[0] = run["observations"] == 3

    state.observation_hook = pause_recording
    operator_module.record(cfg)

    assert len(state.loops) == 2 and len(state.frames) == 16
    assert state.loops[-1]["ended"] - state.loops[-1]["started"] == 1
    assert all(frame["ready"] for frame in state.frames[:2])
    assert all(frame["engaged"] == [True, False] for frame in state.frames[2:])
    np.testing.assert_allclose([frame["action"][7:] for frame in state.frames[2:]],
                               np.repeat(state.frames[2]["action"][None, 7:], 14, axis=0))


def test_reset_pause_persists_until_manual_resume_before_next_episode(timed_recording):
    cfg, state, robots = timed_recording
    cfg.dataset.num_episodes = 2
    cfg.dataset.reset_time_s = 0.25

    def reset_and_resume(run):
        count = run["observations"]
        is_reset = run["dataset"] is None and run["duration"] != float("inf")
        is_second_preparation = run["dataset"] is None and state.created[0].num_episodes == 1
        robots["right_leader"].encoder[0].io_inputs[0] = (
            (is_reset and count == 1) or (is_second_preparation and count == 4)
        )

    state.observation_hook = reset_and_resume
    operator_module.record(cfg)

    assert [run["duration"] for run in state.loops] == [float("inf"), 1, 0.25, float("inf"), 1]
    second_preparation = state.loops[3]
    assert second_preparation["observations"] >= 12  # the paused side did not auto-engage again
    assert [sum(frame["episode"] == index for frame in state.frames) for index in (0, 1)] == [16, 16]
    assert all(frame["ready"] for frame in state.frames)
    rows = episode_rows(cfg)
    assert [row["timestamp"] for row in rows if row["frame_index"] == 0] == [0, 0]


def test_ready_later_episode_does_not_repeat_synchronization(timed_recording):
    cfg, state, _ = timed_recording
    cfg.dataset.num_episodes = 2
    cfg.dataset.reset_time_s = 0.25
    operator_module.record(cfg)
    assert [run["duration"] for run in state.loops] == [float("inf"), 1, 0.25, float("inf"), 1]
    assert state.loops[3]["observations"] == 1  # fresh acknowledgment, without repeating synchronization
    assert len(state.frames) == 32


def test_button_pressed_during_save_is_read_before_next_episode_starts(timed_recording):
    cfg, state, robots = timed_recording
    cfg.dataset.num_episodes = 2

    def press_during_save(dataset):
        if dataset.num_episodes == 1:
            state.now += 20  # no observations/button edges were processed while encoding
            robots["right_leader"].encoder[0].io_inputs[0] = True

    def resume_next_episode(run):
        if state.created[0].num_episodes == 1 and run["dataset"] is None:
            count = run["observations"]
            # Keep the saved-time press for the first fresh observation, then
            # release and press again to resume the deliberately paused side.
            robots["right_leader"].encoder[0].io_inputs[0] = count in (1, 4)

    state.save_hook = press_during_save
    state.observation_hook = resume_next_episode
    operator_module.record(cfg)

    preparation = state.loops[3]
    assert preparation["dataset"] is None and preparation["observations"] >= 12
    first_send = next(send for send in state.sends if send["time"] == preparation["started"])
    assert first_send["engaged"] == [True, False] and not first_send["ready"]
    assert len(state.frames) == 32 and all(frame["ready"] for frame in state.frames)
    assert state.frames[16]["time"] >= preparation["ended"]


def test_rejected_follower_command_during_preparation_never_begins_episode(timed_recording, monkeypatch):
    cfg, state, robots = timed_recording

    def reject(_):
        raise RuntimeError("right follower rejected command")

    def break_right_follower(run):
        monkeypatch.setattr(robots["right_follower"], "command_joint_pos", reject)

    state.observation_hook = break_right_follower
    with pytest.raises(RuntimeError, match="right follower rejected command"):
        operator_module.record(cfg)
    assert not state.frames and not state.sends and not state.homes
    assert not any(message.startswith("Recording episode") for message, _, _ in state.sayings)
    assert state.created[0]._is_finalized and state.created[0].num_episodes == 0


def test_manual_cli_keeps_recording_holds_without_waiting_for_engagement(timed_recording):
    cfg, state, _ = timed_recording
    cfg.teleop.auto_engage = False
    operator_module.record(cfg)
    assert len(state.loops) == 1 and state.loops[0]["dataset"] is state.created[0]
    assert len(state.frames) == 16 and state.frames[0]["time"] == 100
    assert all(frame["engaged"] == [False, False] for frame in state.frames)


@pytest.mark.parametrize("stop_signal", [signal.SIGUSR1, signal.SIGINT], ids=["cooperative", "immediate"])
@pytest.mark.parametrize("when", ["preparation", "ready-announcement"])
def test_dashboard_stop_before_acquisition_saves_no_empty_episode_or_home(
    timed_recording, monkeypatch, stop_signal, when,
):
    cfg, state, robots = timed_recording
    monkeypatch.setenv("YAMKIT_PREVIEW_SESSION", "ready-fixture")
    monkeypatch.setenv("YAMKIT_PREVIEW_TOKEN", "ready-fixture-token")
    previous = signal.getsignal(signal.SIGUSR1)
    signal.signal(signal.SIGUSR1, signal.SIG_DFL)

    def during_preparation(run):
        # Make a return-home attempt observable; initial connection used no home.
        for plugin in (cfg.robot._runtime_robot, cfg.teleop._runtime_teleop):
            for handle in plugin._sides.values():
                handle.home_speed = 50
        if when == "preparation" and run["observations"] == 3:
            signal.raise_signal(stop_signal)

    def at_announcement(message):
        if when == "ready-announcement" and message == "Recording episode 0":
            signal.raise_signal(stop_signal)

    state.observation_hook = during_preparation
    state.say_hook = at_announcement
    try:
        with pytest.raises(SystemExit) as stopped:
            operator_module.record(cfg)
        assert stopped.value.code == 130
    finally:
        signal.signal(signal.SIGUSR1, previous)

    assert len(state.loops) == 1 and not state.frames and not state.homes
    assert len(robots) == 4 and all(robot.closed for robot in robots.values())
    dataset = state.created[0]
    assert dataset._is_finalized and dataset.num_episodes == dataset.num_frames == 0
    assert not dataset.has_pending_frames() and not episode_rows(cfg)


def test_readiness_context_restores_nested_adapters_and_callback_on_error(timed_recording):
    cfg, state, _ = timed_recording
    processor = operator_module.make_teleop_processor(cfg.robot, cfg.teleop, cfg.dataset.fps)
    original_loop, original_say = recorder.record_loop, recorder.log_say
    with (pytest.raises(RuntimeError, match="fixture failure"), operator_module.record_stop_events(),
          operator_module.record_ready_events(processor)):
        outer_loop, outer_say = recorder.record_loop, recorder.log_say
        with operator_module.record_ready_events(processor):
            assert recorder.record_loop is outer_loop and recorder.log_say is outer_say
        assert recorder.record_loop is outer_loop and recorder.log_say is outer_say
        raise RuntimeError("fixture failure")
    assert recorder.record_loop is original_loop and recorder.log_say is original_say
    assert state.processor.steps[0]._on_ready is None


def test_readiness_adapter_accepts_upstream_arguments_positionally(timed_recording, monkeypatch):
    cfg, state, _ = timed_recording
    upstream_record = recorder.record
    positional_names = ("robot", "events", "fps", "teleop_action_processor", "robot_action_processor",
                        "robot_observation_processor", "dataset", "teleop", "control_time_s")

    def positional_record(*args, **kwargs):
        ready_loop = recorder.record_loop

        def positional_loop(**loop_kwargs):
            positional = [loop_kwargs.pop(name, None) for name in positional_names]
            return ready_loop(*positional, **loop_kwargs)

        recorder.record_loop = positional_loop
        try:
            return upstream_record(*args, **kwargs)
        finally:
            recorder.record_loop = ready_loop

    monkeypatch.setattr(recorder, "record", positional_record)
    operator_module.record(cfg)
    assert len(state.loops) == 2 and len(state.frames) == 16
    assert all(frame["ready"] for frame in state.frames)
