"""Actual pinned recording loop with fake YAM plugins and real process SIGINT delivery."""

import signal
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.scripts import lerobot_record as recorder
from lerobot_robot_yamkit import BiYamFollower, BiYamFollowerConfig
from lerobot_teleoperator_yamkit import BiYamLeaderConfig

from yamkit.lerobot_teleop import record, record_stop_events


@pytest.fixture
def recording(rig, fake_connect, monkeypatch):
    rig.control.home_speed = rig.control.leader_home_speed = 0
    rig.save()
    events = {"exit_early": False, "stop_recording": False, "rerecord_episode": False}
    state = SimpleNamespace(frames=[], loops=[], saved=0, finalized=0, events=events,
                            real_create=recorder.LeRobotDataset.create,
                            real_video_manager=recorder.VideoEncodingManager,
                            frame_hook=lambda: None, loop_hook=lambda dataset: None,
                            save_hook=lambda: None, finalize_hook=lambda: None)

    class Dataset:
        num_episodes = 0
        fps = 30
        pending = 0

        def __bool__(self):
            return bool(state.frames)

        def add_frame(self, frame):
            self.pending += 1
            state.frames.append(frame)
            state.frame_hook()

        def has_pending_frames(self):
            return self.pending > 0

        def save_episode(self):
            if not self.has_pending_frames():
                raise ValueError("empty episode buffer")
            state.save_hook()
            self.num_episodes += 1
            state.saved += 1
            self.pending = 0

        def finalize(self):
            state.finalized += 1
            state.finalize_hook()

    dataset = Dataset()

    def create(*args, **kwargs):
        dataset.features = kwargs["features"]
        return dataset

    original_loop = recorder.record_loop

    def track_loop(*args, **kwargs):
        state.loops.append(kwargs.get("dataset"))
        state.loop_hook(kwargs.get("dataset"))
        return original_loop(*args, **kwargs)

    monkeypatch.setattr(recorder, "record_loop", track_loop)
    monkeypatch.setattr(recorder.LeRobotDataset, "create", create)
    monkeypatch.setattr(recorder, "VideoEncodingManager", lambda dataset: nullcontext())
    monkeypatch.setattr(recorder, "init_keyboard_listener", lambda: (None, events))
    monkeypatch.setattr(recorder, "log_say", lambda *args, **kwargs: None)
    cfg = recorder.RecordConfig(
        robot=BiYamFollowerConfig(rig=str(rig.path)), teleop=BiYamLeaderConfig(rig=str(rig.path)),
        play_sounds=False,
        dataset=DatasetRecordConfig(repo_id="yamkit/fixture", single_task="fixture", fps=30, video=False,
                                    push_to_hub=False, num_episodes=3, episode_time_s=1, reset_time_s=1,
                                    no_stamp=True),
    )
    previous_signal = signal.getsignal(signal.SIGINT)
    yield cfg, dataset, state, fake_connect
    assert signal.getsignal(signal.SIGINT) is previous_signal
    assert recorder.record_loop is track_loop
    assert all(robot.closed for robot in fake_connect.values())


def test_first_sigint_saves_partial_episode_and_finishes_successfully(recording):
    cfg, dataset, state, robots = recording
    state.frame_hook = lambda: signal.raise_signal(signal.SIGINT)

    assert record(cfg) is dataset

    assert len(state.frames) == state.saved == state.finalized == 1
    assert state.loops == [dataset]  # no reset loop after Stop
    assert len(robots) == 4


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_stop_before_first_frame_cancels_with_real_empty_dataset(recording, monkeypatch, tmp_path, failure):
    import json

    cfg, _, state, robots = recording
    cfg.dataset.root = tmp_path / "empty-recording"
    created = []
    previous = signal.getsignal(signal.SIGINT)

    def create(*args, **kwargs):
        dataset = state.real_create(*args, **kwargs)
        created.append(dataset)
        if failure:
            finalize = dataset.finalize

            def fail_finalize():
                assert signal.getsignal(signal.SIGINT) is previous
                finalize()
                if failure is KeyboardInterrupt:
                    signal.raise_signal(signal.SIGINT)
                raise failure("empty finalization failed")

            monkeypatch.setattr(dataset, "finalize", fail_finalize)
        return dataset

    monkeypatch.setattr(recorder.LeRobotDataset, "create", create)
    monkeypatch.setattr(recorder, "VideoEncodingManager", state.real_video_manager)
    state.loop_hook = lambda current: signal.raise_signal(signal.SIGINT)

    with pytest.raises(failure or KeyboardInterrupt, match="before any frames" if failure is None else None):
        record(cfg)

    assert len(created) == 1 and not created[0].has_pending_frames()
    assert created[0]._is_finalized
    info = json.loads((cfg.dataset.root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == info["total_frames"] == 0
    assert not list(cfg.dataset.root.glob("data/**/*.parquet"))
    assert len(robots) == 4


def test_second_sigint_interrupts_without_saving_interrupted_episode(recording):
    cfg, _, state, _ = recording

    def two_stops():
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)

    state.frame_hook = two_stops
    with pytest.raises(KeyboardInterrupt):
        record(cfg)

    assert len(state.frames) == state.finalized == 1
    assert state.saved == 0


def test_stop_during_reset_saves_completed_episode(recording):
    cfg, dataset, state, _ = recording
    state.frame_hook = lambda: state.events.update(exit_early=True)
    state.loop_hook = lambda current: signal.raise_signal(signal.SIGINT) if current is None else None

    assert record(cfg) is dataset

    assert len(state.frames) == state.saved == state.finalized == 1
    assert state.loops == [dataset, None]


@pytest.mark.parametrize("phase", ["save_hook", "finalize_hook"])
def test_stop_after_acquisition_retains_immediate_interrupt(recording, phase):
    cfg, _, state, _ = recording
    state.frame_hook = lambda: signal.raise_signal(signal.SIGINT)
    setattr(state, phase, lambda: signal.raise_signal(signal.SIGINT))

    with pytest.raises(KeyboardInterrupt):
        record(cfg)

    assert len(state.frames) == state.finalized == 1


def test_failed_finalization_after_graceful_stop_is_still_failure(recording):
    cfg, _, state, _ = recording
    state.frame_hook = lambda: signal.raise_signal(signal.SIGINT)

    def fail():
        raise RuntimeError("fixture finalization failed")

    state.finalize_hook = fail
    with pytest.raises(RuntimeError, match="finalization failed"):
        record(cfg)

    assert state.saved == state.finalized == 1


def test_startup_interrupt_has_original_signal_behavior(recording, monkeypatch):
    cfg, _, state, _ = recording
    previous = signal.getsignal(signal.SIGINT)

    def stop_connect(robot):
        assert signal.getsignal(signal.SIGINT) is previous
        signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(BiYamFollower, "connect", stop_connect)
    with pytest.raises(KeyboardInterrupt):
        record(cfg)

    assert not state.frames and not state.loops and state.saved == 0


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("nested", [False, True])
def test_saving_phase_observes_pinned_save_without_changing_interrupts(caplog, failure, nested):
    caplog.set_level("INFO", logger="yamkit.lerobot_teleop")
    original = recorder.LeRobotDataset.save_episode
    previous_signal = signal.getsignal(signal.SIGINT)
    dataset = object.__new__(recorder.LeRobotDataset)
    dataset.reader = None
    dataset.meta = SimpleNamespace(total_episodes=3)
    episode_data = {"fixture": object()}
    calls = []

    def save(data, parallel_encoding):
        calls.append((data, parallel_encoding))
        messages = [record.message for record in caplog.records if record.name == "yamkit.lerobot_teleop"]
        assert messages == ["Saving episode 3: encoding videos; followers hold their last command. Stop interrupts saving."]
        assert signal.getsignal(signal.SIGINT) is previous_signal
        if failure is KeyboardInterrupt:
            signal.raise_signal(signal.SIGINT)
        elif failure:
            raise failure("save failed")

    dataset.writer = SimpleNamespace(save_episode=save)
    with (pytest.raises(failure) if failure else nullcontext()), record_stop_events():
        outer = recorder.LeRobotDataset.save_episode
        with record_stop_events() if nested else nullcontext():
            assert recorder.LeRobotDataset.save_episode is outer  # nesting must not duplicate phase logs
            assert dataset.save_episode(episode_data, parallel_encoding=False) is None
    assert calls == [(episode_data, False)]
    assert recorder.LeRobotDataset.save_episode is original
    assert signal.getsignal(signal.SIGINT) is previous_signal


def test_nested_record_adapters_restore_loop_on_cancellation():
    original = recorder.record_loop
    original_save = recorder.LeRobotDataset.save_episode
    with pytest.raises(SystemExit), record_stop_events():
        outer = recorder.record_loop
        outer_save = recorder.LeRobotDataset.save_episode
        with record_stop_events():
            assert recorder.record_loop is not outer
            assert recorder.LeRobotDataset.save_episode is outer_save
        assert recorder.record_loop is outer
        assert recorder.LeRobotDataset.save_episode is outer_save
        raise SystemExit(130)
    assert recorder.record_loop is original
    assert recorder.LeRobotDataset.save_episode is original_save
