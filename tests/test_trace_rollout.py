"""All execution uses fake objects/stubbed runtime methods; no device or cloud calls."""
import importlib.util
import json
import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("trace_rollout", Path(__file__).resolve().parents[1] / "scripts/trace_rollout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
reserve_frames = module.Collector.reserve_frames


@pytest.fixture(autouse=True)
def forbid_hardware(monkeypatch):
    from lerobot_robot_yamkit import yam_follower

    from yamkit.arm import YamArm
    from yamkit.inference import http_transport

    def forbidden(*args, **kwargs):
        pytest.fail("Test attempted an unstubbed hardware/cloud operation")

    monkeypatch.setattr(YamArm, "connect", forbidden)
    monkeypatch.setattr(yam_follower, "make_cameras_from_configs", forbidden)
    monkeypatch.setattr(http_transport, "_make_client", forbidden)
    monkeypatch.setattr(module, "render_report", lambda *_: None)
    # Fake CLI tests need no full-size allocation. Dedicated reservation tests
    # exercise the real method against a tiny pool and synthetic Linux limits.
    monkeypatch.setattr(module.Collector, "reserve_frames", lambda *_: None)
    level = logging.getLogger().level
    yield
    logging.getLogger().setLevel(level)


def test_default_plan_does_not_execute(monkeypatch, capsys):
    monkeypatch.setattr(module, "execute", lambda *_: pytest.fail("plan executed"))
    assert module.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLAN_ONLY" and result["hardware_opened"] is False
    assert module.plan(10)["max_frame_bytes"] == 837734400
    assert result["video_fps"] == 30 and result["max_frame_triplets"] == 153
    assert "return home" in result["run_effects"] and "Stop" in result["run_effects"]


@pytest.mark.parametrize("args", [
    ["--run"], ["--run", "--modal-app", "yamkit-vla-test"],
    ["--run", "--confirm-supervised"],
    ["--run", "--confirm-supervised", "--modal-app", "bad/app"],
    ["--run", "--confirm-supervised", "--modal-app", "yamkit-vla-test", "--duration", "11"],
    ["--plan", "--run"],
])
def test_invalid_run_never_executes(args, monkeypatch):
    monkeypatch.setattr(module, "execute", lambda *_: pytest.fail("invalid run executed"))
    with pytest.raises(SystemExit):
        module.main(args)


def test_explicit_run_keeps_canonical_flags(monkeypatch):
    seen = []
    monkeypatch.setattr(module, "execute", lambda args: seen.append(module.rollout_arguments(args)) or 0)
    assert module.main(["--run", "--duration", "10", "--modal-app", "yamkit-vla-test",
                        "--confirm-supervised"]) == 0
    argv = seen[0]
    assert argv[argv.index("--task") + 1] == module.TASK
    assert argv[argv.index("--duration") + 1] == "10"
    assert "--accept-mapping" in argv and "--confirm-supervised" in argv
    assert "--no-home" not in argv and "--prediction-queue-threshold" not in argv


def test_explicit_rig_is_forwarded_without_environment_changes(monkeypatch):
    monkeypatch.setenv("YAMKIT_PREVIEW_TEST", "managed-session")
    import os

    args = module.parse_args(["--plan", "--rig", "configs/rig.example.yaml"])
    before = dict(os.environ)
    assert module.rollout_arguments(args)[-2:] == ["--rig", "configs/rig.example.yaml"]
    assert dict(os.environ) == before


def test_artifact_directory_refuses_overwrite_escape_and_symlink(tmp_path):
    base = tmp_path / ".context" / "rollout-traces"
    base.mkdir(parents=True)
    selected = base / "unique-session"
    assert module.artifact_directory(tmp_path, selected, "unused") == selected
    assert module.artifact_directory(tmp_path, None, "timestamp") == base / "timestamp"
    selected.mkdir()
    for rejected in (selected, base, tmp_path / "outside-trace-directory"):
        with pytest.raises(ValueError):
            module.artifact_directory(tmp_path, rejected, "unused")
    (base / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        module.artifact_directory(tmp_path, base / "escape" / "new", "unused")


def test_every_observation_is_copied_once_with_original_time_and_caps_memory():
    clock = [0.0]
    collector = module.Collector(5, clock=lambda: clock[0])
    collector.frame_capacity = 3
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {**dict.fromkeys(module.ACTION_NAMES, .25), **dict.fromkeys(module.CAMERAS, frame)}
    collector.start_phase()
    collector.observation(obs)
    frame[:] = 17
    clock[0] = .01
    collector.observation(obs)
    assert len(collector.frames) == 2
    assert collector.frames[0][1][0].max() == 0
    clock[0] = .21
    collector.observation(obs)
    clock[0] = .41
    collector.observation(obs)
    assert len(collector.frames) == 3 and collector.counts["frames_dropped"] == 1
    assert collector.counts["observation_frames_seen"] == 4
    assert collector.frame_observation_indices == [0, 1, 2]
    assert [at for at, _ in collector.frames] == [0, .01, .21]
    assert collector.frames[1][1][0].min() == 17
    collector.end_phase()
    prior = len(collector.events)
    collector.observation(obs)
    assert len(collector.events) == prior


def test_reserved_pool_is_checked_and_prefaulted_before_capture(monkeypatch):
    collector = module.Collector(5)
    collector.frame_capacity = 2
    required = 2 * module.FRAME_TRIPLET_BYTES + module.MEMORY_HEADROOM_BYTES
    monkeypatch.setattr(module, "available_memory_bytes", lambda: required - 1)
    with pytest.raises(MemoryError):
        reserve_frames(collector)
    assert collector.frame_pool is None
    monkeypatch.setattr(module, "available_memory_bytes", lambda: required)
    reserve_frames(collector)
    assert collector.frame_pool.nbytes == 2 * module.FRAME_TRIPLET_BYTES
    assert not collector.frame_pool.any()
    frame = np.full((480, 640, 3), 71, dtype=np.uint8)
    collector.start_phase()
    collector.observation({**dict.fromkeys(module.ACTION_NAMES, .25), **dict.fromkeys(module.CAMERAS, frame)})
    for copied in collector.frames[0][1]:
        assert np.shares_memory(copied, collector.frame_pool)
        assert not np.shares_memory(copied, frame)
        assert copied.min() == 71


@pytest.mark.parametrize("cgroup_version", [1, 2])
def test_available_memory_honors_host_and_ancestor_cgroup_limits(tmp_path, cgroup_version):
    meminfo, process = tmp_path / "meminfo", tmp_path / "process-cgroup"
    meminfo.write_text("MemTotal: 16000000 kB\nMemAvailable: 3000000 kB\n")
    root = tmp_path / "cgroup"
    base = root if cgroup_version == 2 else root / "memory"
    group = base / "parent" / "child"
    group.mkdir(parents=True)
    process.write_text("0::/parent/child\n" if cgroup_version == 2 else "4:memory:/parent/child\n")
    limit = "memory.max" if cgroup_version == 2 else "memory.limit_in_bytes"
    used = "memory.current" if cgroup_version == 2 else "memory.usage_in_bytes"
    (group / limit).write_text("4000000000")
    (group / used).write_text("1000000000")
    (group.parent / limit).write_text("2000000000")
    (group.parent / used).write_text("500000000")
    assert module.available_memory_bytes(meminfo, root, process) == 1500000000
    (group.parent / limit).write_text("max" if cgroup_version == 2 else "9999999999999999")
    assert module.available_memory_bytes(meminfo, root, process) == 3000000000
    (group / limit).write_text("9999999999999999")
    assert module.available_memory_bytes(meminfo, root, process) == 3000000 * 1024
    meminfo.write_text("MemTotal: 16000000 kB\n")
    with pytest.raises(ValueError):
        module.available_memory_bytes(meminfo, root, process)


def test_trace_overflow_and_faults_are_explicit(monkeypatch):
    monkeypatch.setattr(module, "MAX_EVENTS", 2)
    monkeypatch.setattr(module, "MAX_CHUNKS", 1)
    collector = module.Collector(5)
    collector.start_phase()
    for _ in range(4):
        collector.event("extra")
    result = torch.zeros((1, 30, 14))
    collector.capture_chunk(result, 1)
    result[:] = 2
    collector.capture_chunk(result, 2)
    assert collector.chunks[0]["actions"].max() == 0
    assert collector.counts["events_dropped"] == 3 and collector.counts["chunks_dropped"] == 1
    collector.safely(lambda: (_ for _ in ()).throw(ValueError("trace error")))
    assert collector.counts["trace_errors"] == 1
    with pytest.raises(TimeoutError):
        collector.safely(lambda: (_ for _ in ()).throw(TimeoutError()))


def test_hooks_preserve_order_results_and_partial_dispatch_error(monkeypatch):
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot_robot_yamkit.yam_follower import BiYamFollower, _FollowerHandle

    from yamkit import cli
    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy
    from yamkit.remote_rollout import InvalidatableActionQueue, UnguidedRemoteInferenceEngine

    calls = []
    collector = module.Collector(5)
    output = {"joint_1.pos": .05}
    requested = {"joint_1.pos": .3}
    left = SimpleNamespace(spec=SimpleNamespace(name="left_follower"))
    right = SimpleNamespace(spec=SimpleNamespace(name="right_follower"))
    robot = SimpleNamespace(_sides={"left": SimpleNamespace(arm=None), "right": SimpleNamespace(arm=None)},
                            _opened_cameras=[], _camera_lease=None)
    observed = dict.fromkeys(module.ACTION_NAMES, .01)
    policy = SimpleNamespace(_observation_time=123.0)
    chunk = torch.zeros((1, 30, 14))
    expected_error = RuntimeError("right-side failed after left completed")

    def send(handle, action, **kwargs):
        calls.append(handle.spec.name)
        assert action is requested
        if handle is right:
            raise expected_error
        return output

    def run(*args):
        assert collector.active
        assert _FollowerHandle.send(left, requested) is output
        return _FollowerHandle.send(right, requested)

    monkeypatch.setattr(BaseStrategy, "run", run)
    monkeypatch.setattr(BiYamFollower, "connect", lambda self: calls.append("connect"))
    monkeypatch.setattr(BiYamFollower, "get_observation", lambda self: observed)
    monkeypatch.setattr(_FollowerHandle, "send", send)
    monkeypatch.setattr(YamkitRemotePolicy, "predict_action_chunk", lambda *a, **kw: chunk)
    monkeypatch.setattr(UnguidedRemoteInferenceEngine, "_record_merge", lambda *a: None)
    monkeypatch.setattr(InvalidatableActionQueue, "get", lambda self: output)
    original_print = cli._print_inference_result
    with module.install_hooks(collector):
        BiYamFollower.connect(robot)
        collector.active = True
        assert BiYamFollower.get_observation(robot) is observed
        assert YamkitRemotePolicy.predict_action_chunk(policy, {}) is chunk
        assert InvalidatableActionQueue.get(SimpleNamespace(last_action_deadline=9)) is output
        UnguidedRemoteInferenceEngine._record_merge(None, {"accepted_steps": 5})
        cli._print_inference_result({"executed_actions": 1})
        with pytest.raises(RuntimeError) as caught:
            BaseStrategy.run(None, None)
        assert caught.value is expected_error
    assert calls == ["connect", "left_follower", "right_follower"]
    assert not collector.active and collector.released()
    assert _FollowerHandle.send is send and cli._print_inference_result is original_print
    send_events = [event for event in collector.events if event["kind"].startswith("send_")]
    assert [event["kind"] for event in send_events] == ["send_start", "send_end", "send_start", "send_error"]
    assert send_events[0]["requested"] == requested and send_events[1]["postclamp"] == output
    assert send_events[3]["partial_dispatch_possible"] is True
    assert collector.metrics == {"executed_actions": 1}


def test_export_refuses_images_while_resources_remain_open(tmp_path, monkeypatch):
    from lerobot.datasets import image_writer

    collector = module.Collector(5)
    collector.robots.append(SimpleNamespace(_sides={"left": SimpleNamespace(arm=object())},
                                            _opened_cameras=[], _camera_lease=None))
    collector.frames.append((1, (object(), object(), object())))
    collector.metrics = {"large": list(range(10000))}
    monkeypatch.setattr(image_writer, "write_image", lambda *_a, **_k: pytest.fail("wrote image before release"))
    monkeypatch.setattr(module, "encode_timestamped_video", lambda *_a, **_k: pytest.fail("encoded before release"))
    monkeypatch.setattr(module, "render_report", lambda *_a: pytest.fail("rendered before release"))
    summary = module.export(collector, tmp_path)
    assert summary["resources_released"] is False
    assert json.loads((tmp_path / "metrics.json").read_text()) == collector.metrics


def test_export_reuses_lerobot_after_release_and_preserves_timestamp_gaps(tmp_path, monkeypatch):
    from lerobot.datasets import image_writer

    collector = module.Collector(5)
    collector.frames = [(1, ("a", "b", "c")), (1.6, ("d", "e", "f"))]
    collector.phase_started, collector.phase_ended = .9, 1.7
    collector.frame_observation_indices = [0, 3]
    collector.counts["observation_frames_seen"] = 4
    collector.counts["frames_dropped"] = 2
    calls = []
    monkeypatch.setattr(image_writer, "write_image", lambda *a, **kw: calls.append(("image", a[0])))
    monkeypatch.setattr(module, "encode_timestamped_video", lambda images, path, timeline:
                        calls.append(("video", list(images), timeline)))
    monkeypatch.setattr(module, "render_report", lambda *_: calls.append(("render", 0)))
    summary = module.export(collector, tmp_path)
    assert summary["resources_released"] is True
    assert summary["rollout_error"] is None
    assert len(calls) == 10 and sum(call[0] == "video" for call in calls) == 3
    assert calls[-1][0] == "render"
    assert json.loads((tmp_path / "frame_timestamps.json").read_text()) == [1, 1.6]
    timeline = json.loads((tmp_path / "video_timeline.json").read_text())
    assert [row["pts"] for row in timeline["frames"]] == [0, 600000]
    assert [row["observation_index"] for row in timeline["frames"]] == [0, 3]
    assert summary["observation_frames_missing"] == 2 and summary["overflow"] is True
    assert timeline["duration_s"] == pytest.approx(.7)
    assert timeline["policy_phase_offset_s"] == pytest.approx(.1)
    assert calls[2][1] == ["a", "d"] and calls[2][2] == timeline


def test_export_timeout_is_not_swallowed_and_retried(tmp_path, monkeypatch):
    from lerobot.datasets import image_writer

    collector = module.Collector(5)
    collector.frames = [(1, ("a", "b", "c"))]
    collector.phase_started, collector.phase_ended = 1, 1.034
    monkeypatch.setattr(image_writer, "write_image", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):
        module.export(collector, tmp_path)
    assert (tmp_path / "summary.json").exists()


def test_actual_encoder_preserves_irregular_pts_all_original_frames_and_duration(tmp_path):
    import av
    from PIL import Image

    collector = module.Collector(5)
    random = np.random.default_rng(7283)
    frame = random.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    collector.frames = [(at, (np.roll(frame, index, axis=1),) * 3)
                        for index, at in enumerate((100.02, 100.053, 100.14))]
    collector.phase_started, collector.phase_ended = 100, 100.20
    summary = module.export(collector, tmp_path)
    assert summary["video_export_errors"] == {}
    for name in module.CAMERAS:
        with av.open(str(tmp_path / f"{name}.mp4")) as video:
            stream = video.streams.video[0]
            decoded = list(video.decode(stream))
            assert len(decoded) == 3  # No dropped, repeated or interpolated frames.
            assert [float(frame.pts * frame.time_base) for frame in decoded] == pytest.approx([0, .033, .12], abs=1e-6)
            assert float(stream.duration * stream.time_base) == pytest.approx(.18, abs=1e-6)
            assert stream.codec_context.name == "h264" and stream.pix_fmt == "yuv420p"
        for index, (_, images) in enumerate(collector.frames):
            with Image.open(tmp_path / "frames" / name / f"frame-{index:06d}.png") as image:
                np.testing.assert_array_equal(np.asarray(image), images[0])
    timeline = json.loads((tmp_path / "video_timeline.json").read_text())
    assert timeline["policy_phase_offset_s"] == pytest.approx(.02)
    assert [row["duration_ticks"] for row in timeline["frames"]] == [33000, 87000, 60000]


@pytest.mark.parametrize("duration", [5, 10])
def test_full_30hz_video_keeps_every_frame_and_real_duration(tmp_path, duration):
    import av

    timestamps = [100 + index / 30 for index in range(duration * 30)]
    timeline = module.video_timeline(timestamps, 100, 100 + duration)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    destination = tmp_path / "full-30hz.mp4"
    module.encode_timestamped_video((frame for _ in timestamps), destination, timeline)
    with av.open(str(destination)) as video:
        stream = video.streams.video[0]
        decoded = list(video.decode(stream))
        assert len(decoded) == duration * 30
        assert [float(item.pts * item.time_base) for item in decoded] == pytest.approx(
            [at - 100 for at in timestamps], abs=1e-6)
        assert float(stream.duration * stream.time_base) == pytest.approx(duration, abs=1e-6)


@pytest.mark.parametrize("timestamps,start,end,indices", [
    ([1, 1], 1, 2, None), ([1.1, 1], 1, 2, None), ([1, 1.0000001], 1, 2, None),
    ([1], 1.1, 2, None), ([1, 2], 1, 2, None), ([1], 1, 122, None),
    ([float("nan")], 0, 2, None), ([1], 0, float("inf"), None), ([True], 0, 2, None),
    ([1, 1.5], 0, 2, [2, 1]), ([1], 0, 2, []), ([1], 0, 2, [True]),
])
def test_video_timeline_rejects_invalid_bounds_order_precision_and_indices(timestamps, start, end, indices):
    with pytest.raises(ValueError):
        module.video_timeline(timestamps, start, end, indices)


@pytest.mark.parametrize("image_count", [1, 3])
def test_encoder_rejects_more_or_fewer_original_frames_than_timeline(tmp_path, image_count):
    timeline = module.video_timeline([1, 1.1], 1, 1.2)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        module.encode_timestamped_video([frame] * image_count, tmp_path / "invalid.mp4", timeline)


def test_renderer_failure_retains_complete_evidence(tmp_path, monkeypatch):
    collector = module.Collector(5)
    collector.metrics = {"executed_actions": 8}
    collector.event("observation", positions=[0.0] * 14)
    monkeypatch.setattr(module, "render_report", lambda *_: (_ for _ in ()).throw(ValueError("render failed")))
    summary = module.export(collector, tmp_path)
    assert summary["render_error_type"] == "ValueError" and not summary["report_available"]
    assert summary["status"] == "TRACE_SAVED_WITH_EXPORT_ERRORS"
    assert json.loads((tmp_path / "metrics.json").read_text()) == collector.metrics
    assert json.loads((tmp_path / "trace.json").read_text())["events"] == collector.events


@pytest.mark.parametrize("fault", [False, True])
def test_execute_uses_canonical_cli_and_preserves_full_metrics_after_fault(tmp_path, monkeypatch, capsys, fault):
    import os

    from yamkit import cli, paths
    from yamkit.inference.client import RemoteFault

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setenv("YAMKIT_PREVIEW_TEST", "managed-session")
    output = tmp_path / ".context" / "rollout-traces" / "ui-selected-session"
    args = module.parse_args(["--run", "--modal-app", "yamkit-vla-test", "--confirm-supervised",
                              "--output-dir", str(output)])
    metrics = {"executed_actions": 1, "failed": fault, "samples": [{"value": n} for n in range(2000)]}

    def fake_cli(*, args, standalone_mode):
        assert args[0] == "rollout" and standalone_mode is False
        assert os.environ["YAMKIT_PREVIEW_TEST"] == "managed-session"
        cli._print_inference_result(metrics)
        if fault:
            raise RemoteFault("fake inference fault after normal cleanup")

    monkeypatch.setattr(cli, "app", fake_cli)
    assert module.execute(args) == int(fault)
    assert json.loads((output / "metrics.json").read_text()) == metrics
    summary = json.loads((output / "summary.json").read_text())
    assert summary["rollout_error"] == (
        {"type": "RemoteFault", "message": "fake inference fault after normal cleanup"} if fault else None)
    assert json.loads(capsys.readouterr().out)["trace_directory"] == str(output)


@pytest.mark.parametrize("export_failure", [None, "before_summary", "after_summary"])
def test_camera_timeout_keeps_sanitized_cause_after_cleanup_even_if_export_fails(
        tmp_path, monkeypatch, caplog, capsys, export_failure):
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    from yamkit import cli, paths

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    output = tmp_path / ".context" / "rollout-traces" / "camera-timeout"
    args = module.parse_args(["--run", "--modal-app", "yamkit-vla-test", "--confirm-supervised",
                              "--output-dir", str(output)])
    fake_url = "https://private-camera.example.invalid/read?key=synthetic-private-value"
    fake_token = "hf_" + "SyntheticCredentialForSanitizationOnly123456789"
    reason = "left_wrist camera did not deliver a frame within 200 ms"
    message = f"{reason}; source {fake_url}; credential {fake_token}"
    robot = SimpleNamespace(_sides={"left": SimpleNamespace(arm=object())},
                            _opened_cameras=[object()], _camera_lease=object())
    calls = []
    monkeypatch.setattr(BiYamFollower, "connect", lambda _robot: calls.append("connect"))

    def camera_read(_robot):
        calls.append("camera_read")
        raise TimeoutError(message)

    def fake_cli(**_kwargs):
        BiYamFollower.connect(robot)  # The real tracing hook registers this fake robot.
        try:
            BiYamFollower.get_observation(robot)
        finally:
            robot._sides["left"].arm = None
            robot._opened_cameras.clear()
            robot._camera_lease = None
            calls.append("released")

    original_export = module.export

    def export_after_cleanup(collector, directory):
        assert collector.robots == [robot] and collector.released()
        assert calls == ["connect", "camera_read", "released"]
        if export_failure is not None:
            if export_failure == "after_summary":
                module.write_json(directory / "summary.json", {"status": "EXPORTING", "frame_count": 0})
            raise OSError("synthetic export failure after robot cleanup")
        return original_export(collector, directory)

    monkeypatch.setattr(BiYamFollower, "get_observation", camera_read)
    monkeypatch.setattr(cli, "app", fake_cli)
    monkeypatch.setattr(module, "export", export_after_cleanup)
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        assert module.execute(args) == 1
    summary = json.loads((output / "summary.json").read_text())
    error = summary["rollout_error"]
    assert error["type"] == "TimeoutError" and reason in error["message"]
    assert summary["resources_released"] is True
    logs = [record for record in caplog.records if record.name == module.__name__]
    assert len(logs) == 1 and logs[0].levelno == logging.ERROR
    assert reason in logs[0].getMessage() and "TimeoutError" in logs[0].getMessage()
    assert "camera_read" in logs[0].getMessage()  # A usable traceback, beyond only the exception type.
    public_output = capsys.readouterr().out
    assert json.loads(public_output)["exit_status"] == 1
    evidence = (output / "summary.json").read_text() + logs[0].getMessage() + public_output
    if export_failure is not None:
        assert summary["status"] == "EXPORT_FAILED" and summary["error_type"] == "OSError"
        export_error = json.loads((output / "export-error.json").read_text())
        assert export_error["rollout_error"] == error and export_error["resources_released"] is True
        evidence += (output / "export-error.json").read_text()
    for secret in (fake_url, "private-camera.example.invalid", "synthetic-private-value", fake_token):
        assert secret not in evidence


def test_execute_marks_interrupted_export_without_losing_summary(tmp_path, monkeypatch):
    from yamkit import cli, paths

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "app", lambda **kw: None)
    output = tmp_path / ".context" / "rollout-traces" / "interrupted-export"
    args = module.parse_args(["--run", "--modal-app", "yamkit-vla-test", "--confirm-supervised",
                              "--output-dir", str(output)])

    def failed_export(collector, directory):
        module.write_json(directory / "summary.json", {"status": "EXPORTING", "frame_count": 7})
        raise TimeoutError("synthetic export interruption")

    monkeypatch.setattr(module, "export", failed_export)
    assert module.execute(args) == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "EXPORT_FAILED" and summary["frame_count"] == 7
    assert summary["error_type"] == "TimeoutError"
    assert (output / "export-error.json").exists()


def test_insufficient_memory_stops_before_canonical_cli_can_open_hardware(tmp_path, monkeypatch, capsys):
    from yamkit import cli, paths

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(module, "available_memory_bytes", lambda: 0)
    monkeypatch.setattr(module.Collector, "reserve_frames", reserve_frames)
    monkeypatch.setattr(cli, "app", lambda **kw: pytest.fail("CLI started without its trace memory reservation"))
    args = module.parse_args(["--run", "--modal-app", "yamkit-vla-test", "--confirm-supervised"])
    assert module.execute(args) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["rollout_error_type"] == "MemoryError"
    summary = json.loads((Path(result["trace_directory"]) / "summary.json").read_text())
    assert summary["resources_released"] is True and summary["frame_count"] == 0


@pytest.mark.parametrize("preconfigured_warning_handler", [False, True])
def test_fresh_process_hooks_and_reservation_preserve_ui_phase_logs(tmp_path, preconfigured_warning_handler):
    # Import order matters: pytest has already imported LeRobot and configured
    # logging. Only a fresh interpreter reproduces its import-time basicConfig.
    program = textwrap.dedent('''
        import importlib.util
        import logging
        import sys
        from pathlib import Path

        root = Path(sys.argv[1])
        spec = importlib.util.spec_from_file_location('trace_rollout', root / 'scripts/trace_rollout.py')
        trace = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trace)
        from yamkit import cli, paths

        if sys.argv[3] == 'True':
            logging.basicConfig(level=logging.WARNING)
        original_handlers = tuple(logging.getLogger().handlers)
        original_collector = trace.Collector

        class TinyCollector(original_collector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.frame_capacity = 1

        trace.Collector = TinyCollector
        trace.available_memory_bytes = lambda: trace.MEMORY_HEADROOM_BYTES + trace.FRAME_TRIPLET_BYTES
        trace.render_report = lambda *_: None

        def fake_physical_cli(*, args, standalone_mode):
            # execute has installed the real hooks and prefaulted a tiny real
            # frame pool, but only this stub can run instead of the hardware CLI.
            assert args[0] == 'rollout' and standalone_mode is False
            cli._main(verbose=False)
            logger = logging.getLogger('yamkit.remote_rollout')
            for phase in ('running', 'returning_home', 'releasing', 'released'):
                logger.info('[yamkit-rollout] ' + phase)
            logging.getLogger().handle(logging.LogRecord('root', logging.INFO,
                str(paths.ROOT / 'third_party/i2rt/fake_vendor.py'), 1,
                'SYNTHETIC_VENDOR_INFO_MUST_STAY_QUIET', (), None))
            cli._print_inference_result({'synthetic_fixture': True, 'hardware_opened': False})

        cli.app = fake_physical_cli
        paths.ROOT = Path(sys.argv[2])
        args = trace.parse_args(['--run', '--modal-app', 'yamkit-vla-fake-only', '--confirm-supervised'])
        assert trace.execute(args) == 0
        assert logging.getLogger().level == logging.INFO
        if original_handlers:
            assert tuple(logging.getLogger().handlers) == original_handlers
    ''')
    result = subprocess.run([sys.executable, "-c", program, str(Path(__file__).resolve().parents[1]),
                             str(tmp_path), str(preconfigured_warning_handler)],
                            capture_output=True, text=True, timeout=40, check=False)
    assert result.returncode == 0, result.stderr
    for phase in ("running", "returning_home", "releasing", "released"):
        assert sum(line.endswith(f"[yamkit-rollout] {phase}") for line in result.stderr.splitlines()) == 1
    assert "SYNTHETIC_VENDOR_INFO_MUST_STAY_QUIET" not in result.stderr
    final = json.loads(result.stdout.splitlines()[-1])
    summary = json.loads((Path(final["trace_directory"]) / "summary.json").read_text())
    assert summary["memory_preflight"]["frame_bytes"] == module.FRAME_TRIPLET_BYTES
    assert summary["resources_released"] is True and summary["frame_count"] == 0
