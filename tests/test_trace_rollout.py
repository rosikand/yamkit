"""All execution uses fake objects/stubbed runtime methods; no device or cloud calls."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("trace_rollout", Path(__file__).resolve().parents[1] / "scripts/trace_rollout.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


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


def test_default_plan_does_not_execute(monkeypatch, capsys):
    monkeypatch.setattr(module, "execute", lambda *_: pytest.fail("plan executed"))
    assert module.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PLAN_ONLY" and result["hardware_opened"] is False
    assert module.plan(10)["max_frame_bytes"] == 146534400


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


def test_observation_copies_once_at_rate_and_caps_memory():
    clock = [0.0]
    collector = module.Collector(5, clock=lambda: clock[0])
    collector.frame_capacity = 2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    obs = {**dict.fromkeys(module.ACTION_NAMES, .25), **dict.fromkeys(module.CAMERAS, frame)}
    collector.start_phase()
    collector.observation(obs)
    frame[:] = 17
    clock[0] = .01
    collector.observation(obs)
    assert len(collector.frames) == 1
    assert collector.frames[0][1][0].max() == 0
    clock[0] = .21
    collector.observation(obs)
    clock[0] = .41
    collector.observation(obs)
    assert len(collector.frames) == 2 and collector.counts["frames_dropped"] == 1
    assert collector.frames[1][1][0].min() == 17
    collector.end_phase()
    prior = len(collector.events)
    collector.observation(obs)
    assert len(collector.events) == prior


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
    from lerobot.datasets import image_writer, video_utils

    collector = module.Collector(5)
    collector.robots.append(SimpleNamespace(_sides={"left": SimpleNamespace(arm=object())},
                                            _opened_cameras=[], _camera_lease=None))
    collector.frames.append((1, (object(), object(), object())))
    collector.metrics = {"large": list(range(10000))}
    monkeypatch.setattr(image_writer, "write_image", lambda *_a, **_k: pytest.fail("wrote image before release"))
    monkeypatch.setattr(video_utils, "encode_video_frames", lambda *_a, **_k: pytest.fail("encoded before release"))
    monkeypatch.setattr(module, "render_report", lambda *_a: pytest.fail("rendered before release"))
    summary = module.export(collector, tmp_path)
    assert summary["resources_released"] is False
    assert json.loads((tmp_path / "metrics.json").read_text()) == collector.metrics


def test_export_reuses_lerobot_after_release_and_preserves_timestamp_gaps(tmp_path, monkeypatch):
    from lerobot.datasets import image_writer, video_utils

    collector = module.Collector(5)
    collector.frames = [(1, ("a", "b", "c")), (1.6, ("d", "e", "f"))]
    collector.counts["video_sample_slots_missed"] = 2
    calls = []
    monkeypatch.setattr(image_writer, "write_image", lambda *a, **kw: calls.append(("image", a[0])))
    monkeypatch.setattr(video_utils, "encode_video_frames", lambda *a, **kw: calls.append(("video", a[2])))
    monkeypatch.setattr(module, "render_report", lambda *_: calls.append(("render", 0)))
    summary = module.export(collector, tmp_path)
    assert summary["resources_released"] is True
    assert len(calls) == 10 and sum(call[0] == "video" for call in calls) == 3
    assert calls[-1][0] == "render"
    assert json.loads((tmp_path / "frame_timestamps.json").read_text()) == [1, 1.6]


def test_export_timeout_is_not_swallowed_and_retried(tmp_path, monkeypatch):
    from lerobot.datasets import image_writer

    collector = module.Collector(5)
    collector.frames = [(1, ("a", "b", "c"))]
    monkeypatch.setattr(image_writer, "write_image", lambda *a, **kw: (_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):
        module.export(collector, tmp_path)
    assert (tmp_path / "summary.json").exists()


def test_actual_lerobot_encoder_exports_fake_rgb_frames(tmp_path):
    collector = module.Collector(5)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    collector.frames = [(1, (frame, frame, frame))]
    summary = module.export(collector, tmp_path)
    assert summary["video_export_errors"] == {}
    assert all((tmp_path / f"{name}.mp4").stat().st_size > 0 for name in module.CAMERAS)


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
    assert json.loads(capsys.readouterr().out)["trace_directory"] == str(output)


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
