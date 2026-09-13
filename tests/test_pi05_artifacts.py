"""Native capture tests never construct a real arm, camera or CAN handle."""

import builtins
import json
import time
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit import pi05_artifacts as artifacts
from yamkit.inference.mapping import YAM_NAMES
from yamkit.pi05 import rollout
from yamkit.pi05.contract import PROFILE
from yamkit.rollout_artifacts import validate_bundle
from yamkit.ui import catalog


def observation():
    return {"state": np.zeros(14), **{name: np.zeros((480, 640, 3), dtype=np.uint8)
                                    for name in PROFILE.image_keys}}


def report():
    return {"status": "completed", "released": True, "hardware_tested": False,
            "started_at": time.time() - 1, "completed_at": time.time(),
            "execution": {"predicted_rows": 30, "completed_rows": 2, "uncompleted_rows": 28,
                          "interpolation_points": 0, "modified_commands": 0}}


def assert_outcome(directory, *, execution_status="completed", exit_status=0, error=None):
    summary = json.loads((directory / "summary.json").read_text())
    recorded = json.loads((directory / "report.json").read_text())
    meta = json.loads((directory / "meta.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())["native_pi05_rollout"]
    for value in (summary, recorded, meta, metrics):
        assert value["execution_status"] == execution_status
        assert value["exit_status"] == exit_status
        assert value["pipeline_complete"] is (exit_status == 0)
        assert value["postprocess_error"] == error
        assert value["upload_pending"] is False
    assert recorded["status"] == metrics["status"] == execution_status
    assert meta["returncode"] == exit_status
    assert meta["status"] == ("stopped" if execution_status == "stopped"
                               else "success" if exit_status == 0 else "failed")


@pytest.fixture
def bounded_tools(monkeypatch):
    actual = artifacts._trace_tools()
    calls = []

    class Progress:
        def report(self, phase, **kwargs):
            calls.append((phase, kwargs))

    tools = SimpleNamespace(FRAME_TRIPLET_BYTES=actual.FRAME_TRIPLET_BYTES,
                            MEMORY_HEADROOM_BYTES=actual.MEMORY_HEADROOM_BYTES,
                            MAX_EXPORT_WALL_S=240, available_memory_bytes=lambda: 10**12,
                            ExportProgress=Progress, wall_limit=lambda _seconds: nullcontext(),
                            video_timeline=actual.video_timeline,
                            encode_timestamped_video=actual.encode_timestamped_video)
    monkeypatch.setattr(artifacts, "_trace_tools", lambda: tools)
    return tools, calls


def capture_frames(*, count=2):
    now = [100.0]
    capture = artifacts.NativeCapture(task="put cube into bowl", duration_s=0.1,
                                      capture_trace=True, clock=lambda: now[0])
    capture.reserve()
    capture.start()
    sample = observation()
    for index in range(count):
        now[0] += 0.035
        sample["top"].fill(index + 1)
        capture.observation(sample)
    now[0] += 0.035
    capture.end()
    return capture, sample


def test_capture_preserves_images_not_references_and_never_extra_observations(bounded_tools):
    capture, sample = capture_frames()
    sample["top"].fill(99)
    assert np.all(capture.frame_pool[0, 0] == 1)
    assert np.all(capture.frame_pool[1, 0] == 2)
    assert capture.counts["observation_frames_seen"] == 2
    assert [event["observation_index"] for event in capture.events if event["kind"] == "video_sample"] == [0, 1]


def test_capture_admission_is_unchanged_90_second_threshold(bounded_tools):
    tools, _ = bounded_tools
    tools.available_memory_bytes = lambda: 8010125311
    capture = artifacts.NativeCapture(task="cube", duration_s=90, capture_trace=True)
    with pytest.raises(MemoryError, match="Insufficient"):
        capture.reserve()
    assert capture.frame_pool is None
    assert capture.memory_preflight["required_bytes"] == 8010125312


def test_trace_only_recording_never_reserves_frames(monkeypatch):
    monkeypatch.setattr(artifacts, "_trace_tools", lambda: pytest.fail("No RGB reservation requested"))
    capture = artifacts.NativeCapture(task="cube", duration_s=90)
    capture.reserve()
    assert capture.frame_pool is None


def test_instrumentation_overflow_and_bad_image_are_counted_not_control_faults(bounded_tools):
    capture, _ = capture_frames()
    capture.ended = None
    capture.safely(capture.observation, {**observation(), "top": np.zeros((1, 1, 3), dtype=np.uint8)})
    assert capture.counts["trace_errors"] == capture.counts["frames_dropped"] == 1
    capture.events = [{}] * 32768
    capture.safely(capture.event, "dispatch")
    assert capture.counts["events_dropped"] == 1


def test_native_raw_output_and_explicit_projection_remain_separate(bounded_tools):
    capture, _ = capture_frames()
    raw = np.zeros((30, 14))
    raw[0, 6] = 1.000895619392395
    projected = raw.copy()
    projected[0, 6] = 1
    capture.response(raw.tolist(), sequence_id=0, observation_index=0)
    capture.event("chunk_admitted", index=0, policy_observation_index=0, rows=projected.tolist(),
                  raw_rows=raw.tolist(), action_transform="explicit-gripper-projection", gripper_projections=[
                      {"row_index": 0, "column_index": 6, "raw": raw[0, 6], "executed": 1.0}])
    trace = capture._trace()
    assert trace["native_responses"][0]["raw_rows"][0][6] == raw[0, 6]
    assert trace["chunks"][0]["raw_actions"][0][6] == raw[0, 6]
    assert trace["chunks"][0]["actions"][0][6] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_invalid_native_numbers_are_explicit_json_null_not_serialization_failure(bounded_tools, bad):
    capture, _ = capture_frames()
    raw = np.zeros((30, 14))
    raw[0, 6] = bad
    capture.response(raw, sequence_id=0, observation_index=0)
    trace = capture._trace()
    assert trace["native_responses"][0]["raw_rows"][0][6] is None
    assert trace["native_responses"][0]["nonfinite_values"] == 1
    json.dumps(trace, allow_nan=False)


def test_release_failure_prevents_encoder_renderer_and_upload(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    tools, _ = bounded_tools
    tools.encode_timestamped_video = lambda *_: pytest.fail("Cannot encode before release")
    monkeypatch.setattr(artifacts, "_render_native_report", lambda *_: pytest.fail("Cannot render before release"))
    monkeypatch.setattr(artifacts, "upload_rollout", lambda *_a, **_k: pytest.fail("Cannot upload before release"))
    outcome = capture.finalize(tmp_path, {**report(), "released": False}, upload_repo_id="test/private")
    assert outcome["status"] == "EXPORT_SKIPPED_RESOURCES_OPEN"
    assert list(tmp_path.glob("*.mp4")) == []
    assert not (tmp_path / "frames").exists()
    assert json.loads((tmp_path / "report.json").read_text())["released"] is False
    assert_outcome(tmp_path, exit_status=1, error="release_not_confirmed")


def test_real_video_export_native_report_and_ui_catalog_playback(bounded_tools, tmp_path):
    capture, _ = capture_frames()
    directory = tmp_path / "native-test"
    directory.mkdir()
    summary = capture.finalize(directory, report())
    assert summary["status"] == "TRACE_SAVED"
    assert capture.frame_pool is None
    assert all((directory / (camera + ".mp4")).stat().st_size > 0 for camera in artifacts.CAMERAS)
    timeline = json.loads((directory / "video_timeline.json").read_text())
    assert timeline["frames"][0]["observation_index"] == 0
    assert timeline["frames"][1]["pts"] == 35000
    detail = catalog.deployment_detail(tmp_path, directory.name)
    assert detail["recording"]["state"] == "available"
    assert detail["recording"]["video_count"] == 3
    text = (directory / "report.html").read_text()
    assert "fake-arm software run" in text and "timeupdate" in text
    assert "cached command state" not in text
    assert_outcome(directory)


def test_explicit_native_trace_only_run_is_not_mislabeled_missing_recording(bounded_tools, tmp_path):
    capture = artifacts.NativeCapture(task="cube", duration_s=.1, capture_trace=False)
    directory = tmp_path / "without-video"
    directory.mkdir()
    capture.finalize(directory, report())
    detail = catalog.deployment_detail(tmp_path, directory.name)
    assert detail["recording"]["state"] == "not_recorded"
    assert detail["recording"]["requested"] is False
    summary = json.loads((directory / "summary.json").read_text())
    summary["controller_mode"] = "reference"
    (directory / "summary.json").write_text(json.dumps(summary))
    assert catalog.deployment_detail(tmp_path, directory.name)["recording"]["state"] == "unavailable"


def test_native_private_bundle_is_valid_sanitized_and_idempotent(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    directory = tmp_path / "native-bundle"
    directory.mkdir()
    capture.finalize(directory, report(), metadata={"token": "never-include-private-value", "source": "abc123"})
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_private_fixture_value")
    bundle = artifacts.package_native_rollout(directory)
    manifest = validate_bundle(bundle)
    assert manifest["frame_counts"] == dict.fromkeys(artifacts.CAMERAS, 2)
    assert artifacts.package_native_rollout(directory) == bundle
    assert "Native π0.5" in (bundle / "README.md").read_text()
    for path in bundle.glob("*.json"):
        assert "never-include-private-value" not in path.read_text()
        assert "hf_private_fixture_value" not in path.read_text()
    assert all((directory / "frames" / camera / "frame-000000.png").is_file() for camera in artifacts.CAMERAS)


def test_ui_native_bundle_joins_preserved_original_trace_frames(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    source, ui_run = tmp_path / "original-trace", tmp_path / "ui-run"
    source.mkdir()
    ui_run.mkdir()
    capture.finalize(source, report())
    for name in ("summary.json", "meta.json"):
        (ui_run / name).write_bytes((source / name).read_bytes())
    meta = json.loads((ui_run / "meta.json").read_text())
    meta["id"] = ui_run.name
    meta.pop("active")  # DeploymentLog's finalized UI schema has no active field.
    (ui_run / "meta.json").write_text(json.dumps(meta))
    bundle = artifacts.package_native_rollout(ui_run, trace_dir=source)
    manifest = validate_bundle(bundle)
    assert manifest["run_id"] == ui_run.name
    assert manifest["frame_counts"] == dict.fromkeys(artifacts.CAMERAS, 2)
    archived = json.loads((bundle / "meta.json").read_text())
    assert archived["returncode"] == 0 and archived["pipeline_complete"]
    assert archived["execution_status"] == "completed"
    assert (source / "frames/top/frame-000001.png").is_file()
    assert not (ui_run / "frames").exists()


def test_symlink_destination_and_media_are_rejected(bounded_tools, tmp_path):
    directory = tmp_path / "real"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    capture, _ = capture_frames()
    with pytest.raises(ValueError, match="symlink"):
        capture.finalize(alias, report())
    (directory / "summary.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="symlink"):
        capture.finalize(directory, report())


def test_upload_only_after_release_and_export_with_no_retry(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    called = []

    def upload(bundle, *, repo_id):
        called.append(repo_id)
        summary = json.loads((bundle.parent / "summary.json").read_text())
        assert summary["resources_released"] is True
        assert summary["status"] == "TRACE_SAVED"
        meta = json.loads((bundle / "meta.json").read_text())
        assert meta["returncode"] == 0 and meta["upload_pending"]
        assert meta["pipeline_complete"] is False
        assert meta["resources_released"] is True
        raise RuntimeError("Bearer do-not-show-this-private-value")

    monkeypatch.setattr(artifacts, "upload_rollout", upload)
    result = capture.finalize(tmp_path, report(), upload_repo_id="test/private")
    assert called == ["test/private"]
    assert result["upload"]["status"] == "failed"
    status = json.loads((tmp_path / "hf-upload.json").read_text())
    assert status["status"] == "failed"
    assert "do-not-show" not in json.dumps(status)
    assert result["status"] == "TRACE_SAVED"  # Valid local media remains replayable.
    assert_outcome(tmp_path, exit_status=1, error="upload_failed")
    assert json.loads((tmp_path / "report.json").read_text())["released"] is True


@pytest.mark.parametrize("returned", [{"status": "failed"}, {"status": "pending"}, False, None])
def test_upload_unconfirmed_return_is_pipeline_failure_not_success(bounded_tools, monkeypatch, tmp_path, returned):
    capture, _ = capture_frames()
    called = []
    def upload(_bundle, *, repo_id):
        called.append(repo_id)
        return returned
    monkeypatch.setattr(artifacts, "upload_rollout", upload)
    result = capture.finalize(tmp_path, report(), upload_repo_id="test/private")
    assert called == ["test/private"] and result["status"] == "TRACE_SAVED"
    assert result["upload"]["status"] == "failed"
    assert_outcome(tmp_path, exit_status=1, error="upload_failed")


@pytest.mark.parametrize("status", ["uploaded", "already_uploaded"])
def test_confirmed_upload_finalizes_success_without_mutating_archive(bounded_tools, monkeypatch, tmp_path, status):
    capture, _ = capture_frames()
    monkeypatch.setattr(artifacts, "upload_rollout", lambda *_a, **_k: {"status": status, "repo_id": "test/private"})
    capture.finalize(tmp_path, report(), upload_repo_id="test/private")
    assert_outcome(tmp_path)
    assert json.loads((tmp_path / "bundle/meta.json").read_text())["upload_pending"] is True


@pytest.mark.parametrize("status", ["stopped", "failed"])
def test_saved_recording_retains_stop_or_control_failure_identity(bounded_tools, tmp_path, status):
    capture = artifacts.NativeCapture(task="cube", duration_s=.1)
    result = capture.finalize(tmp_path, {**report(), "status": status})
    assert result["status"] == "TRACE_SAVED" and result["resources_released"]
    assert_outcome(tmp_path, execution_status=status, exit_status=130 if status == "stopped" else 1)


def test_render_failure_updates_history_without_relabeling_completed_execution(bounded_tools, monkeypatch, tmp_path):
    capture = artifacts.NativeCapture(task="cube", duration_s=.1)
    monkeypatch.setattr(artifacts, "_render_native_report", lambda *_a: (_ for _ in ()).throw(RuntimeError("renderer")))
    outcome = capture.finalize(tmp_path, report())
    assert outcome["status"] == "EXPORT_FAILED"
    assert_outcome(tmp_path, exit_status=1, error="recording_export_failed")


def test_export_failure_retains_trace_and_prevents_upload(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    tools, _ = bounded_tools
    tools.encode_timestamped_video = lambda *_: (_ for _ in ()).throw(RuntimeError("codec failed"))
    monkeypatch.setattr(artifacts, "upload_rollout", lambda *_a, **_k: pytest.fail("Incomplete export"))
    result = capture.finalize(tmp_path, report(), upload_repo_id="test/private")
    assert result["status"] == "TRACE_SAVED_WITH_EXPORT_ERRORS"
    assert len(result["video_export_errors"]) == 3
    assert (tmp_path / "trace.json").is_file()
    assert (tmp_path / "frames/top/frame-000000.png").is_file()
    assert_outcome(tmp_path, exit_status=1, error="recording_export_failed")


def test_fake_lifecycle_cannot_import_real_robot_and_exports_after_release(bounded_tools, monkeypatch, tmp_path):
    monkeypatch.setattr(rollout, "validate_qualification", lambda *_a, **_k: None)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("yamkit.arm", "lerobot_robot_yamkit", "i2rt", "can", "pyrealsense2")):
            pytest.fail("Fake native rollout attempted a hardware import: " + name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    events, stop = [], Event()

    class Transport:
        def ready(self, _timeout):
            return {"instance_id": "fake", "http_session_expires_at": time.time() + 1000}

        def ensure_session_active(self):
            return None

        def predict_chunk(self, _request, _timeout):
            return {"chunk": np.zeros((30, 14)).tolist()}

        def cancel(self):
            return None

        def close(self):
            events.append("transport_closed")

    class Robot:
        def connect(self):
            events.append("connect")

        def get_observation(self):
            return {**dict.fromkeys(YAM_NAMES, 0.0), **{name: observation()[name] for name in PROFILE.image_keys}}

        def validate_action_target(self, _target):
            return None

        def send_reference_action(self, target, *, dispatch_check):
            dispatch_check()
            events.append("send")
            stop.set()
            return target

        def disconnect(self, *, home):
            assert home is False
            events.append("released")

    actual_finalize = artifacts.NativeCapture.finalize

    def finalize(self, *args, **kwargs):
        assert "released" in events
        events.append("export")
        return actual_finalize(self, *args, **kwargs)

    monkeypatch.setattr(artifacts.NativeCapture, "finalize", finalize)
    result = rollout.run_rollout(Transport(), task="cube", duration_s=.1, rig_path=Path("unused"),
                                qualification={}, accept_mapping=True, confirm_supervised=True,
                                artifact_dir=tmp_path / "run", shutdown_event=stop,
                                robot_factory=lambda *_: Robot(), home=lambda *_: pytest.fail("No home after Stop"),
                                capture_trace=True)
    assert result["hardware_tested"] is False and result["status"] == "stopped"
    assert result["artifact_status"] == "TRACE_SAVED"
    assert result["exit_status"] == 130 and result["pipeline_complete"] is False
    assert result["execution_status"] == "stopped" and result["postprocess_error"] is None
    assert_outcome(tmp_path / "run", execution_status="stopped", exit_status=130)
    assert events.index("released") < events.index("export")


@pytest.mark.parametrize("upload_failure", [False, True])
def test_public_fake_rollout_reports_pipeline_outcome_after_release(bounded_tools, monkeypatch, tmp_path, upload_failure):
    monkeypatch.setattr(rollout, "validate_qualification", lambda *_a, **_k: None)
    events = []

    class Transport:
        def ready(self, _timeout):
            return {"instance_id": "fake", "http_session_expires_at": time.time() + 1000}

        def ensure_session_active(self):
            return None

        def predict_chunk(self, _request, _timeout):
            return {"chunk": np.zeros((30, 14)).tolist()}

        def cancel(self):
            return None

        def close(self):
            events.append("transport_closed")

    class Robot:
        def connect(self):
            events.append("connect")

        def get_observation(self):
            return {**dict.fromkeys(YAM_NAMES, 0.0), **{name: observation()[name] for name in PROFILE.image_keys}}

        def validate_action_target(self, _target):
            return None

        def send_reference_action(self, target, *, dispatch_check):
            dispatch_check()
            return target

        def disconnect(self, *, home):
            assert home is False
            events.append("released")

    def upload(_bundle, *, repo_id):
        assert "released" in events and "transport_closed" in events
        events.append("upload")
        return {"status": "failed" if upload_failure else "uploaded", "repo_id": repo_id}

    monkeypatch.setattr(artifacts, "upload_rollout", upload)
    kwargs = {"task": "cube", "duration_s": .1, "rig_path": Path("unused"), "qualification": {},
              "accept_mapping": True, "confirm_supervised": True, "artifact_dir": tmp_path / "run",
              "robot_factory": lambda *_: Robot(), "home": lambda *_: events.append("home"),
              "capture_trace": True, "upload_repo_id": "test/private"}
    if upload_failure:
        with pytest.raises(RuntimeError, match="retained report"):
            rollout.run_rollout(Transport(), **kwargs)
        assert_outcome(tmp_path / "run", exit_status=1, error="upload_failed")
    else:
        result = rollout.run_rollout(Transport(), **kwargs)
        assert result["status"] == result["execution_status"] == "completed"
        assert result["released"] and result["pipeline_complete"] and result["exit_status"] == 0
        assert result["artifact_status"] == "TRACE_SAVED" and result["postprocess_error"] is None
        assert_outcome(tmp_path / "run")
    assert events.count("connect") == events.count("home") == events.count("released") == events.count("upload") == 1
