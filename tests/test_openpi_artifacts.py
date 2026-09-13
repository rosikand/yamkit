"""Official OpenPI fake-recording tests; never construct physical devices."""

import json
import time
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit.openpi import artifacts
from yamkit.openpi.interface import CONTRACT_ID
from yamkit.rollout_artifacts import validate_bundle


def observation():
    return {"state": np.zeros(14), **{name: np.zeros((480, 640, 3), dtype=np.uint8)
                                    for name in artifacts.CAMERAS}}


def report():
    return {"status": "completed", "released": True, "hardware_tested": False,
            "started_at": time.time() - 1, "completed_at": time.time(),
            "execution": {"predicted_rows": 50, "completed_rows": 25,
                          "intended_unused_rows": 25, "modified_commands": 0,
                          "endpoint_interval_s": {"sample_count": 24, "p50": .02, "p95": .02, "max": .02}}}


def assert_outcome(directory, *, status="completed", code=0, error=None):
    summary = json.loads((directory / "summary.json").read_text())
    recorded = json.loads((directory / "report.json").read_text())
    meta = json.loads((directory / "meta.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())["openpi_rollout"]
    for value in (summary, recorded, meta, metrics):
        assert value["execution_status"] == status
        assert value["exit_status"] == code
        assert value["pipeline_complete"] is (code == 0)
        assert value["postprocess_error"] == error
        assert value["upload_pending"] is False
    assert meta["returncode"] == code
    assert summary["controller_mode"] == CONTRACT_ID
    assert summary["task_success"] is None
    assert meta["hardware_tested"] is False


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


def capture_frames():
    now = [100.0]
    capture = artifacts.OpenPiCapture(task="put the red cube into the black container", duration_s=.1,
                                      capture_trace=True, clock=lambda: now[0])
    capture.reserve()
    capture.start()
    sample = observation()
    for index in range(2):
        now[0] += .02
        sample["top"].fill(index + 1)
        capture.observation(sample, policy_input=True)
    now[0] += .02
    capture.end()
    return capture, sample


def test_capture_retains_original_rgb_and_observation_join(bounded_tools):
    capture, sample = capture_frames()
    sample["top"].fill(99)
    assert np.all(capture.frame_pool[0, 0] == 1)
    assert np.all(capture.frame_pool[1, 0] == 2)
    assert [event["observation_index"] for event in capture.events if event["kind"] == "video_sample"] == [0, 1]


def test_base60_memory_admission_and_legacy_pool_unchanged(bounded_tools):
    from yamkit.pi05_artifacts import NativeCapture

    tools, _ = bounded_tools
    tools.available_memory_bytes = lambda: 5859110911
    capture = artifacts.OpenPiCapture(task="cube", duration_s=60, capture_trace=True)
    with pytest.raises(MemoryError, match="Insufficient"):
        capture.reserve()
    assert capture.frame_pool is None
    assert capture.frame_capacity == 1925
    assert capture.memory_preflight["required_bytes"] == 5859110912
    assert NativeCapture(task="cube", duration_s=90).frame_capacity == 2703


def test_maximum_base_frame_timeline_retains_1925_original_timestamps():
    from yamkit.pi05_artifacts import _trace_tools as legacy_tools

    tools = artifacts._trace_tools()
    times = np.linspace(100, 159.99, 1925).tolist()
    timeline = tools.video_timeline(times, 100., 160.)
    assert len(timeline["frames"]) == 1925
    assert timeline["nominal_fps"] == 30
    assert timeline["frames"][-1]["observation_index"] == 1924
    assert legacy_tools().VIDEO_FPS == 30
    with pytest.raises(ValueError):
        tools.video_timeline(np.linspace(100, 159.99, 1926).tolist(), 100., 160.)


@pytest.mark.parametrize("duration", [0, -1, 60.1, 90, float("nan"), True])
def test_base_capture_rejects_invalid_duration(duration):
    with pytest.raises(ValueError):
        artifacts.OpenPiCapture(task="cube", duration_s=duration)


def test_trace_only_does_not_reserve_rgb(monkeypatch):
    monkeypatch.setattr(artifacts, "_trace_tools", lambda: pytest.fail("Unexpected reservation"))
    capture = artifacts.OpenPiCapture(task="cube", duration_s=60)
    capture.reserve()
    assert capture.frame_pool is None


def test_raw_native32_decoded14_and_actuator_requests_remain_separate(bounded_tools):
    capture, _ = capture_frames()
    normalized = np.arange(1600, dtype=np.float32).reshape(50, 32)
    raw = np.zeros((50, 14)); raw[0, 6] = 1.03
    projected = raw.copy(); projected[0, 6] = 1
    result = {"raw_normalized_chunk": normalized, "chunk": raw, "audit": {"quantile_sha256": "abc"}}
    capture.response(result, sequence_id=3, observation_index=1)
    capture.event("chunk_admitted", chunk_index=0, policy_observation_index=1,
                  predicted_rows=50, committed_prefix_rows=25, rows=projected.tolist(), raw_rows=raw.tolist(),
                  action_transform={"id": "explicit-mechanical-opening"}, gripper_conversions=[
                      {"row_index": 0, "column_index": 6, "raw": 1.03, "executed": 1.0}])
    result["chunk"][0, 6] = 9
    result["raw_normalized_chunk"][0, 0] = -3
    trace = capture._trace()
    native = trace["native_responses"][0]
    assert native["raw_normalized_rows"][0][0] == 0
    assert native["raw_normalized_rows"][-1][-1] == 1599
    assert native["decoded_requested_rows"][0][6] == 1.03
    assert native["policy_input_marked"] is True and native["rgb_frame_index"] == 1
    assert trace["chunks"][0]["raw_actions"][0][6] == 1.03
    assert trace["chunks"][0]["actions"][0][6] == 1
    assert trace["chunks"][0]["gripper_conversions"][0]["raw"] == 1.03


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_reply_preserved_as_explicit_null(bounded_tools, bad):
    capture, _ = capture_frames()
    normalized = np.zeros((50, 32)); normalized[0, 1] = bad
    decoded = np.zeros((50, 14)); decoded[1, 6] = bad
    capture.response({"raw_normalized_chunk": normalized, "chunk": decoded}, sequence_id=0, observation_index=0)
    native = capture._trace()["native_responses"][0]
    assert native["raw_normalized_rows"][0][1] is None
    assert native["decoded_requested_rows"][1][6] is None
    assert native["raw_normalized_rows_nonfinite_values"] == native["decoded_requested_rows_nonfinite_values"] == 1
    json.dumps(capture._trace(), allow_nan=False)


def test_secret_filter_keeps_only_typed_actuator_proof():
    result = artifacts.sanitize({"endpoint": "https://private.invalid", "endpoint_completed": True,
        "nested": {"endpoint": True, "endpoint_interval_s": {"sample_count": 4, "p50": .02}},
        "danger": {"endpoint_interval_s": {"password": "secret-value"}}, "token": "hf_private_token"})
    assert "endpoint" not in result and "token" not in result
    assert result["endpoint_completed"] is True and result["nested"]["endpoint"] is True
    assert result["nested"]["endpoint_interval_s"]["p50"] == .02
    assert "endpoint_interval_s" not in result["danger"]


def test_release_failure_prevents_media_render_and_upload(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    tools, _ = bounded_tools
    tools.encode_timestamped_video = lambda *_: pytest.fail("Encoding before release")
    monkeypatch.setattr(artifacts, "_render_openpi_report", lambda *_: pytest.fail("Rendering before release"))
    monkeypatch.setattr(artifacts, "upload_rollout", lambda *_a, **_k: pytest.fail("Upload before release"))
    result = capture.finalize(tmp_path, {**report(), "released": False}, upload_repo_id="test/private")
    assert result["status"] == "EXPORT_SKIPPED_RESOURCES_OPEN"
    assert not (tmp_path / "frames").exists()
    assert_outcome(tmp_path, code=1, error="release_not_confirmed")


def test_real_three_video_export_preserves_timestamps_and_label(bounded_tools, tmp_path):
    capture, _ = capture_frames()
    result = capture.finalize(tmp_path, report())
    assert result["status"] == "TRACE_SAVED"
    assert capture.frame_pool is None
    assert all((tmp_path / (camera + ".mp4")).stat().st_size > 0 for camera in artifacts.CAMERAS)
    timeline = json.loads((tmp_path / "video_timeline.json").read_text())
    assert timeline["frames"][1]["pts"] == 20000 and timeline["nominal_fps"] == 30
    text = (tmp_path / "report.html").read_text()
    assert "fake-arm software run" in text and "timeupdate" in text
    assert "25 endpoints" in text and "experimental YAM" in text
    assert_outcome(tmp_path)


def test_private_bundle_is_sanitized_hash_checked_and_preserves_originals(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    capture.finalize(tmp_path, report(), metadata={"token": "never-include-fixture-private-value"})
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "get_token", lambda: "hf_private_fixture_value")
    bundle = artifacts.package_openpi_rollout(tmp_path)
    manifest = validate_bundle(bundle)
    assert manifest["frame_counts"] == dict.fromkeys(artifacts.CAMERAS, 2)
    assert artifacts.package_openpi_rollout(tmp_path) == bundle
    for path in bundle.glob("*.json"):
        assert "never-include-fixture-private-value" not in path.read_text()
        assert "hf_private_fixture_value" not in path.read_text()
    assert all((tmp_path / "frames" / camera / "frame-000000.png").is_file() for camera in artifacts.CAMERAS)


@pytest.mark.parametrize("status", ["uploaded", "already_uploaded", "failed", "pending"])
def test_upload_confirms_only_after_release_and_finalization(bounded_tools, monkeypatch, tmp_path, status):
    capture, _ = capture_frames()
    calls = []

    def upload(bundle, *, repo_id):
        calls.append(repo_id)
        meta = json.loads((bundle / "meta.json").read_text())
        assert meta["resources_released"] and meta["upload_pending"] and not meta["pipeline_complete"]
        return {"status": status, "repo_id": repo_id}

    monkeypatch.setattr(artifacts, "upload_rollout", upload)
    capture.finalize(tmp_path, report(), upload_repo_id="test/private")
    assert calls == ["test/private"]
    passed = status in ("uploaded", "already_uploaded")
    assert_outcome(tmp_path, code=0 if passed else 1, error=None if passed else "upload_failed")


def test_upload_exception_is_sanitized_and_never_retried(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    calls = []

    def upload(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("Bearer not-for-logs-private-value")

    monkeypatch.setattr(artifacts, "upload_rollout", upload)
    capture.finalize(tmp_path, report(), upload_repo_id="test/private")
    assert calls == [1]
    assert "not-for-logs" not in (tmp_path / "hf-upload.json").read_text()
    assert_outcome(tmp_path, code=1, error="upload_failed")


def test_export_failure_retains_originals_and_prevents_upload(bounded_tools, monkeypatch, tmp_path):
    capture, _ = capture_frames()
    tools, _ = bounded_tools
    tools.encode_timestamped_video = lambda *_: (_ for _ in ()).throw(RuntimeError("codec failed"))
    monkeypatch.setattr(artifacts, "upload_rollout", lambda *_a, **_k: pytest.fail("Incomplete export"))
    result = capture.finalize(tmp_path, report(), upload_repo_id="test/private")
    assert result["status"] == "TRACE_SAVED_WITH_EXPORT_ERRORS"
    assert len(result["video_export_errors"]) == 3
    assert (tmp_path / "trace.json").is_file() and (tmp_path / "frames/top/frame-000000.png").is_file()
    assert_outcome(tmp_path, code=1, error="recording_export_failed")


@pytest.mark.parametrize("status", ["stopped", "failed"])
def test_saved_fault_or_stop_keeps_execution_identity(bounded_tools, tmp_path, status):
    capture = artifacts.OpenPiCapture(task="cube", duration_s=.1)
    capture.finalize(tmp_path, {**report(), "status": status})
    assert_outcome(tmp_path, status=status, code=130 if status == "stopped" else 1)


def test_bad_image_and_capacity_overflow_cannot_silently_pass(bounded_tools):
    capture, _ = capture_frames()
    capture.ended = None
    capture.safely(capture.observation, {**observation(), "top": np.zeros((1, 1, 3), dtype=np.uint8)}, policy_input=True)
    assert capture.counts["trace_errors"] == capture.counts["frames_dropped"] == 1
    capture.events = [{}] * 32768
    capture.safely(capture.event, "dispatch")
    assert capture.counts["events_dropped"] == 1


def test_linked_destination_rejected(bounded_tools, tmp_path):
    actual = tmp_path / "actual"; actual.mkdir()
    linked = tmp_path / "linked"; linked.symlink_to(actual, target_is_directory=True)
    capture = artifacts.OpenPiCapture(task="cube", duration_s=.1)
    with pytest.raises(ValueError, match="symlink"):
        capture.finalize(linked, report())


def test_50hz_observations_have_30hz_rgb_bins_without_losing_state(bounded_tools):
    now = [0.]
    capture = artifacts.OpenPiCapture(task="cube", duration_s=1, capture_trace=True, clock=lambda: now[0])
    capture.reserve()
    capture.start()
    sample = observation()
    for index in range(50):
        now[0] = index / 50
        sample["state"][0] = index / 100
        capture.observation(sample)
    assert capture.counts["observation_frames_seen"] == 50
    assert len(capture.frames) == capture.counts["cadence_rgb_frames"] == 30
    assert capture.counts["skipped_rgb_observations"] == 20
    assert capture.counts["frames_dropped"] == capture.counts["trace_errors"] == 0
    observed = [event for event in capture.events if event["kind"] == "observation"]
    assert [event["observation_index"] for event in observed] == list(range(50))
    assert [event["positions"][0] for event in observed] == [i / 100 for i in range(50)]
    assert [event["monotonic_s"] for event in observed] == [i / 50 for i in range(50)]


def test_off_cadence_policy_input_rgb_is_always_exact_and_joined(bounded_tools):
    now = [0.]
    capture = artifacts.OpenPiCapture(task="cube", duration_s=.1, capture_trace=True, clock=lambda: now[0])
    capture.reserve()
    capture.start()
    sample = observation()
    capture.observation(sample)  # Cadence bin zero.
    now[0] = .01
    sample["top"].fill(7)
    capture.observation(sample)  # Deliberately omitted non-input RGB; state kept.
    now[0] = .02
    sample["top"].fill(19)
    capture.observation(sample, policy_input=True)  # Same bin, mandatory exact image.
    response = {"raw_normalized_chunk": np.zeros((50, 32)), "chunk": np.zeros((50, 14))}
    capture.response(response, sequence_id=0, observation_index=2)
    sample["top"].fill(99)
    assert np.all(capture.frame_pool[1, 0] == 19)
    assert capture.counts["skipped_rgb_observations"] == 1
    assert capture.counts["policy_input_observations"] == capture.counts["policy_input_rgb_frames"] == 1
    assert capture.counts["extra_policy_input_rgb_frames"] == 1
    native = capture._trace()["native_responses"][0]
    assert native["observation_index"] == 2 and native["rgb_frame_index"] == 1
    assert capture.frames[1] == (.02, 2)


def test_missing_policy_input_join_is_trace_error_not_silent_rgb_omission(bounded_tools):
    now = [0.]
    capture = artifacts.OpenPiCapture(task="cube", duration_s=.1, capture_trace=True, clock=lambda: now[0])
    capture.reserve()
    capture.start()
    capture.observation(observation())
    now[0] = .01
    capture.observation(observation())
    response = {"raw_normalized_chunk": np.zeros((50, 32)), "chunk": np.zeros((50, 14))}
    capture.safely(capture.response, response, sequence_id=0, observation_index=1)
    assert capture.counts["trace_errors"] == 1
    assert capture._trace()["native_responses"][0]["raw_normalized_rows"] is not None
    assert capture._trace()["native_responses"][0]["rgb_frame_index"] is None


def test_rgb_skips_are_disclosed_but_not_export_failure(bounded_tools, tmp_path):
    now = [0.]
    capture = artifacts.OpenPiCapture(task="cube", duration_s=.1, capture_trace=True, clock=lambda: now[0])
    capture.reserve()
    capture.start()
    capture.observation(observation(), policy_input=True)
    now[0] = .01
    capture.observation(observation())
    now[0] = .05
    capture.end()
    result = capture.finalize(tmp_path, report())
    assert result["status"] == "TRACE_SAVED" and result["exit_status"] == 0
    assert result["counts"]["skipped_rgb_observations"] == 1
    assert result["counts"]["frames_dropped"] == 0
    assert result["rgb_sampling"]["mandatory_policy_inputs"] is True
    assert "30 Hz" in result["capture_scope"]
    assert "Omitted non-input RGB ticks" in (tmp_path / "report.html").read_text()
