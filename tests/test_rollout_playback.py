"""Read-only replay catalog and byte serving, with no camera/robot or Hub access."""

import json

import pytest

from tests.test_inference_ui import inference_ui as _inference_ui
from yamkit.ui import catalog

inference_ui = _inference_ui
CAMERAS = ("top", "left_wrist", "right_wrist")


def saved_run(root, *, capture=True, status="success"):
    run = root / "saved-rollout"
    run.mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps({
        "id": run.name, "kind": "rollout", "status": status,
        "task": "put the red cube into the green bowl", "returncode": 0 if status == "success" else None,
    }))
    (run / "run_metadata.json").write_text(json.dumps({"capture": {"requested": capture}}))
    return run


def saved_videos(run, cameras=CAMERAS):
    for camera in cameras:
        (run / f"{camera}.mp4").write_bytes(b"synthetic-byte-serving-fixture")
    (run / "summary.json").write_text(json.dumps({
        "status": "TRACE_SAVED", "frame_count": 150, "resources_released": True,
        "camera_names": list(CAMERAS), "video_export_errors": {},
    }))
    (run / "video_timeline.json").write_text(json.dumps({
        "duration_s": 4.3, "policy_phase_offset_s": .7,
        "origin_monotonic_s": 100.7, "timestamp_basis": "Original observation receipt; not camera exposure",
    }))


def test_recording_catalog_reports_requested_recording_while_capture_or_export_pending(tmp_path):
    run = saved_run(tmp_path, status="running")
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert detail["recording"]["state"] == "pending"
    assert detail["recording"]["video_count"] == 0
    assert detail["recording"]["missing_cameras"] == list(CAMERAS)
    saved_videos(run)
    assert catalog.deployment_detail(tmp_path, run.name)["recording"]["state"] == "pending"
    meta = json.loads((run / "meta.json").read_text())
    (run / "meta.json").write_text(json.dumps({**meta, "status": "success"}))
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert detail["recording"]["state"] == "available"
    assert detail["recording"]["video_count"] == detail["recording"]["expected_video_count"] == 3
    assert detail["recording"]["missing_cameras"] == []
    assert detail["videos"] == [f"{camera}.mp4" for camera in CAMERAS]
    assert detail["recording"]["frame_count"] == 150
    assert detail["recording"]["duration_s"] == 4.3
    assert detail["recording"]["policy_phase_offset_s"] == .7
    assert detail["recording"]["resources_released"] is True
    assert "exposure" in detail["recording"]["timestamp_basis"]
    assert catalog.list_deployments(tmp_path)[0]["recording"]["state"] == "available"


@pytest.mark.parametrize("capture,state", [(False, "not_recorded"), (True, "unavailable")])
def test_successful_exit_does_not_claim_a_missing_recording_succeeded(tmp_path, capture, state):
    run = saved_run(tmp_path, capture=capture)
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert detail["status"] == "success"
    assert detail["recording"]["state"] == state
    assert detail["recording"]["requested"] is capture
    assert detail["videos"] == []


def test_historical_recording_inferred_without_following_original_paths(tmp_path):
    run = saved_run(tmp_path, capture=None)
    (run / "run_metadata.json").write_text(json.dumps({
        "original_paths": {"trace_dir": "/not/a/path/the/catalog/may/open"},
    }))
    assert catalog.deployment_detail(tmp_path, run.name)["recording"]["state"] == "unavailable"
    saved_videos(run)
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert detail["recording"]["requested"] is True
    assert detail["recording"]["state"] == "available"
    assert "/not/a/path" not in json.dumps(detail)


def test_partial_video_export_and_upload_state_are_independent(tmp_path):
    run = saved_run(tmp_path)
    saved_videos(run, cameras=("top",))
    (run / "hf-upload.json").write_text('{"status":"uploaded"}')
    summary = json.loads((run / "summary.json").read_text())
    summary.update(status="TRACE_SAVED_WITH_EXPORT_ERRORS", video_export_errors={
        "left_wrist": "EncoderError SECRET_MUST_NOT_APPEAR",
    })
    (run / "summary.json").write_text(json.dumps(summary))
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert detail["recording"]["state"] == "partial"
    assert detail["recording"]["missing_cameras"] == ["left_wrist", "right_wrist"]
    assert detail["recording"]["errors"] == ["One or more camera videos did not export."]
    assert "SECRET_MUST_NOT_APPEAR" not in json.dumps(detail)


@pytest.mark.parametrize("defect", ["export_error", "trace_error", "overflow", "render_error"])
def test_retained_videos_do_not_hide_other_export_or_capture_errors(tmp_path, defect):
    run = saved_run(tmp_path)
    saved_videos(run)
    summary = json.loads((run / "summary.json").read_text())
    if defect == "export_error":
        (run / "export-error.json").write_text('{"error_type":"TimeoutError"}')
    elif defect == "trace_error":
        summary["trace_error_types"] = ["ValueError"]
    elif defect == "overflow":
        summary["overflow"] = True
    else:
        summary["render_error_type"] = "ValueError"
    (run / "summary.json").write_text(json.dumps(summary))
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert len(detail["videos"]) == 3
    assert detail["recording"]["state"] == "partial"
    assert detail["recording"]["errors"]


def test_invalid_or_missing_timing_does_not_invent_playback_timestamps(tmp_path):
    run = saved_run(tmp_path)
    saved_videos(run)
    (run / "video_timeline.json").write_text('{"duration_s":Infinity,"policy_phase_offset_s":-1}')
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert detail["recording"]["duration_s"] is None
    assert detail["recording"]["policy_phase_offset_s"] is None


def test_catalog_ignores_symlinked_artifacts_and_run_aliases(tmp_path):
    run = saved_run(tmp_path)
    secret = tmp_path / "not-a-run-secret"
    secret.write_text('SECRET_MUST_NOT_APPEAR')
    (run / "top.mp4").symlink_to(secret)
    (run / "left_wrist.mp4").write_bytes(b"")
    for name in ("log.txt", "summary.json", "video_timeline.json"):
        (run / name).symlink_to(secret)
    (tmp_path / "run-alias").symlink_to(run, target_is_directory=True)
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert detail["videos"] == [] and detail["log"] == []
    assert "SECRET_MUST_NOT_APPEAR" not in json.dumps(detail)
    assert catalog.deployment_detail(tmp_path, "run-alias") is None
    assert catalog.deployment_detail(tmp_path, "../not-a-run-secret") is None
    assert [item["id"] for item in catalog.list_deployments(tmp_path)] == [run.name]


@pytest.mark.parametrize("contents", ['[]', 'null', '"not a record"', 'broken json'])
def test_malformed_record_does_not_break_history(tmp_path, contents):
    run = saved_run(tmp_path)
    (run / "meta.json").write_text(contents)
    assert catalog.list_deployments(tmp_path) == []
    assert catalog.deployment_detail(tmp_path, run.name) is None


def test_detail_log_read_is_bounded_and_keeps_latest_lines(tmp_path):
    run = saved_run(tmp_path)
    (run / "log.txt").write_text("old log\n" * 50_000 + "the final line\n")
    detail = catalog.deployment_detail(tmp_path, run.name)
    assert len(detail["log"]) == 300
    assert detail["log"][-1] == "the final line"


def test_replay_routes_support_byte_ranges_without_opening_any_hardware(inference_ui):
    ui = inference_ui
    run = saved_run(ui.root / "outputs" / "ui" / "deployments")
    saved_videos(run)
    detail = ui.client.get(f"/api/deployments/{run.name}").json()
    assert detail["recording"]["state"] == "available"
    for video in detail["videos"]:
        response = ui.client.get(f"/api/deployments/{run.name}/video/{video}", headers={"Range": "bytes=0-9"})
        assert response.status_code == 206
        assert response.content == b"synthetic-"
        assert response.headers["content-type"] == "video/mp4"
        assert response.headers["content-range"].startswith("bytes 0-9/")
    assert not ui.seen and not ui.manager.active and not ui.manager.cameras_owned


def test_replay_routes_reject_symlink_and_cross_run_aliases(inference_ui):
    ui = inference_ui
    root = ui.root / "outputs" / "ui" / "deployments"
    run = saved_run(root)
    saved_videos(run)
    other = root / "other-run"
    other.mkdir()
    (other / "private.mp4").write_bytes(b"PRIVATE_OTHER_RUN")
    (run / "aliased.mp4").symlink_to(other / "private.mp4")
    (root / "run-alias").symlink_to(run, target_is_directory=True)
    assert ui.client.get(f"/api/deployments/{run.name}/video/aliased.mp4").status_code == 404
    assert ui.client.get("/api/deployments/run-alias/video/top.mp4").status_code == 404
    assert ui.client.get("/api/deployments/run-alias/artifact/summary.json").status_code == 404
    assert ui.client.get("/api/deployments/run-alias").status_code == 404
    assert ui.client.get(f"/api/deployments/{run.name}/video/meta.json").status_code == 404
