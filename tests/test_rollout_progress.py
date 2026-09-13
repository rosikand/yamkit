"""Display-only rollout clocks and post-release progress; fake processes/cloud only."""

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.test_ui_rollout_upload import attached_modal as _attached_modal
from tests.test_ui_rollout_upload import inference_ui as _inference_ui
from tests.test_ui_rollout_upload import launch
from tests.test_ui_rollout_upload import upload_trial as _upload_trial
from yamkit import rollout_artifacts
from yamkit.ui import server
from yamkit.ui.sessions import SessionManager, _end_rollout_clock, parse_line, rollout_progress

inference_ui = _inference_ui
attached_modal = _attached_modal
upload_trial = _upload_trial


def status(parsed, **changes):
    return {"mode": "rollout", "active": True, "parsed": parsed,
            "meta": {"operation_id": "fixture", "duration": 20, "capture_trace": True}, **changes}


def test_policy_clock_starts_after_preparation_uses_monotonic_and_freezes_on_home(monkeypatch):
    clock = [10.0]
    wall = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(time, "time", lambda: wall[0])
    parsed = {"rollout_phase_since_monotonic": 5.0}
    preparation = rollout_progress(status(parsed))
    assert preparation["phase"] == "preparing" and preparation["phase_elapsed_s"] == 5.0
    assert preparation["policy_elapsed_s"] is None and preparation["policy_remaining_s"] is None
    parse_line("INFO [yamkit-operator] homing", parsed)
    assert rollout_progress(status(parsed))["phase"] == "homing"
    clock[0] = 12.0
    parse_line("INFO [yamkit-rollout] running", parsed)
    clock[0], wall[0] = 15.0, -9000.0  # Wall-clock correction cannot affect the timer.
    parse_line("INFO [yamkit-rollout] running", parsed)
    running = rollout_progress(status(parsed))
    assert running["policy_elapsed_s"] == 3.0 and running["policy_remaining_s"] == 17.0
    assert running["phase_elapsed_s"] == 3.0 and running["policy_timer_running"]
    clock[0] = 18.0
    parse_line("INFO [yamkit-rollout] returning_home", parsed)
    clock[0] = 25.0
    parse_line("INFO [yamkit-rollout] running", parsed)  # Reordered output cannot restart policy time.
    home = rollout_progress(status(parsed))
    assert home["phase"] == "returning_home" and home["policy_elapsed_s"] == 6.0
    assert not home["policy_timer_running"] and not home["resources_released"]
    parse_line("INFO [yamkit-rollout] releasing", parsed)
    parse_line("INFO [yamkit-rollout] released", parsed)
    clock[0] = 50.0
    final = rollout_progress(status(parsed))
    assert final["policy_elapsed_s"] == 6.0 and final["resources_released"]
    assert final["phase"] == "finalizing"


def test_unknown_clock_and_stop_never_invent_completed_duration(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    parsed = {}
    assert rollout_progress(status(parsed, active=False, returncode=1))["policy_elapsed_s"] is None
    parse_line("[yamkit-rollout] running", parsed)
    clock[0] = 12.0
    _end_rollout_clock(parsed)
    clock[0] = 100.0
    stopped = rollout_progress(status(parsed, active=False, stop_requested=True, returncode=0))
    assert stopped["phase"] == "stopped" and stopped["policy_elapsed_s"] == 2.0
    assert stopped["policy_remaining_s"] == 18.0 and not stopped["policy_timer_running"]
    assert rollout_progress({"mode": "record"}) is None


def test_session_status_is_an_immutable_snapshot():
    manager = SessionManager()
    manager.mode = "rollout"
    manager.meta = {"duration": 5, "arms": ["left_follower"]}
    manager.parsed = {"rollout_export": {"phase": "saving_frames", "completed": 1, "total": 5}}
    snapshot = manager.status()
    manager.parsed["rollout_export"]["completed"] = 2
    manager.meta["arms"].append("right_follower")
    assert snapshot["parsed"]["rollout_export"]["completed"] == 1
    assert snapshot["meta"]["arms"] == ["left_follower"]


def test_new_rollout_has_preparation_clock_before_any_child_marker():
    manager = SessionManager()
    manager.start("rollout", [sys.executable, "-u", "-c", "import time; time.sleep(30)"], {"duration": 5})
    try:
        progress = manager.status()["rollout_progress"]
        assert progress["phase"] == "preparing" and progress["phase_elapsed_s"] >= 0
        assert progress["policy_elapsed_s"] is None and not progress["policy_timer_running"]
    finally:
        manager.stop(grace_s=0.1)
        manager.wait(timeout=5)


def test_stop_freezes_clock_before_child_cleanup_without_altering_signal(monkeypatch):
    manager = SessionManager()
    manager.mode = "rollout"
    manager._proc = SimpleNamespace(poll=lambda: None, pid=1)
    manager.meta = {"duration": 20}
    sent = []
    monkeypatch.setattr(manager, "_signal", lambda proc, sig: sent.append((proc, sig)))
    monkeypatch.setattr(manager, "_escalate", lambda *_: None)
    parse_line("[yamkit-rollout] running", manager.parsed)
    stopped = manager.stop()
    assert not stopped["rollout_progress"]["policy_timer_running"]
    assert stopped["rollout_progress"]["phase"] == "releasing"
    assert sent and sent[0][1].name == "SIGINT"
    assert manager.parsed["rollout_ended_monotonic"] >= manager.parsed["rollout_started_monotonic"]


@pytest.mark.parametrize("changes", [
    {"phase": "running"}, {"completed": -1}, {"completed": 11}, {"completed": True},
    {"total": 100001}, {"total": 1.5}, {"unit": "secret"}, {"camera": "unknown"},
    {"resources_released": False}, {"resources_released": None},
])
def test_export_protocol_rejects_unsupported_or_unreleased_progress(changes):
    parsed = {}
    value = {"phase": "saving_frames", "completed": 1, "total": 10, "unit": "frames",
             "camera": "top", "resources_released": True, **changes}
    parse_line("[yamkit-export] " + json.dumps(value), parsed)
    assert "rollout_export" not in parsed


def test_export_protocol_whitelists_fields_and_does_not_reset_phase_on_each_count(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    parsed = {}
    value = {"phase": "saving_frames", "completed": 1, "total": 10, "unit": "frames",
             "camera": "top", "resources_released": True, "unexpected": "ignored"}
    parse_line("[yamkit-export] " + json.dumps(value), parsed)
    clock[0] = 4.0
    parse_line("[yamkit-export] " + json.dumps({**value, "completed": 4}), parsed)
    progress = rollout_progress(status(parsed))
    assert progress["phase_elapsed_s"] == 3.0 and progress["completed"] == 4
    assert progress["total"] == 10 and progress["resources_released"]
    assert "unexpected" not in json.dumps(parsed)
    parse_line("[yamkit-rollout] running", parsed)
    assert "rollout_started_monotonic" not in parsed


def await_progress(ui, run_id, phase):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = ui.client.get(f"/api/deployments/{run_id}").json().get("rollout_progress")
        if value and value["phase"] == phase:
            return value
        time.sleep(.01)
    pytest.fail(f"progress did not reach {phase}")


def test_pending_packaging_and_upload_remain_visible_after_child_exit(upload_trial, monkeypatch):
    ui = upload_trial.ui
    packaging, uploading, release_package, release_upload = (threading.Event() for _ in range(4))

    def package(run_dir, **_):
        packaging.set()
        assert release_package.wait(timeout=5)
        return run_dir

    def upload(bundle, **_):
        uploading.set()
        assert release_upload.wait(timeout=5)
        return {"status": "uploaded", "repo_id": "owner/rollouts"}

    monkeypatch.setattr(rollout_artifacts, "package_rollout", package)
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", upload)
    response = launch(ui, upload_repo_id="owner/rollouts")
    run_id = response.json()["meta"]["run_id"]
    try:
        assert packaging.wait(timeout=5)
        live = ui.client.get("/api/session").json()
        assert not live["active"] and not live["cameras_owned"]
        assert live["rollout_progress"]["phase"] == "packaging"
        assert live["rollout_progress"]["total"] is None
        release_package.set()
        assert uploading.wait(timeout=5)
        assert await_progress(ui, run_id, "uploading")["completed"] is None
    finally:
        release_package.set()
        release_upload.set()
    done = await_progress(ui, run_id, "done")
    assert done["outcome"] == "completed" and not done["policy_timer_running"]
    assert done["policy_elapsed_s"] is None  # Fake child did not report a policy start.
    assert done["phase_elapsed_s"] == 0.0  # A finished phase does not count forever.
    app = server.create_app(ui.rig.path, outputs_dir=ui.root / "outputs", frontend_dir=server.FRONTEND_DIR)
    with TestClient(app) as client:
        historical = client.get(f"/api/deployments/{run_id}").json()["rollout_progress"]
    assert historical["phase"] == "done" and historical["policy_elapsed_s"] is None


@pytest.mark.parametrize("failed_write", [False, True])
def test_upload_error_is_not_run_success_and_optional_progress_io_cannot_block_upload(upload_trial, monkeypatch, failed_write):
    ui = upload_trial.ui
    monkeypatch.setattr(rollout_artifacts, "package_rollout", lambda run_dir, **_: run_dir)
    calls = []

    def upload(*_args, **_kwargs):
        calls.append(True)
        raise RuntimeError("hf_PRIVATE_UNRELATED_CREDENTIAL")

    monkeypatch.setattr(rollout_artifacts, "upload_rollout", upload)
    if failed_write:
        write_text = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "rollout-progress.json.tmp":
                raise OSError("fake disk failure for optional progress only")
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write)
    response = launch(ui, upload_repo_id="owner/rollouts")
    run_id = response.json()["meta"]["run_id"]
    failed = await_progress(ui, run_id, "failed")
    assert calls and failed["postprocess_error"] == "upload_failed"
    assert failed["outcome"] == "completed"  # Physical outcome is separate, not rewritten.
    assert "PRIVATE" not in json.dumps(failed)
    run = ui.client.get(f"/api/deployments/{run_id}").json()
    assert run["status"] == "success" and run["upload"]["status"] == "failed"


def test_stopped_and_export_error_are_not_promoted_to_done(upload_trial, monkeypatch):
    ui = upload_trial.ui
    original_finalize = server.DeploymentLog.finalize

    def finalize(self, run_dir, status):
        # Simulate a completed motion with a post-release encoder failure.
        summary = run_dir / "summary.json"
        if summary.exists():
            summary.write_text(json.dumps({"resources_released": True, "video_export_errors": {"top": "ValueError"}}))
        original_finalize(self, run_dir, status)

    monkeypatch.setattr(server.DeploymentLog, "finalize", finalize)
    response = launch(ui, capture_trace=True)
    run_id = response.json()["meta"]["run_id"]
    progress = await_progress(ui, run_id, "failed")
    assert progress["postprocess_error"] == "recording_export_incomplete"
    assert progress["outcome"] == "completed"


def test_requested_capture_without_summary_is_not_reported_saved(upload_trial, monkeypatch):
    ui = upload_trial.ui
    original_finalize = server.DeploymentLog.finalize

    def finalize(self, run_dir, status):
        (run_dir / "summary.json").unlink(missing_ok=True)
        original_finalize(self, run_dir, status)

    monkeypatch.setattr(server.DeploymentLog, "finalize", finalize)
    response = launch(ui, capture_trace=True)
    assert response.status_code == 200, response.text
    assert ui.manager.wait(timeout=8) == 0
    run_id = response.json()["meta"]["run_id"]
    progress = ui.client.get(f"/api/deployments/{run_id}").json()["rollout_progress"]
    assert progress["phase"] == "failed"
    assert progress["postprocess_error"] == "recording_summary_unavailable"
    assert progress["outcome"] == "completed"  # Do not rewrite the physical subprocess result.
    assert not ui.manager.active and not ui.manager.cameras_owned


def test_post_release_metadata_write_failure_does_not_leave_progress_finalizing(upload_trial, monkeypatch):
    ui = upload_trial.ui

    def fail_finalize(*_args, **_kwargs):
        raise OSError("fixture metadata write failed")

    monkeypatch.setattr(server.DeploymentLog, "finalize", fail_finalize)
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", lambda *_a, **_k: pytest.fail("No automatic upload after finalization failure"))
    response = launch(ui, capture_trace=True, upload_repo_id="owner/rollouts")
    assert response.status_code == 200, response.text
    assert ui.manager.wait(timeout=8) == 0
    session = ui.client.get("/api/session").json()
    assert not session["active"] and not session["cameras_owned"]
    assert session["returncode"] == 0
    assert session["rollout_progress"]["phase"] == "failed"
    assert session["rollout_progress"]["postprocess_error"] == "recording_finalization_failed"
    assert session["rollout_progress"]["outcome"] == "completed"
    run_id = response.json()["meta"]["run_id"]
    assert (ui.root / "outputs/ui/deployments" / run_id / "summary.json").is_file()


@pytest.mark.parametrize("failed_stage", ["queued", "packaging", "uploading", "failed"])
def test_hf_receipt_disk_failure_is_visible_and_never_retries_upload(upload_trial, monkeypatch, failed_stage):
    ui = upload_trial.ui
    original_write = Path.write_text
    failed = False
    uploads = []

    def write(path, content, *args, **kwargs):
        nonlocal failed
        if path.name == "hf-upload.json.tmp":
            failed |= json.loads(content).get("status") == failed_stage
            if failed:
                raise OSError("fixture HF receipt persistence failed")
        return original_write(path, content, *args, **kwargs)

    def upload(*_args, **_kwargs):
        uploads.append(True)
        raise RuntimeError("fixture HF request failure")

    monkeypatch.setattr(Path, "write_text", write)
    monkeypatch.setattr(rollout_artifacts, "package_rollout", lambda run_dir, **_: run_dir)
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", upload)
    response = launch(ui, upload_repo_id="owner/rollouts")
    assert response.status_code == 200, response.text
    run_id = response.json()["meta"]["run_id"]
    progress = await_progress(ui, run_id, "failed")
    assert failed
    assert progress["postprocess_error"] == ("recording_finalization_failed" if failed_stage == "queued" else "upload_failed")
    assert progress["outcome"] == "completed"
    assert len(uploads) == (1 if failed_stage == "failed" else 0)
    assert not ui.manager.active and not ui.manager.cameras_owned
