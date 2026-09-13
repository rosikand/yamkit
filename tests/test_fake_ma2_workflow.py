"""Explicit fake CLI replay with the actual frozen LeRobot/MA2 runner."""

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_http_policy_readiness import transport  # noqa: F401 — real pipeline, fake HTTP only
from tests.test_http_runtime_binding import TASK
from yamkit import fake_ma2_workflow as workflow
from yamkit.backend_workflow import WorkflowError
from yamkit.inference_workflow import reference_options


@pytest.fixture
def selection(rig, monkeypatch):
    rig.control.home_speed = 1.0
    for spec in rig.arms.values():
        if spec.has_motor_gripper:
            spec.gripper_limits = [0.0, 6.5]
    rig.cameras = {name: {"type": "opencv", "index_or_path": 0, "width": 640, "height": 480, "fps": 30}
                   for name in ("top", "left_wrist", "right_wrist")}
    rig.save()
    observed = {"state": np.zeros(14), **{name: np.zeros((480, 640, 3), dtype=np.uint8) for name in rig.cameras}}
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_k: SimpleNamespace(service="lambda-test"))
    monkeypatch.setattr(workflow, "saved_observations", lambda _target: [observed])
    monkeypatch.setattr(workflow, "require_prepared_current", lambda _selection: None)
    return reference_options(policy="molmoact2", task=TASK, service="lambda-test", rig=rig.path, duration=.15)


def test_current_qualification_is_checked_before_entering_fake_context(selection, monkeypatch, tmp_path):
    def blocked(_selection):
        raise WorkflowError("Current qualification expired")

    monkeypatch.setattr(workflow, "require_prepared_current", blocked)
    monkeypatch.setattr(workflow, "molmo_fake_devices", lambda *_: pytest.fail("No fake or real devices before proof"))
    with pytest.raises(WorkflowError, match="expired"):
        workflow.run_fake_molmoact2(selection, artifact_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change", [
    {"policy": "pi05-yam"}, {"controller_mode": "async"}, {"mapping_accepted": True},
    {"supervised_confirmed": True}, {"duration": 91}, {"fps": 15}, {"center_crop": True},
])
def test_fake_api_rejects_nonreference_selection_and_real_approval_flags(selection, tmp_path, change):
    with pytest.raises(WorkflowError, match="unapproved"):
        workflow.run_fake_molmoact2(replace(selection, **change), artifact_dir=tmp_path / "run")


def test_real_frozen_runner_uses_only_guarded_fake_sdk_and_saved_cameras(selection, transport, tmp_path):  # noqa: F811
    result = workflow.run_fake_molmoact2(selection, artifact_dir=tmp_path / "run")
    assert result["status"] == "completed", result
    assert result["hardware_tested"] is False and result["motion_approval_received"] is False
    assert result["resources_released"] is True
    assert result["fake_sdk_instances"] == 2
    assert result["fake_sdk_commands_including_startup_home"] > 0
    assert result["execution"]["reference_contract"]["id"] == "yam_upstream_literal_v1"
    assert result["execution"]["interpolation_dispatches"] > 0
    assert transport.closed is True
    assert any(request["mode"] != "native_fixture" for request in transport.requests)
    saved = json.loads((tmp_path / "run/fake-result.json").read_text())
    assert saved["physical_task_success"] is None


def test_capture_delegates_unchanged_existing_trace_execute_inside_fake_guard(selection, monkeypatch, tmp_path):
    state = {"inside": False}

    @contextmanager
    def fake(_observations):
        state["inside"] = True
        yield [SimpleNamespace(closed=True, commands=[]) for _ in range(2)]
        state["inside"] = False

    def execute(args):
        assert state["inside"] is True
        assert args.task == TASK and args.controller_mode == "reference"
        assert args.backend == "external" and args.external_service == "lambda-test"
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "summary.json").write_text(json.dumps({"status": "TRACE_SAVED", "resources_released": True}))
        (args.output_dir / "metrics.json").write_text(json.dumps({"reference_execution": {"completed_steps": 30}}))
        (args.output_dir / "trace.json").write_text("{}")
        return 0

    tools = SimpleNamespace(artifact_directory=lambda *_: None, execute=execute,
                            render_report=lambda path: state.update(rendered=path))
    monkeypatch.setattr(workflow, "_tools", lambda: tools)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "molmo_fake_devices", fake)
    result = workflow.run_fake_molmoact2(replace(selection, duration=5), artifact_dir=tmp_path / "run", capture_trace=True)
    assert result["hardware_tested"] is False and result["status"] == "completed"
    assert state["rendered"].parent == tmp_path / ".context/rollout-traces"
    assert Path(result["trace_directory"]) == state["rendered"]
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert summary["synthetic_fixture"] is True and summary["motion_approval_received"] is False


def test_actual_cli_recorder_exports_fake_playback_without_any_real_device(
        selection, transport, monkeypatch, tmp_path):  # noqa: F811 — shared fake HTTP fixture
    from yamkit import external_ops, paths
    from yamkit.ui import catalog

    tools = workflow._tools()
    monkeypatch.setattr(workflow, "_tools", lambda: tools)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(external_ops, "owned_service", lambda _name: {"ready": True})
    result = workflow.run_fake_molmoact2(replace(selection, duration=1), artifact_dir=tmp_path / "playback",
                                        capture_trace=True)
    assert result["exit_status"] == 0, result
    assert result["resources_released"] is True and result["hardware_tested"] is False
    assert result["motion_approval_received"] is False
    assert result["execution"]["interpolation_dispatches"] > 0
    detail = catalog.deployment_detail(tmp_path, "playback")
    assert detail["recording"]["state"] == "available", detail
    assert detail["recording"]["video_count"] == 3
    assert "Generated test data only" in (tmp_path / "playback/report.html").read_text()
    original = Path(result["trace_directory"])
    assert original.is_dir() and (original / "frames/top/frame-000000.png").is_file()


def test_missing_saved_inputs_never_enters_any_hardware_scope(selection, monkeypatch, tmp_path):
    def missing(_target):
        raise WorkflowError("saved_observations required")

    monkeypatch.setattr(workflow, "saved_observations", missing)
    monkeypatch.setattr(workflow, "molmo_fake_devices", lambda *_: pytest.fail("No device context"))
    with pytest.raises(WorkflowError, match="saved_observations"):
        workflow.run_fake_molmoact2(selection, artifact_dir=tmp_path / "run")


def test_unsafe_or_existing_destination_never_overwritten(selection, monkeypatch, tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "original.txt").write_text("preserve")
    with pytest.raises(WorkflowError, match="new"):
        workflow.run_fake_molmoact2(selection, artifact_dir=destination)
    assert (destination / "original.txt").read_text() == "preserve"
    with pytest.raises(WorkflowError, match="checkout"):
        workflow.run_fake_molmoact2(selection, artifact_dir=Path("/tmp/unrelated"))


def test_future_physical_record_helper_never_runs_without_both_fresh_flags(selection, monkeypatch):
    monkeypatch.setattr(workflow, "_tools", lambda: pytest.fail("No recorder before explicit motion confirmation"))
    for options in ({}, {"confirm_supervised": True}, {"accept_mapping": True}):
        with pytest.raises(WorkflowError, match="fresh supervised"):
            workflow.run_recorded_molmoact2(selection, **options)


def test_fake_guards_cannot_be_installed_in_ui_worker_thread(selection, monkeypatch, tmp_path):
    monkeypatch.setattr(workflow.threading, "current_thread", lambda: object())
    with pytest.raises(WorkflowError, match="CLI process"):
        workflow.run_fake_molmoact2(selection, artifact_dir=tmp_path / "run")


@pytest.fixture
def stub_recording(selection, monkeypatch, tmp_path):
    """Only filesystem writes; every actual recorder/device operation is replaced."""
    state = SimpleNamespace(robots=2, execute_error=False, render_error=False, calls=0)

    @contextmanager
    def fake(_observations):
        yield [SimpleNamespace(closed=True, commands=[np.zeros(7)]) for _ in range(state.robots)]

    def execute(args):
        state.calls += 1
        if state.execute_error:
            raise RuntimeError("opaque-private-fixture-do-not-display")
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "summary.json").write_text(json.dumps({"status": "TRACE_SAVED", "resources_released": True}))
        (args.output_dir / "metrics.json").write_text(json.dumps({"reference_execution": {"completed_steps": 30}}))
        (args.output_dir / "trace.json").write_text("{}")
        (args.output_dir / "report.html").write_text("original report")
        return 0

    def render(_directory):
        if state.render_error:
            raise RuntimeError("opaque-private-fixture-do-not-display")

    tools = SimpleNamespace(artifact_directory=lambda *_: None, execute=execute, render_report=render)
    monkeypatch.setattr(workflow, "_tools", lambda: tools)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "molmo_fake_devices", fake)
    monkeypatch.setattr("yamkit.inference.performance.require_physical_modal_rollout", lambda *_a, **_k: None)
    state.selection = replace(selection, duration=5)
    state.run_fake = lambda **kwargs: workflow.run_fake_molmoact2(
        state.selection, artifact_dir=tmp_path / "fake-result", capture_trace=True, **kwargs)
    state.run_physical_stub = lambda **kwargs: workflow.run_recorded_molmoact2(
        state.selection, confirm_supervised=True, accept_mapping=True, **kwargs)
    return state


def assert_finalized_failure(result):
    assert result["status"] == "failed" and result["exit_status"] != 0
    assert result["pipeline_complete"] is False
    meta = json.loads((Path(result["artifact_directory"]) / "meta.json").read_text())
    assert meta["active"] is False and meta["returncode"] != 0 and meta["status"] == "failed"
    assert isinstance(meta["ended_at"], (int, float))
    assert "opaque-private-fixture" not in json.dumps(result)
    return meta


def test_render_failure_updates_cli_and_ui_status_without_invalid_fake_report(stub_recording):
    stub_recording.render_error = True
    result = stub_recording.run_fake()
    meta = assert_finalized_failure(result)
    assert result["execution_status"] == "completed"
    assert meta["postprocess_error"] == "report_render_failed"
    assert not (Path(result["artifact_directory"]) / "report.html").exists()
    assert (Path(result["trace_directory"]) / "trace.json").is_file()


@pytest.mark.parametrize("entry", ["run_fake", "run_physical_stub"])
def test_recording_failure_finalizes_history_without_claiming_release(stub_recording, entry):
    stub_recording.execute_error = True
    result = getattr(stub_recording, entry)()
    assert_finalized_failure(result)
    assert result["resources_released"] is False
    assert result["error_type"] == "RuntimeError"
    assert stub_recording.calls == 1


@pytest.mark.parametrize("entry", ["run_fake", "run_physical_stub"])
def test_upload_failure_marks_entire_pipeline_failed_and_keeps_execution_outcome(
        stub_recording, monkeypatch, entry):
    uploads = []
    monkeypatch.setattr(workflow, "package_rollout", lambda directory, **_k: directory / "bundle")

    def failed(bundle, *, repo_id):
        uploads.append(repo_id)
        assert json.loads((bundle.parent / "meta.json").read_text())["active"] is False
        raise RuntimeError("opaque-private-fixture-do-not-display")

    monkeypatch.setattr(workflow, "upload_rollout", failed)
    result = getattr(stub_recording, entry)(upload_repo_id="test/private")
    meta = assert_finalized_failure(result)
    assert uploads == ["test/private"]
    assert result["execution_status"] == "completed" and result["upload"]["status"] == "failed"
    assert meta["postprocess_error"] == "upload_failed"


@pytest.mark.parametrize("robots", [0, 1])
def test_success_requires_both_fake_robots_to_have_been_constructed(stub_recording, robots):
    stub_recording.robots = robots
    result = stub_recording.run_fake()
    assert_finalized_failure(result)
    assert result["fake_sdk_instances"] == robots
    assert result["hardware_tested"] is False


@pytest.mark.parametrize("entry", ["run_fake", "run_physical_stub"])
def test_artifact_import_failure_still_finalizes_ui_history(stub_recording, monkeypatch, entry):
    def fail(*_args, **_kwargs):
        raise OSError("opaque-private-fixture-do-not-display")

    monkeypatch.setattr(workflow, "_publish_trace", fail)
    result = getattr(stub_recording, entry)()
    meta = assert_finalized_failure(result)
    assert meta["postprocess_error"] == "artifact_import_failed"
    assert result["resources_released"] is True
