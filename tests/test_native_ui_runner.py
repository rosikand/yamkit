"""Real managed UI child/executor/export lifecycle, exclusively with explicit saved fake I/O.

The child command is replaced only in this test. Its production UI entrypoint is
unchanged; the test delegate converts synthetic UI confirmations to FALSE external
approvals plus an exact SavedRobot before calling the actual shared workflow.
No production flag, environment switch, hardware constructor or network is used.
"""

import builtins
import io
import json
import sys
import threading
import time
from contextlib import ExitStack
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_inference_ui import inference_ui as _inference_ui
from tests.test_native_ui import payload
from yamkit.ui import native_inference as native

inference_ui = _inference_ui


def controlled_fake_child(request_path, scenario):
    """Subprocess-only test seam; always uses fake factory and device/network tripwires."""
    from yamkit import backend_workflow, paths, pi05_artifacts, pi05_workflow
    from yamkit.fake_inference import SavedRobot, forbid_device_io
    from yamkit.inference import identity
    from yamkit.inference.http_transport import HttpTransport
    from yamkit.pi05 import rollout
    from yamkit.pi05_workflow import Pi05Selection

    request_path = Path(request_path)
    root = request_path.parents[3]
    qualification = root / "qualification.json"
    with np.load(root / "saved.npz", allow_pickle=False) as saved:
        sample = {key: saved[key].copy() for key in saved.files}
    def validate(target):
        values = np.asarray(list(target.values()), dtype=float)
        assert values.shape == (14,) and np.isfinite(values).all()
        assert 0 <= values[6] <= 1 and 0 <= values[13] <= 1
    robot = SavedRobot([sample], validate)
    cancelled = threading.Event()
    events = []
    diagnostics = {"false_external_approvals": False, "commands_after_stop": 0, "rpc_calls": 0,
                   "hardware_devices_opened": False, "export_after_release": None}
    metadata = {"instance_id": "offline-native-ui-fixture", "http_session_expires_at": time.time() + 3600}

    class Transport:
        def ready(self, _timeout): return metadata
        def ensure_session_active(self): return None
        def close(self): return None
        def cancel(self): cancelled.set()

        def predict_chunk(self, _request, timeout):
            diagnostics["rpc_calls"] += 1
            raw = np.zeros((30, 14))
            raw[0, 6] = 1.000895619392395  # Actual reviewed small native overshoot, not hidden clipping.
            if diagnostics["rpc_calls"] == 2:
                if scenario == "stop":
                    print("OFFLINE_FIXTURE_RPC_INFLIGHT", flush=True)
                    assert cancelled.wait(min(1.8, timeout)), "Test must request UI Stop during this RPC"
                elif scenario == "raw_fault":
                    raw[0, 6] = 1.0101  # Beyond the fixed guard: preserve and reject, never widen.
            return {"chunk": raw.tolist()}

    transport = Transport()
    original_send, original_release = robot.send_reference_action, robot.disconnect
    def send(target, *, dispatch_check):
        diagnostics["commands_after_stop"] += int(cancelled.is_set())
        return original_send(target, dispatch_check=dispatch_check)
    def release(*, home=False):
        assert home is False
        original_release(home=False)
        events.append("released")
        if scenario == "release_failure":
            raise RuntimeError("private_fixture_release_diagnostic")
    robot.send_reference_action, robot.disconnect = send, release
    original_run = pi05_workflow.run_prepared_pi05
    def fake_delegate(selection, *, confirm_supervised, accept_mapping, **kwargs):
        assert confirm_supervised is True and accept_mapping is True  # Synthetic UI boundary only.
        diagnostics["false_external_approvals"] = True
        kwargs["artifact_metadata"] = {"test_only": True, "motion_approval_received": False}
        return original_run(selection, confirm_supervised=False, accept_mapping=False,
                            fake_robot=robot, **kwargs)
    def retained(options):
        return Pi05Selection(options.task, options.rig_path, options.duration, "offline-native-ui", None, qualification), {}
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Real device, HTTP or upload is forbidden in this offline UI test")
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if any(name == key or name.startswith(key + ".") for key in (
                "yamkit.arm", "lerobot_robot_yamkit", "i2rt", "can", "pyrealsense2")):
            forbidden()
        return original_import(name, *args, **kwargs)
    original_finalize = pi05_artifacts.NativeCapture.finalize
    def finalize(self, directory, report, **kwargs):
        diagnostics["export_after_release"] = "released" in events
        assert robot.released and "released" in events
        return original_finalize(self, directory, report, **kwargs)
    tools = pi05_artifacts._trace_tools()
    original_encode = tools.encode_timestamped_video
    def encode(frames, path, timeline):
        assert robot.released and scenario != "release_failure"
        events.append("encode")
        if scenario == "export_failure" and Path(path).name == "top.mp4":
            raise RuntimeError("private_fixture_encoder_diagnostic")
        return original_encode(frames, path, timeline)
    target = backend_workflow.BackendTarget("lambda", "pi05-yam", "offline-native-ui", "http://127.0.0.1:8766",
                                            root / "dummy-token-path-never-read")
    stable_build = identity.inference_build_id()
    with ExitStack() as stack:
        stack.enter_context(forbid_device_io())
        for owner, name, value in (
            (paths, "ROOT", root), (backend_workflow, "ROOT", root), (pi05_workflow, "ROOT", root),
            (identity, "inference_build_id", lambda: stable_build),
            (native, "preparation_context", lambda options: {"selection_key": options.operation_key}),
            (native, "retained_selection", retained),
            (pi05_workflow, "run_prepared_pi05", fake_delegate),
            (pi05_workflow, "load_target", lambda *_a, **_k: target),
            (pi05_workflow, "_readiness", lambda *_a: metadata),
            (pi05_workflow, "_transport", lambda *_a: transport),
            (rollout, "validate_qualification", lambda *_a, **_k: None),
            (rollout, "_make_robot", forbidden), (rollout, "_home", forbidden),
            (HttpTransport, "_invoke", forbidden), (pi05_artifacts, "upload_rollout", forbidden),
            (pi05_artifacts.NativeCapture, "finalize", finalize),
            (tools, "encode_timestamped_video", encode), (builtins, "__import__", guarded_import),
        ):
            stack.enter_context(patch.object(owner, name, value))
        stack.enter_context(patch("yamkit.external_ops._read_private", lambda *_a: "offline-fixture-only"))
        code = native.execute_request(request_path, motion=True)
    diagnostics.update(fake_robot_released=robot.released, commands=len(robot.sent), events=events, exit_code=code)
    (request_path.parent / "test-only-diagnostics.json").write_text(json.dumps(diagnostics))
    return code


@pytest.fixture
def runner_ui(inference_ui, monkeypatch):
    ui = inference_ui
    np.savez(ui.root / "saved.npz", state=np.zeros(14), **{
        name: np.full((480, 640, 3), index * 30, dtype=np.uint8)
        for index, name in enumerate(("top", "left_wrist", "right_wrist"), start=1)})
    (ui.root / "qualification.json").write_text("{}")
    monkeypatch.setattr(native, "preparation_context", lambda value: {"selection_key": value.operation_key})
    monkeypatch.setattr(native, "retained_selection", lambda value: (object(), {
        "ready": True, "selection_key": value.operation_key, "checked_at": time.time(),
        "expires_at": time.time() + 3600, "hardware_tested": False}))
    original_start = ui.manager.start
    ui.scenario = None
    ui.entrypoint_requests = []
    def fake_start(mode, argv, meta, **kwargs):
        assert mode == "rollout" and Path(argv[1]).name == "run_native_inference.py"
        assert ui.scenario in ("stop", "raw_fault", "export_failure", "release_failure")
        ui.entrypoint_requests.append(Path(argv[2]))
        # Only this test substitutes a child, and only its explicit fake delegate
        # is permitted to reach the native runtime. Production argv stays untouched.
        command = [sys.executable, "-u", "-c", ("import sys; from tests.test_native_ui_runner import controlled_fake_child; "
                   "raise SystemExit(controlled_fake_child(sys.argv[1], sys.argv[2]))"), argv[2], ui.scenario]
        return original_start(mode, command, meta, **kwargs)
    monkeypatch.setattr(ui.manager, "start", fake_start)
    return ui


def launch(ui, scenario):
    ui.scenario = scenario
    response = ui.client.post("/api/session/rollout", json=payload(
        confirm_motion=True, mapping_accepted=True, supervised_confirmed=True, capture_trace=True))
    assert response.status_code == 200, response.text
    return response.json()["meta"]["run_id"]


def completed(ui, run_id):
    assert ui.manager.wait(timeout=30) != 0
    status = ui.client.get("/api/session").json()
    assert not status["active"] and not status["cameras_owned"]
    directory = Path(status["meta"]["debug_trace_dir"])
    detail = ui.client.get("/api/deployments/" + run_id).json()
    report = ui.client.get(f"/api/deployments/{run_id}/artifact/report.json").json()
    diagnostics = json.loads((ui.entrypoint_requests[0].parent / "test-only-diagnostics.json").read_text())
    assert report["hardware_tested"] is False and diagnostics["false_external_approvals"]
    assert diagnostics["hardware_devices_opened"] is False and diagnostics["fake_robot_released"]
    assert diagnostics["commands_after_stop"] == 0 and diagnostics["export_after_release"]
    assert "private_fixture_" not in json.dumps(status)
    assert detail["status"] != "success"
    assert len(ui.entrypoint_requests) == 1  # No physical or software retry after a failed/stopped run.
    return status, directory, detail, report, diagnostics


def assert_video(ui, run_id, camera, expected_frames):
    import av

    endpoint = f"/api/deployments/{run_id}/video/{camera}.mp4"
    ranged = ui.client.get(endpoint, headers={"Range": "bytes=0-31"})
    assert ranged.status_code == 206 and len(ranged.content) == 32
    response = ui.client.get(endpoint)
    assert response.status_code == 200 and response.headers["content-type"] == "video/mp4"
    with av.open(io.BytesIO(response.content)) as container:
        frames = list(container.decode(video=0))
    assert len(frames) == expected_frames and all((frame.width, frame.height) == (640, 480) for frame in frames)
    assert all(b.pts > a.pts for a, b in pairwise(frames))


def test_ui_stop_during_real_fake_rpc_releases_then_exports_playable_native_trace(runner_ui):
    ui = runner_ui
    run_id = launch(ui, "stop")
    deadline = time.monotonic() + 8
    while "OFFLINE_FIXTURE_RPC_INFLIGHT" not in ui.manager.log and time.monotonic() < deadline:
        assert ui.manager.active, "Controlled fake child exited before the Stop seam"
        time.sleep(.01)
    assert "OFFLINE_FIXTURE_RPC_INFLIGHT" in ui.manager.log
    receipt = ui.client.post("/api/session/stop", json={}).json()
    assert receipt["stop_requested"]
    status, directory, detail, report, diagnostics = completed(ui, run_id)
    assert status["stop_requested"] and report["status"] == "stopped"
    assert report["released"] and not report["home_attempted"]
    assert diagnostics["rpc_calls"] == 2 and diagnostics["commands"] == 30
    assert report["execution"]["completed_chunks"] == 1
    assert detail["recording"]["state"] == "available" and detail["recording"]["video_count"] == 3
    trace = json.loads((directory / "trace.json").read_text())
    assert trace["chunks"][0]["raw_actions"][0][6] == 1.000895619392395
    assert trace["chunks"][0]["actions"][0][6] == 1
    for camera in ("top", "left_wrist", "right_wrist"):
        assert_video(ui, run_id, camera, detail["recording"]["frame_count"])


def test_ui_actual_native_bound_fault_retains_raw_response_and_playback_after_release(runner_ui):
    ui = runner_ui
    run_id = launch(ui, "raw_fault")
    _status, directory, detail, report, diagnostics = completed(ui, run_id)
    assert report["status"] == "failed" and report["released"] and not report["home_attempted"]
    assert report["execution"]["faults"] == 1 and diagnostics["commands"] == 30
    assert diagnostics["rpc_calls"] == 2 and detail["recording"]["state"] == "available"
    trace = json.loads((directory / "trace.json").read_text())
    assert trace["native_responses"][1]["raw_rows"][0][6] == 1.0101
    assert len(trace["chunks"]) == 1
    assert_video(ui, run_id, "top", detail["recording"]["frame_count"])


def test_ui_real_fake_export_failure_is_partial_not_success_and_keeps_other_playback(runner_ui):
    ui = runner_ui
    run_id = launch(ui, "export_failure")
    _status, directory, detail, report, diagnostics = completed(ui, run_id)
    assert report["status"] == "completed" and report["released"] and report["home_completed"]
    assert diagnostics["commands"] > 0
    assert detail["recording"]["state"] == "partial" and detail["recording"]["video_count"] == 2
    assert detail["recording"]["missing_cameras"] == ["top"] and detail["recording"]["errors"]
    assert (directory / "frames/top/frame-000000.png").is_file()
    assert ui.client.get(f"/api/deployments/{run_id}/video/top.mp4").status_code == 404
    assert_video(ui, run_id, "left_wrist", detail["recording"]["frame_count"])


def test_ui_release_failure_never_encodes_or_advertises_playback(runner_ui):
    ui = runner_ui
    run_id = launch(ui, "release_failure")
    _status, directory, detail, report, diagnostics = completed(ui, run_id)
    assert report["status"] == "release_failed" and not report["released"]
    assert detail["recording"]["state"] == "unavailable" and detail["recording"]["video_count"] == 0
    assert detail["recording"]["export_status"] == "EXPORT_SKIPPED_RESOURCES_OPEN"
    assert "encode" not in diagnostics["events"] and not (directory / "frames").exists()
    assert ui.client.get(f"/api/deployments/{run_id}/video/top.mp4").status_code == 404
