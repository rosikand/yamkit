"""Inference UI integration with real child lifecycle and entirely fake hardware/cloud work."""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from yamkit.ui import server
from yamkit.ui.sessions import SessionManager

UI = Path(__file__).resolve().parents[1] / "ui"


@pytest.fixture
def inference_ui(rig, tmp_path, monkeypatch):
    from yamkit import arm, modal_ops
    from yamkit.inference.http_transport import HttpTransport
    from yamkit.inference.service import ModelRuntime
    from yamkit.ui.camstream import _Camera

    def forbidden(*args, **kwargs):
        pytest.fail("UI opened hardware, loaded weights or contacted a paid service")

    monkeypatch.setattr(arm.YamArm, "connect", forbidden)
    monkeypatch.setattr(ModelRuntime, "load", forbidden)
    monkeypatch.setattr(modal_ops, "prepare", forbidden)
    monkeypatch.setattr(modal_ops, "service_handle", forbidden)
    monkeypatch.setattr(HttpTransport, "_invoke", forbidden)
    monkeypatch.setattr(modal_ops, "owned_service", lambda: None)
    monkeypatch.setattr(_Camera, "ensure_running", forbidden)
    monkeypatch.setattr(server, "ROOT", tmp_path)
    for spec in rig.followers():
        spec.gripper_limits = [0.0, 6.5]
    rig.cameras = {name: {"type": "opencv", "index_or_path": i, "width": 640, "height": 480}
                   for i, name in enumerate(("top", "left_wrist", "right_wrist"))}
    rig.save()
    manager = SessionManager()
    state = SimpleNamespace(manager=manager, rig=rig, seen=[], child="import time; time.sleep(20)", root=tmp_path)

    def argv(*args):
        state.seen.append(args)
        child = state.child
        if (args[0] == "policy-probe" and "--live" in args) or args[0] == "rollout":
            child = ("from yamkit.camera_ownership import claim_from_env; "
                     "lease=claim_from_env(['top','left_wrist','right_wrist']); "
                     "print('CAMERA_ACQUIRED',flush=True); " + child)
        return [sys.executable, "-u", "-c", child]

    monkeypatch.setattr(manager, "yamkit_argv", argv)
    app = server.create_app(rig.path, datasets_dir=tmp_path / "datasets", outputs_dir=tmp_path / "outputs",
                            frontend_dir=UI, session_manager=manager)
    with TestClient(app) as client:
        state.client = client
        try:
            yield state
        finally:
            manager.stop(grace_s=0.1)
            manager.wait(timeout=5)


def payload(**kwargs):
    return {"policy": "molmoact2", "task": "put the cube in the bowl", **kwargs}


def attached_payload(**kwargs):
    return payload(**{"backend": "modal", "modal_app": "yamkit-vla-session-ui-test", "call_mode": "http",
                      "execution_mode": "cuda_graph10", "image_encoding": "rgb8", "duration": 5,
                      "arms": ["left_follower", "right_follower"], **kwargs})


@pytest.fixture
def attached_modal(inference_ui, monkeypatch):
    """Local evidence/credentials only; never create an app or invoke the service."""
    from yamkit import modal_ops
    from yamkit.inference import qualification
    from yamkit.inference.profiles import get_profile

    now = time.time()
    app_name = "yamkit-vla-session-ui-test"
    endpoint = "https://ui-test.r1.modal.host"
    expires = now + 600
    receipt = {
        "app_name": app_name, "status": "ready", "profile_id": "molmoact2",
        "revision": get_profile("molmoact2").revision, "transport": "http",
        "execution_mode": "cuda_graph10", "http_endpoint": endpoint,
        "http_ingress": "tunnel", "http_session_expires_at": expires,
        "metadata": {"http_ingress": "tunnel", "http_endpoint": endpoint,
                     "http_session_expires_at": expires, "instance_id": "ui-fixture-instance"},
    }
    credentials = {"app_name": app_name, "endpoint_url": endpoint, "token": "PRIVATE_UI_TEST_TOKEN"}
    state = SimpleNamespace(ui=inference_ui, receipt=receipt, credentials=credentials,
                            created_at=now, validations=[], options=[], settings_override={}, expected_override={})

    def current_settings(options, *, image_hw, metadata=None):
        state.options.append(options)
        return {
            "profile": "molmoact2", "task": options.task, "modal_app": options.modal_app,
            "image_hw": list(image_hw), "image_encoding": options.image_encoding,
            "call_mode": options.call_mode, "execution_mode": options.execution_mode,
            "crop": "center_16_9" if options.center_crop else "none",
            "prediction_queue_threshold": options.prediction_queue_threshold,
            "http_ingress": receipt["http_ingress"],
            "http_session_expires_at": receipt["http_session_expires_at"],
            "instance_id": receipt["metadata"]["instance_id"], **state.settings_override,
        }

    def validate(settings):
        state.validations.append(dict(settings))
        expected = {
            "task": payload()["task"], "modal_app": app_name, "image_hw": [480, 640],
            "image_encoding": "rgb8", "call_mode": "http", "execution_mode": "cuda_graph10",
            "crop": "none", "prediction_queue_threshold": None,
            "instance_id": "ui-fixture-instance", **state.expected_override,
        }
        if any(settings.get(key) != value for key, value in expected.items()):
            raise qualification.QualificationError("Qualification settings changed; rerun on this host")
        if time.time() - state.created_at > qualification.MAX_AGE_S:
            raise qualification.QualificationError("Qualification is expired; rerun on this host")
        return {"created_unix_s": state.created_at, "settings": dict(settings)}

    monkeypatch.setattr(modal_ops, "owned_service", lambda: receipt)
    monkeypatch.setattr(modal_ops, "_read_http_auth", lambda: credentials)
    monkeypatch.setattr(qualification, "current_settings", current_settings)
    monkeypatch.setattr(qualification, "validate_qualification", validate)
    monkeypatch.setattr(qualification, "is_cloud_host", lambda: False)
    return state


def test_attached_preflight_reads_evidence_without_motion_approval_or_child(attached_modal):
    state = attached_modal
    before = time.time()
    response = state.ui.client.post("/api/inference/preflight", json=attached_payload())
    assert response.status_code == 200
    result = response.json()
    assert result["ready"] is True, result
    assert result["reason"]
    assert result["modal_app"] == state.receipt["app_name"]
    assert result["selection_key"] == state.options[-1].operation_key
    assert before <= result["checked_at"] <= time.time()
    assert result["expires_at"] == state.receipt["http_session_expires_at"]
    assert state.options[-1].rig_path == str(state.ui.rig.path)
    assert state.validations[-1]["image_hw"] == [480, 640]
    assert "PRIVATE_UI_TEST_TOKEN" not in response.text
    assert not state.ui.seen
    assert not state.ui.manager.active
    assert all(cam["suspended_by"] is None for cam in state.ui.client.get("/api/cameras").json())


def test_attached_preflight_expires_with_qualification_before_tunnel(attached_modal):
    from yamkit.inference.qualification import MAX_AGE_S

    state = attached_modal
    state.created_at = time.time() - MAX_AGE_S + 90
    response = state.ui.client.post("/api/inference/preflight", json=attached_payload())
    assert response.json()["ready"] is True
    assert response.json()["expires_at"] == state.created_at + MAX_AGE_S
    state.created_at -= 40
    response = state.ui.client.post("/api/inference/preflight", json=attached_payload())
    assert response.json()["ready"] is False  # Evidence must cover startup, policy execution and return home.
    assert not state.ui.seen


@pytest.mark.parametrize("defect", ["uncalibrated", "wrong_side", "missing_camera", "extra_camera",
                                   "unequal_camera_dimensions"])
def test_attached_preflight_checks_current_rig_without_hardware(attached_modal, defect):
    rig = attached_modal.ui.rig
    if defect == "uncalibrated":
        rig.arm("right_follower").gripper_limits = None
    elif defect == "wrong_side":
        rig.arm("right_follower").side = "left"
    elif defect == "missing_camera":
        del rig.cameras["right_wrist"]
    elif defect == "extra_camera":
        rig.cameras["extra"] = dict(rig.cameras["top"])
    elif defect == "unequal_camera_dimensions":
        rig.cameras["right_wrist"]["width"] = 320
    rig.save()
    response = attached_modal.ui.client.post("/api/inference/preflight", json=attached_payload())
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert not attached_modal.ui.seen


@pytest.mark.parametrize("changes", [
    {"modal_app": None}, {"backend": "local"}, {"call_mode": "remote", "execution_mode": "eager"},
    {"execution_mode": "eager"}, {"image_encoding": "jpeg"}, {"policy": "smolvla"},
    {"task": "a different task"}, {"center_crop": True}, {"prediction_queue_threshold": 15},
    {"arms": ["left_follower"]}, {"fps": 15},
])
def test_attached_preflight_rejects_unqualified_selection_without_launch(attached_modal, changes):
    body = {**attached_payload(), **changes}
    response = attached_modal.ui.client.post("/api/inference/preflight", json=body)
    assert response.status_code == 200
    result = response.json()
    assert result["ready"] is False
    assert result["reason"]
    assert result["expires_at"] is None
    assert not attached_modal.ui.seen


@pytest.mark.parametrize("defect", ["stopped", "wrong_app", "wrong_endpoint", "expired", "short_lifetime",
                                   "expired_qualification", "cloud_host", "changed_instance"])
def test_attached_preflight_rejects_stale_evidence_or_credentials(attached_modal, monkeypatch, defect):
    from yamkit.inference import qualification

    state = attached_modal
    if defect == "stopped":
        state.receipt["status"] = "stopped"
    elif defect == "wrong_app":
        state.credentials["app_name"] = "yamkit-vla-other"
    elif defect == "wrong_endpoint":
        state.credentials["endpoint_url"] = "https://other.r1.modal.host"
    elif defect in ("expired", "short_lifetime"):
        expiry = time.time() + (-1 if defect == "expired" else 20)
        state.receipt["http_session_expires_at"] = expiry
        state.receipt["metadata"]["http_session_expires_at"] = expiry
    elif defect == "expired_qualification":
        state.created_at -= qualification.MAX_AGE_S + 1
    elif defect == "cloud_host":
        monkeypatch.setattr(qualification, "is_cloud_host", lambda: True)
    elif defect == "changed_instance":
        state.receipt["metadata"]["instance_id"] = "replacement-container"
    response = state.ui.client.post("/api/inference/preflight", json=attached_payload())
    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["expires_at"] is None
    assert "PRIVATE_UI_TEST_TOKEN" not in response.text
    assert not state.ui.seen


@pytest.mark.parametrize("missing", ["confirm_motion", "mapping_accepted", "supervised_confirmed"])
def test_attached_rollout_requires_each_independent_confirmation(attached_modal, missing):
    body = attached_payload(confirm_motion=True, mapping_accepted=True, supervised_confirmed=True)
    body[missing] = False
    response = attached_modal.ui.client.post("/api/session/rollout", json=body)
    assert response.status_code == 422
    assert not attached_modal.ui.seen
    assert not attached_modal.ui.manager.active


@pytest.mark.parametrize("defect", ["task", "expired", "stopped", "credentials"])
def test_attached_rollout_rechecks_current_state_after_passing_preflight(attached_modal, defect):
    state = attached_modal
    preflight = state.ui.client.post("/api/inference/preflight", json=attached_payload())
    assert preflight.json()["ready"] is True
    body = attached_payload(confirm_motion=True, mapping_accepted=True, supervised_confirmed=True)
    if defect == "task":
        body["task"] = "a task never measured by this container"
    elif defect == "expired":
        state.receipt["http_session_expires_at"] = time.time() - 1
        state.receipt["metadata"]["http_session_expires_at"] = state.receipt["http_session_expires_at"]
    elif defect == "stopped":
        state.receipt["status"] = "stopped"
    elif defect == "credentials":
        state.credentials["app_name"] = "yamkit-vla-other"
    response = state.ui.client.post("/api/session/rollout", json=body)
    assert response.status_code == 422
    assert not state.ui.seen
    assert not state.ui.manager.active
    assert "PRIVATE_UI_TEST_TOKEN" not in response.text


def test_attached_rollout_passes_reviewed_flags_and_manages_child_camera_handoff(attached_modal):
    ui = attached_modal.ui
    body = attached_payload(confirm_motion=True, mapping_accepted=True, supervised_confirmed=True)
    response = ui.client.post("/api/session/rollout", json=body)
    assert response.status_code == 200, response.text
    assert response.json()["mode"] == "rollout"
    args = ui.seen[-1]
    assert args[0] == "rollout"
    for flag, value in {
        "--modal-app": body["modal_app"], "--backend": "modal", "--call-mode": "http",
        "--execution-mode": "cuda_graph10", "--image-encoding": "rgb8", "--task": body["task"],
        "--duration": "5.0", "--fps": "30.0", "--rig": str(ui.rig.path),
    }.items():
        assert args[args.index(flag) + 1] == value
    assert "--accept-mapping" in args
    assert "--confirm-supervised" in args
    assert [args[index + 1] for index, value in enumerate(args) if value == "--arms"] == body["arms"]
    deadline = time.monotonic() + 5
    while "CAMERA_ACQUIRED" not in ui.manager.log and time.monotonic() < deadline:
        time.sleep(.01)
    assert "CAMERA_ACQUIRED" in ui.manager.log
    assert ui.manager.cameras_owned
    assert all(cam["suspended_by"] for cam in ui.client.get("/api/cameras").json())
    assert ui.client.get("/api/cameras/top/stream").status_code == 409
    assert ui.client.post("/api/session/stop").status_code == 200
    assert ui.manager.wait(timeout=5) is not None
    assert not ui.manager.active
    assert all(cam["suspended_by"] is None for cam in ui.client.get("/api/cameras").json())
    assert not any(command[0] == "modal-shutdown" for command in ui.seen)
    history = ui.client.get("/api/deployments").json()
    assert len(history) == 1
    assert history[0]["kind"] == "rollout"
    assert history[0]["task"] == body["task"]
    assert history[0]["status"] == "stopped"


def test_attached_rollout_denies_child_when_preview_retains_camera(attached_modal, monkeypatch):
    from yamkit.ui.camstream import _Camera

    monkeypatch.setattr(_Camera, "running", property(lambda self: True))
    monkeypatch.setattr(_Camera, "stop", lambda *args, **kwargs: None)
    ui = attached_modal.ui
    response = ui.client.post("/api/session/rollout", json=attached_payload(
        confirm_motion=True, mapping_accepted=True, supervised_confirmed=True,
    ))
    assert response.status_code == 200
    assert ui.manager.wait(timeout=5) != 0
    assert "CAMERA_ACQUIRED" not in ui.manager.log
    assert any("acquisition denied" in line for line in ui.manager.log)


@pytest.mark.parametrize("route", ["/api/inference/preflight", "/api/session/rollout"])
@pytest.mark.parametrize("changes", [
    {"task": "a different qualified task"}, {"duration": 6}, {"center_crop": True},
    {"prediction_queue_threshold": 15}, {"arms": ["right_follower", "left_follower"]},
    {"arms": []}, {"call_mode": "remote", "execution_mode": "eager"},
])
def test_attached_debug_capture_rejects_options_outside_reviewed_bounds(attached_modal, route, changes):
    body = attached_payload(capture_trace=True,
                            task="pick up the orange lid and place it into the black circular container")
    body.update(changes)
    # Let ordinary qualification accept the selected settings: the debug guard must still reject them.
    attached_modal.expected_override.update(
        task=body["task"], crop="center_16_9" if body.get("center_crop") else "none",
        prediction_queue_threshold=body.get("prediction_queue_threshold"),
        call_mode=body["call_mode"], execution_mode=body["execution_mode"],
    )
    if route.endswith("rollout"):
        body.update(confirm_motion=True, mapping_accepted=True, supervised_confirmed=True)
    response = attached_modal.ui.client.post(route, json=body)
    if route.endswith("preflight"):
        assert response.status_code == 200
        assert response.json()["ready"] is False
        assert "Debug capture" in response.json()["reason"]
    else:
        assert response.status_code == 422
        assert "Debug capture" in response.text
    assert not attached_modal.ui.seen
    assert not attached_modal.ui.manager.active
    assert not (attached_modal.ui.root / ".context" / "rollout-traces").exists()


@pytest.mark.parametrize("value", ["true", "false", 1, 0, None])
def test_attached_debug_capture_requires_a_json_boolean(attached_modal, value):
    response = attached_modal.ui.client.post("/api/inference/preflight", json=attached_payload(capture_trace=value))
    assert response.status_code == 422
    assert not attached_modal.ui.seen


def test_attached_http_prepare_is_rejected_without_starting_a_child(attached_modal):
    response = attached_modal.ui.client.post("/api/session/modal-prepare", json=attached_payload())
    assert response.status_code == 422
    assert "Conductor" in response.text
    assert not attached_modal.ui.seen
    assert not attached_modal.ui.manager.active


def test_profile_catalog_get_never_starts_compute_or_hardware(inference_ui, monkeypatch):
    for name in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN"):
        monkeypatch.setenv(name, "test-secret-never-return")
    response = inference_ui.client.get("/api/inference/profiles")
    assert response.status_code == 200
    body = response.json()
    assert body["default_backend"] == "local"
    assert {p["id"] for p in body["profiles"]} == {"smolvla", "molmoact2", "pi05"}
    assert set(body["credentials"].values()) == {"SET"}
    assert "test-secret-never-return" not in response.text
    assert not inference_ui.seen
    assert inference_ui.client.get("/").status_code == 200
    assert inference_ui.client.get("/api/session").json()["active"] is False


@pytest.mark.parametrize("route,body", [
    ("rollout", payload()),
    ("policy-probe", payload(live=True)),
    ("policy-probe", payload()),
    ("policy-probe", payload(live=True, saved="snapshot.npz", confirm_active_read=True)),
    ("modal-prepare", payload(backend="local")),
])
def test_activation_and_mode_confirmations_enforced_before_launch(inference_ui, route, body):
    assert inference_ui.client.post(f"/api/session/{route}", json=body).status_code == 422
    assert inference_ui.seen == []


@pytest.mark.parametrize("route,extra", [("policy-probe", {"live": True, "confirm_active_read": True}),
                                         ("rollout", {"backend": "local", "confirm_motion": True})])
def test_missing_second_arm_calibration_rejected_before_child_launch(inference_ui, route, extra):
    inference_ui.rig.arm("right_follower").gripper_limits = None
    inference_ui.rig.save()
    response = inference_ui.client.post(f"/api/session/{route}", json=payload(**extra))
    assert response.status_code == 422
    assert "gripper" in response.text
    assert inference_ui.seen == []


@pytest.mark.parametrize("policy", ["smolvla", "pi05"])
@pytest.mark.parametrize("backend", ["local", "modal"])
def test_native_only_profiles_cannot_be_launched_for_motion(inference_ui, policy, backend):
    response = inference_ui.client.post("/api/session/rollout", json=payload(
        policy=policy, backend=backend, confirm_motion=True,
    ))
    assert response.status_code == 422
    assert not inference_ui.seen


def test_explicit_live_probe_uses_shared_cli_flags_and_owns_preview_cameras(inference_ui):
    result = inference_ui.client.post("/api/session/policy-probe", json=payload(live=True, confirm_active_read=True))
    assert result.status_code == 200
    status = result.json()
    assert status["mode"] == "policy-probe-live"
    args = inference_ui.seen[-1]
    assert args[0] == "policy-probe"
    assert "--live" in args and "--approve-active-read" in args
    deadline = time.monotonic() + 5
    while "CAMERA_ACQUIRED" not in inference_ui.manager.log and time.monotonic() < deadline:
        time.sleep(.01)
    assert "CAMERA_ACQUIRED" in inference_ui.manager.log
    assert inference_ui.manager.cameras_owned
    assert all(cam["suspended_by"] for cam in inference_ui.client.get("/api/cameras").json())
    assert inference_ui.client.get("/api/cameras/top/stream").status_code == 409


def test_saved_probe_is_distinct_and_does_not_suspend_cameras(inference_ui):
    snapshot = inference_ui.root / "snapshot.npz"
    snapshot.write_bytes(b"validation occurs in the shared CLI runner before readiness")
    result = inference_ui.client.post("/api/session/policy-probe", json=payload(saved="snapshot.npz"))
    assert result.status_code == 200
    assert result.json()["mode"] == "policy-probe"
    args = inference_ui.seen[-1]
    assert "--saved" in args and "--live" not in args
    assert all(cam["suspended_by"] is None for cam in inference_ui.client.get("/api/cameras").json())


def test_duplicate_clicks_and_recording_conflicts_allow_only_one_child(inference_ui):
    first = inference_ui.client.post("/api/session/policy-check", json=payload())
    assert first.status_code == 200
    first_pid = first.json()["pid"]
    assert inference_ui.client.post("/api/session/policy-check", json=payload()).status_code == 409
    assert inference_ui.client.post("/api/session/modal-prepare", json=payload(backend="modal")).status_code == 409
    assert inference_ui.client.post("/api/session/record", json={"name": "test", "task": "test"}).status_code == 409
    assert inference_ui.client.get("/api/session").json()["pid"] == first_pid


def test_physical_modal_performance_gate_blocks_before_launch(inference_ui):
    result = inference_ui.client.post("/api/session/rollout", json=payload(backend="modal", confirm_motion=True))
    assert result.status_code == 422
    assert "Physical Modal rollout BLOCKED" in result.text
    assert not inference_ui.seen
    assert not inference_ui.manager.active


def test_stop_stops_the_local_rollout_child_and_does_not_shutdown_cloud(inference_ui):
    result = inference_ui.client.post("/api/session/rollout", json=payload(backend="local", confirm_motion=True))
    assert result.status_code == 200
    assert inference_ui.manager.active
    stopped = inference_ui.client.post("/api/session/stop")
    assert stopped.status_code == 200
    assert inference_ui.manager.wait(timeout=5) is not None
    assert not inference_ui.manager.active
    assert all(args[0] != "modal-shutdown" for args in inference_ui.seen)
    assert all(cam["suspended_by"] is None for cam in inference_ui.client.get("/api/cameras").json())


def test_preparation_failure_is_finalized_and_new_config_has_new_operation(inference_ui):
    inference_ui.child = "print('preparation failed'); raise SystemExit(2)"
    first = inference_ui.client.post("/api/session/modal-prepare", json=payload(backend="modal"))
    assert first.status_code == 200
    first_meta = first.json()["meta"]
    assert inference_ui.manager.wait(timeout=5) == 2
    history = inference_ui.client.get("/api/deployments").json()
    assert len(history) == 1 and history[0]["status"] == "failed"
    inference_ui.child = "import time; time.sleep(20)"
    second = inference_ui.client.post("/api/session/policy-check", json=payload(task="a changed task"))
    assert second.status_code == 200
    second_meta = second.json()["meta"]
    assert second_meta["operation_id"] != first_meta["operation_id"]
    assert second_meta["profile_key"] != first_meta["profile_key"]
    assert "result" not in second.json()["parsed"]


def test_prior_process_output_must_drain_before_new_operation(inference_ui, monkeypatch):
    manager = inference_ui.manager
    entered, release = threading.Event(), threading.Event()
    original = manager._read_output

    def gated(proc, session, group_gone):
        proc.wait(timeout=5)
        entered.set()
        assert release.wait(timeout=5)
        original(proc, session, group_gone)

    monkeypatch.setattr(manager, "_read_output", gated)
    inference_ui.child = "print('[yamkit-result] {\"operation\": \"old\"}')"
    first = inference_ui.client.post("/api/session/policy-check", json=payload())
    assert first.status_code == 200
    try:
        assert entered.wait(timeout=5)
        assert manager.active  # ownership remains held until output and descendants drain
        response = inference_ui.client.post("/api/session/policy-check", json=payload(task="new"))
        assert response.status_code == 409
        assert manager.meta["operation_id"] == first.json()["meta"]["operation_id"]
    finally:
        release.set()
        manager.wait(timeout=5)


def test_simultaneous_launches_cannot_create_conflicting_children(inference_ui):
    barrier = threading.Barrier(2)

    def launch():
        barrier.wait(timeout=5)
        return inference_ui.client.post("/api/session/policy-check", json=payload()).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(launch) for _ in range(2)]
        assert sorted(future.result(timeout=5) for future in futures) == [200, 409]


@pytest.mark.parametrize("route,extra", [("policy-probe", {"live": True, "confirm_active_read": True}),
                                         ("rollout", {"confirm_motion": True})])
@pytest.mark.parametrize("arms", [["left"], ["right"], ["missing_follower"], []])
def test_incomplete_or_unknown_probe_arm_names_are_validation_errors(inference_ui, route, extra, arms):
    response = inference_ui.client.post(f"/api/session/{route}", json=payload(arms=arms, **extra))
    assert response.status_code == 422
    assert inference_ui.manager._proc is None


def test_preview_that_did_not_release_denies_child_camera_acquisition(inference_ui, monkeypatch):
    from yamkit.ui.camstream import _Camera

    monkeypatch.setattr(_Camera, "running", property(lambda self: True))
    monkeypatch.setattr(_Camera, "stop", lambda *args, **kwargs: None)
    response = inference_ui.client.post("/api/session/policy-probe", json=payload(live=True, confirm_active_read=True))
    assert response.status_code == 200  # launcher may start, acquisition handshake gates cameras
    assert inference_ui.manager.wait(timeout=5) != 0
    assert "CAMERA_ACQUIRED" not in inference_ui.manager.log
    assert any("acquisition denied" in line for line in inference_ui.manager.log)


def test_local_children_exclude_unrelated_secrets(inference_ui, monkeypatch):
    for name in ("YAMKIT_OPENAI_API_KEY", "DATABASE_URL", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"):
        monkeypatch.setenv(name, "SECRET_NEVER_PRINTED")
    inference_ui.child = (
        "import json,os; print('[yamkit-result] '+json.dumps({key:bool(os.getenv(key)) for key in "
        "['YAMKIT_OPENAI_API_KEY','DATABASE_URL','MODAL_TOKEN_ID','MODAL_TOKEN_SECRET']}))"
    )
    response = inference_ui.client.post("/api/session/policy-check", json=payload())
    assert response.status_code == 200
    assert inference_ui.manager.wait(timeout=5) == 0
    result = inference_ui.client.get("/api/session")
    assert "SECRET_NEVER_PRINTED" not in result.text
    assert result.json()["parsed"]["result"] == {
        "YAMKIT_OPENAI_API_KEY": False, "DATABASE_URL": False,
        "MODAL_TOKEN_ID": True, "MODAL_TOKEN_SECRET": True,
    }


@pytest.mark.parametrize("credential", ["MODAL_TOKEN_SECRET", "HF_TOKEN", "YAMKIT_OPENAI_API_KEY", "DATABASE_URL"])
def test_credential_fields_rejected_without_echoing_values(inference_ui, credential):
    response = inference_ui.client.post("/api/session/policy-check", json=payload(**{credential: "SECRET_SENTINEL"}))
    assert response.status_code == 422
    assert "SECRET_SENTINEL" not in response.text
    assert not inference_ui.seen


def test_recording_and_custom_local_rollout_keep_existing_cli_path(inference_ui):
    inference_ui.child = "pass"
    response = inference_ui.client.post("/api/session/record", json={"name": "pick", "task": "pick", "to": "local"})
    assert response.status_code == 200
    assert inference_ui.seen[-1][0] == "record"
    assert inference_ui.manager.wait(timeout=5) == 0
    response = inference_ui.client.post("/api/session/rollout", json=payload(
        policy="outputs/train/custom/pretrained_model", confirm_motion=True, arms=["left"],
    ))
    assert response.status_code == 200
    args = inference_ui.seen[-1]
    assert args[0] == "rollout"
    assert args[args.index("--backend") + 1] == "local"
    assert args[args.index("--arms") + 1] == "left"


@pytest.fixture
def inference_js():
    quickjs = pytest.importorskip("quickjs")
    source = (UI / "app.js").read_text()
    page = source[source.index("pages.inference = {"):source.index("async function renderRunDetail")]
    ctx = quickjs.Context()
    ctx.eval("""
      var pages={}, nodes={}, posts=[], alerts=[], confirmResult=false;
      var session={active:false,meta:{},parsed:{},log:[]};
      function $(id) { return nodes[id] ||= {value:'',checked:false,disabled:false,textContent:'',
        addEventListener:()=>{},innerHTML:''}; }
      var document={getElementById:(id)=>$('#'+id)};
      function pageHead(){return '';} function camsHTML(){return '';} function syncCams(){}
      function esc(s){return s;} function errBanner(s){return s;}
      function confirm(){return confirmResult;} function alert(s){alerts.push(s);}
      function api(path){return Promise.resolve(path==='/inference/profiles'?{profiles:[]}:[]);}
      function post(path,body){posts.push({path,body}); return Promise.resolve({meta:{operation_id:'op1'}});}
      function doPost(path,body){return post(path,body);}
      function refreshSession(){session={active:false,mode:'policy-probe',returncode:0,
        meta:{operation_id:'op1'},parsed:{result:{passed:true}}};return Promise.resolve();}
      $('#inf-policy').value='molmoact2'; $('#inf-task').value='pick';
      $('#inf-backend').value='local'; $('#inf-device').value='cpu'; $('#inf-gpu').value='L40S';
      $('#inf-duration').value='60'; $('#inf-saved').value='data/old.npz';
    """)
    ctx.eval(page)
    ctx.eval("pages.inference.render({innerHTML:''},[])")
    _drain_js(ctx)
    return ctx


def _drain_js(ctx):
    for _ in range(100):
        if not ctx.execute_pending_job():
            return
    pytest.fail("inference page has an unbounded microtask loop")


@pytest.fixture
def attached_browser(inference_js):
    ctx = inference_js
    ctx.eval("""
      var browserNow=1700000000000, deferPreflight=false, resolvePreflight=null;
      Date.now=()=>browserNow;
      var checkedResult={ready:true,reason:'Qualified for this selection',selection_key:'server-selection',
        checked_at:browserNow/1000,expires_at:browserNow/1000+120,modal_app:'yamkit-vla-session-ui-test'};
      post=function(path,body) {
        posts.push({path,body});
        if (path==='/inference/preflight') return deferPreflight
          ? new Promise(resolve=>{resolvePreflight=resolve;}) : Promise.resolve(checkedResult);
        return Promise.resolve({meta:{operation_id:'op1'}});
      };
      pages.inference._profiles=[{id:'molmoact2',mapping_verified:true,
        physical_modal_rollout_allowed:false,physical_modal_rollout_reason:'Host qualification required'}];
      $('#inf-backend').value='modal'; $('#inf-modal-app').value='yamkit-vla-session-ui-test';
      $('#inf-task').value='pick up the orange lid and place it into the black circular container';
      $('#inf-duration').value='5'; $('#inf-mapping').checked=false;
      pages.inference.syncForm();
    """)
    return ctx


def _check_attached_browser(ctx):
    ctx.eval("$('#btn-inf-preflight').onclick()")
    _drain_js(ctx)


def test_browser_attached_start_needs_qualification_for_current_mapping_acceptance(attached_browser):
    ctx = attached_browser
    assert ctx.eval("$('#btn-ro').disabled")
    _check_attached_browser(ctx)
    assert ctx.eval("$('#btn-ro').disabled")  # Evidence never substitutes for mapping acceptance.
    checked = json.loads(ctx.eval("JSON.stringify(posts[0].body)"))
    assert checked["mapping_accepted"] is False
    assert "confirm_motion" not in checked and "supervised_confirmed" not in checked
    ctx.eval("$('#inf-mapping').checked=true; pages.inference.syncForm()")
    assert ctx.eval("$('#btn-ro').disabled")  # The accepted form gets its own fresh check.
    _check_attached_browser(ctx)
    assert ctx.eval("$('#btn-ro').disabled") is False
    assert "Qualified for this selection" in ctx.eval("$('#inf-qualification-status').textContent")


@pytest.mark.parametrize("node,value", [
    ("inf-task", "a changed task"), ("inf-modal-app", "yamkit-vla-other-session"), ("inf-duration", "10"),
])
def test_browser_attached_form_changes_invalidate_qualification(attached_browser, node, value):
    ctx = attached_browser
    ctx.eval("$('#inf-mapping').checked=true")
    _check_attached_browser(ctx)
    assert ctx.eval("$('#btn-ro').disabled") is False
    ctx.eval(f"$({json.dumps('#' + node)}).value={json.dumps(value)}; pages.inference.syncForm()")
    assert ctx.eval("$('#btn-ro').disabled")
    assert "Qualified for this selection" not in ctx.eval("$('#inf-qualification-status').textContent")


def test_browser_attached_trace_selection_invalidates_qualification(attached_browser):
    ctx = attached_browser
    ctx.eval("$('#inf-mapping').checked=true")
    _check_attached_browser(ctx)
    assert ctx.eval("$('#btn-ro').disabled") is False
    ctx.eval("$('#inf-trace').checked=true; pages.inference.syncForm()")
    assert ctx.eval("$('#btn-ro').disabled")
    _check_attached_browser(ctx)
    assert ctx.eval("$('#btn-ro').disabled") is False
    assert json.loads(ctx.eval("JSON.stringify(posts[1].body)"))["capture_trace"] is True


@pytest.mark.parametrize("phase,label,button", [
    ("running", "Policy running — 30 Hz", "Stop and release arms"),
    ("returning_home", "Returning home — keep clear", "Stop and release arms"),
    ("releasing", "Releasing arms", "Stop and release arms"),
    ("released", "Arms released — saving video and joint traces", "Interrupt saving"),
])
def test_browser_managed_rollout_reports_motion_and_export_separately(attached_browser, phase, label, button):
    ctx = attached_browser
    state = {"active": True, "mode": "rollout", "meta": {"capture_trace": True, "task": "pick"},
             "parsed": {"rollout_phase": phase}, "log": []}
    ctx.eval(f"session={json.dumps(state)}; pages.inference.syncForm()")
    assert label in ctx.eval("$('#inf-status').textContent")
    assert ctx.eval("$('#btn-inf-stop').textContent") == button
    assert not ctx.eval("$('#btn-inf-stop').disabled")
    assert ctx.eval("$('#btn-ro').disabled")


def test_browser_attached_start_reserves_startup_duration_and_return_home(attached_browser):
    ctx = attached_browser
    ctx.eval("$('#inf-mapping').checked=true")
    _check_attached_browser(ctx)
    assert ctx.eval("$('#btn-ro').disabled") is False
    ctx.eval("browserNow+=54999; pages.inference.syncForm()")
    assert ctx.eval("$('#btn-ro').disabled") is False
    ctx.eval("browserNow+=1; pages.inference.syncForm()")
    assert ctx.eval("$('#btn-ro').disabled")
    assert "expires too soon" in ctx.eval("$('#inf-qualification-status').textContent")
    ctx.eval("confirmResult=true; $('#btn-ro').onclick({target:$('#btn-ro')})")
    _drain_js(ctx)
    assert ctx.eval("posts.filter(p=>p.path==='/session/rollout').length") == 0


def test_browser_attached_late_preflight_response_cannot_validate_changed_task(attached_browser):
    ctx = attached_browser
    ctx.eval("$('#inf-mapping').checked=true; deferPreflight=true")
    _check_attached_browser(ctx)
    assert ctx.eval("pages.inference._checking")
    ctx.eval("$('#inf-task').value='a new task'; pages.inference.syncForm(); resolvePreflight(checkedResult)")
    _drain_js(ctx)
    assert ctx.eval("pages.inference._checking") is False
    assert ctx.eval("$('#btn-ro').disabled")
    assert ctx.eval("pages.inference._qualification === null")
    assert "Qualified for this selection" not in ctx.eval("$('#inf-qualification-status').textContent")


def test_browser_attached_slow_preflight_response_does_not_extend_session_lifetime(attached_browser):
    ctx = attached_browser
    ctx.eval("$('#inf-mapping').checked=true; deferPreflight=true")
    _check_attached_browser(ctx)
    ctx.eval("browserNow+=86000; resolvePreflight(checkedResult)")
    _drain_js(ctx)
    assert ctx.eval("$('#btn-ro').disabled")
    assert "expires too soon" in ctx.eval("$('#inf-qualification-status').textContent")


def test_browser_attached_supervised_flags_are_sent_only_after_confirmed_start(attached_browser):
    ctx = attached_browser
    ctx.eval("$('#inf-mapping').checked=true")
    _check_attached_browser(ctx)
    ctx.eval("$('#btn-ro').onclick({target:$('#btn-ro')})")
    _drain_js(ctx)
    assert ctx.eval("posts.filter(p=>p.path==='/session/rollout').length") == 0
    assert all("supervised_confirmed" not in post["body"] for post in json.loads(ctx.eval("JSON.stringify(posts)")))
    ctx.eval("confirmResult=true; $('#btn-ro').onclick({target:$('#btn-ro')})")
    _drain_js(ctx)
    runs = json.loads(ctx.eval("JSON.stringify(posts.filter(p=>p.path==='/session/rollout'))"))
    assert len(runs) == 1
    body = runs[0]["body"]
    assert body["confirm_motion"] is True
    assert body["supervised_confirmed"] is True
    assert body["mapping_accepted"] is True
    assert body["modal_app"] == "yamkit-vla-session-ui-test"
    assert body["call_mode"] == "http" and body["execution_mode"] == "cuda_graph10"


@pytest.mark.parametrize("button,flag", [("btn-ro", "confirm_motion"), ("btn-probe-live", "confirm_active_read")])
def test_browser_confirmation_cancel_never_posts_and_accept_is_explicit(inference_js, button, flag):
    ctx = inference_js
    ctx.eval(f"$('#{button}').onclick({{target:$('#{button}')}})")
    assert ctx.eval("posts.length") == 0
    ctx.eval(f"confirmResult=true; $('#{button}').onclick({{target:$('#{button}')}})")
    _drain_js(ctx)
    assert json.loads(ctx.eval("JSON.stringify(posts[0].body)"))[flag] is True


def test_browser_duplicate_clicks_only_post_once(inference_js):
    ctx = inference_js
    ctx.eval("$('#btn-pc').onclick({target:$('#btn-pc')}); $('#btn-pc').onclick({target:$('#btn-pc')})")
    assert ctx.eval("posts.length") == 1
    _drain_js(ctx)


def test_browser_changed_config_or_wrong_operation_hides_old_completion(inference_js):
    ctx = inference_js
    ctx.eval("$('#btn-pc').onclick({target:$('#btn-pc')})")
    _drain_js(ctx)
    assert "passed" in ctx.eval("$('#inf-result').textContent")
    ctx.eval("session.meta.operation_id='old-op';pages.inference.syncForm()")
    assert ctx.eval("$('#inf-result').textContent") == ""
    ctx.eval("session.meta.operation_id='op1';$('#inf-task').value='changed';pages.inference.syncForm()")
    assert ctx.eval("$('#inf-result').textContent") == ""


def test_browser_changed_snapshot_invalidates_prior_probe_result(inference_js):
    ctx = inference_js
    ctx.eval("$('#btn-probe-saved').onclick({target:$('#btn-probe-saved')})")
    _drain_js(ctx)
    assert "passed" in ctx.eval("$('#inf-result').textContent")
    ctx.eval("$('#inf-saved').value='data/new.npz'; pages.inference.syncForm()")
    assert ctx.eval("$('#inf-result').textContent") == ""


def test_spawn_failure_releases_preview_ownership(inference_ui, monkeypatch):
    import subprocess

    def fail_spawn(*args, **kwargs):
        raise OSError('missing interpreter')

    monkeypatch.setattr(subprocess, 'Popen', fail_spawn)
    response = inference_ui.client.post('/api/session/policy-probe',
                                        json=payload(live=True, confirm_active_read=True))
    assert response.status_code == 409
    assert not inference_ui.manager.active
    assert all(camera['suspended_by'] is None for camera in inference_ui.client.get('/api/cameras').json())


def test_custom_local_check_forwards_selected_arm(inference_ui):
    inference_ui.child = 'pass'
    response = inference_ui.client.post('/api/session/policy-check', json=payload(
        policy='outputs/custom', backend='local', arms=['left']))
    assert response.status_code == 200
    argv = inference_ui.seen[-1]
    assert argv[argv.index('--arms') + 1] == 'left'


def test_browser_modal_start_stays_disabled_after_ready_result(inference_js):
    ctx = inference_js
    ctx.eval("""
        pages.inference._profiles=[{id:'molmoact2',mapping_verified:true,
          physical_modal_rollout_allowed:false,physical_modal_rollout_reason:'Physical Modal rollout BLOCKED'}];
        $('#inf-backend').value='modal';
        session={active:false,meta:{},parsed:{result:{ready:true}}};
        pages.inference.syncForm();
    """)
    assert ctx.eval("$('#btn-ro').disabled")
    assert "BLOCKED" in ctx.eval("$('#inf-profile-note').textContent")
