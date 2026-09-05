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
    from yamkit.inference.service import ModelRuntime
    from yamkit.ui.camstream import _Camera

    def forbidden(*args, **kwargs):
        pytest.fail("UI opened hardware, loaded weights or contacted a paid service")

    monkeypatch.setattr(arm.YamArm, "connect", forbidden)
    monkeypatch.setattr(ModelRuntime, "load", forbidden)
    monkeypatch.setattr(modal_ops, "prepare", forbidden)
    monkeypatch.setattr(modal_ops, "service_handle", forbidden)
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
        if args[0] == "policy-probe" and "--live" in args:
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
