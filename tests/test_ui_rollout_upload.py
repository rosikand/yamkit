"""Post-run Hub export must remain independent of hardware ownership and Stop."""

import json
import threading
import time
from pathlib import Path

import pytest

from tests.test_inference_ui import _drain_js, attached_payload
from tests.test_inference_ui import attached_browser as _attached_browser
from tests.test_inference_ui import attached_modal as _attached_modal
from tests.test_inference_ui import inference_js as _inference_js
from tests.test_inference_ui import inference_ui as _inference_ui
from yamkit import rollout_artifacts
from yamkit.ui import server
from yamkit.ui.sessions import _group_alive

inference_ui = _inference_ui
attached_modal = _attached_modal
inference_js = _inference_js
attached_browser = _attached_browser


@pytest.fixture
def upload_trial(attached_modal):
    state = attached_modal
    state.expected_override["task"] = server.TRACE_TASK
    scripts = state.ui.root / "scripts"
    scripts.mkdir()
    (scripts / "trace_rollout.py").write_text('''
import argparse,json,os,pathlib,subprocess,sys,time
from yamkit.camera_ownership import claim_from_env
p=argparse.ArgumentParser()
p.add_argument('--run',action='store_true');p.add_argument('--duration',type=int)
p.add_argument('--modal-app');p.add_argument('--rig');p.add_argument('--output-dir')
p.add_argument('--backend',choices=['modal','external'],default='modal')
p.add_argument('--confirm-supervised',action='store_true')
a=p.parse_args()
lease=claim_from_env(['top','left_wrist','right_wrist'])
root=pathlib.Path(a.rig).parent
runs=list((root/'outputs/ui/deployments').glob('*/run_metadata.json'))
assert len(runs)==1
assert json.loads(runs[0].read_text())['provenance']['kind']=='before_managed_child_launch'
print('ARCHIVE_FIRST '+os.environ.get('HF_TOKEN',''),flush=True)
for i in range(650): print('line-'+str(i),flush=True)
lease.release()
d=pathlib.Path(a.output_dir);d.mkdir(parents=True)
(d/'summary.json').write_text(json.dumps({'resources_released':True}))
(d/'trace.json').write_text('{"events":[]}')
(d/'metrics.json').write_text('{"executed_actions":132}')
print('ARCHIVE_LAST',flush=True)
# The parent must reap this leftover process group before scheduling uploads.
subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
''')
    return state


def launch(ui, **extra):
    return ui.client.post("/api/session/rollout", json=attached_payload(
        task=server.TRACE_TASK, confirm_motion=True, mapping_accepted=True,
        supervised_confirmed=True, **extra))


def await_status(ui, run_id, status):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = ui.client.get(f"/api/deployments/{run_id}").json().get("upload")
        if value and value["status"] == status:
            return value
        time.sleep(.01)
    pytest.fail(f"upload never reached {status}")


def test_upload_starts_after_release_and_finalization_without_blocking_stop_or_next_run(upload_trial, monkeypatch):
    ui = upload_trial.ui
    entered, release = threading.Event(), threading.Event()
    calls = []
    monkeypatch.setenv("HF_TOKEN", "hf_PRIVATE_UPLOAD_FIXTURE_SECRET")

    def package(run_dir, *, trace_dir):
        calls.append((run_dir, trace_dir))
        assert not ui.manager.active and not ui.manager.cameras_owned
        assert not _group_alive(ui.manager._proc.pid)
        assert not ui.manager._lock._is_owned()
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["ended_at"] and meta["returncode"] == 0 and meta["log_complete"]
        assert (run_dir / "summary.json").is_file()
        log = (run_dir / "log.txt").read_text()
        assert "ARCHIVE_FIRST [redacted]" in log and "ARCHIVE_LAST" in log
        assert all(f"line-{index}\n" in log for index in range(650))
        assert "hf_PRIVATE_UPLOAD_FIXTURE_SECRET" not in log
        assert "ARCHIVE_FIRST" not in "\n".join(ui.manager.log)
        snapshot = json.loads((run_dir / "run_metadata.json").read_text())
        assert snapshot["capture"]["log_complete"]
        assert snapshot["model"]["revision"]
        assert "can_serial" not in json.dumps(snapshot) and "notes" not in json.dumps(snapshot)
        assert "http_endpoint" not in json.dumps(snapshot) and "modal_app" not in json.dumps(snapshot)
        entered.set()
        assert release.wait(timeout=5)
        return run_dir / "bundle"

    def upload(bundle_dir, *, repo_id):
        assert len(calls) == 1 and bundle_dir == calls[0][0] / "bundle"
        return {"status": "uploaded", "repo_id": repo_id, "revision": "immutable-commit",
                "url": f"https://huggingface.co/datasets/{repo_id}/tree/main/runs/{bundle_dir.parent.name}"}

    monkeypatch.setattr(rollout_artifacts, "package_rollout", package)
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", upload)
    response = launch(ui, upload_repo_id="owner/private-rollouts")
    assert response.status_code == 200, response.text
    assert response.json()["meta"]["capture_trace"]  # Upload implies complete capture.
    try:
        assert entered.wait(timeout=8)
        assert ui.manager.wait(timeout=2) == 0
        run_id = ui.client.get("/api/deployments").json()[0]["id"]
        started = time.monotonic()
        assert ui.client.post("/api/session/stop").status_code == 200
        assert time.monotonic() - started < .5
        ui.child = "print('next-run')"
        assert ui.client.post("/api/session/policy-check", json={"policy": "smolvla"}).status_code == 200
        assert ui.manager.wait(timeout=3) == 0
        assert json.loads((calls[0][0] / "meta.json").read_text())["kind"] == "rollout"
    finally:
        release.set()
    result = await_status(ui, run_id, "uploaded")
    assert result["revision"] == "immutable-commit"
    assert len(calls) == 1
    # Durable status is a local record, independent of the live session ring.
    assert json.loads((calls[0][0] / "hf-upload.json").read_text())["status"] == "uploaded"


def test_upload_failure_retains_local_artifacts_and_hides_exception_credentials(upload_trial, monkeypatch):
    ui = upload_trial.ui
    monkeypatch.setattr(rollout_artifacts, "package_rollout", lambda run_dir, **_: run_dir / "bundle")

    def fail(*args, **kwargs):
        raise RuntimeError("secret hf_PRIVATE_TOKEN endpoint https://private.modal.host")

    monkeypatch.setattr(rollout_artifacts, "upload_rollout", fail)
    response = launch(ui, upload_repo_id="owner/private-rollouts")
    assert response.status_code == 200, response.text
    ui.manager.wait(timeout=8)
    run_id = ui.client.get("/api/deployments").json()[0]["id"]
    result = await_status(ui, run_id, "failed")
    assert "PRIVATE" not in json.dumps(result) and "modal.host" not in json.dumps(result)
    assert result["retry_command"].startswith("yamkit bundle-rollout ")
    run_dir = ui.root / "outputs/ui/deployments" / run_id
    assert (run_dir / "summary.json").is_file() and (run_dir / "log.txt").is_file()
    assert Path(response.json()["meta"]["debug_trace_dir"]).is_dir()
    assert not ui.manager.active


def test_no_upload_without_per_run_opt_in(upload_trial, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("upload was not selected")

    monkeypatch.setattr(rollout_artifacts, "package_rollout", forbidden)
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", forbidden)
    assert launch(upload_trial.ui, capture_trace=True).status_code == 200
    assert upload_trial.ui.manager.wait(timeout=8) == 0
    assert upload_trial.ui.client.get("/api/deployments").json()[0]["upload"] is None


@pytest.mark.parametrize("repo_id", ["bad", "../outside", "owner/repo/extra", "https://huggingface.co/owner/repo", ""])
def test_invalid_destinations_are_rejected_before_child(attached_modal, repo_id):
    result = launch(attached_modal.ui, upload_repo_id=repo_id)
    assert result.status_code == 422
    assert attached_modal.ui.manager._proc is None


@pytest.mark.parametrize("duration", [31, 46, 61])
def test_upload_requires_supported_complete_capture(attached_modal, duration):
    attached_modal.expected_override["task"] = server.TRACE_TASK
    result = launch(attached_modal.ui, upload_repo_id="owner/private-rollouts", duration=duration)
    assert result.status_code == 422
    assert "5, 10, 20, 30, 45 or 60" in result.text
    assert attached_modal.ui.manager._proc is None


@pytest.mark.parametrize("duration", [30, 45, 60])
def test_upload_capture_keeps_supported_duration_in_trace_command(attached_modal, monkeypatch, duration):
    state = attached_modal
    state.expected_override["task"] = server.TRACE_TASK
    launched = []

    def start(mode, argv, meta):
        launched.append((mode, argv, meta))
        return {"active": True, "mode": mode, "meta": meta}

    monkeypatch.setattr(state.ui.manager, "start", start)
    result = launch(state.ui, upload_repo_id="owner/private-rollouts", duration=duration)
    assert result.status_code == 200, result.text
    mode, argv, meta = launched[0]
    assert mode == "rollout" and Path(argv[1]).name == "trace_rollout.py"
    assert argv[argv.index("--duration") + 1] == str(duration)
    assert meta["capture_trace"] and meta["upload_repo_id"] == "owner/private-rollouts"
    assert state.ui.manager._proc is None  # The test never starts a control child.


def test_browser_upload_implies_capture_and_can_be_opted_out(attached_browser):
    ctx = attached_browser
    ctx.eval("$('#inf-upload').checked=true; $('#inf-upload-repo').value='owner/private-rollouts'; pages.inference.syncForm()")
    selection = json.loads(ctx.eval("JSON.stringify(pages.inference.selection())"))
    assert selection["upload_repo_id"] == "owner/private-rollouts" and selection["capture_trace"]
    assert ctx.eval("$('#inf-trace').disabled")
    ctx.eval("$('#inf-upload').checked=false; pages.inference.syncForm()")
    assert json.loads(ctx.eval("JSON.stringify(pages.inference.selection())"))["upload_repo_id"] is None
    assert not ctx.eval("$('#inf-trace').disabled")


@pytest.mark.parametrize("pending", ["queued", "packaging", "uploading"])
def test_restart_marks_pending_upload_interrupted_without_retry(inference_ui, monkeypatch, pending):
    ui = inference_ui
    run_dir = ui.root / "outputs/ui/deployments/old-finalized-run"
    run_dir.mkdir(parents=True)
    receipt = run_dir / "hf-upload.json"
    receipt.write_text(json.dumps({"status": pending, "repo_id": "owner/private-rollouts",
                                   "retry_command": "yamkit bundle-rollout old-finalized-run --trace-dir saved-trace"}))

    def forbidden(*args, **kwargs):
        pytest.fail("Restart must not upload or launch any process")

    monkeypatch.setattr(rollout_artifacts, "package_rollout", forbidden)
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", forbidden)
    monkeypatch.setattr(ui.manager, "start", forbidden)
    server.create_app(ui.rig.path, outputs_dir=ui.root / "outputs", session_manager=ui.manager)
    status = json.loads(receipt.read_text())
    assert status["status"] == "interrupted" and "--trace-dir saved-trace" in status["retry_command"]
    assert not ui.manager.active


def test_saved_rollout_destination_is_returned_without_enabling_api_upload(inference_ui):
    ui = inference_ui
    ui.rig.hub.rollout_repo = "owner/private-rollouts"
    ui.rig.save()
    assert ui.client.get("/api/inference/profiles").json()["rollout_repo"] == "owner/private-rollouts"
    assert ui.manager._proc is None



def test_failed_child_launch_finalizes_once_and_keeps_original_snapshot(upload_trial, monkeypatch):
    from yamkit.ui import sessions

    ui = upload_trial.ui
    finalized = []
    original = sessions.DeploymentLog.finalize

    def record_finalization(self, run_dir, status):
        finalized.append(status)
        original(self, run_dir, status)

    def fail_launch(*args, **kwargs):
        raise OSError("fixture: executable missing")

    monkeypatch.setattr(sessions.DeploymentLog, "finalize", record_finalization)
    monkeypatch.setattr(sessions.subprocess, "Popen", fail_launch)
    monkeypatch.setattr(rollout_artifacts, "package_rollout", lambda run_dir, **_: run_dir / "bundle")
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", lambda *_, **kwargs: {"status": "uploaded", **kwargs})
    result = launch(ui, upload_repo_id="owner/private-rollouts")
    assert result.status_code == 409
    run = ui.client.get("/api/deployments").json()[0]
    await_status(ui, run["id"], "uploaded")
    assert len(finalized) == 1
    assert run["started_at"] is not None and run["ended_at"] is not None
    assert run["returncode"] == -1 and run["status"] == "failed"
    assert ui.manager._proc is None



def test_runs_list_keeps_polling_while_child_exit_artifacts_are_finalizing(attached_browser):
    ctx = attached_browser
    ctx.eval("""
      function st(){return '';} function fmtDur(){return '';}
      api=()=>Promise.resolve([{id:'finishing-run',status:'running',upload:null}]);
      pages.inference.refreshList();
    """)
    _drain_js(ctx)
    assert ctx.eval("pages.inference._uploadsPending")
    ctx.eval("""
      api=()=>Promise.resolve([{id:'finishing-run',status:'success',upload:{status:'uploaded'}}]);
      pages.inference.refreshList();
    """)
    _drain_js(ctx)
    assert not ctx.eval("pages.inference._uploadsPending")
