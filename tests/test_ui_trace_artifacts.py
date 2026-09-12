"""Managed debug wrapper launch and artifact import using a harmless fake script."""

import json
from pathlib import Path

import pytest

from tests.test_inference_ui import attached_modal as _attached_modal
from tests.test_inference_ui import attached_payload
from tests.test_inference_ui import inference_ui as _inference_ui
from yamkit.ui import server

attached_modal = _attached_modal
inference_ui = _inference_ui
TASK = "put the red cube into the green bowl"


@pytest.mark.parametrize("failure", [None, "symlink", "copy"])
@pytest.mark.parametrize("duration", [5, 20, 30])
def test_managed_trace_exports_only_whitelisted_regular_files_after_exit(attached_modal, failure, duration, monkeypatch):
    state = attached_modal
    ui = state.ui
    state.expected_override["task"] = TASK
    if failure == "copy":
        def fail_copy(*args):
            raise OSError("simulated artifact copy failure")
        monkeypatch.setattr(server.shutil, "copyfile", fail_copy)
    scripts = ui.root / "scripts"
    scripts.mkdir()
    # This substitutes the actual tracked wrapper. No robot or camera constructor occurs.
    (scripts / "trace_rollout.py").write_text('''
import argparse,json,pathlib,time
from yamkit.camera_ownership import claim_from_env
p=argparse.ArgumentParser()
p.add_argument('--run',action='store_true');p.add_argument('--duration',type=int)
p.add_argument('--modal-app');p.add_argument('--rig');p.add_argument('--output-dir')
p.add_argument('--task')
p.add_argument('--backend',choices=['modal','external'],default='modal')
p.add_argument('--confirm-supervised',action='store_true')
a=p.parse_args()
assert a.run and a.confirm_supervised and a.duration in (5,20,30) and a.backend=='modal'
assert pathlib.Path(a.rig).is_file()
lease=claim_from_env(['top','left_wrist','right_wrist'])
print('FAKE_TRACE_CAMERA_ACQUIRED',flush=True)
time.sleep(.05)
lease.release()
d=pathlib.Path(a.output_dir)
d.mkdir(parents=True)
(d/'summary.json').write_text(json.dumps({'resources_released':True,'rig':a.rig}))
(d/'trace.json').write_text('{"events":[]}')
(d/'metrics.json').write_text('{"executed_actions":132}')
(d/'video_timeline.json').write_text('{"nominal_fps":30}')
(d/'top.mp4').write_bytes(b'fake video fixture')
(d/'report.html').write_text('<html>fake report</html>')
(d/'unexpected-secret.json').write_text('PRIVATE_FILE_MUST_NOT_IMPORT')
(d/'frame_timestamps.json').symlink_to(d/'unexpected-secret.json')
''' + ('''
renamed=d.with_name(d.name+'-target')
d.rename(renamed)
d.symlink_to(renamed,target_is_directory=True)
''' if failure == "symlink" else ""))
    response = ui.client.post("/api/session/rollout", json=attached_payload(
        task=TASK, capture_trace=True, confirm_motion=True,
        duration=duration,
        mapping_accepted=True, supervised_confirmed=True,
    ))
    assert response.status_code == 200, response.text
    status = response.json()
    assert status["meta"]["capture_trace"] is True
    trace_dir = Path(status["meta"]["debug_trace_dir"])
    assert trace_dir.parent == ui.root / ".context" / "rollout-traces"
    assert len(trace_dir.name) == 32
    assert ui.manager.wait(timeout=5) == 0
    assert "FAKE_TRACE_CAMERA_ACQUIRED" in ui.manager.log
    assert not ui.manager.cameras_owned
    runs = ui.client.get("/api/deployments").json()
    assert len(runs) == 1 and runs[0]["status"] == "success"
    run_id = runs[0]["id"]
    detail = ui.client.get(f"/api/deployments/{run_id}").json()
    if failure:
        assert detail["artifacts"] == []
        assert detail["videos"] == []
        assert detail["log"]
        if failure == "copy":
            assert any("artifact import incomplete" in line for line in detail["log"])
        return
    assert detail["artifacts"] == ["metrics.json", "report.html", "summary.json", "trace.json", "video_timeline.json"]
    assert detail["videos"] == ["top.mp4"]
    artifact = ui.client.get(f"/api/deployments/{run_id}/artifact/summary.json")
    assert artifact.status_code == 200
    assert artifact.json() == {"resources_released": True, "rig": str(ui.rig.path)}
    assert ui.client.get(f"/api/deployments/{run_id}/artifact/video_timeline.json").json() == {"nominal_fps": 30}
    assert ui.client.get(f"/api/deployments/{run_id}/video/top.mp4").content == b"fake video fixture"
    report = ui.client.get(f"/api/deployments/{run_id}/artifact/report.html")
    assert "sandbox" in report.headers["content-security-policy"]
    for denied in ("unexpected-secret.json", "frame_timestamps.json", "log.txt"):
        assert ui.client.get(f"/api/deployments/{run_id}/artifact/{denied}").status_code == 404
    run_dir = ui.root / "outputs" / "ui" / "deployments" / run_id
    (run_dir / "trace.json").unlink()
    (run_dir / "trace.json").symlink_to(trace_dir / "unexpected-secret.json")
    assert ui.client.get(f"/api/deployments/{run_id}/artifact/trace.json").status_code == 404
    assert "PRIVATE_FILE_MUST_NOT_IMPORT" not in json.dumps(detail)


def test_http_attachment_cannot_prepare_a_new_gpu(inference_ui):
    result = inference_ui.client.post("/api/session/modal-prepare", json=attached_payload())
    assert result.status_code == 422
    assert "Conductor" in result.text
    assert not inference_ui.seen
