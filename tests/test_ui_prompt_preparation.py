"""UI-owned prompt preparation is bounded software work, never implicit physical approval."""

import dataclasses
import importlib.util
import json
import time
from pathlib import Path

import pytest

from tests.test_external_ops import NAME, TOKEN, attach
from tests.test_external_ops import attachment as _attachment
from tests.test_http_runtime_binding import runtime_metadata
from tests.test_inference_ui import inference_ui as _inference_ui
from yamkit import external_ops, modal_qualification, paths
from yamkit.deployment import InferenceOptions
from yamkit.inference import qualification
from yamkit.ui import server
from yamkit.ui.sessions import HARDWARE_MODES, parse_line

attachment = _attachment
inference_ui = _inference_ui
NEW_TASK = "pick the red cube out of the green browl and place it on the table"
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_inference_prompt.py"
spec = importlib.util.spec_from_file_location("prepare_inference_prompt", SCRIPT)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


@pytest.fixture
def prompt_ui(inference_ui, attachment, monkeypatch):
    attachment.metadata["graph_warmup"] = runtime_metadata(image_hw=(480, 640))["graph_warmup"]
    receipt = attach(attachment)
    monkeypatch.setattr(qualification, "is_cloud_host", lambda: False)
    monkeypatch.setattr(qualification, "DATA_DIR", inference_ui.root / "data")
    monkeypatch.setattr(paths, "ROOT", inference_ui.root)
    monkeypatch.setattr(qualification, "validate_qualification", lambda *_args, **_kwargs: {
        "created_unix_s": time.time(), "assessment": {"qualified": True}})
    inference_ui.receipt = receipt
    inference_ui.old_task = receipt["metadata"]["graph_warmup"]["signature"]["task"]
    return inference_ui


def body(task, **extra):
    return {"policy": "molmoact2", "task": task, "backend": "external", "external_service": NAME,
            "call_mode": "http", "execution_mode": "cuda_graph10", "image_encoding": "rgb8",
            "controller_mode": "reference", "arms": ["left_follower", "right_follower"], "duration": 20,
            "mapping_accepted": False, "supervised_confirmed": False, **extra}


def fake_child(ui, code):
    directory = ui.root / "scripts"
    directory.mkdir(exist_ok=True)
    (directory / "prepare_inference_prompt.py").write_text(code)


def test_existing_ready_task_is_reused_without_child_or_gpu(prompt_ui, monkeypatch):
    ui = prompt_ui
    monkeypatch.setattr(server, "_prompt_preparation_context", lambda *_: pytest.fail("Ready settings should skip extra preparation checks"))
    assert ui.client.post("/api/inference/preflight", json=body(ui.old_task)).json()["ready"]
    response = ui.client.post("/api/inference/prepare", json=body(ui.old_task))
    assert response.status_code == 200, response.text
    assert response.json()["ready"] and response.json()["reused"]
    assert not response.json()["preparing"] and not ui.manager.active and not ui.seen
    assert not (ui.root / ".context/inference-preparation").exists()


def test_successful_preparation_child_does_not_launch_physical_work(prompt_ui):
    ui = prompt_ui
    fake_child(ui, """
import json,sys
request=json.load(open(sys.argv[1]))
print('[yamkit-prepare] ready',flush=True)
print('[yamkit-result] '+json.dumps({'ready':True,'hardware_tested':False,
      'selection_key':request['expected']['selection_key']}),flush=True)
""")
    response = ui.client.post("/api/inference/prepare", json=body(NEW_TASK))
    assert response.status_code == 200
    assert ui.manager.wait(timeout=5) == 0
    final = ui.client.get("/api/session").json()
    assert final["mode"] == "inference-prepare" and not final["active"] and not final["cameras_owned"]
    assert final["parsed"]["result"]["ready"] and not final["parsed"]["result"]["hardware_tested"]
    assert final["parsed"]["result"]["selection_key"] == response.json()["selection_key"]
    assert not ui.seen  # No fallback CLI rollout or automatic new SessionManager launch.


def test_changed_prompt_is_managed_nonhardware_work_with_no_approval(prompt_ui):
    ui = prompt_ui
    fake_child(ui, "import time; print('[yamkit-prepare] warming_and_qualifying',flush=True); time.sleep(30)")
    preflight = ui.client.post("/api/inference/preflight", json=body(NEW_TASK, mapping_accepted=True)).json()
    assert not preflight["ready"] and preflight["can_prepare"]
    response = ui.client.post("/api/inference/prepare", json=body(NEW_TASK))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["preparing"] and not result["ready"]
    assert result["session"]["mode"] == "inference-prepare" and "inference-prepare" not in HARDWARE_MODES
    meta = result["session"]["meta"]
    assert meta["task"] == NEW_TASK and not meta["mapping_accepted"] and not meta["supervised_confirmed"]
    assert meta["profile_key"] == result["selection_key"]
    assert meta["profile_key"] != preflight["selection_key"]  # Existing form approval is not carried forward.
    request = json.loads((Path(meta["preparation_dir"]) / "request.json").read_text())
    assert request["options"]["task"] == NEW_TASK and TOKEN not in json.dumps(request)
    assert not ui.manager.cameras_owned
    assert ui.client.post("/api/inference/prepare", json=body(NEW_TASK)).status_code == 409
    stopped = ui.client.post("/api/session/stop").json()
    assert stopped["stop_requested"]
    ui.manager.wait(timeout=5)
    assert not ui.manager.active and not ui.manager.cameras_owned and not ui.seen
    assert ui.client.get("/api/session").json()["mode"] == "inference-prepare"


@pytest.mark.parametrize("extra", [
    {"mapping_accepted": True}, {"supervised_confirmed": True}, {"confirm_motion": True},
    {"controller_mode": "async"}, {"center_crop": True}, {"image_encoding": "jpeg"},
    {"prediction_queue_threshold": 30}, {"jpeg_quality": 84}, {"fps": 15},
    {"external_service": "other-service"},
])
def test_preparation_rejects_motion_approval_and_nonfrozen_settings_before_child(prompt_ui, extra):
    response = prompt_ui.client.post("/api/inference/prepare", json=body(NEW_TASK, **extra))
    assert response.status_code == 422
    assert not prompt_ui.manager.active and prompt_ui.manager._proc is None


@pytest.mark.parametrize("defect", ["memory", "shape", "source", "expiry", "active"])
def test_nonqualification_blockers_are_not_silently_repaired(prompt_ui, monkeypatch, defect):
    ui = prompt_ui
    request = body(NEW_TASK)
    if defect == "memory":
        request["capture_trace"] = True
        monkeypatch.setattr(server, "_capture_memory_preflight", lambda _duration: {"admission_passes": False})
    elif defect == "shape":
        ui.rig.cameras["top"]["width"] = 320
        ui.rig.save()
    elif defect in ("source", "expiry"):
        receipt = external_ops.owned_service(NAME)
        if defect == "source":
            receipt["metadata"]["inference_build_id"] = "different-build"
        else:
            receipt["metadata"]["http_session_expires_at"] = time.time() - 1
        external_ops._save(external_ops._directory(NAME) / "receipt.json", receipt)
    else:
        ui.manager.start("rollout", [ui.manager._python, "-c", "import time; time.sleep(30)"])
    preflight = ui.client.post("/api/inference/preflight", json=request).json()
    assert not preflight["ready"] and not preflight["can_prepare"]
    response = ui.client.post("/api/inference/prepare", json=request)
    assert response.status_code == (409 if defect == "active" else 422)
    assert not (ui.root / ".context/inference-preparation").exists()


def helper_request(ui):
    values = body(NEW_TASK)
    options = InferenceOptions(**{**values, "arms": tuple(values["arms"]), "rig_path": str(ui.rig.path)})
    directory = ui.root / ".context/inference-preparation" / ("a" * 32)
    directory.mkdir(parents=True)
    request = directory / "request.json"
    request.write_text(json.dumps({"options": dataclasses.asdict(options), "capture_trace": False,
                                   "expected": server._prompt_preparation_context(options, ui.rig)}))
    old = {"settings": {"profile": "molmoact2", "backend": "external", "external_service_name": NAME,
                         "controller_mode": "reference"}, "assessment": {"qualified": True}, "original": "preserve me"}
    qualification.save_qualification(old)
    return request, old


def test_helper_preserves_old_evidence_exact_task_and_fixed_collection_settings(prompt_ui, monkeypatch, capsys):
    request, old = helper_request(prompt_ui)
    calls = []

    def collect(policy, **kwargs):
        calls.append((policy, kwargs))
        assert json.loads((request.parent / "previous-qualification.json").read_text()) == old
        assert (request.parent / "previous-attachment.json").exists()
        print(TOKEN)  # An optional library diagnostic cannot leak the external bearer.
        receipt = external_ops.owned_service(NAME)
        metadata = receipt["metadata"]
        metadata["graph_warmup"] = runtime_metadata(task=kwargs["task"], image_hw=(480, 640))["graph_warmup"]
        external_ops.update_ready(NAME, metadata, expected_instance_id=metadata["instance_id"])
        record = {"settings": old["settings"], "assessment": {"qualified": True}, "hardware_tested": False}
        qualification.save_qualification(record)
        return record

    monkeypatch.setattr(modal_qualification, "collect_qualification", collect)
    assert helper.execute(request) == 0
    result = json.loads((request.parent / "result.json").read_text())
    assert result["ready"] and result["hardware_tested"] is False
    assert result["selection_key"] == json.loads(request.read_text())["expected"]["selection_key"]
    assert calls == [("molmoact2", {"requests": 50, "rig_path": prompt_ui.rig.path, "backend": "external",
                                  "external_service": NAME, "image_encoding": "rgb8", "jpeg_quality": 85,
                                  "call_mode": "http", "center_crop": False, "prediction_queue_threshold": None,
                                  "execution_mode": "cuda_graph10", "controller_mode": "reference", "task": NEW_TASK})]
    output = capsys.readouterr()
    assert TOKEN not in output.out + output.err and "[REDACTED_SECRET]" in output.out
    assert "[yamkit-prepare] ready" in output.out and len(output.out) < 3000
    assert not prompt_ui.manager.active and not prompt_ui.manager.cameras_owned


@pytest.mark.parametrize("failure", ["assessment", "exception", "cancelled", "changed_instance"])
def test_helper_failure_or_cancellation_never_reports_ready_or_runs_hardware(prompt_ui, monkeypatch, capsys, failure):
    request, _old = helper_request(prompt_ui)
    called = []

    def collect(*args, **kwargs):
        called.append(True)
        if failure == "exception":
            raise ValueError("diagnostic " + TOKEN)
        if failure == "cancelled":
            raise KeyboardInterrupt
        return {"hardware_tested": False, "assessment": {"qualified": False, "reasons": ["fixture failed"]}}

    monkeypatch.setattr(modal_qualification, "collect_qualification", collect)
    if failure == "changed_instance":
        readiness = external_ops.owned_service(NAME)["metadata"]
        readiness["instance_id"] = "new-instance"
        monkeypatch.setattr(external_ops, "_probe_ready", lambda *_: readiness)
    assert helper.execute(request) != 0
    output = capsys.readouterr()
    assert TOKEN not in output.out + output.err
    result = json.loads((request.parent / "result.json").read_text())
    assert not result["ready"] and not result["hardware_tested"]
    assert not prompt_ui.manager.active and not prompt_ui.manager.cameras_owned
    assert bool(called) is (failure != "changed_instance")


def test_helper_refuses_outside_request_without_writing_result(tmp_path, capsys):
    request = tmp_path / "request.json"
    request.write_text("{}")
    assert helper.execute(request) == 1
    assert not (tmp_path / "result.json").exists()
    assert '"ready": false' in capsys.readouterr().out


def test_preparation_phase_protocol_is_bounded():
    parsed = {}
    parse_line("[yamkit-prepare] warming_and_qualifying", parsed)
    assert parsed["preparation_phase"] == "warming_and_qualifying"
    parse_line("[yamkit-prepare] start_robot", parsed)
    assert parsed["preparation_phase"] == "warming_and_qualifying"
