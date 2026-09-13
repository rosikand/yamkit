"""Native UI preparation/launch routing with only saved files and mocked physical delegates."""

import dataclasses
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_inference_ui import _check_attached_browser, _drain_js, _posted, _start_preparation
from tests.test_inference_ui import attached_browser as _attached_browser
from tests.test_inference_ui import inference_js as _inference_js
from tests.test_inference_ui import inference_ui as _inference_ui
from tests.test_inference_ui import preparation_browser as _preparation_browser
from yamkit import backend_workflow, paths, pi05_workflow
from yamkit.ui import native_inference as native
from yamkit.ui import server

inference_ui = _inference_ui
inference_js = _inference_js
attached_browser = _attached_browser
preparation_browser = _preparation_browser


def payload(**extra):
    return {"policy": "pi05-yam", "backend": "lambda", "task": "put the red cube into the green bowl",
            "controller_mode": "pi05_reference", "async_chunks": False, "call_mode": "http",
            "execution_mode": "eager", "duration": 5, "arms": ["left_follower", "right_follower"], **extra}


def options(**extra):
    values = payload(**extra)
    values["arms"] = tuple(values["arms"])
    return native.NativeUIOptions(**values)


@pytest.mark.parametrize("change", [{"controller_mode": "reference"}, {"execution_mode": "cuda_graph10"},
                                    {"backend": "external"}, {"center_crop": True}, {"rtc": True},
                                    {"async_chunks": True}, {"fps": 15}, {"duration": 91},
                                    {"arms": ["left_follower"]}, {"prediction_queue_threshold": 30}])
def test_native_options_never_inherit_ma2_or_weaken_the_native_contract(change):
    with pytest.raises(ValueError):
        options(**change).validate()


@pytest.mark.parametrize("policy", ["pi05-base", "pi05_base"])
def test_official_base_has_precise_no_motion_blocker(policy):
    with pytest.raises(ValueError, match="normalization, joint/gripper coordinates"):
        options(policy=policy).validate()


def test_full_selection_key_binds_recording_destination_and_fresh_approvals():
    value = options()
    for changed in (dataclasses.replace(value, capture_trace=True),
                    dataclasses.replace(value, capture_trace=True, upload_repo_id="owner/private"),
                    dataclasses.replace(value, task="different task"), dataclasses.replace(value, duration=20),
                    dataclasses.replace(value, mapping_accepted=True)):
        assert changed.operation_key != value.operation_key
    with pytest.raises(ValueError, match="supervised"):
        value.validate(motion=True)


@pytest.fixture
def native_ui(inference_ui, monkeypatch):
    ui = inference_ui
    monkeypatch.setattr(native, "preparation_context", lambda value: {"selection_key": value.operation_key})
    def retained(value):
        return object(), {"ready": True, "selection_key": value.operation_key,
                          "checked_at": time.time(), "expires_at": time.time() + 3600,
                          "hardware_tested": False}
    monkeypatch.setattr(native, "retained_selection", retained)
    ui.launched = []
    def start(mode, argv, meta, **_ownership):
        ui.launched.append((mode, argv, meta))
        return {"active": True, "mode": mode, "meta": meta}
    monkeypatch.setattr(ui.manager, "start", start)
    return ui


def test_native_preflight_is_passive_and_official_base_cannot_prepare(native_ui):
    ui = native_ui
    result = ui.client.post("/api/inference/preflight", json=payload()).json()
    assert result["ready"] and result["prepare_before_start"] and not result["can_prepare"]
    result = ui.client.post("/api/inference/preflight", json=payload(policy="pi05_base")).json()
    assert not result["ready"] and not result["can_prepare"]
    assert "No hardware was opened" in result["reason"]
    assert not ui.launched and not ui.manager.cameras_owned


def test_native_expired_proof_can_prepare_but_not_launch(native_ui, monkeypatch):
    def stale(_):
        raise ValueError("Native proof expired; prepare this task")
    monkeypatch.setattr(native, "retained_selection", stale)
    ui = native_ui
    result = ui.client.post("/api/inference/preflight", json=payload()).json()
    assert not result["ready"] and result["can_prepare"]
    result = ui.client.post("/api/session/rollout", json=payload(
        confirm_motion=True, mapping_accepted=True, supervised_confirmed=True))
    assert result.status_code == 422 and not ui.launched


def test_preparation_routes_to_nonhardware_child_with_false_approvals(native_ui):
    ui = native_ui
    result = ui.client.post("/api/inference/prepare", json=payload(capture_trace=True))
    assert result.status_code == 200, result.text
    mode, argv, meta = ui.launched[0]
    assert mode == "inference-prepare" and Path(argv[1]).name == "prepare_native_inference.py"
    request = json.loads(Path(argv[2]).read_text())
    assert not request["options"]["supervised_confirmed"] and not request["options"]["mapping_accepted"]
    assert request["options"]["capture_trace"] and not meta["hardware_tested"]
    assert request["expected"]["selection_key"] == result.json()["selection_key"]
    assert ui.manager._proc is None and not ui.manager.cameras_owned


@pytest.mark.parametrize("extra", [{"confirm_motion": True}, {"mapping_accepted": True}, {"supervised_confirmed": True}])
def test_preparation_cannot_carry_approval(native_ui, extra):
    assert native_ui.client.post("/api/inference/prepare", json=payload(**extra)).status_code == 422
    assert not native_ui.launched


@pytest.mark.parametrize("missing", ["confirm_motion", "mapping_accepted", "supervised_confirmed"])
def test_native_launch_requires_all_fresh_confirmations(native_ui, missing):
    request = payload(confirm_motion=True, mapping_accepted=True, supervised_confirmed=True)
    request[missing] = False
    assert native_ui.client.post("/api/session/rollout", json=request).status_code == 422
    assert not native_ui.launched


def test_native_mock_launch_has_one_fresh_trace_and_parent_upload(native_ui):
    ui = native_ui
    result = ui.client.post("/api/session/rollout", json=payload(
        confirm_motion=True, mapping_accepted=True, supervised_confirmed=True, upload_repo_id="owner/private"))
    assert result.status_code == 200, result.text
    mode, argv, meta = ui.launched[0]
    assert mode == "rollout" and Path(argv[1]).name == "run_native_inference.py"
    request = json.loads(Path(argv[2]).read_text())
    trace = Path(request["trace_dir"])
    assert trace.parent == ui.root / ".context/rollout-traces" and not trace.exists()
    assert meta["capture_trace"] and meta["upload_repo_id"] == "owner/private"
    assert request["options"]["capture_trace"] and ui.manager._proc is None
    snapshot = json.loads((Path(meta["session_log_path"]).parent / "run_metadata.json").read_text())
    assert snapshot["model"]["id"] == "pi05-yam" and snapshot["software"]["pi05_build_id"]


def test_native_recording_memory_is_never_bypassed(native_ui, monkeypatch):
    monkeypatch.setattr(server, "_capture_memory_preflight", lambda _: {"admission_passes": False})
    request = payload(capture_trace=True)
    preflight = native_ui.client.post("/api/inference/preflight", json=request).json()
    assert not preflight["ready"] and not preflight["can_prepare"]
    assert native_ui.client.post("/api/inference/prepare", json=request).status_code == 422
    request.update(confirm_motion=True, mapping_accepted=True, supervised_confirmed=True)
    assert native_ui.client.post("/api/session/rollout", json=request).status_code == 422
    assert not native_ui.launched


def test_native_parent_upload_routes_only_after_finalization_and_imports_report(native_ui, monkeypatch):
    from yamkit import pi05_artifacts, rollout_artifacts

    ui, done, calls = native_ui, threading.Event(), []
    def package(run_dir, *, trace_dir):
        assert not ui.manager.active and not ui.manager.cameras_owned
        meta = json.loads((run_dir / "meta.json").read_text())
        assert meta["policy"] == "pi05-yam" and meta["returncode"] == 0 and meta["ended_at"]
        assert (run_dir / "report.json").is_file()
        calls.append((run_dir, trace_dir))
        return run_dir / "bundle"
    def upload(bundle, *, repo_id):
        assert calls and bundle == calls[0][0] / "bundle" and repo_id == "owner/private"
        done.set()
        return {"status": "uploaded", "repo_id": repo_id}
    monkeypatch.setattr(pi05_artifacts, "package_native_rollout", package)
    monkeypatch.setattr(rollout_artifacts, "package_rollout", lambda *_a, **_k: pytest.fail("native run routed to MA2 packager"))
    monkeypatch.setattr(rollout_artifacts, "upload_rollout", upload)
    response = ui.client.post("/api/session/rollout", json=payload(
        confirm_motion=True, mapping_accepted=True, supervised_confirmed=True, upload_repo_id="owner/private"))
    assert response.status_code == 200 and not calls
    mode, _argv, meta = ui.launched[0]
    trace = Path(meta["debug_trace_dir"])
    trace.mkdir(parents=True)
    (trace / "summary.json").write_text(json.dumps({"resources_released": True, "status": "TRACE_SAVED"}))
    (trace / "report.json").write_text('{"released":true,"hardware_tested":false}')
    ui.manager.on_exit({"mode": mode, "meta": meta, "active": False, "returncode": 0,
                        "started_at": time.time() - 1, "ended_at": time.time(), "log": []})
    assert done.wait(5) and len(calls) == 1
    assert calls[0][1] == trace and ui.manager._proc is None


@pytest.mark.parametrize("route", ["policy-check", "policy-probe"])
def test_native_does_not_fall_into_legacy_hardware_helpers(native_ui, route):
    assert native_ui.client.post("/api/session/" + route, json=payload()).status_code == 422
    assert not native_ui.launched


@pytest.fixture
def helper(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(backend_workflow, "ROOT", tmp_path)
    monkeypatch.setattr(native, "preparation_context", lambda value: {"selection_key": value.operation_key})
    monkeypatch.setattr(native, "retained_selection", lambda value: (value, {
        "ready": True, "selection_key": value.operation_key, "expires_at": time.time() + 3600,
        "hardware_tested": False}))
    calls = []
    def prepare(**kwargs):
        calls.append(("prepare", kwargs))
        print("private_fixture_token must never escape optional backend diagnostics")
        return object(), {"reused": True, "evidence_directory": str(tmp_path / "evidence")}
    def run(selection, **kwargs):
        calls.append(("run", kwargs))
        assert selection.mapping_accepted and selection.supervised_confirmed
        return {"status": "completed", "released": True}
    monkeypatch.setattr(pi05_workflow, "prepare_pi05", prepare)
    monkeypatch.setattr(pi05_workflow, "run_prepared_pi05", run)
    def request(*, motion=False, **extra):
        value = options(rig_path=str(tmp_path / "rig.yaml"), **extra)
        directory = tmp_path / ".context" / ("native-inference" if motion else "inference-preparation") / ("a" * 32)
        directory.mkdir(parents=True)
        body = {"options": dataclasses.asdict(value), "expected": native.preparation_context(value)}
        if motion:
            body["trace_dir"] = str(tmp_path / ".context/rollout-traces" / ("b" * 32))
        path = directory / "request.json"
        path.write_text(json.dumps(body))
        return path
    return SimpleNamespace(request=request, calls=calls, root=tmp_path)


def test_helper_preparation_reuses_shared_workflow_and_never_runs(helper, capsys):
    request = helper.request()
    assert native.execute_request(request) == 0
    assert [kind for kind, _ in helper.calls] == ["prepare"]
    kwargs = helper.calls[0][1]
    assert kwargs["own_preparation_dir"] == request.parent and kwargs["backend"] == "lambda"
    assert "private_fixture_token" not in capsys.readouterr().out
    result = json.loads((request.parent / "result.json").read_text())
    assert result["ready"] and not result["hardware_tested"]


def test_helper_rejects_approved_preparation_before_workflow(helper):
    assert native.execute_request(helper.request(mapping_accepted=True)) == 2
    assert not helper.calls


def test_mock_physical_helper_consumes_exact_request_once_and_does_not_upload(helper):
    request = helper.request(motion=True, mapping_accepted=True, supervised_confirmed=True,
                             capture_trace=True, upload_repo_id="owner/private")
    assert native.execute_request(request, motion=True) == 0
    assert helper.calls[0][0] == "run" and helper.calls[0][1]["upload_repo_id"] is None
    assert helper.calls[0][1]["capture_trace"] and helper.calls[0][1]["confirm_supervised"]
    assert native.execute_request(request, motion=True) == 2
    assert len(helper.calls) == 1


@pytest.mark.parametrize("report", [{"status": "failed", "released": True},
                                     {"status": "completed", "released": False},
                                     {"status": "completed", "released": True, "exit_status": 2},
                                     {"status": "completed", "released": True, "artifact_status": "EXPORT_FAILED"},
                                     {"status": "stopped", "released": True}])
def test_native_returned_fault_or_stop_cannot_be_reported_as_success(helper, monkeypatch, report):
    monkeypatch.setattr(pi05_workflow, "run_prepared_pi05", lambda *_a, **_k: report)
    request = helper.request(motion=True, mapping_accepted=True, supervised_confirmed=True)
    assert native.execute_request(request, motion=True) != 0
    result = json.loads((request.parent / "result.json").read_text())
    assert not result["ready"] and result["status"] == report["status"]


def test_native_explicit_stopped_exit_status_stays_130(helper, monkeypatch):
    monkeypatch.setattr(pi05_workflow, "run_prepared_pi05", lambda *_a, **_k: {
        "status": "stopped", "released": True, "exit_status": 130, "artifact_status": "TRACE_SAVED"})
    request = helper.request(motion=True, mapping_accepted=True, supervised_confirmed=True)
    assert native.execute_request(request, motion=True) == 130
    assert not json.loads((request.parent / "result.json").read_text())["ready"]


def test_helper_changed_selection_or_arbitrary_error_never_exposes_diagnostics(helper, monkeypatch, capsys):
    request = helper.request()
    monkeypatch.setattr(native, "preparation_context", lambda _: (_ for _ in ()).throw(RuntimeError("secret_fixture_value")))
    assert native.execute_request(request) == 2
    assert not helper.calls and "secret_fixture_value" not in capsys.readouterr().out


def test_browser_native_selection_uses_native_defaults_and_optional_recording(attached_browser):
    ctx = attached_browser
    ctx.eval("pages.inference.applyDefaults({policy:'pi05-yam',backend:'lambda',controller_mode:'pi05_reference',task:'green bowl',duration:5});pages.inference.syncForm();")
    value = json.loads(ctx.eval("JSON.stringify(pages.inference.selection())"))
    assert value["backend"] == "lambda" and value["controller_mode"] == "pi05_reference"
    assert value["execution_mode"] == "eager" and value["call_mode"] == "http"
    assert not value["async_chunks"] and not value["capture_trace"] and not value["mapping_accepted"]
    assert ctx.eval("$('#inf-controller').disabled") and not ctx.eval("$('#inf-trace').disabled")
    ctx.eval("$('#inf-upload').checked=true;$('#inf-upload-repo').value='owner/private';pages.inference.syncForm();")
    value = json.loads(ctx.eval("JSON.stringify(pages.inference.selection())"))
    assert value["capture_trace"] and value["upload_repo_id"] == "owner/private"


@pytest.mark.parametrize("cancel", [False, True])
def test_browser_native_preparation_requires_new_confirmation_after_result(preparation_browser, cancel):
    ctx = preparation_browser
    ctx.eval("pages.inference.applyDefaults({policy:'pi05-yam',backend:'lambda',controller_mode:'pi05_reference',task:'green bowl',duration:5});$('#inf-mapping').checked=true;pages.inference.formChanged('inf-mapping');")
    _check_attached_browser(ctx)
    _start_preparation(ctx)
    request = _posted(ctx, "/inference/prepare")[0]["body"]
    assert request["backend"] == "lambda" and request["controller_mode"] == "pi05_reference"
    assert not request["mapping_accepted"] and not request["supervised_confirmed"]
    assert not _posted(ctx, "/session/rollout") and ctx.eval("confirmMessages.length") == 0
    ctx.eval("confirmResult=" + ("false" if cancel else "true") + ";finishPrep();")
    _drain_js(ctx)
    launched = _posted(ctx, "/session/rollout")
    assert len(launched) == (0 if cancel else 1)
    if launched:
        assert launched[0]["body"]["supervised_confirmed"] and launched[0]["body"]["confirm_motion"]
        assert launched[0]["body"]["policy"] == "pi05-yam"


@pytest.fixture
def retained_files(tmp_path, monkeypatch):
    from tests.test_pi05_qualification import metadata
    from yamkit import external_ops
    from yamkit.inference import identity
    from yamkit.inference.identity import inference_build_id
    from yamkit.pi05 import admission

    stable_build = inference_build_id()
    monkeypatch.setattr(identity, "inference_build_id", lambda: stable_build)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(backend_workflow, "ROOT", tmp_path)
    rig = tmp_path / "rig.yaml"
    rig.write_text("saved rig fixture")
    directory = tmp_path / "data/inference/native/native-test"
    directory.mkdir(parents=True)
    token = tmp_path / "data/inference/native.token"
    token.write_text("PRIVATE_FILE_NEVER_READ_BY_PREFLIGHT")
    token.chmod(0o600)
    config = tmp_path / "data/inference/backends.json"
    external_ops._save(config, {"version": 1, "backends": {"lambda": {"policies": {"pi05-yam": {
        "service": "native-test", "endpoint": "http://127.0.0.1:8766", "token_file": str(token)}}}}})
    private_read = external_ops._read_private
    def read(path, *args, **kwargs):
        assert path != token, "Passive UI preflight read a bearer token"
        return private_read(path, *args, **kwargs)
    monkeypatch.setattr(external_ops, "_read_private", read)
    monkeypatch.setattr(admission, "passive_target_validator", lambda _: lambda _target: None)
    def validate(report, service, *, task, rig_path):
        assert rig_path == rig
        if report["task"] != task or report["instance_id"] != service["instance_id"]:
            raise ValueError("Exact task or runtime proof changed")
    monkeypatch.setattr(admission, "validate_qualification", validate)
    service = {**metadata(), "http_ingress": "ssh", "http_endpoint": "http://127.0.0.1:8766",
               "inference_build_id": stable_build, "external_service": {
                   "provider": "lambda", "service_id": "native-test", "host_id": "a" * 64,
                   "region": "Georgia", "region_source": "operator_declared"}}
    receipt = {"status": "ready", "profile": "pi05-yam", "service": "native-test", "metadata": service}
    external_ops._save(directory / "receipt.json", receipt)
    report = tmp_path / "qualification.json"
    report.write_text(json.dumps({"task": payload()["task"], "instance_id": service["instance_id"],
                                  "completed_at": time.time()}))
    external_ops._save(directory / "qualification.json", {"path": str(report)})
    return SimpleNamespace(root=tmp_path, rig=rig, directory=directory, receipt=receipt,
                           report=report, options=options(rig_path=str(rig)), config=config)


def test_retained_preflight_reads_only_local_proof_not_token_http_or_devices(retained_files, monkeypatch):
    from yamkit.inference.http_transport import HttpTransport

    monkeypatch.setattr(HttpTransport, "_invoke", lambda *_a, **_k: pytest.fail("passive check contacted GPU"))
    value = retained_files
    before = {path: path.read_bytes() for path in value.root.rglob("*") if path.is_file()}
    selection, current = native.retained_selection(value.options)
    after = {path: path.read_bytes() for path in value.root.rglob("*") if path.is_file()}
    assert before == after and current["ready"] and not current["hardware_tested"]
    assert selection.qualification_path == value.report
    assert "PRIVATE_FILE" not in json.dumps(current)
    with pytest.raises(ValueError, match="task"):
        native.retained_selection(dataclasses.replace(value.options, task="different"))


@pytest.mark.parametrize("change", ["service", "endpoint", "build", "expiry", "rig", "config"])
def test_retained_local_identity_and_prepare_context_bindings(retained_files, change):
    from yamkit.external_ops import _save

    value = retained_files
    before = native.preparation_context(value.options)
    if change in ("rig", "config"):
        path = value.rig if change == "rig" else value.config
        path.write_text(path.read_text() + "\n")
        assert native.preparation_context(value.options) != before
        return
    meta = value.receipt["metadata"]
    if change == "service":
        meta["external_service"]["service_id"] = "another-service"
    elif change == "endpoint":
        meta["http_endpoint"] = "http://127.0.0.1:8767"
    elif change == "build":
        meta["inference_build_id"] = "changed"
    else:
        meta["http_session_expires_at"] = time.time() + 30
    _save(value.directory / "receipt.json", value.receipt)
    with pytest.raises(ValueError):
        native.retained_selection(value.options)


@pytest.fixture
def sanitized_runtime_receipt(retained_files):
    """Actual service readiness builder and workflow sanitizer, not invented persisted metadata."""
    from yamkit.external_ops import _save
    from yamkit.inference.standalone_service import ServiceConfig, readiness_metadata
    from yamkit.pi05.transport import validate_readiness
    from yamkit.rollout_artifacts import sanitize

    value = retained_files
    original = value.receipt["metadata"]
    config = ServiceConfig("native-test", "lambda", "Georgia", 8766, "unused-private-file",
                           value.options.task)
    live = readiness_metadata(SimpleNamespace(ready=lambda: original), config,
                              expires_at=original["http_session_expires_at"], host_id="a" * 64,
                              provenance={"packages": {"tokenizers": "0.22.2"}},
                              build_id=original["inference_build_id"])
    live["execution_mode"] = "eager"  # The native service's make_application.ready wrapper.
    validate_readiness(live)
    value.receipt = sanitize({**value.receipt, "metadata": live})
    validate_readiness(value.receipt["metadata"])
    assert "http_endpoint" in live and "http_endpoint" not in value.receipt["metadata"]
    _save(value.directory / "receipt.json", value.receipt)
    return value


def test_ui_retained_proof_accepts_the_actual_sanitized_runtime_receipt(sanitized_runtime_receipt, monkeypatch):
    from yamkit.inference.http_transport import HttpTransport

    value = sanitized_runtime_receipt
    monkeypatch.setattr(HttpTransport, "_invoke", lambda *_a, **_k: pytest.fail("UI local preflight contacted the GPU"))
    original = (value.directory / "receipt.json").read_bytes()
    selection, ready = native.retained_selection(value.options)
    assert ready["ready"] and not ready["hardware_tested"]
    assert selection.qualification_path == value.report
    assert (value.directory / "receipt.json").read_bytes() == original
    assert "http_endpoint" not in value.receipt["metadata"]  # No mutation or credential/address export.
    assert "127.0.0.1" not in json.dumps(ready)


def test_live_native_workflow_still_requires_the_advertised_origin(sanitized_runtime_receipt, monkeypatch):
    from yamkit.inference import http_transport

    class NoNetworkTransport:
        def __init__(self, *_args, **_kwargs): pass
        def _invoke(self, *_args): return sanitized_runtime_receipt.receipt["metadata"]
        def close(self): pass

    monkeypatch.setattr(http_transport, "HttpTransport", NoNetworkTransport)
    target = SimpleNamespace(service="native-test", endpoint="http://127.0.0.1:8766")
    with pytest.raises(ValueError, match="explicit endpoint"):
        pi05_workflow._readiness(target, "unused-offline-fixture")


@pytest.mark.parametrize("field,replacement", [
    ("http_endpoint", "http://127.0.0.1:8767"), ("http_endpoint", None), ("http_endpoint", ""),
    ("http_session_expires_at", None), ("http_session_expires_at", 0),
    ("http_session_expires_at", 10**20), ("http_ingress", "asgi"),
])
def test_sanitized_cached_view_still_rejects_expiry_and_advertised_origin_defects(sanitized_runtime_receipt, field, replacement):
    from yamkit.external_ops import _save

    value = sanitized_runtime_receipt
    value.receipt["metadata"][field] = replacement
    _save(value.directory / "receipt.json", value.receipt)
    with pytest.raises(ValueError):
        native.retained_selection(value.options)
