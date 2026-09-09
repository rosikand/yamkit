"""External GPU selection keeps the existing supervised runner and UI gates."""

import json

import pytest
from typer.testing import CliRunner

from tests.test_inference_ui import (  # noqa: F401 — shared hardware-forbidden fixtures
    _drain_js,
    attached_browser,
    attached_modal,
    attached_payload,
    inference_js,
    inference_ui,
)
from yamkit.cli import app
from yamkit.deployment import InferenceOptions


def external_options(**changes):
    return InferenceOptions(**{
        "policy": "molmoact2", "backend": "external", "external_service": "lambda-georgia",
        "call_mode": "http", "execution_mode": "cuda_graph10", **changes,
    })


@pytest.mark.parametrize("changes", [
    {"external_service": None}, {"external_service": "../host"}, {"external_service": "Host Name"},
    {"modal_app": "yamkit-vla-other"}, {"call_mode": "remote"}, {"rtc": True},
    {"async_chunks": False}, {"duration": 0}, {"image_encoding": "jpeg"},
])
def test_external_options_cannot_bypass_remote_contract(changes):
    with pytest.raises(ValueError):
        external_options(**changes).validate()


def test_external_service_is_bound_into_exact_selection_and_cli():
    selected = external_options().validate()
    other = external_options(external_service="lambda-other").validate()
    assert selected.operation_key != other.operation_key
    args = selected.cli_args()
    assert args[args.index("--external-service") + 1] == "lambda-georgia"
    assert "--modal-app" not in args and "--gpu" not in args


@pytest.mark.parametrize("duration", [10, 20, 30])
def test_external_trace_preserves_canonical_task_duration_and_supervision(duration):
    from tests.test_trace_rollout import module

    args = module.parse_args(["--run", "--backend", "external", "--external-service", "lambda-georgia",
                              "--duration", str(duration), "--confirm-supervised"])
    argv = module.rollout_arguments(args)
    assert argv[argv.index("--external-service") + 1] == "lambda-georgia"
    assert argv[argv.index("--task") + 1] == module.TASK
    assert argv[argv.index("--duration") + 1] == str(duration)
    assert "--modal-app" not in argv
    assert "--confirm-supervised" in argv and "--accept-mapping" in argv
    with pytest.raises(SystemExit):
        module.parse_args(["--run", "--backend", "external", "--external-service", "lambda-georgia"])


def test_external_cli_attachment_does_not_request_motion(monkeypatch, tmp_path):
    from yamkit import external_ops

    calls = []
    monkeypatch.setattr(external_ops, "attach_service", lambda *a, **kw: calls.append((a, kw)) or {"ready": True})
    result = CliRunner().invoke(app, ["external-attach", "--name", "lambda-georgia",
                                    "--endpoint", "http://127.0.0.1:8765", "--token-file", str(tmp_path / "token")])
    assert result.exit_code == 0, result.output
    assert calls == [(("lambda-georgia", "http://127.0.0.1:8765", tmp_path / "token"), {"provider": "lambda"})]


@pytest.fixture
def attached_external(attached_modal, monkeypatch):  # noqa: F811 — imported pytest fixture
    from yamkit import external_ops
    from yamkit.inference import qualification

    state = attached_modal
    state.receipt["http_ingress"] = "ssh"
    state.expected_override.update(modal_app=None, external_service_name="lambda-georgia")
    original = qualification.current_settings

    def settings(options, **kwargs):
        return {**original(options, **kwargs), "external_service_name": options.external_service}

    def credentials(name):
        if name != "lambda-georgia":
            raise ValueError("No matching external attachment")
        return {"token": "PRIVATE_EXTERNAL_TOKEN"}

    monkeypatch.setattr(qualification, "current_settings", settings)
    monkeypatch.setattr(external_ops, "http_credentials", credentials)
    return state


def external_payload(**changes):
    return attached_payload(**{"backend": "external", "modal_app": None,
                               "external_service": "lambda-georgia", **changes})


def test_external_ui_preflight_reads_only_and_rejects_changed_service(attached_external):
    state = attached_external
    checked = state.ui.client.post("/api/inference/preflight", json=external_payload())
    assert checked.json()["ready"], checked.text
    assert checked.json()["external_service"] == "lambda-georgia"
    assert "PRIVATE_EXTERNAL_TOKEN" not in checked.text
    changed = state.ui.client.post("/api/inference/preflight", json=external_payload(external_service="lambda-other"))
    assert changed.json()["ready"] is False
    assert not state.ui.seen


@pytest.mark.parametrize("missing", ["confirm_motion", "mapping_accepted", "supervised_confirmed"])
def test_external_ui_requires_each_motion_confirmation(attached_external, missing):
    body = external_payload(confirm_motion=True, mapping_accepted=True, supervised_confirmed=True)
    body[missing] = False
    response = attached_external.ui.client.post("/api/session/rollout", json=body)
    assert response.status_code == 422
    assert not attached_external.ui.seen


def test_external_ui_dispatches_existing_managed_rollout(attached_external):
    ui = attached_external.ui
    response = ui.client.post("/api/session/rollout", json=external_payload(
        confirm_motion=True, mapping_accepted=True, supervised_confirmed=True))
    assert response.status_code == 200, response.text
    args = ui.seen[-1]
    assert args[args.index("--backend") + 1] == "external"
    assert args[args.index("--external-service") + 1] == "lambda-georgia"
    assert "--modal-app" not in args
    assert "--accept-mapping" in args and "--confirm-supervised" in args
    assert ui.client.post("/api/session/stop").status_code == 200
    assert ui.manager.wait(timeout=5) is not None


def test_external_browser_keeps_service_selection_and_invalidates_ready(attached_browser):  # noqa: F811
    ctx = attached_browser
    ctx.eval("$('#inf-backend').value='external'; $('#inf-external-service').value='lambda-georgia';"
             "$('#inf-mapping').checked=true; pages.inference.syncForm(); $('#btn-inf-preflight').onclick();")
    _drain_js(ctx)
    selected = json.loads(ctx.eval("JSON.stringify(pages.inference.selection())"))
    assert selected["external_service"] == "lambda-georgia" and selected["modal_app"] is None
    assert ctx.eval("$('#btn-ro').disabled") is False
    assert ctx.eval("$('#inf-gpu').disabled") is True
    assert ctx.eval("$('#btn-cloud-stop').disabled") is True
    ctx.eval("$('#inf-external-service').value='lambda-other'; pages.inference.syncForm();")
    assert ctx.eval("$('#btn-ro').disabled") is True
