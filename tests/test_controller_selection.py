"""Controller selection reaches managed execution without bypassing approval or qualification."""

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


def reference_options(**changes):
    return InferenceOptions(**{
        "policy": "molmoact2", "backend": "modal", "call_mode": "http",
        "execution_mode": "cuda_graph10", "controller_mode": "reference", **changes,
    })


@pytest.mark.parametrize("async_chunks", [True, False])
def test_reference_selection_overrides_only_legacy_async_flag(async_chunks):
    options = reference_options(async_chunks=async_chunks).validate()
    argv = options.cli_args()
    assert argv[argv.index("--controller-mode") + 1] == "reference"
    assert "--controller-mode" not in options.cli_args(include_controller=False)
    assert options.operation_key != reference_options(controller_mode="async").operation_key
    with pytest.raises(ValueError, match="Async controller"):
        reference_options(controller_mode="async", async_chunks=False).validate()


@pytest.mark.parametrize("changes", [
    {"backend": "local"}, {"controller_mode": "other"}, {"policy": "smolvla"},
    {"call_mode": "remote"}, {"execution_mode": "eager"}, {"image_encoding": "jpeg"},
    {"center_crop": True}, {"fps": 15}, {"rtc": True},
])
def test_reference_rejects_unreviewed_execution_options(changes):
    with pytest.raises(ValueError):
        reference_options(**changes).validate()


def test_proxy_configuration_keeps_controller_and_model_contract():
    from yamkit.remote_policy import YamkitRemoteConfig

    config = YamkitRemoteConfig(controller_mode="reference", call_mode="http", execution_mode="cuda_graph10")
    assert config.controller_mode == "reference"
    assert len(config.action_feature_names) == 14
    for changes in ({"controller_mode": "other"}, {"execution_mode": "eager"},
                    {"profile": "pi05"}, {"center_crop": True}):
        with pytest.raises(ValueError):
            YamkitRemoteConfig(**{"controller_mode": "reference", "call_mode": "http",
                                  "execution_mode": "cuda_graph10", **changes})


def test_cli_reference_reaches_lerobot_config_without_legacy_async(rig, monkeypatch):
    seen = []
    monkeypatch.setattr("yamkit.inference.performance.require_physical_modal_rollout", lambda *a, **kw: None)
    monkeypatch.setattr("yamkit.modal_ops.owned_service", lambda: {"app_name": "yamkit-vla-test"})
    monkeypatch.setattr("yamkit.remote_rollout.run_remote_rollout", lambda cfg: seen.append(cfg) or {})
    result = CliRunner().invoke(app, [
        "rollout", "--policy", "molmoact2", "--backend", "modal", "--task", "put cube in bowl",
        "--rig", str(rig.path), "--call-mode", "http", "--execution-mode", "cuda_graph10",
        "--controller-mode", "reference", "--no-async", "--confirm-supervised", "--accept-mapping",
    ])
    assert result.exit_code == 0, result.output
    assert len(seen) == 1 and seen[0].policy.controller_mode == "reference"
    assert seen[0].policy.supervised_confirmed and seen[0].policy.mapping_accepted
    assert seen[0].fps == 30


@pytest.mark.parametrize("command,extra", [
    ("modal-qualify", ["--call-mode", "http", "--execution-mode", "cuda_graph10"]),
    ("external-qualify", ["--service", "lambda-georgia"]),
])
def test_qualification_cli_passes_exact_controller(command, extra, monkeypatch):
    calls = []
    monkeypatch.setattr("yamkit.modal_qualification.collect_qualification",
                        lambda *args, **kwargs: calls.append(kwargs) or {})
    result = CliRunner().invoke(app, [command, "--task", "put cube in bowl", "--controller-mode", "reference", *extra])
    assert result.exit_code == 0, result.output
    assert calls[0]["controller_mode"] == "reference"
    assert calls[0]["task"] == "put cube in bowl"


def test_reference_selection_requires_matching_qualification(attached_modal):  # noqa: F811
    state = attached_modal
    body = attached_payload(controller_mode="reference", async_chunks=False)
    response = state.ui.client.post("/api/inference/preflight", json=body)
    assert response.status_code == 200 and response.json()["ready"] is False
    assert "settings changed" in response.json()["reason"]
    assert not state.ui.seen
    state.expected_override["controller_mode"] = "reference"
    response = state.ui.client.post("/api/inference/preflight", json=body)
    assert response.json()["ready"] is True, response.text
    assert not state.ui.seen


@pytest.mark.parametrize("missing", ["confirm_motion", "mapping_accepted", "supervised_confirmed"])
def test_reference_keeps_each_motion_approval(attached_modal, missing):  # noqa: F811
    state = attached_modal
    state.expected_override["controller_mode"] = "reference"
    body = attached_payload(controller_mode="reference", async_chunks=False, confirm_motion=True,
                            mapping_accepted=True, supervised_confirmed=True)
    body[missing] = False
    response = state.ui.client.post("/api/session/rollout", json=body)
    assert response.status_code == 422
    assert not state.ui.seen and not state.ui.manager.active


@pytest.mark.parametrize("capture", [False, True])
def test_reference_managed_launch_and_snapshot_keep_selection(attached_modal, capture, monkeypatch):  # noqa: F811
    state = attached_modal
    task = "put the red cube into the green bowl"
    state.expected_override.update(controller_mode="reference", task=task)
    ui = state.ui
    launched = []
    original = ui.manager.start

    def fake_start(mode, argv, meta=None, **kwargs):
        launched.append(list(argv))
        # The command is captured for inspection; only the fixture's harmless child runs.
        return original(mode, ui.manager.yamkit_argv("rollout"), meta, **kwargs)

    monkeypatch.setattr(ui.manager, "start", fake_start)
    response = ui.client.post("/api/session/rollout", json=attached_payload(
        controller_mode="reference", async_chunks=False, task=task,
        capture_trace=capture, confirm_motion=True, mapping_accepted=True, supervised_confirmed=True,
    ))
    assert response.status_code == 200, response.text
    assert response.json()["meta"]["controller_mode"] == "reference"
    argv = launched[0] if capture else next(args for args in ui.seen if "--controller-mode" in args)
    assert argv[argv.index("--controller-mode") + 1] == "reference"
    assert argv[argv.index("--task") + 1] == task
    if capture:
        assert "--run" in argv and "--confirm-supervised" in argv
    snapshots = list((ui.root / "outputs/ui/deployments").glob("*/run_metadata.json"))
    assert len(snapshots) == 1
    metadata = json.loads(snapshots[0].read_text())
    assert metadata["configuration"]["controller_mode"] == "reference"
    assert metadata["configuration"]["async_chunks"] is False


def test_browser_controller_change_invalidates_qualified_start(attached_browser):  # noqa: F811
    ctx = attached_browser
    ctx.eval("$('#inf-mapping').checked=true; pages.inference.syncForm(); $('#btn-inf-preflight').onclick();")
    _drain_js(ctx)
    assert ctx.eval("$('#btn-ro').disabled") is False
    ctx.eval("$('#inf-controller').value='reference'; pages.inference.syncForm();")
    assert ctx.eval("$('#btn-ro').disabled") is True
    selected = json.loads(ctx.eval("JSON.stringify(pages.inference.selection())"))
    assert selected["controller_mode"] == "reference" and selected["async_chunks"] is False
    ctx.eval("$('#btn-inf-preflight').onclick();")
    _drain_js(ctx)
    assert ctx.eval("$('#btn-ro').disabled") is False
    ctx.eval("session.active=true; pages.inference.syncForm();")
    assert ctx.eval("$('#inf-controller').disabled") is True
