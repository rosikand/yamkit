"""Native workflow glue: strict selection, saved observations, no fallback or motion."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from yamkit import backend_workflow as backend
from yamkit import cli, inference_workflow
from yamkit import pi05_workflow as workflow
from yamkit.backend_workflow import BackendTarget, WorkflowError
from yamkit.pi05 import admission, qualification


@pytest.fixture
def native(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(inference_workflow, "ROOT", tmp_path)
    rig = tmp_path / "rig.yaml"
    rig.write_text("fixture rig only")
    token = tmp_path / "private.token"
    token.write_text("private_fixture_token_never_printed_1234567890")
    token.chmod(0o600)
    observation = tmp_path / "observation.npz"
    np.savez(observation, state=np.zeros(14), **{name: np.zeros((8, 12, 3), dtype=np.uint8)
                                               for name in ("top", "left_wrist", "right_wrist")})
    target = BackendTarget("lambda", "pi05-yam", "native-test", "http://127.0.0.1:8766", token,
                           saved_observations=(observation,))
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_kw: target)
    monkeypatch.setattr(admission, "passive_target_validator", lambda *_: lambda _: None)
    metadata = {"instance_id": "native-instance", "http_session_expires_at": time.time() + 3600}
    monkeypatch.setattr(workflow, "connect_configured_runtime", lambda *_a, **_kw: metadata)
    monkeypatch.setattr(workflow, "_readiness", lambda *_: metadata)
    transports = []
    def transport(*_):
        instance = SimpleNamespace(closed=False)
        instance.close = lambda: setattr(instance, "closed", True)
        transports.append(instance)
        return instance
    monkeypatch.setattr(workflow, "_transport", transport)
    record = {"qualified": True, "hardware_tested": False, "instance_id": metadata["instance_id"], "reasons": [],
              "direct_warm_round_trip_s": {"p50": 0.3, "p95": 0.4, "max": 0.5},
              "integrated": {"predicted_rows": 1500, "completed_rows": 1500, "dropped_rows": 0,
                             "modified_commands": 0, "coherence_violations": 0, "completed_chunks": 50, "faults": 0}}
    calls = []
    def collect(_transport, **kwargs):
        calls.append(kwargs)
        return record
    monkeypatch.setattr(qualification, "collect_qualification", collect)
    def validate(report, *_a, **_kw):
        if report.get("qualified") is not True:
            raise ValueError("fixture did not qualify")
    monkeypatch.setattr(admission, "validate_qualification", validate)
    return SimpleNamespace(root=tmp_path, rig=rig, target=target, record=record, transports=transports, calls=calls)


def test_native_preparation_uses_only_its_own_collector_with_saved_real_schema(native, monkeypatch):
    from yamkit import modal_qualification
    monkeypatch.setattr(modal_qualification, "collect_qualification", lambda *_a, **_kw: pytest.fail("MA2 collector forbidden"))
    selection, result = workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    assert selection.controller_mode == "pi05_reference" and selection.policy == "pi05-yam"
    assert result["ready"] and not result["hardware_tested"] and not result["motion_approval_received"]
    assert native.calls[0]["requests"] == 50 and native.calls[0]["rig_path"] == native.rig
    assert native.calls[0]["observations"][0]["state"].shape == (14,)
    assert native.transports[0].closed and selection.qualification_path.is_file()
    assert "private_fixture_token" not in json.dumps(result)


def test_current_native_proof_is_reused_without_requalification(native):
    selection, _ = workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    second, result = workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    assert result["reused"] and len(native.calls) == 1
    assert second.qualification_path == selection.qualification_path


def test_failed_native_qualification_saved_but_never_current_or_ready(native):
    native.record["qualified"] = False
    native.record["reasons"] = ["fixture failure"]
    with pytest.raises(WorkflowError, match="did not pass") as caught:
        workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    paths = list((native.root / ".context/pi05-qualification").glob("*/qualification.json"))
    assert len(paths) == 1 and str(paths[0]) in str(caught.value)
    assert not (native.root / "data/inference/native/native-test/qualification.json").exists()
    assert native.transports[0].closed


def test_native_failure_names_actual_gripper_without_clipping(native):
    native.record.update(qualified=False, failure={"native_response": {"gripper_bound_violations": [{
        "row_index": 0, "column_index": 6, "name": "untrusted text must not be displayed",
        "value": 1.000895619392395}]}})
    with pytest.raises(WorkflowError, match=r"left_gripper.pos=1.000895619392395 at row 0") as caught:
        workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    assert "untrusted text" not in str(caught.value)
    assert "outside [0,1]" in str(caught.value) and "No motion was started" in str(caught.value)
    assert not (native.root / "data/inference/native/native-test/qualification.json").exists()


def test_native_failure_summary_keeps_tiny_overshoot_visible():
    record = {"failure": {"native_response": {"gripper_bound_violations": [{
        "row_index": 0, "column_index": 6, "value": 1.0000000001}]}}}
    message = workflow._qualification_failure_message(record, ".context/report.json")
    assert "left_gripper.pos=1.0000000001 at row 0" in message


@pytest.mark.parametrize("value", ["private_fixture_token", float("nan"), float("inf"), 1 << 1024, True])
def test_native_failure_summary_never_formats_untrusted_values(value):
    record = {"failure": {"native_response": {"gripper_bound_violations": [{
        "row_index": 0, "column_index": 6, "name": "private_fixture_token", "value": value}]}}}
    message = workflow._qualification_failure_message(record, ".context/report.json")
    assert message == "Native π0.5 qualification did not pass. No motion was started. Report: .context/report.json"


def test_native_expired_session_fails_before_collection(native, monkeypatch):
    monkeypatch.setattr(workflow, "connect_configured_runtime", lambda *_a, **_kw: {
        "instance_id": "native", "http_session_expires_at": time.time() + 20})
    with pytest.raises(WorkflowError, match="expires too soon"):
        workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    assert not native.calls


def test_native_backend_failure_explains_gated_dependency_without_secret(native, monkeypatch):
    def unavailable(*_a, **_kw): raise WorkflowError("Model readiness failed (RemoteFault)")
    monkeypatch.setattr(workflow, "connect_configured_runtime", unavailable)
    with pytest.raises(WorkflowError, match="authorized access to google/paligemma") as exc:
        workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    assert "private_fixture" not in str(exc.value) and not native.calls


def test_native_high_level_alias_never_changes_generic_base_selection(native):
    from yamkit.inference.profiles import get_profile
    selection, result = inference_workflow.prepare_inference(backend="lambda", policy="pi05", task="cube", rig=native.rig)
    assert selection.policy == "pi05-yam" and result["ready"]
    assert get_profile("pi05").repo_id == "lerobot/pi05_base"


def test_native_cli_dry_run_has_no_physical_delegate(native, monkeypatch):
    monkeypatch.setattr(workflow, "run_prepared_pi05", lambda *_a, **_kw: pytest.fail("no physical native calls"))
    result = CliRunner().invoke(cli.app, ["rollout", "--backend", "lambda", "--policy", "pi05",
                                         "--task", "cube", "--rig", str(native.rig), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert '"policy": "pi05-yam"' in result.output and "no rollout was launched" in result.output


def test_native_physical_delegate_requires_confirmation_before_any_connection(native, monkeypatch):
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_kw: pytest.fail("no connection without GO"))
    with pytest.raises(WorkflowError, match="supervised"):
        workflow.run_prepared_pi05(None, confirm_supervised=False, accept_mapping=False)


@pytest.mark.parametrize("failure", [RuntimeError, OSError])
def test_native_fault_error_is_actionable_without_private_chain_or_release_claim(native, monkeypatch, failure):
    from yamkit.pi05 import rollout

    selection, _ = workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    def failed(*_a, **_kw):
        raise failure("private endpoint or SDK diagnostic")
    monkeypatch.setattr(rollout, "run_rollout", failed)
    with pytest.raises(WorkflowError, match="No automatic physical retry") as caught:
        workflow.run_prepared_pi05(selection, confirm_supervised=True, accept_mapping=True)
    assert "private endpoint" not in str(caught.value)
    assert "released" not in str(caught.value)
    assert "outputs/ui/deployments/pi05-" in str(caught.value)
    assert native.transports[-1].closed


def test_delayed_native_confirmation_checks_fresh_lease_before_physical_delegate(native, monkeypatch):
    from yamkit.pi05 import rollout

    selection, _ = workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    monkeypatch.setattr(workflow, "_readiness", lambda *_: {
        "instance_id": "same-instance", "http_session_expires_at": time.time() + 20})
    monkeypatch.setattr(rollout, "run_rollout", lambda *_a, **_kw: pytest.fail("expired confirmation wait must not open hardware"))
    with pytest.raises(WorkflowError, match="expires too soon after confirmation"):
        workflow.run_prepared_pi05(selection, confirm_supervised=True, accept_mapping=True)


def test_native_qualification_disappearing_during_confirmation_is_actionable(native, monkeypatch):
    from yamkit.pi05 import rollout

    selection, _ = workflow.prepare_pi05(backend="lambda", task="cube", rig=native.rig)
    selection.qualification_path.unlink()
    monkeypatch.setattr(rollout, "run_rollout", lambda *_a, **_kw: pytest.fail("no hardware without retained proof"))
    with pytest.raises(WorkflowError, match="evidence changed after confirmation"):
        workflow.run_prepared_pi05(selection, confirm_supervised=True, accept_mapping=True)
