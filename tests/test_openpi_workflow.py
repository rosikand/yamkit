"""Official-base workflow with explicit fake services, evidence and rollout seams."""

import copy
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit import backend_workflow as backend
from yamkit.backend_workflow import BackendTarget, WorkflowError
from yamkit.openpi import admission, qualification, rollout, service, workflow
from yamkit.openpi.interface import CONTRACT_ID
from yamkit.openpi.yam_candidate import CandidateQuantiles, CandidateStatistics


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    rig = tmp_path / "rig.yaml"
    rig.write_text("explicit fake rig; never passed to hardware")
    token = tmp_path / "private.token"
    token.write_text("fake_private_credential_never_logged_1234567890")
    token.chmod(0o600)
    observed = tmp_path / "saved.npz"
    np.savez(observed, state=np.zeros(14), **{name: np.zeros((2, 3, 3), dtype=np.uint8)
                                             for name in ("top", "left_wrist", "right_wrist")})
    target = BackendTarget("lambda", "pi05-base", "lambda-openpi", "http://127.0.0.1:8767", token,
                           saved_observations=(observed,))
    q = CandidateQuantiles([-1] * 14, [1] * 14, "explicit fake workflow data", "a" * 64)
    statistics = CandidateStatistics(q, q)
    runtime = service.NativeRuntime(SimpleNamespace(), statistics, expires_at=time.time() + 3600,
                                    source_sha="b" * 40, statistics_file_sha256="c" * 64)
    metadata = runtime.ready()
    binding = {"host_id": "explicit fake host", "rig_sha256": "d" * 64}
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_kw: target)
    monkeypatch.setattr(workflow, "load_statistics", lambda: statistics)
    monkeypatch.setattr(workflow, "rig_contract", lambda _: binding)
    monkeypatch.setattr(admission, "rig_contract", lambda _: binding)
    monkeypatch.setattr(admission, "adapter_build_id", lambda: "e" * 64)
    connection_calls, transports, collections, rollouts = [], [], [], []

    def connect(*_args, **_kwargs):
        connection_calls.append(True)
        return copy.deepcopy(metadata)

    monkeypatch.setattr(workflow, "connect_configured_runtime", connect)
    monkeypatch.setattr(workflow, "readiness", lambda *_: copy.deepcopy(metadata))

    def transport(*_args, **_kwargs):
        value = SimpleNamespace(closed=False, ready=lambda *_a, **_kw: copy.deepcopy(metadata))
        value.close = lambda: setattr(value, "closed", True)
        transports.append(value)
        return value

    monkeypatch.setattr(workflow, "make_transport", transport)
    modifications = {}

    def collect(_transport, **kwargs):
        collections.append(kwargs)
        report = {"profile": "pi05-base", "qualified": True, "reasons": [], "hardware_tested": False,
                  "controller_mode": CONTRACT_ID, "adapter_build_id": "e" * 64,
                  "statistics_sha256": statistics.metadata()["candidate_statistics_sha256"],
                  "service_identity": admission.service_binding(copy.deepcopy(metadata)),
                  "task": kwargs["task"], "robot_host": binding, "completed_warm_samples": 50,
                  "completed_at": time.time(), "expires_at": metadata["session_expires_at"],
                  "direct_warm_round_trip_s": {"p50": .08, "p95": .10, "max": .12, "sample_count": 50},
                  "bounds_checked": True, "direct_chunks_validated": 50,
                  "integrated": {"completed_chunks": 50, "predicted_rows": 2500, "completed_rows": 1250,
                                 "admitted_chunks": 50, "inference_calls": 50,
                                 "admitted_rows": 1250, "faults": 0, "modified_commands": 0,
                                 "coherence_violations": 0, "invalid_chunks": 0, "unknown_partial_dispatches": 0,
                                 "late_response_rows_discarded": 0, "intended_unused_rows": 1250,
                                 "uncompleted_committed_rows": 0, "completed_points": 1250, "attempted_points": 1250,
                                 "reordered_rows": 0, "prefix_dropped_rows": 0, "rows_discarded_on_stop_or_deadline": 0,
                                 "rows_discarded_on_fault": 0, "stopped_rpc_exceptions": 0, "stop_requested": False,
                                 "inserted_transition_points": 0},
                  "stop_proof": {"stop_requested_during_inflight_rpc": True, "commands_after_stop": 0,
                                 "total_fake_commands": 0, "cancelled_before_transport_return": True,
                                 "execution": {"inference_calls": 1, "completed_points": 0, "attempted_points": 0,
                                               "faults": 0, "stop_requested": True, "stopped_rpc_exceptions": 1,
                                               "late_response_rows_discarded": 0},
                                 "all_fake_robots_released": True}}
        report.update(copy.deepcopy(modifications))
        kwargs["directory"].mkdir(parents=True, exist_ok=False)
        (kwargs["directory"] / "qualification.json").write_text(json.dumps(report))
        return report

    monkeypatch.setattr(qualification, "collect", collect)
    monkeypatch.setattr(rollout, "run_rollout", lambda transport, **kwargs: rollouts.append(kwargs) or {"fake": True})
    return SimpleNamespace(root=tmp_path, rig=rig, target=target, metadata=metadata, statistics=statistics,
                           connections=connection_calls, transports=transports, collections=collections,
                           modifications=modifications, rollouts=rollouts)


def prepare(fixture, **kwargs):
    return workflow.prepare(backend="lambda", task="put the red cube into the black container",
                            rig=fixture.rig, **kwargs)


def test_cold_preparation_collects_only_official_base_saved_fake_workflow(prepared, monkeypatch):
    from yamkit import modal_qualification
    from yamkit.pi05 import qualification as pi_qualification

    monkeypatch.setattr(modal_qualification, "collect_qualification", lambda *_a, **_kw: pytest.fail("no MA2 substitution"))
    monkeypatch.setattr(pi_qualification, "collect_qualification", lambda *_a, **_kw: pytest.fail("no PI-YAM substitution"))
    selection, result = prepare(prepared)
    assert selection.policy == "pi05-base" and selection.controller_mode == CONTRACT_ID
    assert selection.qualification_path.is_file()
    assert result["ready"] and not result["reused"]
    assert result["hardware_tested"] is False and result["motion_approval_received"] is False
    assert len(prepared.collections) == 1 and prepared.collections[0]["rig_path"] == prepared.rig
    assert prepared.collections[0]["observations"][0]["state"].shape == (14,)
    assert prepared.transports[0].closed and not prepared.rollouts
    assert "fake_private_credential" not in json.dumps(result)


def test_warm_current_qualification_reuses_without_collection_or_motion(prepared):
    first, _ = prepare(prepared)
    prepared.metadata.update(busy=True, warm_signatures=["different transient warm cache"],
                             runtime_provenance={"load_s": 99})
    second, result = prepare(prepared)
    assert result["reused"] and first.qualification_path == second.qualification_path
    assert len(prepared.collections) == 1 and not prepared.rollouts


@pytest.mark.parametrize("field,value", [("instance_id", "new-model-instance"), ("openpi_service_build_id", "f" * 64),
                                      ("statistics_file_sha256", "f" * 64), ("source_sha", "f" * 40),
                                      ("session_expires_at", "new_expiry")])
def test_changed_stable_service_identity_cannot_reuse_cached_qualification(prepared, field, value):
    old, _ = prepare(prepared)
    prepared.metadata[field] = time.time() + 3500 if value == "new_expiry" else value
    new, result = prepare(prepared)
    assert not result["reused"] and new.qualification_path != old.qualification_path
    assert old.qualification_path.exists() and len(prepared.collections) == 2


def test_force_requalifies_without_overwriting_old_evidence(prepared):
    first, _ = prepare(prepared)
    original = first.qualification_path.read_bytes()
    second, result = prepare(prepared, force=True)
    assert not result["reused"] and len(prepared.collections) == 2
    assert first.qualification_path.read_bytes() == original
    assert first.qualification_path != second.qualification_path


def test_invalid_cached_proof_requalifies_and_failed_new_proof_never_becomes_current(prepared):
    selection, _ = prepare(prepared)
    value = json.loads(selection.qualification_path.read_text())
    value["qualified"] = False
    selection.qualification_path.write_text(json.dumps(value))
    prepared.modifications.update(qualified=False, reasons=["explicit fake failure"])
    with pytest.raises(WorkflowError, match="qualification failed"):
        prepare(prepared)
    assert len(prepared.collections) == 2 and not prepared.rollouts
    pointer = prepared.root / "data/inference/openpi/lambda-openpi/qualification.json"
    assert json.loads(pointer.read_text())["path"] == str(selection.qualification_path)
    assert all(value.closed for value in prepared.transports)


@pytest.mark.parametrize("field,value", [("integrated", None), ("stop_proof", []),
                                      ("direct_warm_round_trip_s", None)])
def test_malformed_cached_nested_proof_is_requalified_not_uncaught(prepared, field, value):
    selection, _ = prepare(prepared)
    report = json.loads(selection.qualification_path.read_text())
    report[field] = value
    selection.qualification_path.write_text(json.dumps(report))
    _, result = prepare(prepared)
    assert not result["reused"] and len(prepared.collections) == 2


@pytest.mark.parametrize("kwargs", [{"backend": "modal"}, {"duration": 0}, {"duration": 61},
                                   {"duration": True}, {"duration": float("nan")}, {"task": ""},
                                   {"arms": ("left_follower",)}, {"arms": ("right_follower", "left_follower")}])
def test_invalid_selection_stops_before_network_or_saved_data(prepared, kwargs):
    arguments = {"backend": "lambda", "task": "cube", "rig": prepared.rig, **kwargs}
    with pytest.raises(WorkflowError):
        workflow.prepare(**arguments)
    assert not prepared.connections and not prepared.collections


def test_missing_statistics_or_saved_inputs_never_starts_runtime(prepared, monkeypatch):
    def missing():
        raise ValueError("Install reviewed statistics")

    monkeypatch.setattr(workflow, "load_statistics", missing)
    with pytest.raises(ValueError, match="statistics"):
        prepare(prepared)
    assert not prepared.connections
    monkeypatch.setattr(workflow, "load_statistics", lambda: prepared.statistics)
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_kw: replace(prepared.target, saved_observations=()))
    with pytest.raises(WorkflowError, match="saved observations"):
        prepare(prepared)
    assert not prepared.connections


@pytest.mark.parametrize("confirm,mapping", [(False, False), (True, False), (False, True), (1, True), (True, 1)])
def test_no_motion_approval_is_inferred_from_truthy_values(prepared, monkeypatch, confirm, mapping):
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_kw: pytest.fail("no backend before exact booleans"))
    with pytest.raises(WorkflowError, match="confirmation"):
        workflow.run_prepared(None, confirm_supervised=confirm, accept_mapping=mapping)


@pytest.mark.parametrize("fake", [1, "true", "false", None])
def test_fake_mode_must_be_an_explicit_boolean(prepared, monkeypatch, fake):
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_kw: pytest.fail("malformed fake flag must fail first"))
    with pytest.raises((ValueError, WorkflowError), match="boolean|confirmation"):
        workflow.run_prepared(None, fake_hardware=fake)


def test_fake_run_passes_only_explicit_fake_saved_inputs_without_approval(prepared):
    selection, _ = prepare(prepared)
    workflow.run_prepared(selection, fake_hardware=True)
    assert len(prepared.rollouts) == 1
    arguments = prepared.rollouts[0]
    assert arguments["fake_hardware"] is True
    assert len(arguments["fake_observations"]) == 1
    assert "confirm_supervised" not in arguments and "accept_mapping" not in arguments
    assert prepared.transports[-1].closed


def test_fake_mode_cannot_carry_physical_flags(prepared, monkeypatch):
    monkeypatch.setattr(workflow, "load_target", lambda *_a, **_kw: pytest.fail("no connection"))
    with pytest.raises(WorkflowError, match="cannot carry physical approval"):
        workflow.run_prepared(None, fake_hardware=True, confirm_supervised=True)


def test_invalid_proof_blocks_approved_delegate_before_hardware(prepared):
    selection, _ = prepare(prepared)
    report = json.loads(selection.qualification_path.read_text())
    report["stop_proof"]["commands_after_stop"] = 1
    selection.qualification_path.write_text(json.dumps(report))
    with pytest.raises(WorkflowError, match="execution failed.*no automatic retry"):
        workflow.run_prepared(selection, confirm_supervised=True, accept_mapping=True)
    assert not prepared.rollouts and prepared.transports[-1].closed


def test_late_confirmation_rechecks_session_margin_before_hardware(prepared):
    selection, _ = prepare(prepared)
    prepared.metadata["session_expires_at"] = time.time() + 30
    with pytest.raises(ValueError):
        workflow.run_prepared(selection, confirm_supervised=True, accept_mapping=True)
    assert not prepared.rollouts and prepared.transports[-1].closed


def test_disappearing_proof_is_actionable_and_no_hardware(prepared):
    selection, _ = prepare(prepared)
    selection.qualification_path.unlink()
    with pytest.raises(WorkflowError, match="execution failed.*FileNotFoundError.*no automatic retry"):
        workflow.run_prepared(selection, confirm_supervised=True, accept_mapping=True)
    assert not prepared.rollouts and prepared.transports[-1].closed


def test_runtime_fault_does_not_expose_private_exception_or_claim_release(prepared, monkeypatch):
    selection, _ = prepare(prepared)

    def failed(*_args, **_kwargs):
        raise RuntimeError("fake_private_credential_never_logged_1234567890")

    monkeypatch.setattr(rollout, "run_rollout", failed)
    with pytest.raises(WorkflowError) as caught:
        workflow.run_prepared(selection, fake_hardware=True)
    assert "fake_private_credential" not in str(caught.value)
    assert "released" not in str(caught.value)
    assert prepared.transports[-1].closed


def test_collection_fault_keeps_transports_closed_and_error_sanitized(prepared, monkeypatch):
    def failed(*_args, **_kwargs):
        raise RuntimeError("fake_private_credential_never_logged_1234567890")

    monkeypatch.setattr(qualification, "collect", failed)
    with pytest.raises(WorkflowError) as caught:
        prepare(prepared)
    assert "fake_private_credential" not in str(caught.value)
    assert not prepared.rollouts and all(value.closed for value in prepared.transports)


def test_symlinked_prepared_proof_is_rejected_before_delegate(prepared):
    selection, _ = prepare(prepared)
    alias = prepared.root / "qualification-alias.json"
    alias.symlink_to(selection.qualification_path)
    with pytest.raises(WorkflowError):
        workflow.run_prepared(replace(selection, qualification_path=alias), fake_hardware=True)
    assert not prepared.rollouts and prepared.transports[-1].closed


@pytest.mark.parametrize("section,field,value", [
    ("integrated", "completed_rows", 1249), ("integrated", "predicted_rows", 2499),
    ("integrated", "unknown_partial_dispatches", 1), ("integrated", "prefix_dropped_rows", 1),
    ("integrated", "uncompleted_committed_rows", 1), ("integrated", "attempted_points", 1251),
    ("stop_proof", "total_fake_commands", 1), ("stop_proof", "cancelled_before_transport_return", False),
])
def test_incomplete_execution_or_stop_proof_never_reaches_rollout(prepared, section, field, value):
    selection, _ = prepare(prepared)
    report = json.loads(selection.qualification_path.read_text())
    report[section][field] = value
    selection.qualification_path.write_text(json.dumps(report))
    with pytest.raises(WorkflowError, match="execution failed.*no automatic retry"):
        workflow.run_prepared(selection, fake_hardware=True)
    assert not prepared.rollouts and prepared.transports[-1].closed
