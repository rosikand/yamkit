"""Shared validation, CLI routing and cloud ownership without credentials or hardware."""

import asyncio
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from yamkit.cli import app
from yamkit.deployment import InferenceOptions
from yamkit.inference_check import run_check


def test_local_default_and_custom_checkpoint():
    options = InferenceOptions(policy="outputs/custom", task="move cube").validate(motion=True)
    assert options.backend == "local" and options.rtc is False
    assert "--backend" in options.cli_args()


@pytest.mark.parametrize("changes", [
    {"backend": "bad"}, {"duration": float("nan")}, {"fps": float("inf")}, {"task": ""},
    {"backend": "modal", "rtc": True}, {"backend": "modal", "async_chunks": False},
    {"backend": "modal", "gpu": "H100"}, {"modal_app": "unrelated-app"},
])
def test_invalid_options_fail_offline(changes):
    with pytest.raises(ValueError):
        InferenceOptions(**{"policy": "molmoact2", **changes}).validate()


@pytest.mark.parametrize("backend", ["modal", "local"])
@pytest.mark.parametrize("policy", ["smolvla", "pi05"])
def test_native_forward_pass_does_not_authorize_physical_mapping(policy, backend):
    InferenceOptions(policy=policy, backend=backend).validate()
    with pytest.raises(ValueError, match="mapping"):
        InferenceOptions(policy=policy, backend=backend).validate(motion=True)


@pytest.mark.parametrize("qualification_enabled", [False, True])
def test_modal_performance_gate_is_independent_of_source_mapping_and_readiness(monkeypatch, qualification_enabled):
    from yamkit.inference import performance
    from yamkit.inference.profiles import get_profile
    from yamkit.remote_policy import YamkitRemoteConfig
    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy

    monkeypatch.setattr(performance, "QUALIFICATION_GATE_ENABLED", qualification_enabled)
    monkeypatch.setattr("yamkit.inference.qualification.is_cloud_host", lambda: True)
    metadata = get_profile("molmoact2").metadata()
    assert metadata["mapping_verified"]
    assert metadata["physical_validation"] == "not performed"
    assert metadata["physical_modal_rollout_allowed"] is False
    InferenceOptions("molmoact2", backend="modal").validate()  # probes/checks still available
    with pytest.raises(ValueError, match="BLOCKED"):
        InferenceOptions("molmoact2", backend="modal").validate(motion=True)
    monkeypatch.setattr("yamkit.remote_policy.modeling_yamkit_remote.make_transport",
                        lambda cfg: pytest.fail("gate must run before readiness or paid transport"))
    with pytest.raises(ValueError, match="BLOCKED"):
        YamkitRemotePolicy(YamkitRemoteConfig(modal_app="fake-app"))
    result = CliRunner().invoke(app, ["rollout", "--backend", "modal", "--policy", "molmoact2",
                                     "--task", "pick cube", "--rig", "/missing"])
    assert result.exit_code != 0 and "BLOCKED" in result.output


def test_profile_id_changes_with_configuration():
    a = InferenceOptions("molmoact2", backend="modal")
    b = InferenceOptions("molmoact2", backend="modal", center_crop=True)
    assert a.operation_key != b.operation_key
    assert a.operation_key == InferenceOptions("molmoact2", backend="modal").operation_key


def test_cli_local_rollout_keeps_lerobot_command(rig, monkeypatch):
    called = []
    monkeypatch.setattr("yamkit.cli._exec_lerobot", lambda *args: called.append(args))
    result = CliRunner().invoke(app, ["rollout", "--policy", "outputs/custom", "--task", "pick cube",
                                     "--rig", str(rig.path), "--dry-run"])
    assert result.exit_code == 0, result.output
    script, argv, dry = called[0]
    assert script == "lerobot_rollout" and dry
    assert "--policy.path=outputs/custom" in argv
    assert "--robot.type=bi_yam_follower" in argv
    assert not any("inference.type" in x for x in argv)


def test_cli_remote_invalid_combination_fails_before_reading_rig():
    result = CliRunner().invoke(app, ["rollout", "--backend", "modal", "--policy", "molmoact2",
                                     "--task", "pick cube", "--rtc", "--rig", "/missing"])
    assert result.exit_code != 0 and "unguided" in result.output


def test_three_check_calls_are_fresh_and_processors_are_not_overridden(monkeypatch):
    from yamkit.inference.profiles import get_profile
    from yamkit.inference.service import ModelRuntime

    profile = get_profile("smolvla")
    seen = []

    def predict(request):
        seen.append(request)
        response = {key: request[key] for key in (
            "protocol_version", "profile", "model_revision", "session_id", "sequence_id", "observation_time")}
        response.update(chunk=[[0.] * 6] * 50, action_names=list(profile.action_names),
                        action_units="checkpoint_native",
                        timing={"preprocess_s": 0., "inference_s": 0., "postprocess_s": 0., "total_s": 0.})
        return response

    monkeypatch.setattr(ModelRuntime, "load", lambda *a, **k: SimpleNamespace(ready=dict, predict_chunk=predict))
    result = run_check("smolvla")
    assert len(seen) == len(result["fresh_chunks"]) == 3
    assert [x["sequence_id"] for x in seen] == [0, 1, 2]
    assert len({x["session_id"] for x in seen}) == 1
    assert all(x["mode"] == "native_fixture" for x in seen)
    assert result["warm"]["sample_count"] == 2 and result["queue_depth"] is None


def test_modal_call_deadline_cancels_once():
    from yamkit.modal_ops import call

    cancelled = []

    async def get():
        await asyncio.sleep(10)

    async def cancel():
        cancelled.append(True)

    async def spawn(*args):
        return SimpleNamespace(get=SimpleNamespace(aio=get), cancel=SimpleNamespace(aio=cancel))

    with pytest.raises(TimeoutError):
        call(SimpleNamespace(spawn=SimpleNamespace(aio=spawn)), timeout=.01, call_mode="spawn")
    assert cancelled == [True]


def test_cloud_shutdown_requires_owned_receipt(monkeypatch, tmp_path):
    from yamkit import modal_ops

    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned.json")
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *a, **k: pytest.fail("must not stop unrelated apps"))
    assert modal_ops.shutdown()["status"] == "stopped"
    modal_ops._save({"app_name": "unrelated-app", "status": "ready"})
    with pytest.raises(ValueError):
        modal_ops.shutdown()


def test_owned_shutdown_uses_validated_argv_and_excludes_unrelated_credentials(monkeypatch, tmp_path):
    from yamkit import modal_ops

    monkeypatch.setattr(modal_ops, "receipt_path", lambda: tmp_path / "owned.json")
    modal_ops._save({"app_name": "yamkit-vla-test", "app_id": "ap-test", "status": "ready"})
    monkeypatch.setenv("YAMKIT_OPENAI_API_KEY", "unrelated-test-value")
    monkeypatch.setenv("DATABASE_URL", "unrelated-test-database")
    seen = []
    monkeypatch.setattr(modal_ops.subprocess, "run", lambda *a, **k: seen.append((a, k)) or SimpleNamespace(returncode=0, stdout="[]"))
    assert modal_ops.shutdown()["status"] == "stopped"
    argv, kwargs = seen[0]
    assert argv[0][-1] == "ap-test" and "--yes" in argv[0]
    assert "YAMKIT_OPENAI_API_KEY" not in kwargs["env"] and "DATABASE_URL" not in kwargs["env"]
