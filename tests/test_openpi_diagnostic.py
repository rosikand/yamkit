"""Explicitly fake native policy, normalized arrays only, no robot-side imports."""

import ast
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yamkit.openpi import contract, diagnostic, runtime


def images():
    return {key: np.full((480, 640, 3), index * 37, dtype=np.uint8)
            for index, key in enumerate(contract.IMAGE_KEYS)}


def native_fixture():
    calls = []

    def infer(observation, *, noise=None):
        calls.append(observation)
        assert observation["state"].shape == (32,)
        assert not observation["state"].any()
        # Out-of-normalized-range values must NOT be clipped or projected.
        return {"actions": np.arange(1600, dtype=np.float32).reshape(50, 32) / 500 - 1.25}

    policy = SimpleNamespace(infer=infer)
    wrapper = runtime.OfficialPi05Diagnostic(policy, contract.identity())
    return wrapper, calls


def test_saved_rgb_mapping_uses_no_measured_state_or_other_statistics(tmp_path):
    source = tmp_path / "saved.npz"
    np.savez(source, top=np.zeros((480, 640, 3), dtype=np.uint8),
             left_wrist=np.full((480, 640, 3), 60, dtype=np.uint8),
             right_wrist=np.full((480, 640, 3), 90, dtype=np.uint8),
             state=np.array([{"malformed_state_must_not_load": True}], dtype=object))
    result = diagnostic.load_saved_images(source)
    assert result["left_wrist_0_rgb"][0, 0, 0] == 60
    assert result["right_wrist_0_rgb"][0, 0, 0] == 90
    observation = runtime.normalized_observation(result, "put cube in bowl")
    np.testing.assert_array_equal(observation["state"], np.zeros(32, dtype=np.float32))
    assert "actions" not in observation


@pytest.mark.parametrize("shape,dtype", [((3, 480, 640), np.uint8), ((480, 640, 3), np.float32),
                                          ((0, 640, 3), np.uint8), ((2161, 1, 3), np.uint8)])
def test_rejects_ambiguous_image_contract(shape, dtype):
    rgb = images()
    rgb["base_0_rgb"] = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError, match="HWC uint8"):
        runtime.normalized_observation(rgb, "task")


@pytest.mark.parametrize("task", ["", "   ", None, 123, "a" * 513])
def test_rejects_invalid_task(task):
    with pytest.raises(ValueError, match="nonempty diagnostic task"):
        runtime.normalized_observation(images(), task)


@pytest.mark.parametrize("shape", [(30, 14), (50, 14), (30, 32), (1, 50, 32)])
def test_no_first14_or_other_shape_fallback(shape):
    with pytest.raises(ValueError, match="50x32"):
        runtime.validate_normalized_chunk(np.zeros(shape, dtype=np.float32))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_is_diagnostic_failure_not_repaired(value):
    chunk = np.zeros((50, 32), dtype=np.float32)
    chunk[0, 6] = value
    with pytest.raises(ValueError, match="non-finite"):
        runtime.validate_normalized_chunk(chunk)


def test_native_shape_and_all_values_are_preserved():
    wrapper, calls = native_fixture()
    result = wrapper.predict(images(), "task")
    assert result.shape == (50, 32)
    assert result[0, 0] == -1.25
    assert result[-1, -1] > 1.0
    assert len(calls) == 1


def test_successful_normalized_diagnostic_still_blocks_yam(tmp_path):
    wrapper, calls = native_fixture()
    output = tmp_path / "evidence"
    result = diagnostic.run_diagnostic(wrapper, [images(), images()], "task", 3, output)
    assert result["software_inference_passed"] is True
    assert result["same_noise_native_wrapper_parity"]["bitwise_equal"] is True
    assert result["completed_warm_samples"] == 3
    assert result["model_rows_returned"] == 150
    assert result["model_rows_sent_to_robot"] == 0
    assert result["saved_measured_states_used"] is False
    assert result["fake_yam_rollout_attempted"] is False
    assert result["physical_ready"] is False
    assert result["qualified_for_yam"] is False
    assert result["hardware_tested"] is False
    assert result["physical_task_success"] is None
    assert len(calls) == 5  # Two identical-noise parity calls, three native random calls.
    assert len(list(output.glob("native-chunk-*.json"))) == 3
    saved = json.loads((output / "native-chunk-000.json").read_text())
    assert np.asarray(saved["chunk"]).shape == (50, 32)
    assert saved["chunk"][0][0] == -1.25
    assert json.loads((output / "diagnostic.json").read_text())["physical_ready"] is False


def test_malicious_runtime_metadata_cannot_promote_readiness(tmp_path):
    wrapper, _ = native_fixture()
    wrapper.provenance.update(physical_ready=True, hardware_tested=True, qualified_for_yam=True)
    result = diagnostic.run_diagnostic(wrapper, [images()], "task", 1, tmp_path / "evidence")
    assert not result["physical_ready"]
    assert not result["hardware_tested"]
    assert not result["qualified_for_yam"]


def test_failure_saved_without_untrusted_exception_text(tmp_path):
    wrapper, _ = native_fixture()

    def fail(*args, **kwargs):
        raise RuntimeError("secret-shaped-fixture-value-must-not-be-reported")

    wrapper.predict = fail
    result = diagnostic.run_diagnostic(wrapper, [images()], "task", 1, tmp_path / "evidence")
    assert result["failure_type"] == "RuntimeError"
    assert result["software_inference_passed"] is False
    assert "secret-shaped-fixture" not in json.dumps(result)
    assert not result["physical_ready"]


def test_ctrl_c_saves_incomplete_evidence_and_propagates(tmp_path):
    wrapper, _ = native_fixture()

    def stop(*args, **kwargs):
        raise KeyboardInterrupt

    wrapper.predict = stop
    output = tmp_path / "evidence"
    with pytest.raises(KeyboardInterrupt):
        diagnostic.run_diagnostic(wrapper, [images()], "task", 1, output)
    result = json.loads((output / "diagnostic.json").read_text())
    assert not result["software_inference_passed"]
    assert not result["physical_ready"]


def test_no_evidence_overwrite(tmp_path):
    wrapper, _ = native_fixture()
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(wrapper, [images()], "task", 1, tmp_path)


@pytest.mark.parametrize("requests", [0, -1, 51, True])
def test_request_budget_is_bounded(tmp_path, requests):
    wrapper, _ = native_fixture()
    with pytest.raises(ValueError, match="1..50"):
        diagnostic.run_diagnostic(wrapper, [images()], "task", requests, tmp_path / "evidence")
    assert not (tmp_path / "evidence").exists()


def test_configure_environment_refuses_escaping_cache_symlink(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "data/openpi").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "data/openpi/cache").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="inside this checkout"):
        runtime.configure_environment(repo)


def test_diagnostic_package_has_no_robot_imports():
    package = Path(runtime.__file__).parent
    forbidden = ("yamkit.arm", "yamkit.teleop", "yamkit.camera", "yamkit.pi05.rollout",
                 "lerobot.robots", "lerobot.cameras", "i2rt", "can", "cv2")
    for file in package.glob("*.py"):
        tree = ast.parse(file.read_text())
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(name.startswith(forbidden) for name in imports)


def test_upstream_revision_and_lock_are_recorded_not_a_lerobot_substitute():
    assert contract.UPSTREAM_REVISION == "215abfb217dbac7d5f1273282331b9b1866c0479"
    assert contract.identity()["runtime"] == "official OpenPI JAX"
    assert contract.identity()["manifest_sha256"] == hashlib.sha256(contract.MANIFEST_PATH.read_bytes()).hexdigest()


def test_oversized_file_fails_before_loading_archive(tmp_path, monkeypatch):
    source = tmp_path / "large.npz"
    with source.open("wb") as stream:
        stream.truncate(diagnostic.MAX_SAVED_FILE_BYTES + 1)
    monkeypatch.setattr(diagnostic.np, "load", lambda *_a, **_kw: pytest.fail("no allocation"))
    with pytest.raises(ValueError, match="16 MiB"):
        diagnostic.load_saved_images(source)


def test_compressed_archive_expansion_is_bounded(tmp_path, monkeypatch):
    source = tmp_path / "compressed.npz"
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("top.npy", b"0" * (diagnostic.MAX_SAVED_FILE_BYTES + 1))
    assert source.stat().st_size < diagnostic.MAX_SAVED_FILE_BYTES
    monkeypatch.setattr(diagnostic.np, "load", lambda *_a, **_kw: pytest.fail("no decompression"))
    with pytest.raises(ValueError, match="member sizes"):
        diagnostic.load_saved_images(source)


@pytest.mark.parametrize("failure,exit_code", [(RuntimeError("credential-shaped-value"), 2),
                                               (KeyboardInterrupt(), 130)])
def test_main_loader_failure_saved_sanitized_and_physical_block_preserved(tmp_path, monkeypatch, capsys,
                                                                        failure, exit_code):
    source = tmp_path / "saved.npz"
    np.savez(source, **{name: np.zeros((1, 1, 3), dtype=np.uint8) for name in contract.SAVED_IMAGE_MAP})
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", ["diagnostic", "--root", str(tmp_path), "--saved-observation", str(source),
                                     "--task", "task", "--output", str(output)])

    def fail(_root):
        raise failure

    monkeypatch.setattr(runtime.OfficialPi05Diagnostic, "load", fail)
    with pytest.raises(SystemExit) as exited:
        diagnostic.main()
    assert exited.value.code == exit_code
    report = json.loads((output / "diagnostic.json").read_text())
    assert report["failure_phase"] == "official_runtime_load"
    assert report["failure_type"] == type(failure).__name__
    assert not report["physical_ready"] and not report["qualified_for_yam"]
    assert "YAM" in " ".join(report["blockers"])
    assert "credential-shaped-value" not in json.dumps(report)
    assert "credential-shaped-value" not in capsys.readouterr().out


def test_main_too_many_inputs_fails_before_reading_or_model_load(tmp_path, monkeypatch):
    argv = ["diagnostic", "--root", str(tmp_path), "--task", "task", "--output", "evidence"]
    argv += [argument for _ in range(51) for argument in ("--saved-observation", "saved.npz")]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(diagnostic, "load_saved_images", lambda *_: pytest.fail("no file reads"))
    monkeypatch.setattr(runtime.OfficialPi05Diagnostic, "load", lambda *_: pytest.fail("no model load"))
    with pytest.raises(SystemExit) as exited:
        diagnostic.main()
    assert exited.value.code == 2
    report = json.loads((tmp_path / "evidence/diagnostic.json").read_text())
    assert report["failure_phase"] == "saved_input_validation"
    assert not report["physical_ready"]
