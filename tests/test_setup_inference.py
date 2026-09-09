"""Exercise bootstrap orchestration with fake uv; no downloads or GPU/device calls."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/setup_inference.sh"


@pytest.fixture
def inference_checkout(tmp_path):
    root = tmp_path / "checkout with spaces"
    for relative in ("scripts/setup_inference.sh", "configs/modal-requirements.txt",
                     "src/yamkit/inference/service.py",
                     "plugins/lerobot_robot_yamkit/lerobot_robot_yamkit/yam_follower.py"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    shutil.copyfile(SOURCE, root / "scripts/setup_inference.sh")
    tools = root / ".tools"
    tools.mkdir()
    fake_uv = tools / "uv"
    # Avoid the parent test environment's yamkit_env.pth rewriting the fake checkout's caches.
    fake_uv.write_text(f"#!{sys.executable} -S\n" + textwrap.dedent('''
        import json, os, pathlib, sys
        root = pathlib.Path.cwd()
        args = sys.argv[1:]
        with (root / 'uv-calls.jsonl').open('a') as handle:
            handle.write(json.dumps({'argv': args, 'environment': {
                key: value for key, value in os.environ.items()
                if key.startswith(('UV_', 'HF_', 'TORCH_', 'XDG_', 'TRITON_', 'CUDA_'))
                or key in ('TMPDIR', 'PYTHONPATH', 'VIRTUAL_ENV', 'PYTHONNOUSERSITE')
            }}) + '\\n')
        assert args[0] == '--no-config'
        if args[1:3] == ['python', 'install']:
            binary = root / '.uv-python' / 'fake-python' / 'bin' / 'python'
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text('#!/bin/sh\\nexit 0\\n')
            binary.chmod(0o755)
        elif args[1] == 'venv':
            environment = root / '.venv-inference'
            (environment / 'bin').mkdir(parents=True)
            (environment / 'pyvenv.cfg').write_text('include-system-site-packages = false\\n')
            (environment / 'bin' / 'python').symlink_to(root / '.uv-python/fake-python/bin/python')
        elif args[1] != 'pip':
            raise AssertionError('Unexpected bootstrap operation')
    '''))
    fake_uv.chmod(0o755)
    return root


def run_setup(root, *arguments):
    environment = {**os.environ, "HF_HOME": "/ignored-parent-hf-cache", "PYTHONPATH": "/ignored-parent-python"}
    return subprocess.run(["bash", "scripts/setup_inference.sh", *arguments], cwd=root, env=environment,
                          capture_output=True, text=True, timeout=15, check=False)


def test_bootstrap_is_local_and_repeatable_without_cpu_project_configuration(inference_checkout):
    root = inference_checkout
    for _ in range(2):
        result = run_setup(root)
        assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (root / "uv-calls.jsonl").read_text().splitlines()]
    assert sum(call["argv"][1] == "venv" for call in calls) == 1
    for call in calls:
        assert call["argv"][0] == "--no-config"
        for key in ("UV_PYTHON_INSTALL_DIR", "UV_PYTHON_BIN_DIR", "UV_CACHE_DIR", "UV_TOOL_DIR",
                    "UV_TOOL_BIN_DIR", "UV_CREDENTIALS_DIR", "HF_HOME", "HF_LEROBOT_HOME", "TORCH_HOME",
                    "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "TRITON_CACHE_DIR",
                    "CUDA_CACHE_PATH", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV"):
            assert Path(call["environment"][key]).is_relative_to(root), key
        assert call["environment"]["PYTHONNOUSERSITE"] == "1"
    syncs = [call["argv"] for call in calls if call["argv"][1:3] == ["pip", "sync"]]
    assert len(syncs) == 2 and all(args[-1] == "configs/modal-requirements.txt" for args in syncs)
    installs = [call["argv"] for call in calls if call["argv"][1:3] == ["pip", "install"]]
    assert all("--constraint" in args and "uvicorn==0.52.4" in args and "h11==0.16.0" in args
               for args in installs)
    assert not (root / ".venv").exists()
    result = subprocess.run(["bash", "-c", 'source data/inference/env.sh; printf "%s\\n" "$VIRTUAL_ENV" "$HF_HOME"'],
                            cwd=root, capture_output=True, text=True, timeout=5, check=True)
    assert result.stdout.splitlines() == [str(root / ".venv-inference"), str(root / "data/hf")]


def test_bootstrap_refuses_cache_escape_before_mutation(inference_checkout, tmp_path):
    root = inference_checkout
    outside = tmp_path / "outside-checkout"
    outside.mkdir()
    (root / "data").symlink_to(outside, target_is_directory=True)
    result = run_setup(root)
    assert result.returncode == 1 and "outside this checkout" in result.stderr
    assert list(outside.iterdir()) == [] and not (root / "uv-calls.jsonl").exists()


def test_bootstrap_preserves_unknown_existing_environment(inference_checkout):
    root = inference_checkout
    environment = root / ".venv-inference"
    environment.mkdir()
    marker = environment / "user-file"
    marker.write_text("keep")
    result = run_setup(root)
    assert result.returncode == 1 and "refusing to overwrite" in result.stderr
    assert marker.read_text() == "keep"
    calls = [json.loads(line) for line in (root / "uv-calls.jsonl").read_text().splitlines()]
    assert all(call["argv"][1] != "pip" for call in calls)


@pytest.mark.parametrize("argument,expected", [("--help", 0), ("--system", 2)])
def test_bootstrap_help_and_bad_arguments_do_not_start_installation(inference_checkout, argument, expected):
    root = inference_checkout
    result = run_setup(root, argument)
    assert result.returncode == expected
    assert not (root / "uv-calls.jsonl").exists() and not (root / "data").exists()
