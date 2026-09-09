"""HTTP qualification binds the exact local command validation and shaping inputs."""

import shutil
from pathlib import Path

import pytest

from yamkit import paths
from yamkit.inference import identity


@pytest.fixture
def mirrored_http_source(tmp_path, monkeypatch):
    original = Path(identity.__file__).resolve().parents[1]
    root = Path(paths.ROOT)
    before = identity.inference_build_id()
    # Mirror the image's Python source copy without importing any driver or SDK.
    package = tmp_path / "src/yamkit"
    for source in original.rglob("*.py"):
        target = package / source.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for relative in ("configs/modal-requirements.txt", identity.FOLLOWER_SOURCE_RELATIVE):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, target)
    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(identity, "__file__", str(package / "inference/identity.py"))
    assert identity.inference_build_id() == before
    return tmp_path, before


@pytest.mark.parametrize("relative", ["src/yamkit/arm.py", identity.FOLLOWER_SOURCE_RELATIVE])
def test_hardware_command_validation_source_changes_invalidate_identity(mirrored_http_source, relative):
    root, before = mirrored_http_source
    source = root / relative
    source.write_bytes(source.read_bytes() + b"\n# changed command validation/limits\n")
    assert identity.inference_build_id() != before


def test_http_source_identity_fails_closed_without_follower_validation_source(mirrored_http_source):
    root, _ = mirrored_http_source
    (root / identity.FOLLOWER_SOURCE_RELATIVE).unlink()
    with pytest.raises(FileNotFoundError):
        identity.inference_build_id()
