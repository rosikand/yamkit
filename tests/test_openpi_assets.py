"""Public asset lifecycle with fake HTTP; never downloads weights or opens hardware."""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import ClassVar

import pytest

from yamkit.openpi import assets, contract


@pytest.fixture
def pinned(monkeypatch):
    raw = b"official checkpoint fixture"
    item = {"bucket": "openpi-assets", "name": "checkpoints/pi05_base/params/test",
            "generation": "1234", "size": str(len(raw)),
            "md5": base64.b64encode(hashlib.md5(raw, usedforsecurity=False).digest()).decode()}
    monkeypatch.setattr(assets, "manifest", lambda: {"objects": [item]})

    class Response(io.BytesIO):
        headers: ClassVar[dict[str, str]] = {"x-goog-generation": "1234"}

    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return Response(raw)

    monkeypatch.setattr(assets.urllib.request, "urlopen", download)
    return item, raw, calls, Response


def test_pinned_manifest_is_official_base_not_yam_or_lerobot():
    saved = assets.manifest()
    assert saved["checkpoint"] == "gs://openpi-assets/checkpoints/pi05_base"
    assert saved["runtime_revision"] == contract.UPSTREAM_REVISION
    assert len(saved["objects"]) == 30
    assert sum(int(item["size"]) for item in saved["objects"]) == 12446013604
    robot_assets = [item["name"].split("/")[3] for item in saved["objects"] if "/assets/" in item["name"]]
    assert set(robot_assets) == {"arx", "arx_mobile", "droid", "fibocom_mobile", "franka",
                                 "trossen", "trossen_mobile", "ur5e", "ur5e_dual"}
    assert contract.MODEL_CONFIG["action_horizon"] == 50
    assert contract.MODEL_CONFIG["action_dim"] == 32
    assert contract.identity()["physical_ready"] is False
    with pytest.raises(ValueError, match="experimental YAM normalization/decoder"):
        contract.require_yam_contract()


def test_acquisition_pins_generation_and_rehashes_warm_cache(tmp_path, pinned):
    item, raw, calls, _ = pinned
    result = assets.download_assets(tmp_path)
    assert result["all_objects_verified"]
    assert result["physical_ready"] is False
    assert assets.cache_path(tmp_path, item).read_bytes() == raw
    assert len(calls) == 1
    assert calls[0][1] == 30
    assert calls[0][0].endswith("?alt=media&generation=1234")
    assert "%2Fpi05_base%2F" in calls[0][0]
    assert assets.download_assets(tmp_path) == result
    assert len(calls) == 1


def test_wrong_generation_fails_without_committing_checkpoint(tmp_path, pinned):
    item, _, calls, response = pinned
    response.headers = {"x-goog-generation": "4321"}
    with pytest.raises(ValueError, match="different official checkpoint generation"):
        assets.download_assets(tmp_path)
    assert len(calls) == 1
    assert not assets.cache_path(tmp_path, item).exists()
    assert not (tmp_path / "data/openpi/asset-receipt.json").exists()


@pytest.mark.parametrize("field,value", [("size", "1"), ("size", "5000"), ("md5", "bad")])
def test_wrong_length_or_digest_fails(tmp_path, pinned, field, value):
    item, _, calls, _ = pinned
    item[field] = value
    with pytest.raises(ValueError):
        assets.download_assets(tmp_path)
    assert len(calls) == 1
    assert not assets.cache_path(tmp_path, item).exists()


def test_existing_without_receipt_is_preserved_not_trusted(tmp_path, pinned):
    item, _, calls, _ = pinned
    path = assets.cache_path(tmp_path, item)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"historical")
    with pytest.raises(ValueError, match="no immutable acquisition receipt"):
        assets.download_assets(tmp_path)
    assert path.read_bytes() == b"historical"
    assert not calls


def test_existing_partial_is_preserved_never_overwritten(tmp_path, pinned):
    item, _, _, _ = pinned
    path = assets.cache_path(tmp_path, item)
    path.parent.mkdir(parents=True)
    partial = assets.partial_path(tmp_path, item)
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"prior partial")
    with pytest.raises(FileExistsError):
        assets.download_assets(tmp_path)
    assert partial.read_bytes() == b"prior partial"
    assert not path.exists()


def test_modified_cached_object_is_not_silently_repaired(tmp_path, pinned):
    item, raw, calls, _ = pinned
    assets.download_assets(tmp_path)
    assets.cache_path(tmp_path, item).write_bytes(b"X" * len(raw))
    with pytest.raises(ValueError, match="checksum"):
        assets.verify_assets(tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["object", "partial", "receipt"])
def test_object_and_sidecar_symlink_escape_rejected(tmp_path, pinned, kind):
    item, _, calls, _ = pinned
    repo = tmp_path / "repo"
    repo.mkdir()
    path = {"object": assets.cache_path, "partial": assets.partial_path,
            "receipt": assets.object_receipt_path}[kind](repo, item)
    path.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"protected")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="inside this checkout"):
        assets.download_assets(repo)
    assert outside.read_bytes() == b"protected"
    assert not calls


def test_receipt_symlink_escape_rejected(tmp_path, pinned):
    repo = tmp_path / "repo"
    (repo / "data/openpi").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("{}")
    (repo / "data/openpi/asset-receipt.json").symlink_to(outside)
    with pytest.raises(ValueError, match="inside this checkout"):
        assets.download_assets(repo)


def test_whole_manifest_changed_receipt_fails(tmp_path, pinned):
    assets.download_assets(tmp_path)
    path = tmp_path / "data/openpi/asset-receipt.json"
    receipt = json.loads(path.read_text())
    receipt["manifest_sha256"] = "changed"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="pinned manifest"):
        assets.verify_assets(tmp_path)


def test_setup_uses_separate_pinned_environment_and_no_model_launch():
    script = (Path(__file__).parents[1] / "scripts/setup_openpi_inference.sh").read_text()
    assert "--frozen --no-dev --python 3.11.13" in script
    assert 'UV_PROJECT_ENVIRONMENT="$OPENPI_ROOT/data/openpi/venv"' in script
    assert 'UV_LINK_MODE=copy' in script
    assert contract.UPSTREAM_REVISION in script
    assert "sudo" not in script
    assert "yamkit rollout" not in script


def test_receipts_and_partials_never_enter_orbax_scanned_native_directories(tmp_path, pinned):
    item, raw, _, _ = pinned
    item["name"] = "checkpoints/pi05_base/params/array_metadatas/process_0"
    assets.download_assets(tmp_path)
    native = assets.checkpoint_path(tmp_path)
    # This is the exact PathResolver glob in Orbax 0.11.13, not a yamkit-only filter.
    scanned = list((native / "params").glob("array_metadatas/process_*"))
    assert scanned == [native / "params/array_metadatas/process_0"]
    assert scanned[0].read_bytes() == raw
    assert {path.relative_to(native).as_posix() for path in native.rglob("*") if path.is_file()} == {
        "params/array_metadatas/process_0"}
    assert assets.object_receipt_path(tmp_path, item).is_file()
    assert not assets.object_receipt_path(tmp_path, item).is_relative_to(native)
    assert not assets.partial_path(tmp_path, item).is_relative_to(native)
    assets.verify_assets(tmp_path)


def test_legacy_orbax_receipt_sidecar_rejected_before_native_restore(tmp_path, pinned):
    item, _, _, _ = pinned
    item["name"] = "checkpoints/pi05_base/params/array_metadatas/process_0"
    assets.download_assets(tmp_path)
    legacy = assets.cache_path(tmp_path, item).with_name("process_0.yamkit-receipt.json")
    legacy.write_text('{"generation":"1234"}')
    with pytest.raises(ValueError, match="foreign files"):
        assets.verify_assets(tmp_path)
    assert legacy.read_text() == '{"generation":"1234"}'
