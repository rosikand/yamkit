"""Public, generation-pinned official assets; never handles authentication tokens."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

from .contract import MANIFEST_PATH, identity


def local_path(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise ValueError("OpenPI assets and evidence must stay inside this checkout")
    return resolved


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def cache_path(root: Path, item: dict) -> Path:
    return local_path(root, Path("data/openpi/cache") / item["bucket"] / item["name"])


def checkpoint_path(root: Path) -> Path:
    return local_path(root, Path("data/openpi/cache/openpi-assets/checkpoints/pi05_base"))


def object_receipt_path(root: Path, item: dict) -> Path:
    key = item["bucket"] + "/" + item["name"]
    name = hashlib.sha256(key.encode()).hexdigest() + ".json"
    return local_path(root, Path("data/openpi/object-receipts") / name)


def partial_path(root: Path, item: dict) -> Path:
    key = item["bucket"] + "/" + item["name"]
    name = hashlib.sha256(key.encode()).hexdigest() + ".partial"
    return local_path(root, Path("data/openpi/partials") / name)


def verify_native_inventory(root: Path) -> None:
    """Orbax scans process_* files: no receipt, partial, or foreign file may enter its tree."""
    checkpoint = checkpoint_path(root)
    prefix = "checkpoints/pi05_base/"
    expected = {item["name"][len(prefix):] for item in manifest()["objects"]
                if item["bucket"] == "openpi-assets" and item["name"].startswith(prefix)}
    actual = {path.relative_to(checkpoint).as_posix()
              for path in checkpoint.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != expected:
        raise ValueError("Official checkpoint tree contains missing or foreign files; move acquisition "
                         "receipts/partials outside the native tree before model loading")


def _hash_file(path: Path) -> tuple[str, str, int]:
    sha, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha.hexdigest(), base64.b64encode(md5.digest()).decode("ascii"), size


def verify_assets(root: Path) -> dict:
    """Rehash every stored object against the receipt from generation-pinned acquisition."""
    root = root.resolve()
    receipt_path = local_path(root, Path("data/openpi/asset-receipt.json"))
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("manifest_sha256") != identity()["manifest_sha256"]:
        raise ValueError("OpenPI asset receipt does not match the pinned manifest")
    verify_native_inventory(root)
    entries = receipt.get("objects", {})
    for item in manifest()["objects"]:
        key = item["bucket"] + "/" + item["name"]
        expected = entries.get(key, {})
        path = cache_path(root, item)
        sha, md5, size = _hash_file(path)
        if (size != int(item["size"]) or expected.get("sha256") != sha
                or expected.get("generation") != item["generation"]
                or (item.get("md5") is not None and md5 != item["md5"])):
            raise ValueError("Official OpenPI asset checksum or immutable generation differs")
    return receipt


def download_assets(root: Path) -> dict:
    """One HTTP attempt/object, finite socket timeout; failed partial files are retained.

    GCS generations pin immutable bytes even for composite objects without MD5.
    A SHA-256 receipt is committed after all objects pass generation and size checks.
    Existing complete receipts are verified without downloading. Unreceipted existing
    files are refused, not silently trusted or overwritten.
    """
    root = root.resolve()
    receipt_path = local_path(root, Path("data/openpi/asset-receipt.json"))
    if receipt_path.exists():
        return verify_assets(root)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {**identity(), "objects": {}, "all_objects_verified": False}
    for item in manifest()["objects"]:
        path = cache_path(root, item)
        object_receipt = object_receipt_path(root, item)
        key = item["bucket"] + "/" + item["name"]
        if path.exists():
            if not object_receipt.is_file():
                raise ValueError("Existing official asset has no immutable acquisition receipt; preserve and review it")
            saved = json.loads(object_receipt.read_text())
            sha, md5, size = _hash_file(path)
            if (saved.get("generation") != item["generation"] or saved.get("sha256") != sha
                    or size != int(item["size"]) or (item.get("md5") and md5 != item["md5"])):
                raise ValueError("Existing official asset differs from its pinned generation or checksum")
            receipt["objects"][key] = saved
            continue
        partial = partial_path(root, item)
        path.parent.mkdir(parents=True, exist_ok=True)
        object_receipt.parent.mkdir(parents=True, exist_ok=True)
        partial.parent.mkdir(parents=True, exist_ok=True)
        url = ("https://storage.googleapis.com/download/storage/v1/b/" + item["bucket"]
               + "/o/" + urllib.parse.quote(item["name"], safe="")
               + "?alt=media&generation=" + item["generation"])
        sha, md5, size = hashlib.sha256(), hashlib.md5(usedforsecurity=False), 0
        with urllib.request.urlopen(url, timeout=30) as response, partial.open("xb") as stream:
            if response.headers.get("x-goog-generation") != item["generation"]:
                raise ValueError("GCS returned a different official checkpoint generation")
            while chunk := response.read(8 * 1024 * 1024):
                stream.write(chunk)
                sha.update(chunk)
                md5.update(chunk)
                size += len(chunk)
                if size > int(item["size"]):
                    raise ValueError("Official asset exceeded its pinned size")
        if (size != int(item["size"])
                or (item.get("md5") and base64.b64encode(md5.digest()).decode("ascii") != item["md5"])):
            raise ValueError("Official asset size/checksum does not match its immutable manifest")
        saved = {"generation": item["generation"], "size": size, "sha256": sha.hexdigest()}
        with object_receipt.open("x") as stream:
            json.dump(saved, stream, indent=2)
        os.replace(partial, path)
        receipt["objects"][key] = saved
        print(json.dumps({"object": key, "bytes": size, "verified_generation": item["generation"]}), flush=True)
    receipt["all_objects_verified"] = True
    verify_native_inventory(root)
    with receipt_path.open("x") as stream:
        json.dump(receipt, stream, indent=2)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    receipt = verify_assets(args.root) if args.verify_only else download_assets(args.root)
    print(json.dumps({"all_objects_verified": receipt["all_objects_verified"],
                      "objects": len(receipt["objects"]), "physical_ready": False}))


if __name__ == "__main__":
    main()
