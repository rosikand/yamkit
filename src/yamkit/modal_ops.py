"""Explicit deployment and shutdown of this workspace's dedicated Modal service.

No module import, catalog query or page load calls Modal. The ownership receipt contains
identifiers and public profile metadata only; credentials are read by the outbound SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

from .paths import OUTPUT_DIR, ROOT


def credential_status() -> dict[str, str]:
    return {key: "SET" if os.environ.get(key) else "MISSING"
            for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN")}


def receipt_path() -> Path:
    return OUTPUT_DIR / "modal" / "owned-service.json"


def owned_service() -> dict | None:
    try:
        return json.loads(receipt_path().read_text())
    except FileNotFoundError:
        return None


def _save(receipt: dict) -> None:
    path = receipt_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


def call(method, *args, timeout: float = 120):
    """Bound submission + response, with best-effort cancellation and no retry."""
    async def invoke():
        handle = None
        try:
            async with asyncio.timeout(timeout):
                handle = await method.spawn.aio(*args)
                return await handle.get.aio()
        except BaseException:
            if handle is not None:
                try:
                    async with asyncio.timeout(3):
                        await handle.cancel.aio()
                except Exception:  # noqa: BLE001, S110 — preserve original error; server has finite timeouts
                    pass
            raise
    return asyncio.run(invoke())


def service_handle(app_name: str, profile_id: str):
    import modal

    return modal.Cls.from_name(app_name, "PolicyService")()


def prepare(profile_name: str, *, gpu: str = "L40S", development: bool = False,
            cache_volume_name: str = "yamkit-policy-weights") -> dict:
    """Deploy and warm explicitly. A failed prepare shuts down only its owned app."""
    from .inference.modal_service import create_app
    from .inference.profiles import get_profile

    profile = get_profile(profile_name)
    prior = owned_service()
    if prior and prior.get("status") != "stopped":
        if prior.get("profile_id") != profile.id or prior.get("revision") != profile.revision:
            raise ValueError("shut down the owned cloud service before preparing a different model")
        if prior.get("status") != "ready":
            raise ValueError("previous preparation is incomplete; shut down the owned service before retrying")
        metadata = call(service_handle(prior["app_name"], profile.id).ready, timeout=300)
        _validate_ready(metadata, profile)
        return {**prior, "metadata": metadata}
    app_name = "yamkit-vla-" + uuid.uuid4().hex[:16]
    receipt = {"app_name": app_name, "app_id": None, "profile_id": profile.id,
               "revision": profile.revision, "status": "preparing", "created_at": time.time(),
               "development": development, "gpu": gpu, "scaledown_window": 15 if development else 300}
    _save(receipt)
    try:
        app = create_app(profile_id=profile.id, gpu=gpu, development=development, app_name=app_name,
                         cache_volume_name=cache_volume_name)
        async def deploy():
            async with asyncio.timeout(600):
                await app.deploy.aio(name=app_name)
        asyncio.run(deploy())
        receipt["app_id"] = app.app_id
        _save(receipt)
        metadata = call(service_handle(app_name, profile.id).ready, timeout=300 if development else 660)
        _validate_ready(metadata, profile)
        receipt.update(status="ready", metadata=metadata)
        _save(receipt)
        return receipt
    except BaseException:
        # If deployment failed before returning its ID, the unique owned name remains sufficient.
        shutdown()
        raise


def _validate_ready(metadata: dict, profile) -> None:
    if metadata.get("profile_id") != profile.id or metadata.get("revision") != profile.revision:
        raise ValueError("service readiness does not match the requested profile revision")


def shutdown() -> dict:
    """Stop only the exact app named in our ownership receipt; robot Stop is separate."""
    receipt = owned_service()
    if receipt is None or receipt.get("status") == "stopped":
        return {"status": "stopped", "owned_app": None}
    name = receipt.get("app_name", "")
    if not name.startswith("yamkit-vla-"):
        raise ValueError("invalid owned app receipt")
    env = {k: v for k, v in os.environ.items() if k not in ("YAMKIT_OPENAI_API_KEY", "DATABASE_URL", "HF_TOKEN")}
    env["MODAL_CONFIG_PATH"] = str(ROOT / ".context" / "modal.toml")
    completed = subprocess.run([sys.executable, "-m", "modal", "app", "stop", "--yes",
                                receipt.get("app_id") or name],
                               env=env, capture_output=True, text=True, timeout=30, check=False)
    app_id = receipt.get("app_id")
    if app_id:
        # Stop is asynchronous at the control plane. Confirm container retirement with a
        # finite poll before a subsequent model can acquire this workspace's pool.
        containers = None
        for attempt in range(6):
            inventory = subprocess.run([sys.executable, "-m", "modal", "container", "list",
                                        "--app-id", app_id, "--json"], env=env, capture_output=True,
                                       text=True, timeout=15, check=False)
            if inventory.returncode == 0:
                containers = json.loads(inventory.stdout)
                if not containers:
                    break
            if attempt < 5:
                time.sleep(1)
        if containers != []:
            receipt["status"] = "shutdown_failed"
            _save(receipt)
            raise RuntimeError("owned Modal containers have not retired; no other model may be prepared")
        receipt["remaining_containers"] = 0
    elif completed.returncode:
        receipt["status"] = "shutdown_failed"
        _save(receipt)
        raise RuntimeError("owned Modal app shutdown failed; use the app ID in the local ownership receipt")
    receipt.update(status="stopped", stopped_at=time.time())
    _save(receipt)
    return receipt
