"""Explicit deployment and shutdown of this workspace's dedicated Modal service.

No module import, catalog query or page load calls Modal. The ownership receipt contains
identifiers and public profile metadata only; credentials are read by the outbound SDK.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .paths import OUTPUT_DIR, ROOT


def credential_status() -> dict[str, str]:
    return {key: "SET" if os.environ.get(key) else "MISSING"
            for key in ("MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "HF_TOKEN")}


def _safe_sdk_error(error: Exception, operation: str) -> Exception:
    """Keep actionable failure categories without SDK messages, payloads or credentials."""
    if isinstance(error, TimeoutError):
        return TimeoutError(f"Modal {operation} timed out")
    if isinstance(error, PermissionError):
        return PermissionError(f"Modal {operation} authorization failed")
    if isinstance(error, ValueError):
        return ValueError(f"Modal {operation} rejected the request or configuration")
    return RuntimeError(f"Modal {operation} failed; verify the dedicated service status")


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


@contextmanager
def _ownership_lock():
    """Serialize CLI processes as well as UI workers; never wait behind a paid operation.

    Keep the lock file in place: unlinking it would let another process lock a new
    inode while the old owner still holds its lock. The OS releases flock on exit.
    """
    path = receipt_path().with_suffix(".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another Modal preparation or shutdown is already in progress") from None
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def call(method, *args, timeout: float = 120, call_mode: str = "remote"):
    """Bound submission + response, with best-effort cancellation and no retry."""
    async def invoke():
        handle = None
        try:
            async with asyncio.timeout(timeout):
                if call_mode == "remote":
                    return await method.remote.aio(*args)
                if call_mode != "spawn":
                    raise ValueError("Modal call mode must be remote or spawn")
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
    try:
        return asyncio.run(invoke())
    except Exception as error:  # noqa: BLE001 — SDK error bodies must not cross into CLI/browser logs
        raise _safe_sdk_error(error, "request") from None


def service_handle(app_name: str, profile_id: str):
    import modal

    try:
        return modal.Cls.from_name(app_name, "PolicyService")()
    except Exception as error:  # noqa: BLE001 — lookup can fail before the sanitized request boundary
        raise _safe_sdk_error(error, "service lookup") from None


def prepare(profile_name: str, *, gpu: str = "L40S", development: bool = False,
            cache_volume_name: str = "yamkit-policy-weights", region: str | None = "us-west",
            routing_region: str = "us-west", memory_mib: int = 65536) -> dict:
    """Deploy and warm explicitly. A failed prepare shuts down only its owned app."""
    with _ownership_lock():
        return _prepare_locked(profile_name, gpu=gpu, development=development,
                               cache_volume_name=cache_volume_name, region=region, routing_region=routing_region,
                               memory_mib=memory_mib)


def _prepare_locked(profile_name: str, *, gpu: str, development: bool, cache_volume_name: str,
                    region: str | None, routing_region: str, memory_mib: int) -> dict:
    from .inference.modal_service import create_app
    from .inference.profiles import get_profile

    profile = get_profile(profile_name)
    prior = owned_service()
    if prior and prior.get("status") != "stopped":
        if prior.get("profile_id") != profile.id or prior.get("revision") != profile.revision:
            raise ValueError("shut down the owned cloud service before preparing a different model")
        if prior.get("status") != "ready":
            raise ValueError("previous preparation is incomplete; shut down the owned service before retrying")
        if (prior.get("region") != region or prior.get("routing_region", "us-east") != routing_region
                or prior.get("cache_volume_name", "yamkit-policy-weights") != cache_volume_name
                or prior.get("memory_mib", 65536) != memory_mib):
            raise ValueError("shut down the owned cloud service before changing placement or its weight cache")
        metadata = call(service_handle(prior["app_name"], profile.id).ready, timeout=300)
        _validate_ready(metadata, profile)
        return {**prior, "metadata": metadata}
    app_name = "yamkit-vla-" + uuid.uuid4().hex[:16]
    receipt = {"app_name": app_name, "app_id": None, "profile_id": profile.id,
               "revision": profile.revision, "status": "preparing", "created_at": time.time(),
               "development": development, "gpu": gpu, "scaledown_window": 15 if development else 300,
               "region": region, "routing_region": routing_region, "cache_volume_name": cache_volume_name,
               "memory_mib": memory_mib,
               "deployment_started": False}
    _save(receipt)
    app = None
    try:
        try:
            app = create_app(profile_id=profile.id, gpu=gpu, development=development, app_name=app_name,
                             cache_volume_name=cache_volume_name, region=region, routing_region=routing_region,
                             memory_mib=memory_mib)
        except Exception as error:  # noqa: BLE001 — image/secret construction also invokes the SDK
            raise _safe_sdk_error(error, "application construction") from None
        receipt["deployment_started"] = True
        _save(receipt)
        async def deploy():
            try:
                async with asyncio.timeout(600):
                    await app.deploy.aio(name=app_name)
            except Exception as error:  # noqa: BLE001 — never display deployment transport error bodies
                raise _safe_sdk_error(error, "deployment") from None
        asyncio.run(deploy())
        receipt["app_id"] = app.app_id
        _save(receipt)
        metadata = call(service_handle(app_name, profile.id).ready, timeout=300 if development else 660)
        _validate_ready(metadata, profile)
        receipt.update(status="ready", metadata=metadata)
        _save(receipt)
        return receipt
    except BaseException:
        # Modal can learn the ID before a later build/deployment step fails.
        if app is not None and getattr(app, "app_id", None):
            receipt["app_id"] = app.app_id
            _save(receipt)
        # If deployment failed before returning its ID, the unique owned name remains sufficient.
        _shutdown_locked()
        raise


def _validate_ready(metadata: dict, profile) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("service readiness must contain validated metadata")  # noqa: TRY004
    if metadata.get("profile_id") != profile.id or metadata.get("revision") != profile.revision:
        raise ValueError("service readiness does not match the requested profile revision")
    for key in ("ready", "fresh_chunk", "saved_processors"):
        if metadata.get(key) is not True:
            raise ValueError(f"service readiness requires {key}=true")
    if metadata.get("repo_id") != profile.repo_id or metadata.get("model_revision") != profile.revision:
        raise ValueError("service readiness model identity does not match the pinned checkpoint")
    for key in ("state_names", "action_names", "image_keys", "native_image_keys"):
        actual = metadata.get(key)
        if not isinstance(actual, (list, tuple)) or tuple(actual) != tuple(getattr(profile, key)):
            raise ValueError(f"service readiness {key} does not match the ordered profile schema")
    for key, expected in (("chunk_size", profile.chunk_size), ("max_chunk_steps", profile.chunk_size),
                          ("fps", profile.fps)):
        if type(metadata.get(key)) is not int or metadata[key] != expected:
            raise ValueError(f"service readiness {key} does not match the profile")
    expected_units = "robot" if profile.mapping_verified else "checkpoint_native"
    if metadata.get("mapping_verified") is not profile.mapping_verified or metadata.get("action_units") != expected_units:
        raise ValueError("service readiness mapping or action units do not match the profile")
    if metadata.get("supports_rtc") is not False:
        raise ValueError("service readiness must not advertise unverified RTC guidance")


def shutdown() -> dict:
    """Stop only the exact app named in our ownership receipt; robot Stop is separate."""
    with _ownership_lock():
        return _shutdown_locked()


def _shutdown_locked() -> dict:
    receipt = owned_service()
    if receipt is None or receipt.get("status") == "stopped":
        return {"status": "stopped", "owned_app": None}
    name = receipt.get("app_name", "")
    if not isinstance(name, str) or not re.fullmatch(r"yamkit-vla-[a-z0-9-]{1,80}", name):
        raise ValueError("invalid owned app receipt")
    if receipt.get("deployment_started") is False and not receipt.get("app_id"):
        receipt.update(status="stopped", stopped_at=time.time(), remaining_containers=0,
                       shutdown_verification="deployment was never invoked")
        _save(receipt)
        return receipt
    env = {k: v for k, v in os.environ.items() if k not in ("YAMKIT_OPENAI_API_KEY", "DATABASE_URL", "HF_TOKEN")}
    env["MODAL_CONFIG_PATH"] = str(ROOT / ".context" / "modal.toml")

    def run(*args: str, timeout: int = 15):
        try:
            return subprocess.run([sys.executable, "-m", "modal", *args], env=env,
                                  capture_output=True, text=True, timeout=timeout, check=False)
        except (subprocess.TimeoutExpired, OSError):
            raise RuntimeError("Modal shutdown control operation failed or timed out; retirement is unverified") from None

    def inventory(result) -> list[dict]:
        if result.returncode:
            raise RuntimeError("Modal shutdown inventory failed; retirement is unverified")
        try:
            rows = json.loads(result.stdout)
        except (json.JSONDecodeError, TypeError):
            raise RuntimeError("Modal shutdown inventory was malformed; retirement is unverified") from None
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise RuntimeError("Modal shutdown inventory was malformed; retirement is unverified")
        return rows

    try:
        app_id = receipt.get("app_id")
        if not app_id:
            matches = [row for row in inventory(run("app", "list", "--json")) if row.get("description") == name]
            if len(matches) != 1:
                raise RuntimeError("cannot uniquely identify the owned Modal app; shutdown is unverified")
            app_id = matches[0].get("app_id")
            if not isinstance(app_id, str) or not re.fullmatch(r"ap-[A-Za-z0-9-]{1,80}", app_id):
                raise RuntimeError("owned Modal app ID is invalid; shutdown is unverified")
            receipt["app_id"] = app_id
            _save(receipt)
        elif not isinstance(app_id, str) or not re.fullmatch(r"ap-[A-Za-z0-9-]{1,80}", app_id):
            raise RuntimeError("owned Modal app ID is invalid; shutdown is unverified")
        completed = run("app", "stop", "--yes", app_id, timeout=30)
        if completed.returncode:
            # The CLI returns nonzero for an already stopped app too. An empty
            # container inventory alone cannot distinguish stopped from idle.
            matches = [row for row in inventory(run("app", "list", "--json")) if row.get("app_id") == app_id]
            if len(matches) != 1 or matches[0].get("state") != "stopped":
                raise RuntimeError("owned Modal app stop was not confirmed; shutdown is unverified")
        # Stop is asynchronous at the control plane. Confirm container retirement with a
        # finite poll before a subsequent model can acquire this workspace's pool.
        containers = None
        for attempt in range(6):
            containers = inventory(run("container", "list", "--app-id", app_id, "--json"))
            if not containers:
                break
            if attempt < 5:
                time.sleep(1)
        if containers != []:
            raise RuntimeError("owned Modal containers have not retired; no other model may be prepared")
        receipt["remaining_containers"] = 0
    except RuntimeError:
        receipt["status"] = "shutdown_unverified"
        _save(receipt)
        raise
    receipt.update(status="stopped", stopped_at=time.time())
    _save(receipt)
    return receipt
