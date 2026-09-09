"""Explicit attachment to a user-owned inference server; never manages its VM.

Only ``attach_service`` contacts the server, and only for readiness. Receipts and
credentials remain separate private files under this checkout's data directory.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import re
import stat
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .paths import DATA_DIR


def _name(name: str) -> str:
    if type(name) is not str or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", name) is None:
        raise ValueError("An explicit bounded external service name is required")
    return name


def _directory(name: str, *, create=False) -> Path:
    path = DATA_DIR
    for part in ("inference", "external", _name(name)):
        path = path / part
        if create:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            details = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(details.st_mode) or details.st_uid != os.geteuid():
            raise ValueError("External attachment directories must be owned directories, without symlinks")
    return path


def _read_private(path: Path, maximum: int) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r") as stream:
            details = os.fstat(stream.fileno())
            if (not stat.S_ISREG(details.st_mode) or details.st_mode & 0o077
                    or details.st_uid != os.geteuid() or not 0 < details.st_size <= maximum):
                raise ValueError("External attachment files require private permissions and bounded size")
            value = stream.read(maximum + 1)
        if len(value) > maximum:
            raise ValueError("External attachment file exceeds its bound")
        return value
    except (OSError, UnicodeError):
        raise ValueError("External attachment file is missing or invalid") from None


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(_read_private(path, 262144))
    except json.JSONDecodeError:
        raise ValueError("External attachment metadata is invalid") from None
    if type(value) is not dict:
        raise ValueError("External attachment metadata must be a mapping")
    return value


def _save(path: Path, value: dict) -> None:
    content = json.dumps(value, allow_nan=False, indent=2) + "\n"
    if len(content.encode()) > 262144:
        raise ValueError("External attachment metadata exceeds its bound")
    temporary = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _lock(name: str):
    path = _directory(name, create=True) / "attachment.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_uid != os.geteuid() or details.st_mode & 0o077:
            raise ValueError("External attachment lock must be a private owned regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    except BlockingIOError:
        raise ValueError("Another external attachment operation is in progress") from None
    finally:
        os.close(descriptor)


def owned_service(name: str) -> dict | None:
    """Read the named local receipt, without contacting a host or opening hardware."""
    path = _directory(name) / "receipt.json"
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    value = _read_json(path)
    if value.get("name") != name or value.get("backend") != "external" or value.get("schema_version") != 1:
        raise ValueError("External attachment receipt identity is invalid")
    return value


def _validated_metadata(name: str, endpoint_url: str, metadata: dict, provider: str) -> dict:
    from .inference.identity import external_service_binding, http_runtime_binding
    from .inference.profiles import get_profile
    from .modal_ops import _validate_http_ready, _validate_ready

    profile = get_profile("molmoact2")
    _validate_ready(metadata, profile)
    external = external_service_binding(metadata)
    if external["service_id"] != name or external["provider"] != provider or metadata.get("http_ingress") != "ssh":
        raise ValueError("External service name, provider or SSH ingress differs from the attachment")
    mode = metadata.get("execution_mode")
    _validate_http_ready(metadata, mode)
    # Initial attachment verifies runtime identity without claiming that any task
    # has been warmed. Real task/image warmup remains mandatory for qualification.
    http_runtime_binding(profile, metadata, execution_mode=mode, task="external attachment validation",
                         image_hw=(480, 640), require_warmup=False, endpoint_url=endpoint_url)
    allowed = set(profile.metadata()) | {
        "ready", "instance_id", "execution_mode", "execution_identity", "graph_warmup", "device", "load_s",
        "fresh_chunk", "prediction_count", "runtime_age_s", "supports_rtc", "continuation", "image_encoding",
        "preferred_image_encoding", "supported_image_encodings", "jpeg_quality", "memory", "model_execution",
        "saved_processors", "session_state", "transport", "inference_build_id", "http_wire_version",
        "http_wire_codec", "supported_call_modes", "http_ingress", "http_endpoint", "http_session_expires_at",
        "external_service", "runtime_provenance",
    }
    return copy.deepcopy({key: value for key, value in metadata.items() if key in allowed})


def _probe_ready(name: str, endpoint_url: str, token: str) -> dict:
    from .inference.http_transport import HttpTransport

    # Bootstrap discovers the advertised lease using readiness ONLY. The temporary
    # local 120-second bound cannot authorize prediction or a robot connection.
    transport = HttpTransport(name, "molmoact2", endpoint_url=endpoint_url, token=token,
                              http_ingress="ssh", http_session_expires_at=time.time() + 120)
    try:
        return transport._invoke("ready", None, 120)
    finally:
        transport.close()


def attach_service(name: str, endpoint_url: str, token_file: str | Path, provider: str = "lambda") -> dict:
    """Attach an already running server through an existing verified SSH forward."""
    from .inference.http_service import validate_http_token
    from .inference.http_transport import validate_endpoint_url

    name = _name(name)
    endpoint_url = validate_endpoint_url(endpoint_url, http_ingress="ssh")
    if provider != "lambda":
        raise ValueError("This external runtime currently supports the reviewed Lambda provider")
    token = _read_private(Path(token_file), 257).strip()
    validate_http_token(token)
    with _lock(name):
        metadata = _validated_metadata(name, endpoint_url, _probe_ready(name, endpoint_url, token), provider)
        if token in json.dumps(metadata, allow_nan=False):
            raise ValueError("Readiness metadata unexpectedly contains a private credential")
        receipt = {"schema_version": 1, "backend": "external", "name": name, "service_id": name,
                   "attachment_id": uuid.uuid4().hex, "status": "ready", "provider": provider,
                   "profile_id": metadata["profile_id"], "revision": metadata["revision"],
                   "transport": "http", "execution_mode": metadata["execution_mode"],
                   "http_ingress": "ssh", "http_endpoint": endpoint_url,
                   "http_session_expires_at": metadata["http_session_expires_at"],
                   "attached_at": time.time(), "metadata": metadata}
        directory = _directory(name)
        _save(directory / "http-auth.json", {"name": name, "attachment_id": receipt["attachment_id"],
                                             "endpoint_url": endpoint_url, "token": token})
        _save(directory / "receipt.json", receipt)
        return receipt


def http_credentials(name: str) -> dict:
    """Return private credentials only for the current unexpired attachment."""
    from .inference.http_service import validate_http_token

    receipt = owned_service(name) or {}
    if receipt.get("status") != "ready" or receipt.get("transport") != "http":
        raise ValueError("Attach the matching external HTTP service first")
    auth = _read_json(_directory(name) / "http-auth.json")
    if (set(auth) != {"name", "attachment_id", "endpoint_url", "token"} or auth.get("name") != name
            or auth.get("attachment_id") != receipt.get("attachment_id")
            or auth.get("endpoint_url") != receipt.get("http_endpoint")):
        raise ValueError("External credentials differ from their attachment receipt")
    validate_http_token(auth["token"])
    metadata = _validated_metadata(name, auth["endpoint_url"], receipt.get("metadata", {}), receipt.get("provider"))
    if receipt.get("http_session_expires_at") != metadata["http_session_expires_at"]:
        raise ValueError("External attachment expiry differs from its service")
    return {"endpoint_url": auth["endpoint_url"], "token": auth["token"], "http_ingress": "ssh",
            "http_session_expires_at": metadata["http_session_expires_at"]}


def update_ready(name: str, metadata: dict, *, expected_instance_id: str) -> dict:
    """Refresh warmed metadata only if the exact attached server is unchanged."""
    with _lock(name):
        receipt = owned_service(name) or {}
        credentials = http_credentials(name)
        old = receipt["metadata"]
        metadata = _validated_metadata(name, receipt["http_endpoint"], metadata, receipt["provider"])
        if credentials["token"] in json.dumps(metadata, allow_nan=False):
            raise ValueError("Readiness metadata unexpectedly contains a private credential")
        if (old.get("instance_id") != expected_instance_id or metadata.get("instance_id") != expected_instance_id
                or any(old.get(key) != metadata.get(key) for key in (
                    "external_service", "runtime_provenance", "http_session_expires_at", "execution_identity"))):
            raise ValueError("External service changed during qualification")
        updated = {**receipt, "metadata": metadata}
        _save(_directory(name) / "receipt.json", updated)
        return updated


def detach_service(name: str) -> dict:
    """Invalidate only the local attachment; never stop its process, tunnel or VM."""
    with _lock(name):
        receipt = owned_service(name)
        if receipt is None:
            return {"name": name, "backend": "external", "status": "not_attached"}
        receipt = {**receipt, "status": "detached", "detached_at": time.time()}
        _save(_directory(name) / "receipt.json", receipt)
        (_directory(name) / "http-auth.json").unlink(missing_ok=True)
        return receipt
