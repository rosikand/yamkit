"""Isolated, authenticated official OpenPI service; no hardware or model changes.

Only the explicit experimental input adapter surrounds the pinned native policy.
Every native 50x32 output is returned unchanged, including out-of-range values.
The service cannot dispatch actions, authorize motion, or qualify a robot host.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import hmac
import json
import math
import multiprocessing
import os
import re
import signal
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .assets import local_path
from .contract import CHECKPOINT, IMAGE_KEYS, MODEL_CONFIG, UPSTREAM_REVISION, identity
from .runtime import OfficialPi05Diagnostic, validate_normalized_chunk
from .yam_candidate import CANDIDATE_ID, CandidateQuantiles, CandidateStatistics, encode_state

WIRE_VERSION = 1
MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_SESSION_SECONDS = 86400
MAX_PREDICT_SECONDS = 2.0
MAX_WARM_SECONDS = 120.0
_HASH = re.compile(r"[0-9a-f]{64}")
_CLIENT = re.compile(r"[A-Za-z0-9_-]{1,64}")


def build_id() -> str:
    """Content identity for this isolated boundary, not MA2 or fine-tuned PI."""
    digest = hashlib.sha256(b"yamkit-official-openpi-service-v1\0")
    package = Path(__file__).parent
    for name in ("assets.py", "contract.py", "pi05_base_manifest.json", "runtime.py",
                 "service.py", "transport.py", "yam_candidate.py"):
        digest.update(name.encode() + b"\0" + hashlib.sha256((package / name).read_bytes()).digest())
    return digest.hexdigest()


def _number(value, maximum: float, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
        raise ValueError(f"{label} must be finite and within its declared bound")
    return float(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def decode_json(body: bytes) -> dict:
    if not 0 < len(body) <= MAX_BODY_BYTES:
        raise ValueError("OpenPI message exceeds its size bound")
    value = json.loads(body, object_pairs_hook=_unique_object,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("OpenPI message must be an object")
    return value


def encode_json(value: dict) -> bytes:
    body = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(body) > MAX_BODY_BYTES:
        raise ValueError("OpenPI message exceeds its size bound")
    return body


def load_statistics(root: Path, path: Path) -> tuple[CandidateStatistics, str]:
    path = local_path(root, path)
    if not path.is_file() or path.stat().st_size > 128 * 1024:
        raise ValueError("OpenPI requires a bounded repository-local statistics file")
    body = path.read_bytes()
    value = decode_json(body)
    if value.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("OpenPI statistics belong to a different experimental adapter")
    statistics = CandidateStatistics(*(CandidateQuantiles(**value[key]) for key in ("state", "actions")))
    if value.get("candidate_statistics_sha256") != statistics.metadata()["candidate_statistics_sha256"]:
        raise ValueError("OpenPI statistics content identity does not match")
    return statistics, hashlib.sha256(body).hexdigest()


def read_token_file(root: Path, path: Path) -> str:
    path = local_path(root, path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        with os.fdopen(os.open(path, flags), "r") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or stat.S_IMODE(info.st_mode) & 0o077 or not 32 <= info.st_size <= 257):
                raise ValueError("OpenPI credential requires an owned private regular file")
            token = stream.read(258).strip()
    except OSError:
        raise ValueError("OpenPI private credential file could not be opened") from None
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token):
        raise ValueError("OpenPI credential has an invalid format")
    return token


def encode_images(images: dict[str, np.ndarray]) -> dict:
    if type(images) is not dict or set(images) != set(IMAGE_KEYS):
        raise ValueError("OpenPI requires exactly the three native RGB image roles")
    result = {}
    for key, value in images.items():
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3
                or value.shape[2] != 3 or not 1 <= value.shape[0] <= 720 or not 1 <= value.shape[1] <= 1280):
            raise ValueError("OpenPI images require bounded HWC uint8 RGB")
        result[key] = {"encoding": "rgb8", "shape": list(value.shape),
                       "data": base64.b64encode(value.tobytes(order="C")).decode("ascii")}
    return result


def decode_images(images: dict, *, image_hw: tuple[int, int]) -> dict[str, np.ndarray]:
    if type(images) is not dict or set(images) != set(IMAGE_KEYS):
        raise ValueError("OpenPI requires exactly the three native RGB image roles")
    result = {}
    shape = [*image_hw, 3]
    size = math.prod(shape)
    for key, value in images.items():
        if (type(value) is not dict or set(value) != {"encoding", "shape", "data"}
                or value["encoding"] != "rgb8" or value["shape"] != shape
                or any(type(item) is not int for item in value["shape"])
                or type(value["data"]) is not str or len(value["data"]) != 4 * ((size + 2) // 3)):
            raise ValueError("OpenPI RGB wire schema or configured image shape differs")
        try:
            raw = base64.b64decode(value["data"], validate=True)
        except (ValueError, UnicodeEncodeError):
            raise ValueError("OpenPI RGB wire data is malformed") from None
        if len(raw) != size:
            raise ValueError("OpenPI RGB wire byte count differs")
        result[key] = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    return result


def observation_request(images: dict[str, np.ndarray], measured_state, task: str) -> dict:
    """Pure encoding of saved or explicitly supplied observations, never acquisition."""
    state = np.asarray(measured_state)
    if state.shape != (14,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
        raise ValueError("OpenPI requires exactly fourteen finite measured YAM state values")
    if type(task) is not str or not task.strip() or len(task) > 512:
        raise ValueError("OpenPI requires a nonempty task of at most 512 characters")
    return {"images": encode_images(images), "state": state.tolist(), "task": task}


@dataclass(frozen=True)
class ServiceConfig:
    root: str
    token_file: str
    statistics: str
    service: str = "lambda-openpi"
    port: int = 8767
    session_seconds: float = 28800
    image_height: int = 480
    image_width: int = 640
    source_sha: str | None = None

    def validate(self) -> None:
        root = Path(self.root).resolve()
        if not root.is_dir():
            raise ValueError("OpenPI root must be an existing checkout")
        local_path(root, Path(self.token_file))
        local_path(root, Path(self.statistics))
        if not _CLIENT.fullmatch(self.service) or type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("OpenPI service name or loopback port is invalid")
        _number(self.session_seconds, MAX_SESSION_SECONDS, "OpenPI session lifetime")
        if (type(self.image_height) is not int or not 1 <= self.image_height <= 720
                or type(self.image_width) is not int or not 1 <= self.image_width <= 1280):
            raise ValueError("OpenPI service requires explicit bounded RGB dimensions")
        if self.source_sha is not None and re.fullmatch(r"[0-9a-f]{40}", self.source_sha) is None:
            raise ValueError("OpenPI source SHA must identify a source commit")


class NativeRuntime:
    """Serialized native policy, immutable statistics, finite session and sequences."""

    def __init__(self, policy, statistics: CandidateStatistics, *, expires_at: float,
                 source_sha: str, statistics_file_sha256: str, provenance: dict | None = None,
                 service_id: str = "lambda-openpi", image_hw: tuple[int, int] = (480, 640)):
        _number(expires_at - time.time(), MAX_SESSION_SECONDS, "OpenPI remaining lifetime")
        if (re.fullmatch(r"[0-9a-f]{40}", source_sha) is None
                or _HASH.fullmatch(statistics_file_sha256) is None):
            raise ValueError("OpenPI source and statistics file identities are required")
        self.policy, self.statistics = policy, statistics
        self.expires_at = expires_at
        self._deadline = time.monotonic() + expires_at - time.time()
        self.instance_id = str(uuid.uuid4())
        self.image_hw = image_hw
        self._lock = threading.Lock()
        self._sequences: dict[str, int] = {}
        self._warm_signatures: set[str] = set()
        self._identity = {
            "wire_version": WIRE_VERSION, "policy": "pi05-base", "checkpoint": CHECKPOINT,
            "runtime_revision": UPSTREAM_REVISION, "manifest_sha256": identity()["manifest_sha256"],
            "model_config": dict(MODEL_CONFIG), "num_inference_steps": 10,
            "service_id": service_id, "instance_id": self.instance_id, "source_sha": source_sha,
            "openpi_service_build_id": build_id(), "adapter_id": CANDIDATE_ID,
            "statistics_sha256": statistics.metadata()["candidate_statistics_sha256"],
            "statistics_file_sha256": statistics_file_sha256,
            "session_expires_at": expires_at, "image_hw": list(image_hw),
            "native_output_shape": [50, 32], "model_weights_modified": False,
            "native_inference_modified": False, "hardware_tested": False,
        }
        self.provenance = dict(provenance or {})

    @classmethod
    def load(cls, config: ServiceConfig, *, expires_at: float):
        config.validate()
        root = Path(config.root).resolve()
        statistics, file_hash = load_statistics(root, Path(config.statistics))
        source = config.source_sha
        if source is None:
            source = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                    text=True, check=True, timeout=10).stdout.strip()
        loaded = OfficialPi05Diagnostic.load(root)
        provenance = {key: value for key, value in loaded.provenance.items() if key in
                      ("packages", "asset_objects_verified", "load_s", "input_transforms", "output_transforms",
                       "model_native_defaults_verified", "jax_preallocate")}
        provenance.update(state_source="actual supplied measured YAM state; 14D normalized before tokenization",
                          output_space="untouched native normalized 50x32; robot decoding is client-side")
        return cls(loaded.policy, statistics, expires_at=expires_at, source_sha=source,
                   statistics_file_sha256=file_hash, provenance=provenance, service_id=config.service,
                   image_hw=(config.image_height, config.image_width))

    def ensure_active(self) -> None:
        if time.time() >= self.expires_at or time.monotonic() >= self._deadline:
            raise TimeoutError("OpenPI service session expired")

    def ready(self) -> dict:
        self.ensure_active()
        # Metadata remains readable while inference runs; it does not claim that
        # current inference is idle or that robot-side qualification is complete.
        return {**copy.deepcopy(self._identity), "ready": True, "busy": self._lock.locked(),
                "warm_signatures": sorted(self._warm_signatures), "runtime_provenance": copy.deepcopy(self.provenance)}

    @staticmethod
    def signature(task: str, image_hw: tuple[int, int]) -> str:
        return hashlib.sha256(encode_json({"task": task, "image_hw": list(image_hw),
                                         "encoding": "rgb8"})).hexdigest()

    def predict(self, request: dict, *, warm: bool = False) -> dict:
        started = time.monotonic()
        expected = {"wire_version", "images", "state", "task", "client_id", "sequence_id", "timeout_s",
                    "instance_id", "statistics_sha256", "openpi_service_build_id"}
        if (type(request) is not dict or set(request) != expected or type(request["wire_version"]) is not int
                or request["wire_version"] != WIRE_VERSION):
            raise ValueError("OpenPI request schema differs")
        timeout = _number(request["timeout_s"], MAX_WARM_SECONDS if warm else MAX_PREDICT_SECONDS,
                          "OpenPI request timeout")
        client, sequence = request["client_id"], request["sequence_id"]
        if (type(client) is not str or _CLIENT.fullmatch(client) is None or type(sequence) is not int
                or not 0 <= sequence < 2**53 or request["instance_id"] != self.instance_id
                or request["statistics_sha256"] != self._identity["statistics_sha256"]
                or request["openpi_service_build_id"] != self._identity["openpi_service_build_id"]):
            raise ValueError("OpenPI request identity or sequence differs")
        task = request["task"]
        if type(task) is not str or not task.strip() or len(task) > 512:
            raise ValueError("OpenPI task must be nonempty and bounded")
        state = np.asarray(request["state"])
        normalized = encode_state(state, self.statistics)
        images = decode_images(request["images"], image_hw=self.image_hw)
        self.ensure_active()
        if time.monotonic() - started >= timeout:
            raise TimeoutError("OpenPI request expired before inference")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("OpenPI inference is already in flight")
        try:
            self.ensure_active()
            if (sequence <= self._sequences.get(client, -1)
                    or (client not in self._sequences and len(self._sequences) >= 4096)):
                raise ValueError("OpenPI request sequence is stale or client budget is exhausted")
            self._sequences[client] = sequence
            signature = self.signature(task, self.image_hw)
            if not warm and signature not in self._warm_signatures:
                raise ValueError("OpenPI actual task and image shape must be warmed before prediction")
            # Do not prepad: pinned ModelTransformFactory tokenizes the 14-value
            # state, then PadStatesAndActions supplies the native 32D tensor.
            observation = {"image": images, "image_mask": dict.fromkeys(IMAGE_KEYS, np.True_),
                           "state": normalized, "prompt": task}
            model_started = time.monotonic()
            result = self.policy.infer(observation)
            chunk = validate_normalized_chunk(result["actions"])
            model_s = time.monotonic() - model_started
            self.ensure_active()
            if time.monotonic() - started >= timeout:
                raise TimeoutError("OpenPI response expired after inference")
            if warm:
                if len(self._warm_signatures) >= 256 and signature not in self._warm_signatures:
                    raise ValueError("OpenPI warm task budget is exhausted")
                self._warm_signatures.add(signature)
            return {**self._identity, "client_id": client, "sequence_id": sequence,
                    "state": state.tolist(), "raw_normalized_chunk": chunk.tolist(),
                    "raw_dtype": str(chunk.dtype), "warm_signature": signature,
                    "model_s": model_s, "server_s": time.monotonic() - started, "warm": warm}
        finally:
            self._lock.release()


def make_server(runtime: NativeRuntime, *, token: str, port: int) -> ThreadingHTTPServer:
    """Bounded loopback HTTP; no access logs, traceback bodies, or credential echo."""
    if type(token) is not str or re.fullmatch(r"[A-Za-z0-9_-]{32,256}", token) is None:
        raise ValueError("OpenPI service requires a valid private bearer credential")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):
            pass

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def _reply(self, code: int, value: dict):
            body = encode_json(value)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(body)

        def _handle(self, method: str):
            try:
                if (len(self.headers.get_all("Authorization", [])) != 1
                        or not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)):
                    self._reply(401, {"error": "unauthorized"})
                    return
                runtime.ensure_active()
                if method == "GET" and self.path == "/ready":
                    self._reply(200, {"ok": True, "result": runtime.ready()})
                    return
                if method != "POST" or self.path not in ("/warm", "/predict"):
                    self._reply(404, {"error": "unknown operation"})
                    return
                lengths = self.headers.get_all("Content-Length", [])
                if (len(lengths) != 1 or not re.fullmatch(r"[0-9]{1,8}", lengths[0])
                        or not 0 < int(lengths[0]) <= MAX_BODY_BYTES
                        or self.headers.get("Transfer-Encoding") is not None
                        or self.headers.get("Content-Encoding", "identity") != "identity"
                        or self.headers.get("Content-Type") != "application/json"):
                    self._reply(400, {"error": "invalid message framing"})
                    return
                body = self.rfile.read(int(lengths[0]))
                if len(body) != int(lengths[0]):
                    raise ValueError("OpenPI request body is incomplete")
                result = runtime.predict(decode_json(body), warm=self.path == "/warm")
                self._reply(200, {"ok": True, "result": result})
            except TimeoutError:
                self._reply(408, {"error": "deadline expired"})
            except ValueError:
                self._reply(400, {"error": "invalid request"})
            except RuntimeError:
                self._reply(409, {"error": "inference unavailable"})
            except Exception:  # noqa: BLE001 — never expose request, key, token or model exception text
                self._reply(500, {"error": "inference failed"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    class BoundedServer(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False
        request_queue_size = 8

        def __init__(self):
            self._slots = threading.BoundedSemaphore(8)
            super().__init__(("127.0.0.1", port), Handler)

        def process_request(self, request, client_address):
            if not self._slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except BaseException:
                self._slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                self._slots.release()

        def handle_error(self, request, client_address):
            pass  # Connection resets or model failures never print request context.

    return BoundedServer()


def _worker(config_values: dict, expires_at: float, supervisor_pid: int) -> None:
    try:
        os.setsid()
        # The supervisor is this child's sole owner. A SIGKILL/crash of that
        # supervisor must not leave an unbounded GPU process behind on Linux.
        import ctypes

        if (ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0
                or os.getppid() != supervisor_pid):
            raise RuntimeError("Official OpenPI supervisor ownership could not be established")
        config = ServiceConfig(**config_values)
        token = read_token_file(Path(config.root), Path(config.token_file))
        runtime = NativeRuntime.load(config, expires_at=expires_at)
        server = make_server(runtime, token=token, port=config.port)
        print(json.dumps({"event": "official_openpi_service_ready", "instance_id": runtime.instance_id,
                          "service_id": config.service, "session_expires_at": expires_at}), flush=True)
        server.serve_forever(poll_interval=0.2)
    except BaseException as exc:  # noqa: BLE001 — intentionally omit untrusted exception text
        print(json.dumps({"event": "official_openpi_service_failed", "error_type": type(exc).__name__}), flush=True)
        raise SystemExit(1) from None


def supervise(config: ServiceConfig) -> int:
    """Own one finite process group; never provision, discover, or stop other services."""
    config.validate()
    read_token_file(Path(config.root), Path(config.token_file))
    expires_at = time.time() + config.session_seconds
    deadline = time.monotonic() + config.session_seconds
    child = multiprocessing.get_context("spawn").Process(target=_worker, args=(asdict(config), expires_at, os.getpid()),
                                                         name="yamkit-owned-official-openpi", daemon=True)
    stop = threading.Event()
    handlers = {}
    started = False
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda *_: stop.set())
        child.start()
        started = True
        while child.is_alive() and not stop.is_set() and time.monotonic() < deadline:
            child.join(min(0.2, max(0, deadline - time.monotonic())))
        return 0 if stop.is_set() or time.monotonic() >= deadline or child.exitcode == 0 else 1
    finally:
        if started and child.is_alive():
            child.terminate()
            child.join(5)
            if child.is_alive():
                child.kill()
                child.join(5)
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--service", default="lambda-openpi")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--token-file", required=True)
    parser.add_argument("--statistics", required=True)
    parser.add_argument("--session-seconds", type=float, default=28800)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--source-sha")
    return supervise(ServiceConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    raise SystemExit(main())
