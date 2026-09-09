"""Portable diagnostic snapshots and optional, post-finalization private Hub upload.

This module never imports a robot or runs in the control loop. Original recordings
remain untouched; only explicitly named artifacts enter a separate sanitized bundle.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
CAMERAS = ("top", "left_wrist", "right_wrist")
ARTIFACTS = (
    "meta.json", "summary.json", "trace.json", "metrics.json", "frame_timestamps.json",
    "video_timeline.json", "log.txt", "report.html", "joints-left.png", "joints-right.png",
    "top.mp4", "left_wrist.mp4", "right_wrist.mp4", "export-error.json", "run_metadata.json", "plan.json",
)
REQUIRED = tuple(name for name in ARTIFACTS if name not in ("export-error.json", "run_metadata.json", "plan.json"))
METADATA_FIELDS = frozenset({
    "source", "model", "rig", "configuration", "runtime", "packages", "provenance",
    "original_paths", "operator_feedback", "known_missing_data", "capture", "software", "environment",
})
_SENSITIVE_KEY = re.compile(
    r"token|secret|password|passwd|credential|authorization|authentication|api.?key|"
    r"endpoint|private.?url|session.?url|cookie|(^|_)headers?($|_)|(^|_)auth($|_)", re.IGNORECASE,
)
_URL = re.compile(r"(?:https?|wss?|s3|gs)://[^\s\"'<>]+", re.IGNORECASE)
_TOKEN = re.compile(r"\b(?:hf_[A-Za-z0-9]{8,}|(?:ak|as|sk)[_-][A-Za-z0-9_-]{12,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b")
_ASSIGNMENT = re.compile(
    r"(?i)(\b[\w-]*(?:authorization|bearer|password|passwd|token|secret|api[_-]?key|credential)"
    r"[\w-]*[\"']?[\s:=]+)(?:Bearer\s+)?(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
)
_FLAG = re.compile(r"(?i)(--(?:[\w-]*(?:token|secret|password|credential|api-key|endpoint)[\w-]*))(?:=|\s+)(?:\"[^\"]*\"|'[^']*'|[^\s]+)")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}")
_MAX_TEXT_BYTES = 32 * 1024 * 1024


def sanitize_text(value: str, *, secrets: tuple[str, ...] = ()) -> str:
    """Remove service addresses and common credential forms even inside log strings."""
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            value = value.replace(secret, "[REDACTED_SECRET]")
    value = _URL.sub("[REDACTED_URL]", value)
    value = _TOKEN.sub("[REDACTED_SECRET]", value)
    value = re.sub(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----",
                   "[REDACTED_PRIVATE_KEY]", value, flags=re.DOTALL)
    value = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED_SECRET]", value)
    value = _FLAG.sub(r"\1 [REDACTED_SECRET]", value)
    return _ASSIGNMENT.sub(r"\1[REDACTED_SECRET]", value)


def sanitize(value: Any, *, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {sanitize_text(str(key), secrets=secrets): sanitize(item, secrets=secrets) for key, item in value.items()
                if not _SENSITIVE_KEY.search(str(key))}
    if isinstance(value, list):
        return [sanitize(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return sanitize_text(value, secrets=secrets)
    return value


def _safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Artifact paths cannot traverse parent directories")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("Artifact paths cannot contain symlinks")
    return path


def _read(path: Path, *, bounded: bool = True) -> bytes:
    path = _safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or (bounded and info.st_size > _MAX_TEXT_BYTES):
            raise ValueError("Artifact must be a bounded regular file")
        result = stream.read(_MAX_TEXT_BYTES + 1 if bounded else -1)
    if bounded and len(result) > _MAX_TEXT_BYTES:
        raise ValueError("Artifact exceeded its size bound")
    return result


def _json(path: Path) -> Any:
    def invalid(_value):
        raise ValueError("Artifact JSON contains non-finite numbers")

    return json.loads(_read(path), parse_constant=invalid)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    fd = os.open(_safe_path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Artifact must be a regular file")
        while data := stream.read(1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> None:
    """Copy binary media without following symlinks or opening devices/FIFOs."""
    fd = os.open(_safe_path(source), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("Artifact must be a regular file")
        with target.open("wb") as destination:
            shutil.copyfileobj(stream, destination)


def _readme(run_id: str, missing: list[str], frame_counts: dict[str, int]) -> str:
    absent = "\n".join(f"- {item}" for item in missing) or "- No expected artifact files are missing."
    return f"""# Rollout {run_id}

Self-contained diagnostic recording, not a training dataset. Open `report.html`
directly, or run `python -m http.server 8000` in this directory and visit
`http://localhost:8000/report.html`. Videos and plots use relative paths.
The process exit status describes execution/cleanup, **not manipulation success**;
see `meta.json.task_success` and operator feedback when present.

## Files and schema (version {SCHEMA_VERSION})

- `top.mp4`, `left_wrist.mp4`, `right_wrist.mp4`: timestamped H.264 playback.
  `summary.json.video_fps` is the nominal rate; actual variable presentation times
  are in `video_timeline.json`. Compression is not pixel-exact.
- `frames/<camera>/frame-NNNNNN.png`: original captured RGB arrays, lossless PNG;
  counts: {', '.join(f'{key}={value}' for key, value in frame_counts.items())}.
- `frame_timestamps.json`: frame-index-ordered host monotonic receipt times.
- `video_timeline.json.frames[i]`: `source_index` names PNG/video frame i;
  `observation_index` joins the observation event; `receipt_monotonic_s` is host
  receipt time. `pts * time_base[0] / time_base[1]` gives playback seconds.
  Add `origin_monotonic_s` to align video with trace clocks; subtract
  `summary.json.phase_started_monotonic_s` for policy-relative seconds.
  Gaps remain real gaps; no invented frames. Older recordings may lack this map.
- `trace.json.events`: `observation` contains measured positions in the ordered
  14-element `summary.json.action_names`; `video_sample` joins frame_index and
  observation_index. `action_dequeued` records queue deadlines. `send_start`
  contains requested arm targets, and `send_end.postclamp` contains the targets
  sent after joint/gripper speed clamps. Pair sends by arm and chronological
  order; two follower sends normally make one bimanual action. A send target is
  not proof that the physical joint reached it. `send_error` may mean partial
  dispatch; never count it as an ordinary completed send.
- `trace.json.chunks`: chunk_index, observation_monotonic_s,
  returned_monotonic_s, and 30 x 14 predicted robot-unit targets for MolmoAct2,
  after the recorded policy postprocessing (not raw normalized model outputs). A chunk is
  predicted from the latest observation available to inference; its timestamp
  can be matched to the preceding observation receipt, subject to host timing
  precision. There is no exact request-to-observation ID in this schema.
  `chunk_merge` events describe queue merges, discarded expired/overlap prefixes
  and accepted tails; this strategy preserves retained old queued actions.
  not every predicted target is executed. Target index j has a nominal deadline
  `observation_monotonic_s + (j + 1) / 30` for this policy. Approximate matching
  by deadlines and requested target values can reconstruct joins, but there is
  no explicit foreign key. `action_dequeued` and ordered sends are execution
  evidence, not a one-to-one join by chunk_index.
- `metrics.json`: complete available inference/queue/control/cleanup metrics;
  `summary.json`: capture scope, phase boundaries, dropped data and export errors.
  Newer remote runs include `command_shaping`: limits, counts and bounded samples
  joining each `dispatch_index` and host monotonic timestamp to the original
  14-key `requested` policy target, acceleration-limited `shaped` target, and
  successfully returned post-clamp `sent` command. These are command targets,
  not measured joint trajectories. In these runs, trace `send_start.requested`
  is already shaped; use the metric sample's `requested` for model-chunk joins.
  Any samples omitted by the bound are counted in `samples_dropped`.
- `meta.json`: launcher task, times, return code and any operator feedback.
  `log.txt`: available child stdout/stderr; `plan.json`: saved capture plan if available.
  `run_metadata.json`: sanitized
  rig/control settings, model/source versions and provenance when recorded.
- `manifest.json`: SHA-256 and size for every bundled file (excluding itself),
  missing-data inventory and frame counts. Originals remain on the robot host.

All joint positions/targets are radians; gripper `.pos` is normalized 0..1.
Host monotonic times cannot be compared directly with another computer's clock.
Wall timestamps in meta describe the whole child process, including startup,
homing and export; policy/video duration is shorter. JSON and text are sanitized;
credential fields and service URLs are removed/redacted. Images/video are copied
unchanged and can contain visible scene details.
Service-identity digests (such as `http_endpoint_sha256`) are intentionally
removed along with endpoints; numerical latency/action evidence is retained.

## Missing data and interpretation limits

{absent}

This bundle preserves diagnostic formats; it does not convert to LeRobot training
episodes, fill gaps, interpolate targets or infer whether the task succeeded.
"""


def package_rollout(
    run_dir: Path, *, trace_dir: Path | None = None, metadata: dict[str, Any] | None = None,
) -> Path:
    """Snapshot one finalized run. Repeated identical snapshots are idempotent."""
    run_dir = _safe_path(run_dir)
    # Local token lookup only, never a Hub request. Redact known credential
    # values even if they appear under an innocent key or have no usual prefix.
    from huggingface_hub import get_token

    hub_token = get_token()
    secrets = tuple({value for key, value in os.environ.items() if _SENSITIVE_KEY.search(key) and len(value) >= 4}
                    | ({hub_token} if hub_token else set()))
    meta = _json(run_dir / "meta.json")
    if (not isinstance(meta, dict) or meta.get("ended_at") is None or meta.get("returncode") is None
            or meta.get("active") or meta.get("status") == "running"):
        raise ValueError("Only a finalized run can be packaged")
    run_id = meta.get("id", run_dir.name)
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id) or run_id in (".", ".."):
        raise ValueError("Invalid rollout run ID")
    bundle = _safe_path(run_dir / "bundle")
    if bundle.exists() and trace_dir is None and metadata is None:
        # A retry uploads the immutable completed snapshot, without requiring
        # original frames to be recopied or consulting any recorded source path.
        validate_bundle(bundle)
        return bundle
    trace_dir = _safe_path(trace_dir) if trace_dir is not None else run_dir
    sources: dict[str, Path] = {}
    for name in ARTIFACTS:
        for directory in (run_dir, trace_dir):
            candidate = _safe_path(directory / name)
            if candidate.exists():
                sources[name] = candidate
                break
    summary = _json(sources["summary.json"]) if "summary.json" in sources else {}
    if summary and (summary.get("resources_released") is not True or summary.get("status") == "EXPORTING"):
        raise ValueError("Trace resources must be released and export finalized before packaging")
    missing = [f"Missing artifact: {name}" for name in REQUIRED if name not in sources]
    missing.extend([
        "Camera exposure timestamps are not recorded; image times are host receipt times.",
        "Startup and return-home RGB/video and joint trajectories are outside the policy capture scope.",
        "Motor currents, torques, velocities and hardware acknowledgement timestamps are not captured.",
        "Exact prediction request IDs and an explicit chunk-to-executed-action index join are not captured.",
        "Cloud service stdout/stderr and internal model tensors are not part of the robot-host recording.",
    ])
    if not sources.get("run_metadata.json") and not metadata:
        missing.append("Configuration/model-version metadata was not captured for this run.")
    supplied = _json(sources["run_metadata.json"]) if "run_metadata.json" in sources else {}
    if not isinstance(supplied, dict):
        raise ValueError("Run metadata must be a JSON object")  # noqa: TRY004 — artifact validation boundary.
    supplied.update(metadata or {})
    supplied = sanitize({key: value for key, value in supplied.items() if key in METADATA_FIELDS}, secrets=secrets)
    missing.extend(str(item) for item in supplied.get("known_missing_data", []) if isinstance(item, str))
    if meta.get("log_complete") is False or supplied.get("capture", {}).get("log_complete") is False:
        missing.append("Session log is incomplete; retained output may be truncated.")
    for key, value in summary.get("counts", {}).items():
        if value and (key.endswith("_dropped") or key == "trace_errors"):
            missing.append(f"Capture reported {key}={value}.")
    if summary.get("video_export_errors") or summary.get("render_error_type"):
        missing.append("Capture reported export/render errors; inspect summary.json.")
    with tempfile.TemporaryDirectory(prefix=".bundle-build-", dir=run_dir) as temporary:
        stage = Path(temporary)
        for name, source in sources.items():
            if name == "run_metadata.json":
                continue
            if source.suffix == ".json":
                _write_json(stage / name, sanitize(_json(source), secrets=secrets))
            elif source.suffix in (".txt", ".html"):
                (stage / name).write_text(sanitize_text(_read(source).decode("utf-8"), secrets=secrets))
            else:
                _copy(source, stage / name)
        _write_json(stage / "run_metadata.json", supplied)
        frame_counts = {}
        expected = summary.get("frame_count")
        for camera in CAMERAS:
            frame_root = _safe_path(trace_dir / "frames" / camera)
            if not frame_root.exists() and trace_dir != run_dir:
                frame_root = _safe_path(run_dir / "frames" / camera)
            files = sorted(frame_root.iterdir()) if frame_root.is_dir() else []
            selected = [path for path in files if re.fullmatch(r"frame-\d{6}\.png", path.name)]
            frame_counts[camera] = len(selected)
            if not selected:
                missing.append(f"Missing original RGB frames: {camera}.")
            elif isinstance(expected, int) and len(selected) != expected:
                missing.append(f"Original RGB count mismatch: {camera} has {len(selected)}, expected {expected}.")
            indices = [int(path.stem.split("-")[1]) for path in selected]
            if indices != list(range(len(indices))):
                missing.append(f"Original RGB frame indices have gaps: {camera}.")
            if selected:
                destination = stage / "frames" / camera
                destination.mkdir(parents=True)
                for source in selected:
                    _copy(source, destination / source.name)
        missing = sanitize(missing, secrets=secrets)
        (stage / "README.md").write_text(_readme(run_id, missing, frame_counts))
        manifest = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "sanitized": True,
                    "missing_data": missing, "frame_counts": frame_counts, "files": {}}
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest["files"][path.relative_to(stage).as_posix()] = {
                    "sha256": _hash(path), "size_bytes": path.stat().st_size,
                }
        _write_json(stage / "manifest.json", manifest)
        if bundle.exists():
            validate_bundle(bundle)
            if _read(bundle / "manifest.json") != _read(stage / "manifest.json"):
                raise ValueError("A different bundle already exists; preserve it and choose a new run snapshot")
            return bundle
        stage.rename(bundle)
    return bundle


def validate_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Refuse changed, extra or linked files before any network upload."""
    bundle_dir = _safe_path(bundle_dir)
    manifest = _json(bundle_dir / "manifest.json")
    if (manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("sanitized") is not True
            or not _RUN_ID.fullmatch(str(manifest.get("run_id", "")))):
        raise ValueError("Invalid rollout bundle manifest")
    files = manifest.get("files", {})
    if not isinstance(files, dict) or "README.md" not in files:
        raise ValueError("Bundle manifest has no file inventory")
    actual = set()
    for path in bundle_dir.rglob("*"):
        _safe_path(path)
        if not path.is_dir():
            actual.add(path.relative_to(bundle_dir).as_posix())
    if actual != set(files) | {"manifest.json"}:
        raise ValueError("Bundle contains missing or unexpected files")
    allowed = set(ARTIFACTS) | {"README.md"}
    for name, expected in files.items():
        if name not in allowed and not re.fullmatch(r"frames/(?:top|left_wrist|right_wrist)/frame-\d{6}\.png", name):
            raise ValueError("Bundle manifest contains a disallowed artifact")
        path = _safe_path(bundle_dir / name)
        if _hash(path) != expected.get("sha256") or path.stat().st_size != expected.get("size_bytes"):
            raise ValueError(f"Bundle integrity check failed: {name}")
    return manifest


def upload_rollout(bundle_dir: Path, *, repo_id: str, api=None) -> dict[str, Any]:
    """Upload once to a private dataset, with durable status and conflict-safe retry.

    Call from a CLI or post-run worker only, after hardware release/finalization.
    Authentication uses the Hub's existing token store; no token enters the bundle.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(repo_id)
    if repo_id.count("/") != 1:
        raise ValueError("An explicit namespace/dataset destination is required")
    bundle_dir = _safe_path(bundle_dir)
    manifest = validate_bundle(bundle_dir)
    run_id = manifest["run_id"]
    prefix = f"runs/{run_id}"
    manifest_hash = _hash(bundle_dir / "manifest.json")
    status = {"status": "uploading", "repo_id": repo_id, "run_id": run_id, "path_in_repo": prefix,
              "manifest_sha256": manifest_hash, "revision": None,
              "url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{prefix}"}
    status_path = _safe_path(bundle_dir.parent / "hf-upload.json")

    def save():
        temporary = status_path.with_suffix(".json.tmp")
        _safe_path(temporary)
        _write_json(temporary, status)
        temporary.replace(status_path)

    save()
    try:
        api = api or HfApi()
        try:
            info = api.dataset_info(repo_id=repo_id)
        except RepositoryNotFoundError:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
            info = api.dataset_info(repo_id=repo_id)
        if info.private is not True:
            raise ValueError("Rollout uploads require a private dataset repository")
        revision = info.sha
        if not revision:
            raise ValueError("Dataset has no revision for conflict-safe upload")
        remote_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
        existing = {name for name in remote_files if name.startswith(prefix + "/")}
        expected = {f"{prefix}/{name}" for name in manifest["files"]} | {f"{prefix}/manifest.json"}
        if existing:
            if existing != expected:
                raise ValueError("Remote run already exists with different or incomplete artifacts")
            with tempfile.TemporaryDirectory(prefix=".hf-manifest-", dir=bundle_dir.parent) as temporary:
                remote_manifest = api.hf_hub_download(
                    repo_id=repo_id, filename=f"{prefix}/manifest.json", repo_type="dataset",
                    revision=revision, local_dir=temporary,
                )
                if _hash(Path(remote_manifest)) != manifest_hash:
                    raise ValueError("Remote run already exists with different content; refusing overwrite")
            status.update(status="already_uploaded", revision=revision)
        else:
            result = api.upload_folder(
                repo_id=repo_id, repo_type="dataset", folder_path=bundle_dir, path_in_repo=prefix,
                commit_message=f"Add finalized rollout {run_id}", parent_commit=revision,
                allow_patterns=sorted(manifest["files"]) + ["manifest.json"],
            )
            status.update(status="uploaded", revision=result.oid)
        status["url"] = f"https://huggingface.co/datasets/{repo_id}/tree/{status['revision']}/{prefix}"
        save()
        return status
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=sanitize_text(str(exc)))
        save()
        raise
