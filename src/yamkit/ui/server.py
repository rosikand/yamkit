"""FastAPI app behind `yamkit ui`.

Read-only endpoints (rig, CAN, datasets, models, deployments) touch only files and sysfs.
Hardware endpoints spawn the unmodified `yamkit` CLI via `SessionManager` — starting the server
or opening any page never connects to (and never energises) an arm.
"""

from __future__ import annotations

import dataclasses
import json
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, StrictBool

from .. import hub
from ..can import bringup_commands, list_can_interfaces
from ..config import RigConfig
from ..paths import DATASETS_DIR, DEFAULT_RIG, OUTPUT_DIR, ROOT
from ..preview import MJPEG_MEDIA_TYPE, STALE_S
from . import catalog
from .camstream import CameraHub, CameraStreamingResponse
from .preview_proxy import PreviewStreamingResponse, PreviewUnavailable, fetch_status, open_stream
from .sessions import DeploymentLog, SessionManager

FRONTEND_DIR = ROOT / "ui"
TRACE_TASK = "pick up the orange lid and place it into the black circular container"
TRACE_FILES = frozenset({"summary.json", "trace.json", "metrics.json", "frame_timestamps.json", "video_timeline.json", "export-error.json",
                         "top.mp4", "left_wrist.mp4", "right_wrist.mp4",
                         "report.html", "joints-left.png", "joints-right.png"})


def _rollout_metadata(options, rig: RigConfig) -> dict:
    """Capture source/configuration identity before launch without storing host credentials."""
    from ..inference.identity import inference_build_id
    from ..inference.profiles import LEROBOT_VERSION, get_profile

    arm_fields = {"role", "side", "arm_type", "gripper", "gripper_limits", "rest_pose", "joint_offsets"}
    camera_fields = {"type", "width", "height", "fps", "color_mode", "rotation"}
    option_fields = {"policy", "task", "backend", "device", "gpu", "call_mode", "execution_mode",
                     "image_encoding", "jpeg_quality", "prediction_queue_threshold", "center_crop",
                     "async_chunks", "duration", "fps", "rtc", "arms"}
    try:
        model = dataclasses.asdict(get_profile(options.policy))
    except ValueError:
        model = {"requested_policy": options.policy, "revision": None}
    software = {"lerobot_version": LEROBOT_VERSION, "inference_build_id": inference_build_id()}
    try:
        software["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, timeout=3, text=True).strip()
        software["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
            stderr=subprocess.DEVNULL, timeout=3, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        software["git_commit"] = None
    return {"provenance": {"captured_at": time.time(), "kind": "before_managed_child_launch"},
            "model": model, "software": software,
            "configuration": {key: value for key, value in dataclasses.asdict(options).items()
                              if key in option_fields},
            "rig": {"arms": {name: {key: value for key, value in dataclasses.asdict(arm).items()
                                    if key in arm_fields} for name, arm in rig.arms.items()},
                    "cameras": {name: {key: value for key, value in config.items() if key in camera_fields}
                                for name, config in rig.cameras.items()},
                    "control": dataclasses.asdict(rig.control)}}


# --------------------------------------------------------------------------- request bodies --
class RestBody(BaseModel):
    arms: list[str] | None = None


class ReadBody(BaseModel):
    arms: list[str] | None = None
    hz: float = 5.0


class TeleopBody(BaseModel):
    pairs: list[str] | None = None
    auto_engage: bool = True
    bilateral_kp: float | None = None
    duration: float | None = None


class RecordBody(BaseModel):
    name: str
    task: str
    episodes: int = 10
    episode_s: float = 30.0
    reset_s: float = 10.0
    fps: int = 30
    arms: list[str] | None = None
    resume: bool = False
    to: str | None = None  # local | hub | both (default: hub.datasets in the rig)


class HubLoginBody(BaseModel):
    token: str


class HubTransferBody(BaseModel):
    name: str  # dataset name / repo id, or a checkpoint path under outputs/ for push-model
    remove_local: bool = False


class InferenceBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: str
    task: str = "pick up the object"
    backend: str = "local"
    device: str = "cpu"
    gpu: str = "L40S"
    modal_app: str | None = None
    call_mode: str = "remote"
    execution_mode: str = "eager"
    image_encoding: str = "rgb8"
    jpeg_quality: int = 85
    prediction_queue_threshold: int | None = None
    mapping_accepted: StrictBool = False
    supervised_confirmed: StrictBool = False
    capture_trace: StrictBool = False
    upload_repo_id: str | None = None
    center_crop: bool = False
    async_chunks: bool = True
    duration: float = 60.0
    fps: float = 30.0
    rtc: bool = False
    arms: list[str] | None = None


class RolloutBody(InferenceBody):
    confirm_motion: bool = False


class PolicyCheckBody(InferenceBody):
    pass


class ProbeBody(InferenceBody):
    saved: str | None = None
    live: bool = False
    confirm_active_read: bool = False


class ConfigBody(BaseModel):
    """Either a full raw-YAML replacement of the rig file or a structured `control` update."""

    yaml_text: str | None = None
    control: dict[str, Any] | None = None
    hub: dict[str, Any] | None = None
    validate_only: bool = False


def create_app(
    rig_path: Path | None = None,
    *,
    datasets_dir: Path | None = None,
    outputs_dir: Path | None = None,
    frontend_dir: Path | None = None,
    session_manager: SessionManager | None = None,
) -> FastAPI:
    rig_path = Path(rig_path or DEFAULT_RIG)
    datasets_dir = Path(datasets_dir or DATASETS_DIR)
    outputs_dir = Path(outputs_dir or OUTPUT_DIR)
    frontend_dir = Path(frontend_dir or FRONTEND_DIR)
    deployments = DeploymentLog(outputs_dir / "ui" / "deployments")

    @asynccontextmanager
    async def lifespan(app):
        import asyncio

        try:
            yield
        finally:
            await asyncio.to_thread(sessions.close)
            await asyncio.to_thread(cameras.close)

    app = FastAPI(title="yamkit ui", docs_url=None, redoc_url=None, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Validation errors ordinarily echo invalid input. Inference accepts no credentials,
        # and must not reflect an accidentally submitted token back to the browser.
        errors = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                  for error in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": errors})

    def load_rig() -> RigConfig | None:
        try:
            return RigConfig.load(rig_path)
        except Exception:  # noqa: BLE001 — UI must render without a rig file
            return None

    rig0 = load_rig()
    cameras = CameraHub(rig0.cameras if rig0 else {})
    run_dirs: dict[str, Path] = {}
    inference_launch_lock = threading.RLock()

    def upload_status(run_dir: Path) -> dict | None:
        path = run_dir / "hf-upload.json"
        try:
            value = json.loads(path.read_text()) if path.is_file() and not path.is_symlink() else None
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def write_upload_status(run_dir: Path, value: dict) -> None:
        path = run_dir / "hf-upload.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({**value, "updated_at": time.time()}, indent=2) + "\n")
        temporary.replace(path)

    def upload_retry(run_dir: Path, repo_id: str, trace_dir: Path | None = None) -> str:
        args = ["yamkit", "bundle-rollout", str(run_dir), "--upload-to", repo_id]
        if trace_dir:
            args += ["--trace-dir", str(trace_dir)]
        return shlex.join(args)

    # A UI restart cannot continue its old daemon workers. Retain an explicit interrupted
    # receipt so operators can retry the finalized local bundle without starting hardware.
    for old_run in deployments.root.iterdir() if deployments.root.is_dir() else ():
        if not old_run.is_dir() or old_run.is_symlink():
            continue
        previous = upload_status(old_run)
        if previous and previous.get("status") in ("queued", "packaging", "uploading"):
            previous.update(status="interrupted", error="Dashboard restarted before upload completion; local artifacts retained")
            write_upload_status(old_run, previous)

    def upload_finalized_run(run_dir: Path, trace_dir: Path | None, repo_id: str) -> None:
        # This worker is not a managed hardware session. No session lock, child process or
        # motor/camera ownership is held while copying, hashing or contacting the Hub.
        pending = {"repo_id": repo_id, "retry_command": upload_retry(run_dir, repo_id, trace_dir)}
        try:
            from ..rollout_artifacts import package_rollout, upload_rollout

            write_upload_status(run_dir, {"status": "packaging", **pending})
            bundle = package_rollout(run_dir, trace_dir=trace_dir)
            write_upload_status(run_dir, {"status": "uploading", **pending})
            result = upload_rollout(bundle, repo_id=repo_id)
            write_upload_status(run_dir, result)
        except Exception as exc:  # noqa: BLE001 — post-run failures never affect hardware cleanup
            # Exceptions from HTTP clients can contain credentials and private endpoints.
            write_upload_status(run_dir, {"status": "failed", **pending,
                                          "error": f"{type(exc).__name__}: upload did not complete; local artifacts retained"})

    def finalize_run(run_dir: Path, status: dict) -> None:
        """Import known debug artifacts after the managed child and descendants exit."""
        try:
            source_value = status.get("meta", {}).get("debug_trace_dir")
            if source_value:
                source = Path(source_value)
                trace_root = ROOT / ".context" / "rollout-traces"
                if (source.parent == trace_root and not source.is_symlink()
                        and not trace_root.is_symlink() and not trace_root.parent.is_symlink()
                        and source.resolve().parent == trace_root.resolve()
                        and len(source.name) == 32 and all(c in "0123456789abcdef" for c in source.name)):
                    for name in TRACE_FILES:
                        artifact = source / name
                        if artifact.is_file() and not artifact.is_symlink() and artifact.stat().st_size <= 256 * 1024 * 1024:
                            shutil.copyfile(artifact, run_dir / name)
        except OSError:
            message = "Debug artifact import incomplete; original files remain in the trace directory."
            with (run_dir / "log.txt").open("a") as logfile:
                logfile.write(message + "\n")
            status = {**status, "log": [*status.get("log", []), message]}
        finally:
            deployments.finalize(run_dir, status)
        metadata_path = run_dir / "run_metadata.json"
        try:
            if metadata_path.is_file():
                snapshot = json.loads(metadata_path.read_text())
                snapshot["capture"] = {"log_complete": status.get("log_complete", False)}
                metadata_path.write_text(json.dumps(snapshot, indent=2) + "\n")
        except (OSError, ValueError, TypeError):
            # Preserve the original snapshot and let packaging report a durable failure.
            pass
        repo_id = status.get("meta", {}).get("upload_repo_id")
        if repo_id and status.get("mode") == "rollout":
            write_upload_status(run_dir, {"status": "queued", "repo_id": repo_id,
                                          "retry_command": upload_retry(run_dir, repo_id, Path(source_value) if source_value else None)})
            threading.Thread(target=upload_finalized_run,
                             args=(run_dir, Path(source_value) if source_value else None, repo_id),
                             name=f"rollout-upload-{run_dir.name}", daemon=True).start()

    def on_exit(status: dict[str, Any]) -> None:
        if status.get("mode") in ("push", "pull", "record"):
            hub.clear_cache()  # what is on the Hub may just have changed
        with inference_launch_lock:
            run_dir = run_dirs.pop(status.get("meta", {}).get("operation_id", ""), None)
        if run_dir is not None:
            finalize_run(run_dir, status)

    sessions = session_manager or SessionManager()
    sessions.on_camera_acquire = cameras.suspend
    sessions.on_camera_release = cameras.resume
    sessions.on_exit = on_exit

    def camera_statuses() -> list[dict[str, Any]]:
        statuses = cameras.statuses()
        owned = cameras.suspended_by is not None
        reg = sessions.preview_registration()
        preview_status = {}
        unavailable = False
        if reg is not None:
            try:
                preview_status = fetch_status(reg, sessions.preview_is_current)
            except PreviewUnavailable:
                unavailable = True
        # A release/acquire can happen while the status request is in flight.
        if owned != (cameras.suspended_by is not None) or (reg and not sessions.preview_is_current(reg)):
            return [{**s, "preview_state": "waiting", "preview_source": "session" if cameras.suspended_by else "direct",
                     "preview_generation": sessions.preview_generation, "frame_age_s": None} for s in cameras.statuses()]
        for status in statuses:
            status["preview_source"] = "session" if owned else "direct"
            status["preview_generation"] = sessions.preview_generation
            if owned:
                child = preview_status.get(status["name"], {})
                status["preview_state"] = child.get("state", "unavailable" if unavailable or reg or not sessions.preview_starting else "waiting")
                status["frame_age_s"] = child.get("age_s")
                status["source_seq"] = child.get("source_seq")
                status["preview_seq"] = child.get("seq")
                status["streaming"] = status["preview_state"] == "live"
                status["error"] = "preview unavailable" if status["preview_state"] == "unavailable" else None
                status["suspended_by"] = sessions.mode or "session"
            else:
                age = status["frame_age_s"]
                status["preview_state"] = ("unavailable" if status["error"] else "waiting" if age is None
                                           else "stale" if age > STALE_S else "live")
        return statuses

    def require_rig() -> RigConfig:
        rig = load_rig()
        if rig is None:
            raise HTTPException(409, f"rig file not found or invalid: {rig_path}")
        return rig

    def start(mode: str, argv: list[str], meta: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return sessions.start(mode, argv, meta)
        except RuntimeError as e:
            raise HTTPException(409, str(e)) from None

    # ---------------------------------------------------------------------------- overview --
    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        rig = load_rig()
        ifaces = list_can_interfaces()
        rig_serials = {a.can_serial for a in rig.arms.values()} if rig else set()
        return {
            "root": str(ROOT),
            "rig": {
                "path": str(rig_path),
                "found": rig is not None,
                "arms": {
                    n: {
                        "role": a.role,
                        "side": a.side,
                        "gripper": a.gripper,
                        "can_serial": a.can_serial,
                        "adapter_present": any(i.serial == a.can_serial for i in ifaces) if a.can_serial else None,
                    }
                    for n, a in (rig.arms.items() if rig else {}.items())
                },
                "pairs": [{"leader": p.leader, "follower": p.follower} for p in (rig.pairs if rig else [])],
                "control": {
                    "teleop_hz": rig.control.teleop_hz,
                    "max_joint_speed": rig.control.max_joint_speed,
                    "sync_seconds": rig.control.sync_seconds,
                }
                if rig
                else None,
                "problems": rig.validate() if rig else [],
            },
            "can": [
                {
                    "name": i.name,
                    "up": i.up,
                    "bitrate": i.bitrate,
                    "serial": i.serial,
                    "product": i.product,
                    "in_rig": i.serial in rig_serials,
                    "bus_errors": i.bus_errors,
                }
                for i in ifaces
            ],
            "can_bringup": bringup_commands([i.name for i in ifaces if not i.up]),
            "video_devices": sorted(p.name for p in Path("/dev").glob("video*")),
            "cameras": camera_statuses(),
            "session": {"active": sessions.active, "mode": sessions.mode if sessions.active else None},
            "hub": {"logged_in": bool(hub.get_token()), **(dataclasses.asdict(rig.hub) if rig else {})},
        }

    # ----------------------------------------------------------------------------- cameras --
    @app.get("/api/cameras")
    def camera_list() -> list[dict[str, Any]]:
        return camera_statuses()

    @app.get("/api/cameras/{name}/stream")
    def camera_stream(name: str) -> StreamingResponse:
        """Use the active owner's authenticated preview while it holds the devices."""
        cam = cameras.get(name)
        if cam is None:
            raise HTTPException(404, f"no camera {name!r} in the rig")
        if cameras.suspended_by:
            reg = sessions.preview_registration(name)
            if reg is None:
                raise HTTPException(409, "waiting for session camera preview")
            try:
                stream = open_stream(reg, name, sessions.preview_is_current)
            except PreviewUnavailable:
                raise HTTPException(503, "session camera preview unavailable") from None
            return PreviewStreamingResponse(stream)
        return CameraStreamingResponse(cam, media_type=MJPEG_MEDIA_TYPE)

    # ---------------------------------------------------------------------------- sessions --
    @app.get("/api/session")
    def session_status() -> dict[str, Any]:
        return sessions.status()

    @app.post("/api/session/stop")
    def session_stop() -> dict[str, Any]:
        return sessions.stop()

    @app.post("/api/session/rest")
    def session_rest(body: RestBody) -> dict[str, Any]:
        """Park: every arm (or the given ones) moves slowly to its home pose and is released there."""
        require_rig()
        args = ["rest", *(body.arms or []), "--rig", str(rig_path)]
        return start("rest", sessions.yamkit_argv(*args))

    @app.post("/api/session/read")
    def session_read(body: ReadBody) -> dict[str, Any]:
        require_rig()
        args = ["read", *(body.arms or []), "--rig", str(rig_path), "--hz", str(body.hz)]
        return start("read", sessions.yamkit_argv(*args))

    @app.post("/api/session/teleop")
    def session_teleop(body: TeleopBody) -> dict[str, Any]:
        require_rig()
        # --print-state adds per-arm q/gripper lines to the child's output so the Live page
        # can show joint state during teleop (same format `yamkit read` prints)
        args = ["teleop", "--rig", str(rig_path), "--print-state"]
        for p in body.pairs or []:
            args += ["--pair", p]
        if body.auto_engage:
            args.append("--auto-engage")
        if body.bilateral_kp is not None:
            args += ["--bilateral-kp", str(body.bilateral_kp)]
        if body.duration is not None:
            args += ["--duration", str(body.duration)]
        return start("teleop", sessions.yamkit_argv(*args))

    @app.post("/api/session/record")
    def session_record(body: RecordBody) -> dict[str, Any]:
        require_rig()
        args = [
            "record",
            "--rig", str(rig_path),
            "--auto-engage",
            "--name", body.name,
            "--task", body.task,
            "--episodes", str(body.episodes),
            "--episode-s", str(body.episode_s),
            "--reset-s", str(body.reset_s),
            "--fps", str(body.fps),
        ]
        for a in body.arms or []:
            args += ["--arms", a]
        if body.resume:
            args.append("--resume")
        if body.to:
            if body.to not in hub.DESTINATIONS:
                raise HTTPException(422, f"to must be one of {hub.DESTINATIONS}")
            args += ["--to", body.to]
        meta = {"name": body.name, "task": body.task, "episodes": body.episodes,
                "episode_s": body.episode_s, "reset_s": body.reset_s, "fps": body.fps, "to": body.to}
        return start("record", sessions.yamkit_argv(*args), meta)

    # --------------------------------------------------------------------------------- hub --
    def rig_hub():
        rig = load_rig()
        return rig.hub if rig else None

    @app.get("/api/hub")
    def hub_status() -> dict[str, Any]:
        h = rig_hub()
        return {**hub.status(), "settings": dataclasses.asdict(h) if h else None, "token_path": str(hub.token_path())}

    @app.post("/api/hub/login")
    def hub_login(body: HubLoginBody) -> dict[str, Any]:
        try:
            name = hub.login(body.token)
        except Exception as e:  # noqa: BLE001 — bad token / offline
            raise HTTPException(400, f"sign-in failed: {e}") from None
        return {"username": name}

    @app.post("/api/hub/logout")
    def hub_logout() -> dict[str, Any]:
        hub.logout()
        return hub.status()

    @app.post("/api/hub/push-dataset")
    def hub_push_dataset(body: HubTransferBody) -> dict[str, Any]:
        dataset_dir(body.name)
        args = ["push-dataset", body.name, "--rig", str(rig_path)] + (["--remove-local"] if body.remove_local else [])
        return start("push", sessions.yamkit_argv(*args), {"name": body.name})

    @app.post("/api/hub/pull-dataset")
    def hub_pull_dataset(body: HubTransferBody) -> dict[str, Any]:
        return start("pull", sessions.yamkit_argv("pull-dataset", body.name, "--rig", str(rig_path)), {"name": body.name})

    @app.post("/api/hub/push-model")
    def hub_push_model(body: HubTransferBody) -> dict[str, Any]:
        d = (outputs_dir / body.name).resolve()
        if outputs_dir.resolve() not in d.parents or not (d / "config.json").is_file():
            raise HTTPException(404, f"no checkpoint at outputs/{body.name}")
        return start("push", sessions.yamkit_argv("push-model", str(d), "--rig", str(rig_path)), {"name": body.name})

    def inference_options(body: InferenceBody, *, motion: bool = False):
        from ..deployment import InferenceOptions

        values = {field.name: getattr(body, field.name) for field in dataclasses.fields(InferenceOptions)
                  if hasattr(body, field.name)}
        values["arms"] = tuple(body.arms or ())
        values["rig_path"] = str(rig_path)
        try:
            return InferenceOptions(**values).validate(motion=motion)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    def modal_attachment(options, rig: RigConfig) -> dict:
        """Inspect an exact retained session locally; never contact cloud or open hardware."""
        from ..inference.profiles import get_profile
        from ..inference.qualification import (
            MAX_AGE_S,
            is_cloud_host,
            settings_from_rig,
            validate_qualification,
        )
        from ..modal_ops import http_credentials
        from ..probes import preflight_live_probe

        if (options.backend != "modal" or get_profile(options.policy).id != "molmoact2"
                or not options.modal_app or options.call_mode != "http"
                or options.execution_mode != "cuda_graph10" or options.image_encoding != "rgb8"):
            raise ValueError("Attach the exact Conductor-prepared MolmoAct2 HTTP cuda_graph10 session with raw RGB")
        if is_cloud_host():
            raise ValueError("Check and start the retained session on the Lenovo robot host")
        profile = get_profile(options.policy)
        specs, _ = preflight_live_probe(rig, options.arms or None, expected_state_names=profile.state_names)
        if (len(specs) != 2 or any(spec.side != side or spec.arm_type != "yam"
                                 or spec.gripper != "linear_4310"
                                 for side, spec in zip(("left", "right"), specs))):
            raise ValueError("Select the physically verified left and right YAM followers with LINEAR_4310 grippers")
        if set(rig.cameras) != set(profile.image_keys):
            raise ValueError("Rig cameras must exactly match the retained policy profile")
        if options.fps != profile.fps:
            raise ValueError("The retained policy requires 30 Hz actions")
        settings = settings_from_rig(options)
        if settings.get("http_ingress") != "tunnel":
            raise ValueError("Browser attachment requires a bounded retained HTTP tunnel")
        record = validate_qualification(settings)
        http_credentials(options.modal_app)  # Verify locally; never include this private object in the response.
        expires = min(settings["http_session_expires_at"], record["created_unix_s"] + MAX_AGE_S)
        checked = time.time()
        if checked + options.duration + 60 >= expires:
            raise ValueError("Retained session expires too soon for this duration, 30 seconds of startup and 30 seconds of return home; refresh it in Conductor")
        return {"ready": True, "reason": "Qualified for these settings; mapping acceptance and supervised Start are still required",
                "selection_key": options.operation_key, "checked_at": checked,
                "expires_at": expires, "modal_app": options.modal_app}

    def validate_trace(body: InferenceBody) -> None:
        if body.upload_repo_id is not None:
            from huggingface_hub.utils import validate_repo_id

            validate_repo_id(body.upload_repo_id)
            if body.upload_repo_id.count("/") != 1:
                raise ValueError("Upload destination must be a namespace/repository private dataset ID")
            body.capture_trace = True  # All camera frames and trace data must exist for the upload.
        if body.capture_trace and (
                body.backend != "modal" or body.policy not in ("molmoact2", "lerobot/MolmoAct2-BimanualYAM-LeRobot")
                or body.task != TRACE_TASK or body.duration not in (5, 10)
                or body.call_mode != "http" or body.execution_mode != "cuda_graph10"
                or body.image_encoding != "rgb8" or body.center_crop or body.rtc or not body.async_chunks
                or body.prediction_queue_threshold not in (None, 30)
                or body.arms not in (None, ["left_follower", "right_follower"])):
            raise ValueError("Debug capture requires the orange-lid task, 5 or 10 seconds, both named followers, and the unchanged raw-RGB HTTP graph settings")
        if (body.capture_trace and body.arms is None
                and [pair.follower for pair in require_rig().pairs] != ["left_follower", "right_follower"]):
            raise ValueError("Debug capture requires the rig's default followers to be left_follower then right_follower")

    @app.post("/api/inference/preflight")
    def inference_preflight(body: InferenceBody) -> dict:
        """Read-only exact-form qualification check, without operator approval or a child process."""
        options = None
        try:
            options = inference_options(body)
            validate_trace(body)
            if sessions.active:
                raise ValueError("Wait for the current UI session to finish before checking another rollout")
            return modal_attachment(options, require_rig())
        except HTTPException as exc:
            if exc.status_code != 422:
                raise
            return {"ready": False, "reason": exc.detail, "selection_key": None,
                    "checked_at": time.time(), "expires_at": None}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else "Retained session metadata or qualification is unavailable"
            return {"ready": False, "reason": reason, "selection_key": options.operation_key,
                    "checked_at": time.time(), "expires_at": None}

    def inference_start(mode: str, args: list[str], options, *, argv_override=None, extra_meta=None) -> dict:
        with inference_launch_lock:
            if sessions.active:
                raise HTTPException(409, "Wait for the current UI session to finish")
            operation_id = uuid.uuid4().hex
            meta = {**dataclasses.asdict(options), "operation_id": operation_id,
                    "profile_key": options.operation_key, **(extra_meta or {})}
            # Reserve the history and snapshot provenance before the child can open hardware.
            run_dir = deployments.create({"active": True, "mode": mode, "meta": meta,
                                          "started_at": time.time()})
            meta["session_log_path"] = str(run_dir / "log.txt")
            if mode == "rollout":
                snapshot = _rollout_metadata(options, require_rig())
                snapshot["original_paths"] = {"deployment_dir": str(run_dir), "rig_path": str(rig_path),
                                               "trace_dir": meta.get("debug_trace_dir")}
                (run_dir / "run_metadata.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            run_dirs[operation_id] = run_dir
            try:
                st = start(mode, argv_override if argv_override is not None else sessions.yamkit_argv(*args), meta)
            except HTTPException:
                pending = run_dirs.pop(operation_id, None)
                if pending is not None:
                    deployments.finalize(pending, {"active": False, "mode": mode, "meta": meta,
                                                   "ended_at": time.time(), "returncode": -1,
                                                   "log": ["Managed child did not start."]})
                # Popen failure may already have invoked on_exit synchronously. Its complete
                # snapshot owns finalization; never overwrite it while its upload is starting.
                raise
            # Immediate exits are finalized by the callback after this launch lock releases.
            return st

    @app.get("/api/inference/profiles")
    def inference_profiles() -> dict:
        from ..inference.profiles import list_profiles
        from ..modal_ops import credential_status, owned_service

        rig = load_rig()
        return {"profiles": list_profiles(), "credentials": credential_status(),
                "owned_service": owned_service(), "default_backend": "local",
                "rollout_repo": rig.hub.rollout_repo if rig else None}

    @app.post("/api/session/rollout")
    def session_rollout(body: RolloutBody) -> dict[str, Any]:
        rig = require_rig()
        options = inference_options(body, motion=True)
        if not body.confirm_motion:
            raise HTTPException(422, "explicit motion confirmation is required")
        try:
            validate_trace(body)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        if body.backend == "modal":
            try:
                modal_attachment(options, rig)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                reason = str(exc) if isinstance(exc, ValueError) else "Retained session metadata or qualification is unavailable"
                raise HTTPException(422, reason) from None
        if body.backend == "modal" or body.policy in ("molmoact2", "lerobot/MolmoAct2-BimanualYAM-LeRobot"):
            from ..inference.profiles import get_profile
            from ..probes import preflight_live_probe

            try:
                profile = get_profile(body.policy)
                preflight_live_probe(rig, body.arms, expected_state_names=profile.state_names)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
        args = ["rollout", "--rig", str(rig_path), *options.cli_args(),
                "--duration", str(body.duration), "--fps", str(body.fps)]
        if body.rtc:
            args.append("--rtc")
        for a in body.arms or []:
            args += ["--arms", a]
        if body.capture_trace:
            trace_dir = ROOT / ".context" / "rollout-traces" / uuid.uuid4().hex
            trace_args = [sys.executable, str(ROOT / "scripts" / "trace_rollout.py"), "--run",
                          "--duration", str(int(body.duration)), "--modal-app", str(body.modal_app),
                          "--rig", str(rig_path), "--output-dir", str(trace_dir), "--confirm-supervised"]
            return inference_start("rollout", args, options, argv_override=trace_args,
                                   extra_meta={"capture_trace": True, "debug_trace_dir": str(trace_dir),
                                               "upload_repo_id": body.upload_repo_id})
        return inference_start("rollout", args, options)

    @app.post("/api/session/policy-check")
    def session_policy_check(body: PolicyCheckBody) -> dict[str, Any]:
        options = inference_options(body)
        args = ["policy-check", "--rig", str(rig_path), *options.cli_args()]
        for arm in body.arms or []:
            args += ["--arms", arm]
        return inference_start("policy-check", args, options)

    @app.post("/api/session/modal-prepare")
    def session_modal_prepare(body: PolicyCheckBody) -> dict:
        options = inference_options(body)
        if options.backend != "modal":
            raise HTTPException(422, "select Modal before preparing a cloud service")
        if options.call_mode == "http" or options.execution_mode == "cuda_graph10":
            raise HTTPException(422, "Prepare and qualify the retained HTTP session in Conductor, then attach here")
        return inference_start("modal-prepare", ["modal-prepare", "--policy", options.policy,
                                                "--gpu", options.gpu], options)

    @app.post("/api/session/modal-shutdown")
    def session_modal_shutdown() -> dict:
        return start("modal-shutdown", sessions.yamkit_argv("modal-shutdown"))

    @app.post("/api/session/policy-probe")
    def session_policy_probe(body: ProbeBody) -> dict:
        options = inference_options(body)
        if body.live == bool(body.saved):
            raise HTTPException(422, "choose a saved snapshot or live active read")
        args = ["policy-probe", "--rig", str(rig_path), *options.cli_args()]
        if body.live:
            if not body.confirm_active_read:
                raise HTTPException(422, "explicit GRAVITY-COMPENSATION ACTIVE READ confirmation is required")
            from ..inference.profiles import get_profile
            from ..probes import preflight_live_probe

            try:
                profile = get_profile(body.policy)
                profile.require_robot_mapping()
                preflight_live_probe(require_rig(), body.arms, expected_state_names=profile.state_names)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from None
            args += ["--live", "--approve-active-read"]
        else:
            snapshot = (ROOT / body.saved).resolve()
            if ROOT not in snapshot.parents or snapshot.suffix != ".npz" or not snapshot.is_file():
                raise HTTPException(422, "saved snapshot must be an existing .npz file inside this repository")
            args += ["--saved", str(snapshot)]
        for arm in body.arms or []:
            args += ["--arms", arm]
        return inference_start("policy-probe-live" if body.live else "policy-probe", args, options)

    # ------------------------------------------------------------------------------ config --
    def config_payload() -> dict[str, Any]:
        rig = load_rig()
        text = rig_path.read_text() if rig_path.is_file() else ""
        return {
            "path": str(rig_path),
            "found": rig is not None,
            "yaml": text,
            "control": dataclasses.asdict(rig.control) if rig else None,
            "arms": {
                n: {k: v for k, v in dataclasses.asdict(a).items() if k != "name"}
                for n, a in (rig.arms.items() if rig else {}.items())
            },
            "pairs": [dataclasses.asdict(p) for p in (rig.pairs if rig else [])],
            "cameras": rig.cameras if rig else {},
            "hub": dataclasses.asdict(rig.hub) if rig else None,
            "problems": rig.validate() if rig else [],
        }

    def validate_rig_yaml(text: str) -> RigConfig:
        """Parse + validate raw YAML as a rig file; raise HTTPException(422) with details if bad."""
        import yaml as pyyaml

        try:
            data = pyyaml.safe_load(text)
        except pyyaml.YAMLError as e:
            raise HTTPException(422, f"YAML syntax error: {e}") from None
        if not isinstance(data, dict):
            raise HTTPException(422, "rig file must be a YAML mapping")
        try:
            cfg = RigConfig.from_dict(data)
        except (TypeError, ValueError, KeyError) as e:
            raise HTTPException(422, f"invalid rig config: {e}") from None
        problems = cfg.validate()
        if problems:
            raise HTTPException(422, "invalid rig config: " + "; ".join(problems))
        return cfg

    @app.get("/api/config")
    def config_get() -> dict[str, Any]:
        return config_payload()

    @app.post("/api/config")
    def config_save(body: ConfigBody) -> dict[str, Any]:
        if body.validate_only:
            if body.yaml_text is None:
                raise HTTPException(422, "validate_only needs yaml_text")
            validate_rig_yaml(body.yaml_text)
            return {"valid": True}
        if sessions.active:
            raise HTTPException(409, f"a {sessions.mode!r} session is running — stop it before editing the rig")
        if body.yaml_text is not None:
            validate_rig_yaml(body.yaml_text)
            rig_path.parent.mkdir(parents=True, exist_ok=True)
            rig_path.write_text(body.yaml_text)  # verbatim: keeps the user's comments/ordering
        elif body.control is not None:
            rig = require_rig()
            known = set(dataclasses.asdict(rig.control))
            unknown = set(body.control) - known
            if unknown:
                raise HTTPException(422, f"unknown control field(s): {sorted(unknown)}")
            try:
                # Reconstruct through ControlSpec's validator before saving. Mutating its
                # fields bypasses validation, and coercion can turn invalid buttons or
                # booleans into apparently valid safety settings.
                rig.control = dataclasses.replace(rig.control, **body.control)
            except (TypeError, ValueError, OverflowError) as exc:
                raise HTTPException(422, f"invalid rig config: {exc}") from None
            problems = rig.validate()
            if problems:
                raise HTTPException(422, "invalid rig config: " + "; ".join(problems))
            rig.save(rig_path)
        elif body.hub is not None:
            rig = require_rig()
            unknown = set(body.hub) - set(dataclasses.asdict(rig.hub))
            if unknown:
                raise HTTPException(422, f"unknown hub field(s): {sorted(unknown)}")
            try:
                from ..config import HubSpec

                rig.hub = HubSpec(**{**dataclasses.asdict(rig.hub), **{k: (v or None) if k == "username" else v for k, v in body.hub.items()}})
            except (TypeError, ValueError) as e:
                raise HTTPException(422, str(e)) from None
            rig.save(rig_path)
        else:
            raise HTTPException(422, "provide yaml_text or control")
        rig = load_rig()
        cameras.reload(rig.cameras if rig else {})
        return config_payload()

    # ---------------------------------------------------------------------------- datasets --
    def dataset_dir(name: str) -> Path:
        d = (datasets_dir / name).resolve()
        if datasets_dir.resolve() not in d.parents or not d.is_dir():
            raise HTTPException(404, f"no dataset {name!r}")
        return d

    @app.get("/api/datasets")
    def datasets() -> list[dict[str, Any]]:
        """Local datasets and the account's Hub datasets in one list (`where`: local | cloud | both)."""
        h = rig_hub()
        local = [{**d, "where": "local"} for d in catalog.list_datasets(datasets_dir)]
        by_name = {d["name"]: d for d in local}
        for c in hub.list_datasets(h.username if h else None) if hub.get_token() else []:
            if c["name"] in by_name:
                by_name[c["name"]].update(where="both", repo_id=c["repo_id"], url=c["url"], private=c["private"])
            else:
                local.append({**c, "where": "cloud"})
        return local

    @app.get("/api/datasets/{name}")
    def dataset(name: str) -> dict[str, Any]:
        d = catalog.dataset_detail(dataset_dir(name))
        if d is None:
            raise HTTPException(404, f"{name!r} has no readable meta/info.json")
        return d

    @app.get("/api/datasets/{name}/episodes/{index}")
    def episode(name: str, index: int) -> dict[str, Any]:
        s = catalog.episode_series(dataset_dir(name), index)
        if s is None:
            raise HTTPException(404, f"episode {index} not found in {name!r} (or pyarrow unavailable)")
        return s

    @app.get("/api/datasets/{name}/video/{camera}/{index}")
    def episode_video(name: str, camera: str, index: int, request: Request) -> Response:
        r = catalog.episode_video_file(dataset_dir(name), camera, index)
        if r is None:
            raise HTTPException(404, "no video for this camera/episode")
        return _range_response(r[0], request, media_type="video/mp4")

    # ------------------------------------------------------------------- deployments/models --
    @app.get("/api/deployments")
    def deployments_list() -> list[dict[str, Any]]:
        return [{**entry, "upload": upload_status(deployments.root / entry["id"])}
                for entry in catalog.list_deployments(deployments.root)]

    @app.get("/api/deployments/{run_id}")
    def deployment(run_id: str) -> dict[str, Any]:
        d = catalog.deployment_detail(deployments.root, run_id)
        if d is None:
            raise HTTPException(404, f"no deployment {run_id!r}")
        directory = deployments.root / run_id
        d["upload"] = upload_status(directory)
        d["artifacts"] = sorted(name for name in TRACE_FILES if not name.endswith(".mp4")
                                and (directory / name).is_file() and not (directory / name).is_symlink())
        return d

    @app.get("/api/deployments/{run_id}/artifact/{filename}")
    def deployment_artifact(run_id: str, filename: str) -> FileResponse:
        directory = (deployments.root / run_id).resolve()
        path = directory / filename
        if (directory.parent != deployments.root.resolve() or filename not in TRACE_FILES
                or path.is_symlink() or not path.is_file() or path.resolve().parent != directory):
            raise HTTPException(404, "no such debug artifact")
        headers = {"X-Content-Type-Options": "nosniff"}
        if filename.endswith(".html"):
            headers["Content-Security-Policy"] = "sandbox; default-src 'none'; img-src 'self' data:; media-src 'self'; style-src 'unsafe-inline'"
        return FileResponse(path, headers=headers)

    @app.get("/api/deployments/{run_id}/video/{filename}")
    def deployment_video(run_id: str, filename: str, request: Request) -> Response:
        p = (deployments.root / run_id / filename).resolve()
        if deployments.root.resolve() not in p.parents or not p.is_file() or p.suffix != ".mp4":
            raise HTTPException(404, "no such video")
        return _range_response(p, request, media_type="video/mp4")

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        """Local checkpoints and the account's Hub models in one list (`where`: local | cloud | both)."""
        h = rig_hub()
        local = [{**m, "where": "local", "name": m["path"].split("/")[1] if m["path"].startswith("train/") and m["path"].count("/") >= 1 else m["path"]} for m in catalog.list_models(outputs_dir)]
        by_name = {m["name"]: m for m in local}
        for c in hub.list_models(h.username if h else None) if hub.get_token() else []:
            if c["name"] in by_name:
                by_name[c["name"]].update(where="both", repo_id=c["repo_id"], url=c["url"], private=c["private"])
            else:
                local.append({**c, "where": "cloud"})
        return local

    @app.get("/api/hub/models/{repo:path}")
    def hub_model(repo: str) -> dict[str, Any]:
        d = hub.model_detail(repo)
        if d is None:
            raise HTTPException(404, f"{repo!r} is not a LeRobot policy on the Hub (or the Hub is unreachable)")
        return d

    @app.get("/api/models/{model_path:path}")
    def model(model_path: str) -> dict[str, Any]:
        d = (outputs_dir / model_path).resolve()
        if outputs_dir.resolve() not in d.parents or not d.is_dir():
            raise HTTPException(404, f"no checkpoint at {model_path!r}")
        detail = catalog.model_detail(outputs_dir, d)
        if detail is None:
            raise HTTPException(404, f"{model_path!r} does not look like a checkpoint directory")
        return detail

    @app.get("/api/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    if frontend_dir.is_dir():

        @app.middleware("http")
        async def _no_stale_page_code(request: Request, call_next):
            # The page is a few local files; a browser that keeps yesterday's app.js talks to today's
            # server and shows blank tiles. Always revalidate them (the API responses are untouched).
            response = await call_next(request)
            if not request.url.path.startswith("/api/"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        @app.get("/", include_in_schema=False)
        @app.get("/index.html", include_in_schema=False)
        def index() -> Response:
            # app.js / style.css are referenced with their modification time as a version, so a
            # browser can never keep running an older app.js against a newer server.
            html = (frontend_dir / "index.html").read_text()
            for asset in ("app.js", "style.css"):
                try:
                    v = int((frontend_dir / asset).stat().st_mtime)
                except OSError:
                    continue
                html = html.replace(f'"{asset}"', f'"{asset}?v={v}"')
            return Response(html, media_type="text/html", headers={"Cache-Control": "no-cache"})

        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="ui")
    return app


def _range_response(path: Path, request: Request, media_type: str) -> Response:
    """Serve a file with HTTP Range support (video seeking)."""
    size = path.stat().st_size
    range_header = request.headers.get("range")
    headers = {"accept-ranges": "bytes"}
    if range_header and range_header.startswith("bytes="):
        try:
            start_s, _, end_s = range_header[len("bytes=") :].partition("-")
            start = int(start_s) if start_s else 0
            end = min(int(end_s) if end_s else size - 1, size - 1)
        except ValueError:
            raise HTTPException(416, "bad range") from None
        if start > end or start >= size:
            raise HTTPException(416, "range out of bounds")
        with open(path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start + 1)
        headers["content-range"] = f"bytes {start}-{end}/{size}"
        return Response(chunk, status_code=206, media_type=media_type, headers=headers)
    return Response(path.read_bytes(), media_type=media_type, headers=headers)


def run(rig_path: Path | None = None, host: str = "127.0.0.1", port: int = 8400) -> None:
    import uvicorn

    uvicorn.run(create_app(rig_path), host=host, port=port, log_level="info")
