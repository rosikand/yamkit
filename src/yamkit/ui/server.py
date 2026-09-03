"""FastAPI app behind `yamkit ui`.

Read-only endpoints (rig, CAN, datasets, models, deployments) touch only files and sysfs.
Hardware endpoints spawn the unmodified `yamkit` CLI via `SessionManager` — starting the server
or opening any page never connects to (and never energises) an arm.
"""

from __future__ import annotations

import dataclasses
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..can import bringup_commands, list_can_interfaces
from ..config import RigConfig
from ..paths import DATASETS_DIR, DEFAULT_RIG, OUTPUT_DIR, ROOT
from . import catalog
from .camstream import CameraHub
from .sessions import CAMERA_MODES, DeploymentLog, SessionManager

FRONTEND_DIR = ROOT / "ui"


# --------------------------------------------------------------------------- request bodies --
class ReadBody(BaseModel):
    arms: list[str] | None = None
    hz: float = 5.0


class TeleopBody(BaseModel):
    pairs: list[str] | None = None
    auto_engage: bool = False
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


class RolloutBody(BaseModel):
    policy: str
    task: str
    duration: float = 60.0
    fps: float = 30.0
    rtc: bool = True
    arms: list[str] | None = None


class PolicyCheckBody(BaseModel):
    policy: str
    task: str = "pick up the object"
    device: str = "cpu"


class ConfigBody(BaseModel):
    """Either a full raw-YAML replacement of the rig file or a structured `control` update."""

    yaml_text: str | None = None
    control: dict[str, float | int] | None = None
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

    app = FastAPI(title="yamkit ui", docs_url=None, redoc_url=None)

    def load_rig() -> RigConfig | None:
        try:
            return RigConfig.load(rig_path)
        except Exception:  # noqa: BLE001 — UI must render without a rig file
            return None

    rig0 = load_rig()
    cameras = CameraHub(rig0.cameras if rig0 else {})
    run_dir_box: dict[str, Path | None] = {"dir": None}

    def on_start(mode: str) -> None:
        if mode in CAMERA_MODES:
            cameras.suspend(mode)

    def on_exit(status: dict[str, Any]) -> None:
        cameras.resume()
        if run_dir_box["dir"] is not None:
            deployments.finalize(run_dir_box["dir"], status)
            run_dir_box["dir"] = None

    sessions = session_manager or SessionManager(on_start=on_start, on_exit=on_exit)
    if session_manager is not None:  # injected (tests): still wire the camera/deployment hooks
        sessions.on_start = on_start
        sessions.on_exit = on_exit

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
            "cameras": cameras.statuses(),
            "session": {"active": sessions.active, "mode": sessions.mode if sessions.active else None},
        }

    # ----------------------------------------------------------------------------- cameras --
    @app.get("/api/cameras")
    def camera_list() -> list[dict[str, Any]]:
        return cameras.statuses()

    @app.get("/api/cameras/{name}/stream")
    def camera_stream(name: str) -> StreamingResponse:
        cam = cameras.get(name)
        if cam is None:
            raise HTTPException(404, f"no camera {name!r} in the rig")
        if cameras.suspended_by:
            raise HTTPException(409, f"cameras are in use by the {cameras.suspended_by} session")
        from starlette.background import BackgroundTask

        stop = threading.Event()
        return StreamingResponse(
            cam.frames(stop),
            media_type="multipart/x-mixed-replace; boundary=yamkitframe",
            background=BackgroundTask(stop.set),
        )

    # ---------------------------------------------------------------------------- sessions --
    @app.get("/api/session")
    def session_status() -> dict[str, Any]:
        return sessions.status()

    @app.post("/api/session/stop")
    def session_stop() -> dict[str, Any]:
        return sessions.stop()

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
        meta = {"name": body.name, "task": body.task, "episodes": body.episodes,
                "episode_s": body.episode_s, "reset_s": body.reset_s, "fps": body.fps}
        return start("record", sessions.yamkit_argv(*args), meta)

    @app.post("/api/session/rollout")
    def session_rollout(body: RolloutBody) -> dict[str, Any]:
        require_rig()
        args = ["rollout", "--rig", str(rig_path), "--policy", body.policy, "--task", body.task,
                "--duration", str(body.duration), "--fps", str(body.fps)]
        if body.rtc:
            args.append("--rtc")
        for a in body.arms or []:
            args += ["--arms", a]
        meta = {"policy": body.policy, "task": body.task, "duration": body.duration}
        st = start("rollout", sessions.yamkit_argv(*args), meta)
        run_dir_box["dir"] = deployments.create(sessions.status())
        return st

    @app.post("/api/session/policy-check")
    def session_policy_check(body: PolicyCheckBody) -> dict[str, Any]:
        require_rig()
        args = ["policy-check", "--rig", str(rig_path), "--policy", body.policy,
                "--task", body.task, "--device", body.device]
        meta = {"policy": body.policy, "task": body.task}
        st = start("policy-check", sessions.yamkit_argv(*args), meta)
        run_dir_box["dir"] = deployments.create(sessions.status())
        return st

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
            for k, v in body.control.items():
                setattr(rig.control, k, int(v) if k == "engage_button" else float(v))
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
        return catalog.list_datasets(datasets_dir)

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
        return catalog.list_deployments(deployments.root)

    @app.get("/api/deployments/{run_id}")
    def deployment(run_id: str) -> dict[str, Any]:
        d = catalog.deployment_detail(deployments.root, run_id)
        if d is None:
            raise HTTPException(404, f"no deployment {run_id!r}")
        return d

    @app.get("/api/deployments/{run_id}/video/{filename}")
    def deployment_video(run_id: str, filename: str, request: Request) -> Response:
        p = (deployments.root / run_id / filename).resolve()
        if deployments.root.resolve() not in p.parents or not p.is_file() or p.suffix != ".mp4":
            raise HTTPException(404, "no such video")
        return _range_response(p, request, media_type="video/mp4")

    @app.get("/api/models")
    def models() -> list[dict[str, Any]]:
        return catalog.list_models(outputs_dir)

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
