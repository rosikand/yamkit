"""FastAPI app behind `yamkit ui`.

Read-only endpoints (rig, CAN, datasets, models, deployments) touch only files and sysfs.
Hardware endpoints spawn the unmodified `yamkit` CLI via `SessionManager` — starting the server
or opening any page never connects to (and never energises) an arm.
"""

from __future__ import annotations

import dataclasses
import threading
from contextlib import asynccontextmanager
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .. import hub
from ..can import bringup_commands, list_can_interfaces
from ..config import RigConfig
from ..paths import DATASETS_DIR, DEFAULT_RIG, OUTPUT_DIR, ROOT
from ..preview import MJPEG_MEDIA_TYPE, STALE_S
from . import catalog
from .camstream import CameraHub
from .preview_proxy import PreviewStreamingResponse, PreviewUnavailable, fetch_status, open_stream
from .sessions import DeploymentLog, SessionManager

FRONTEND_DIR = ROOT / "ui"


# --------------------------------------------------------------------------- request bodies --
class RestBody(BaseModel):
    arms: list[str] | None = None


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
    control: dict[str, float | int] | None = None
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
    run_dir_box: dict[str, Path | None] = {"dir": None}
    inference_launch_lock = threading.Lock()

    def on_exit(status: dict[str, Any]) -> None:
        if status.get("mode") in ("push", "pull", "record"):
            hub.clear_cache()  # what is on the Hub may just have changed
        if run_dir_box["dir"] is not None:
            deployments.finalize(run_dir_box["dir"], status)
            run_dir_box["dir"] = None

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
        from starlette.background import BackgroundTask

        if cameras.suspended_by:
            reg = sessions.preview_registration(name)
            if reg is None:
                raise HTTPException(409, "waiting for session camera preview")
            try:
                stream = open_stream(reg, name, sessions.preview_is_current)
            except PreviewUnavailable:
                raise HTTPException(503, "session camera preview unavailable") from None
            return PreviewStreamingResponse(stream)
        stop = threading.Event()
        return StreamingResponse(
            cam.frames(stop),
            media_type=MJPEG_MEDIA_TYPE,
            headers={"Cache-Control": "no-store"},
            background=BackgroundTask(stop.set),
        )

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

        values = {field.name: getattr(body, field.name) for field in dataclasses.fields(InferenceOptions)}
        values["arms"] = tuple(body.arms or ())
        try:
            return InferenceOptions(**values).validate(motion=motion)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    def inference_start(mode: str, args: list[str], options) -> dict:
        with inference_launch_lock:
            operation_id = uuid.uuid4().hex
            meta = {**dataclasses.asdict(options), "operation_id": operation_id,
                    "profile_key": options.operation_key}
            st = start(mode, sessions.yamkit_argv(*args), meta)
            # The child can exit between start() and writing its history record. Finalize that
            # snapshot here too so an immediate failure cannot stay marked as running.
            run_dir = deployments.create(st)
            run_dir_box["dir"] = run_dir
            current = sessions.status()
            if not current["active"]:
                deployments.finalize(run_dir, current)
                run_dir_box["dir"] = None
            return st

    @app.get("/api/inference/profiles")
    def inference_profiles() -> dict:
        from ..inference.profiles import list_profiles
        from ..modal_ops import credential_status, owned_service

        return {"profiles": list_profiles(), "credentials": credential_status(),
                "owned_service": owned_service(), "default_backend": "local"}

    @app.post("/api/session/rollout")
    def session_rollout(body: RolloutBody) -> dict[str, Any]:
        rig = require_rig()
        options = inference_options(body, motion=True)
        if not body.confirm_motion:
            raise HTTPException(422, "explicit motion confirmation is required")
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
            for k, v in body.control.items():
                setattr(rig.control, k, int(v) if k == "engage_button" else float(v))
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
