"""Shared CLI/UI action probes. Predictions are inspected and never executed.

Saved observations never open hardware. Live capture uses the existing ``yamkit read`` arm
lifecycle, which ACTIVATES gravity compensation. It deliberately bypasses the follower plugin's
homing connect/disconnect. Policy preprocessing belongs to the supplied predictor, so probes and
rollout use the same image/mapping/normalization boundary.
"""

from __future__ import annotations

import json
import logging
import math
import time
import zipfile
from collections.abc import Callable, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import N_JOINTS, ArmSpec, RigConfig

log = logging.getLogger(__name__)
ACTIVE_READ_LABEL = "GRAVITY-COMPENSATION ACTIVE READ"
SAVED_LABEL = "SAVED OBSERVATION — zero arm activation"
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
MAX_CAMERAS = 8
CAMERA_MAX_AGE_MS = 200
MAX_ACTION_VALUES = 64_000


@dataclass
class ProbeObservation:
    state: np.ndarray
    state_names: tuple[str, ...]
    images: dict[str, np.ndarray]
    source: str
    captured_at: float | None
    captured_monotonic: float | None = None
    mode: str = "saved"
    camera_max_age_s: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def age_s(self) -> float | None:
        """Live age uses this host's monotonic clock; saved age is an explicitly labeled estimate."""
        if self.captured_monotonic is not None:
            return max(0.0, time.monotonic() - self.captured_monotonic) + self.camera_max_age_s
        if self.captured_at is None:
            return None
        return time.time() - self.captured_at + self.camera_max_age_s

    def validate(self) -> None:
        if not isinstance(self.state, np.ndarray):
            raise TypeError("observation state must be a bounded NumPy array")
        state = self.state
        if not 1 <= len(self.state_names) <= 64 or state.shape != (len(self.state_names),):
            raise ValueError("observation state must exactly match its ordered state_names (1–64 values)")
        if not all(isinstance(n, str) and 1 <= len(n) <= 128 for n in self.state_names) or len(set(self.state_names)) != len(self.state_names):
            raise ValueError("state_names must contain unique nonempty names")
        if state.dtype.kind not in "fiu" or not np.isfinite(state).all():
            raise ValueError("observation state must contain finite numbers")
        if self.mode not in ("saved", "live"):
            raise ValueError("probe mode must be saved or live")
        if self.captured_at is not None and not math.isfinite(self.captured_at):
            raise ValueError("captured_at must be a finite Unix timestamp or null")
        if self.captured_monotonic is not None and not math.isfinite(self.captured_monotonic):
            raise ValueError("captured_monotonic must be finite or null")
        if not math.isfinite(self.camera_max_age_s) or self.camera_max_age_s < 0:
            raise ValueError("camera age allowance must be finite and nonnegative")
        if len(self.images) > MAX_CAMERAS:
            raise ValueError(f"at most {MAX_CAMERAS} named images are supported")
        nbytes = state.nbytes
        for name, value in self.images.items():
            if not isinstance(value, np.ndarray):
                raise TypeError("images must be bounded NumPy arrays")
            frame = value
            if not isinstance(name, str) or not name or frame.dtype != np.uint8:
                raise ValueError("images require nonempty names and uint8 RGB pixels")
            if frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
                raise ValueError(f"image {name!r} must have shape (height, width, 3)")
            nbytes += frame.nbytes
        if nbytes > MAX_SNAPSHOT_BYTES:
            raise ValueError("observation exceeds the 32 MiB uncompressed payload limit")


def preflight_live_probe(
    rig: RigConfig,
    arm_names: Sequence[str] | None = None,
    *,
    expected_state_names: Sequence[str] | None = None,
) -> tuple[list[ArmSpec], tuple[str, ...]]:
    """Validate ALL arm calibration and physical dimensions before resolving/opening any bus.

    For two arms the given order has the same meaning as the existing bimanual plugin's
    ``left`` / ``right`` configuration. Default order follows rig.pairs, just like CLI rollout.
    Calibration endpoints are [closed, open]; their numeric order can legitimately be reversed.
    """
    selected = list(arm_names) if arm_names is not None else [p.follower for p in rig.pairs]
    if not 1 <= len(selected) <= 2 or len(set(selected)) != len(selected):
        raise ValueError("select one follower or an ordered left/right pair of distinct followers")
    specs = [rig.arm(name) for name in selected]
    names: list[str] = []
    errors: list[str] = []
    for i, spec in enumerate(specs):
        if spec.role != "follower":
            errors.append(f"{spec.name}: action probes require a follower")
        if spec.has_motor_gripper:
            try:
                raw_limits = spec.gripper_limits
                valid = (
                    isinstance(raw_limits, (tuple, list)) and len(raw_limits) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in raw_limits)
                    and abs(float(raw_limits[1] - raw_limits[0])) > 1e-6
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                errors.append(f"{spec.name}: valid saved [closed, open] gripper_limits required; auto-calibration is forbidden during probes")
        prefix = ("left_" if i == 0 else "right_") if len(specs) == 2 else ""
        names.extend(f"{prefix}joint_{j}.pos" for j in range(1, N_JOINTS + 1))
        if spec.has_motor_gripper:
            names.append(f"{prefix}gripper.pos")
    if expected_state_names is not None and tuple(names) != tuple(expected_state_names):
        errors.append("physical state mapping differs from the selected profile; no truncation, padding, or reordering is permitted")
    if errors:
        raise ValueError("; ".join(errors))
    return specs, tuple(names)


@contextmanager
def _preview_ownership(camera_hub: Any | None):
    """Use the existing UI hub when called in-process; CLI children use SessionManager's hook."""
    if camera_hub is None:
        yield
        return
    if camera_hub.suspended_by is not None:
        raise RuntimeError(f"cameras already owned by {camera_hub.suspended_by}")
    owner = "policy-probe-live"
    camera_hub.suspend(owner)
    try:
        if any(camera.running for camera in camera_hub.cams.values()):
            raise RuntimeError("camera preview did not release its device; active read was not started")
        yield
    finally:
        if camera_hub.suspended_by == owner:
            camera_hub.resume()


def _make_cameras(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from lerobot.cameras import make_cameras_from_configs

    from .cameras import camera_configs_from_dicts

    return make_cameras_from_configs(camera_configs_from_dicts(configs))


def _close_camera(camera: Any) -> None:
    # A camera can fail partway through connect. LeRobot disconnect releases partial resources.
    try:
        camera.disconnect()
    except Exception:
        log.exception("camera cleanup failed during active-read probe")


def capture_live_observation(
    rig: RigConfig,
    arm_names: Sequence[str] | None = None,
    *,
    approved: bool = False,
    expected_state_names: Sequence[str] | None = None,
    camera_hub: Any | None = None,
) -> ProbeObservation:
    """Capture one explicitly approved gravity-compensation read, then close before inference.

    Caller must prepare/validate its predictor first and hold the existing session ownership for
    the entire operation. UI subprocess callers suspend previews through CAMERA_MODES; direct
    in-process UI callers supply their existing CameraHub. No separate camera/driver framework
    is created here. No policy target, move_to or home command is issued.
    """
    if approved is not True:
        raise PermissionError(f"explicit operator approval required for {ACTIVE_READ_LABEL}")
    specs, names = preflight_live_probe(rig, arm_names, expected_state_names=expected_state_names)
    from .arm import YamArm, resolve_channel

    # Resolve every bus before activation too (lookup only, no SDK construction).
    channels = [resolve_channel(spec) for spec in specs]
    log.warning("%s: motors are energised; arms can move. No policy positions or homing.", ACTIVE_READ_LABEL)
    with _preview_ownership(camera_hub), ExitStack() as cleanup:
        cameras = _make_cameras(rig.cameras)
        # Open cameras before arms: a busy/invalid capture device cannot trigger motor activation.
        for camera in cameras.values():
            cleanup.callback(_close_camera, camera)
            camera.connect()
            camera.async_read(timeout_ms=2000)
        arms = []
        for spec, channel in zip(specs, channels, strict=True):
            arm = YamArm.connect(
                spec, channel, zero_gravity=True,
                max_joint_speed=rig.control.max_joint_speed,
                max_gripper_speed=rig.control.max_gripper_speed,
            )
            cleanup.callback(arm.close)
            arms.append(arm)
        captured_at, captured_monotonic = time.time(), time.monotonic()
        values = []
        for arm, spec in zip(arms, specs, strict=True):
            state = arm.read()
            q = np.asarray(state.q, dtype=float)
            if q.shape != (N_JOINTS,):
                raise ValueError(f"{spec.name}: expected exactly {N_JOINTS} observed joint positions")
            values.extend(q)
            if spec.has_motor_gripper:
                if state.gripper is None:
                    raise ValueError(f"{spec.name}: missing observed gripper position")
                values.append(state.gripper)
        images = {name: camera.read_latest(max_age_ms=CAMERA_MAX_AGE_MS).copy() for name, camera in cameras.items()}
        obs = ProbeObservation(
            np.asarray(values, dtype=float), names, images,
            source="live:" + ",".join(spec.name for spec in specs),
            captured_at=captured_at, captured_monotonic=captured_monotonic, mode="live",
            camera_max_age_s=CAMERA_MAX_AGE_MS / 1000 if cameras else 0.0,
            metadata={"arm_names": [spec.name for spec in specs]},
        )
        obs.validate()
    return obs


def save_observation(path: str | Path, observation: ProbeObservation) -> Path:
    """Portable NPZ snapshot with JSON metadata and no executable/pickled objects."""
    observation.validate()
    path = Path(path)
    names = list(observation.images)
    metadata = {
        "version": 1, "state_names": list(observation.state_names), "image_names": names,
        "source": observation.source, "captured_at": observation.captured_at,
        "camera_max_age_s": observation.camera_max_age_s,
    }
    arrays = {f"image_{i}": observation.images[name] for i, name in enumerate(names)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as out:
        np.savez_compressed(out, state=observation.state, metadata=np.asarray(json.dumps(metadata)), **arrays)
    return path


def load_saved_observation(path: str | Path) -> ProbeObservation:
    """Read a bounded snapshot without accessing arms or cameras; keep capture source and age."""
    path = Path(path)
    if path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise ValueError("saved observation exceeds 32 MiB")
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_SNAPSHOT_BYTES:
            raise ValueError("uncompressed saved observation exceeds 32 MiB")
    with np.load(path, allow_pickle=False) as values:
        metadata = json.loads(str(values["metadata"].item()))
        if metadata.get("version") != 1:
            raise ValueError("unsupported saved observation version")
        image_names = metadata["image_names"]
        if len(image_names) > MAX_CAMERAS or len(set(image_names)) != len(image_names):
            raise ValueError("saved observation has too many or duplicate image names")
        obs = ProbeObservation(
            state=values["state"].copy(), state_names=tuple(metadata["state_names"]),
            images={name: values[f"image_{i}"].copy() for i, name in enumerate(image_names)},
            source=f"{path} (captured from {metadata.get('source', 'unknown')})",
            captured_at=metadata.get("captured_at"), mode="saved",
            camera_max_age_s=float(metadata.get("camera_max_age_s", 0)),
        )
    obs.validate()
    return obs


def _numbers(values: Any) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def _bounded_chunk(raw: Any) -> np.ndarray:
    """Check shape/count before a conversion or a GPU-to-CPU copy can allocate large arrays."""
    if hasattr(raw, "shape"):
        shape = tuple(raw.shape)
        if len(shape) > 3 or math.prod(shape) > MAX_ACTION_VALUES:
            raise ValueError("predicted action chunk exceeds diagnostic bounds")
        if isinstance(raw, np.ndarray) and raw.dtype.kind not in "fiu":
            raise ValueError("predicted action chunk must contain real numerical arrays")
        if hasattr(raw, "detach"):
            raw = raw.detach().cpu().numpy()
    else:
        count = 0

        def check(value: Any, depth: int = 0) -> None:
            nonlocal count
            if isinstance(value, (list, tuple)):
                if depth >= 3 or len(value) > MAX_ACTION_VALUES:
                    raise ValueError("predicted action chunk exceeds diagnostic bounds")
                for child in value:
                    check(child, depth + 1)
            else:
                count += 1
                if count > MAX_ACTION_VALUES or not isinstance(value, (int, float, np.integer, np.floating)):
                    raise ValueError("predicted action chunk must contain bounded numerical arrays")

        check(raw)
    return np.asarray(raw, dtype=float)


def probe_observation(
    observation: ProbeObservation,
    predict: Callable[[ProbeObservation], Any],
    *,
    action_names: Sequence[str],
    profile_id: str,
    revision: str,
    transforms: Sequence[str] = (),
    mapping_validated: bool = False,
    max_age_s: float = 1.0,
    max_joint_delta_rad: float = 0.5,
    expected_state_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Inspect exactly one fresh predicted chunk in robot units, before ANY clipping.

    ``predict`` must use the same saved processing/profile as rollout and return a fresh
    postprocessed robot-unit chunk (T,D or 1,T,D). The report contains summaries, never an
    executable chunk. Even a clean report grants no motion approval and cannot be replayed.
    Plausibility thresholds are diagnostics, not physical joint limits.
    """
    observation.validate()
    if not 1 <= len(action_names) <= 64 or not all(isinstance(n, str) and 1 <= len(n) <= 128 for n in action_names):
        raise ValueError("action names must contain 1–64 bounded strings")
    if len(set(action_names)) != len(action_names):
        raise ValueError("action names must be unique")
    if expected_state_names is not None and observation.state_names != tuple(expected_state_names):
        raise ValueError("saved/live state names differ from profile mapping; refusing to infer")
    if not math.isfinite(max_age_s) or max_age_s <= 0 or not math.isfinite(max_joint_delta_rad) or max_joint_delta_rad <= 0:
        raise ValueError("freshness and plausibility thresholds must be finite and positive")
    started = time.monotonic()
    age_before = observation.age_s()
    raw = predict(observation)
    chunk = _bounded_chunk(raw)
    raw_shape = list(chunk.shape)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    age_after = observation.age_s()
    issues: list[str] = []
    if not mapping_validated:
        issues.append("physical_mapping_unvalidated")
    if age_before is None:
        issues.append("observation_age_unknown")
    elif age_before < 0:
        issues.append("observation_timestamp_in_future")
    elif age_after is not None and age_after > max_age_s:
        issues.append("observation_stale")
    names = tuple(action_names)
    report: dict[str, Any] = {
        "mode": observation.mode,
        "activation": ACTIVE_READ_LABEL if observation.mode == "live" else SAVED_LABEL,
        "source": observation.source, "captured_at": observation.captured_at,
        "age_basis": "local monotonic" if observation.captured_monotonic is not None else "saved wall-clock estimate",
        "observation_age_before_s": age_before, "observation_age_after_s": age_after,
        "profile_id": profile_id, "revision": revision, "transforms": list(transforms),
        "state_names": list(observation.state_names), "state": _numbers(observation.state),
        "action_names": list(names), "raw_action_shape": raw_shape,
        "units": ["normalized gripper (0 closed, 1 open)" if n.endswith("gripper.pos") else "rad" for n in names],
        "clipped": False, "mapping_validated": mapping_validated,
        "inference_ms": (time.monotonic() - started) * 1000,
        "issues": issues, "motion_approved": False, "replay_permitted": False,
        "next_step": "Motion requires separate approval and fresh observations; this probe never executes its chunk.",
    }
    if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != len(names):
        issues.append("malformed_action_shape")
    else:
        finite = np.isfinite(chunk)
        if not finite.all():
            issues.append("nonfinite_actions")
        report["chunk_steps"] = chunk.shape[0]
        report["first_targets"] = _numbers(chunk[0])
        report["chunk_min"] = _numbers(np.min(chunk, axis=0))
        report["chunk_max"] = _numbers(np.max(chunk, axis=0))
        report["grippers"] = {
            name: {"first_target": _numbers([chunk[0, i]])[0], "min": report["chunk_min"][i], "max": report["chunk_max"][i]}
            for i, name in enumerate(names) if name.endswith("gripper.pos")
        }
        if observation.state_names != names:
            issues.append("state_action_mapping_mismatch")
        else:
            deltas = chunk - observation.state[None, :]
            report["first_signed_deltas"] = _numbers(deltas[0])
            for i, name in enumerate(names):
                if name.endswith("gripper.pos"):
                    report["grippers"][name]["state"] = float(observation.state[i])
                    if not 0 <= observation.state[i] <= 1:
                        issues.append(f"gripper_state_out_of_range:{name}")
                    if ((chunk[:, i] < 0) | (chunk[:, i] > 1)).any():
                        issues.append(f"gripper_target_out_of_range:{name}")
                elif np.any(np.abs(deltas[:, i]) > max_joint_delta_rad):
                    issues.append(f"large_joint_delta:{name}")
        report["finite"] = bool(finite.all())
    report["passed_diagnostics"] = not issues
    return report


def format_probe_report(report: dict[str, Any]) -> str:
    """CLI-readable JSON retains signed numbers and all per-joint extrema for UI/log reuse."""
    return json.dumps(report, indent=2, allow_nan=False)
