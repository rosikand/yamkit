"""Explicit, non-learned actuator boundary for the frozen OpenPI YAM experiment.

This is NOT an upstream ALOHA adapter or a claim of pretrained YAM support.
Native model arrays and normalization are untouched. A finite requested gripper
opening is interpreted as a bounded physical opening: requests beyond either
mechanical endpoint request that endpoint. Every such conversion is returned,
including unused model rows. There is no fitted extrapolation percentage and
there is no projection of arm joints. NaNs, infinities and malformed arrays are
always errors. Callers must separately check actual robot joint bounds.
"""

from __future__ import annotations

import math

import numpy as np

from yamkit.inference.mapping import YAM_NAMES

CONTRACT_ID = "openpi_frozen_yam_endpoint_fifo_v1"
MODEL_ROWS = 50
COMMITTED_ROWS = 25
NOMINAL_HZ = 50.0
# Mirror the existing YAM ordinary-command cap; do not import a hardware module.
MAX_COMMAND_DT = 0.01
STALE_COMMAND_S = 0.5
GRIPPERS = (6, 13)
ACTION_TRANSFORM = {
    "id": "explicit_yam_mechanical_opening_endpoints_v1",
    "columns": list(GRIPPERS),
    "input": "finite unconstrained requested normalized opening",
    "executed_range": [0.0, 1.0],
    "rule": "below closed -> closed; above open -> open; interior unchanged",
    "raw_anomaly_percentage": None,
    "native_openpi_transform": False,
    "all_conversions_logged": True,
    "joint_projection": False,
}
EXECUTION_CONTRACT = {
    "id": CONTRACT_ID,
    "predicted_rows": MODEL_ROWS,
    "committed_prefix_rows": COMMITTED_ROWS,
    "nominal_max_row_hz": NOMINAL_HZ,
    "ordinary_command_max_dt_s": MAX_COMMAND_DT,
    "stale_command_reset_s": STALE_COMMAND_S,
    "replan": "only after all 25 committed endpoints complete",
    "unused_rows": "rows 25..49 intentionally unused, as in the native ALOHA client",
    "transition": "synchronized linear substeps only when ordinary YAM speed caps require",
    "timing": "no catch-up; explicit row time dilation preserves every committed endpoint",
    "model_action_origin": "one measured observation at each inference request",
    "initial_transition_origin": "fresh measured state after RPC if the prior command is stale",
    "sdk_speed_limit_enabled": True,
    "action_during_rpc": False,
    "hardware_tested": False,
}


class OpenPiExecutionFault(ValueError):
    """The experimental interface cannot safely execute the supplied data."""


def finite_array(value, shape, label: str) -> np.ndarray:
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise OpenPiExecutionFault(f"{label} requires finite real values with exact shape {shape}")
    return result.astype(np.float64, copy=True)


def executable_state(value) -> np.ndarray:
    state = finite_array(value, (14,), "Measured YAM state")
    if np.any(state[list(GRIPPERS)] < 0) or np.any(state[list(GRIPPERS)] > 1):
        raise OpenPiExecutionFault("Measured gripper state is outside its calibrated [0,1] range")
    return state


def prepare_chunk(value) -> dict:
    """Return all 50 raw/actuator rows and an exact, explicit conversion ledger.

Joint bounds are not available here. The executor prevalidates all 25 committed
rows against the actual rig before sending any point. Unused rows retain their
raw joint values for audit; they are never made into physical commands.
    """
    raw = finite_array(value, (MODEL_ROWS, 14), "OpenPI absolute action chunk")
    rows = raw.copy()
    rows[:, list(GRIPPERS)] = np.clip(raw[:, list(GRIPPERS)], 0.0, 1.0)
    conversions = [
        {"row_index": int(row), "column_index": int(column), "name": YAM_NAMES[column],
         "raw": float(raw[row, column]), "executed": float(rows[row, column]),
         "delta": float(rows[row, column] - raw[row, column]),
         "committed_prefix": bool(row < COMMITTED_ROWS)}
        for row, column in zip(*np.nonzero(raw != rows), strict=True)
    ]
    return {"raw_rows": raw, "rows": rows, "gripper_conversions": conversions,
            "action_transform": dict(ACTION_TRANSFORM), "joint_projection": False}


def speed_vector(max_joint_speed: float, max_gripper_speed: float) -> np.ndarray:
    for value in (max_joint_speed, max_gripper_speed):
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 < value <= 3.0:
            raise ValueError("OpenPI YAM speeds must be finite, positive and no greater than 3.0")
    result = np.full(14, max_joint_speed, dtype=np.float64)
    result[list(GRIPPERS)] = max_gripper_speed
    return result


def linear_transition(origin, target, speeds) -> np.ndarray:
    """Reach the exact endpoint with bounded synchronized 14D linear substeps.

The .01-second cap is independent of observed output magnitudes. Subdivision
does not drop, average or shorten a policy row. Final values equal the target
exactly. A row requiring subdivision takes longer, and the executor logs that
time dilation. This is an explicit actuator-interface deviation from direct
ALOHA dispatch, not a change to the model's output or native flow sampling.
    """
    origin = executable_state(origin)
    target = executable_state(target)
    speeds = finite_array(speeds, (14,), "Actuator speeds")
    if np.any(speeds <= 0) or np.any(speeds > 3):
        raise OpenPiExecutionFault("Actuator speeds must be in (0,3]")
    delta = target - origin
    caps = speeds * MAX_COMMAND_DT
    ratio = float(np.max(np.abs(delta) / caps))
    if not math.isfinite(ratio) or ratio > 9000:
        raise OpenPiExecutionFault("A single transition exceeds the bounded 90-second execution budget")
    count = max(1, math.ceil(ratio))
    # Floating-point division can round an exact-boundary ratio down. Check the
    # actual segments rather than permitting an epsilon-sized cap overrun.
    while True:
        points = origin + (np.arange(1, count + 1, dtype=np.float64) / count)[:, None] * delta
        points[-1] = target
        if np.all(np.abs(np.diff(np.vstack((origin, points)), axis=0)) <= caps):
            return points
        count += 1
