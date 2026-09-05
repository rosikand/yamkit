"""Offline deployment gate and measurement summaries; neither authorizes motion."""

from __future__ import annotations

import math
from collections.abc import Iterable


def percentile_summary(values: Iterable[float]) -> dict:
    """Linear sample quantiles with counts, including empty/incomplete measurements."""
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("Performance samples must be finite")
    if not ordered:
        return {"sample_count": 0}

    def quantile(fraction):
        position = fraction * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return {"sample_count": len(ordered), "min": ordered[0], "max": ordered[-1],
            **{f"p{percent}": quantile(percent / 100) for percent in (50, 95, 99)}}


def summarize_measurements(samples: list[dict], *, minimum_warm_samples: int = 100) -> dict:
    """Keep first-request cost separate; never infer a cold start from its position.

    Missing SDK internals remain missing, not zero. Nested stage names are retained
    so the report can include each camera and saved-processor step independently.
    """
    if minimum_warm_samples < 1:
        raise ValueError("minimum_warm_samples must be positive")
    warm = samples[1:]
    stages: dict[str, list[float]] = {}

    def collect(value, prefix=""):
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                collect(item, name)
            elif ((key.endswith(("_s", "_bytes")) or prefix.endswith("_s")) and type(item) in (int, float)
                  and "timestamp" not in key and "monotonic" not in key):
                # Monotonic timestamps belong in raw records, not latency percentiles.
                stages.setdefault(name, []).append(item)

    for sample in warm:
        collect(sample)
    instances = {sample["instance_id"] for sample in samples if sample.get("instance_id")}
    return {"sample_count": len(samples), "warm_sample_count": len(warm),
            "minimum_warm_samples": minimum_warm_samples,
            "warm_sample_requirement_met": len(warm) >= minimum_warm_samples,
            "first_request": samples[0] if samples else None,
            "warm_round_trip_s": percentile_summary(sample["round_trip_s"] for sample in warm),
            "warm_stages": {name: percentile_summary(values) for name, values in sorted(stages.items())},
            "container_instance_count": len(instances),
            "cold_start_note": "First observed request is separated, not assumed cold; inspect readiness and lifecycle."}

PHYSICAL_MODAL_ROLLOUT_REASON = (
    "Physical Modal rollout BLOCKED: integrated real-service queue performance is unvalidated. "
    "Recorded Molmo warm RPC (~1.48 s) exceeds its 1 s action horizon; "
    "saved-observation inference and fake-robot diagnostics remain available."
)
QUALIFICATION_GATE_ENABLED = True
PHYSICAL_MODAL_QUALIFICATION_REASON = (
    "Physical Modal rollout BLOCKED until this robot host has a matching qualification from the last 24 hours, "
    "with accepted mapping and explicit supervised confirmation. Cloud measurements cannot qualify the robot host."
)


def physical_modal_status() -> dict:
    # An offline catalog has neither host evidence nor operator confirmation.
    # Only the guarded rollout entry points can validate those requirements.
    return {"physical_modal_rollout_allowed": False,
            "physical_modal_rollout_reason": PHYSICAL_MODAL_QUALIFICATION_REASON
            if QUALIFICATION_GATE_ENABLED else PHYSICAL_MODAL_ROLLOUT_REASON,
            "qualification_required": QUALIFICATION_GATE_ENABLED}


def require_physical_modal_rollout(settings=None, *, supervised_confirmed=False, mapping_accepted=False) -> None:
    if not QUALIFICATION_GATE_ENABLED:
        raise ValueError(PHYSICAL_MODAL_ROLLOUT_REASON)
    from .qualification import QualificationError, is_cloud_host, validate_qualification

    if is_cloud_host():
        raise QualificationError("Physical Modal rollout BLOCKED in cloud workspaces; qualify on the Lenovo robot host")
    if supervised_confirmed is not True or mapping_accepted is not True:
        raise QualificationError("Physical Modal rollout BLOCKED: explicit supervised confirmation and mapping acceptance required")
    if callable(settings):
        settings = settings()
    if not isinstance(settings, dict):
        raise QualificationError("Physical Modal rollout BLOCKED: current host, model and image settings required")
    validate_qualification(settings)
