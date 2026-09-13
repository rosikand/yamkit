"""Pure offline arithmetic for an UNQUALIFIED frozen-π0.5 YAM hypothesis.

No model, robot, file, socket, service or qualification entrypoints live here.
The declared 14D joint-delta/closure interpretation is an engineering hypothesis,
not an official pretrained YAM contract. Numerical success never enables motion.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from yamkit.inference.mapping import YAM_NAMES

CANDIDATE_ID = "experimental_yam_chunk_delta_closure_quantiles_v1"
HORIZON = 50
MODEL_DIM = 32
YAM_DIM = 14
GRIPPERS = (6, 13)
JOINTS = tuple(index for index in range(YAM_DIM) if index not in GRIPPERS)
NATIVE_EPSILON = 1e-6
# Explicit numerical audit choice, not a training statistic or physical safety limit.
MIN_QUANTILE_SPAN = 100 * NATIVE_EPSILON
MAX_WINDOWS = 20_000


def _array(value, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.array(value, copy=True)
    if result.shape != shape or result.dtype.kind not in "fiu" or not np.isfinite(result).all():
        raise ValueError(f"{label} requires finite real values with exact shape {shape}")
    return result


def _gripper_range(values: np.ndarray, label: str) -> None:
    grips = values[..., list(GRIPPERS)]
    if np.any((grips < 0) | (grips > 1)):
        raise ValueError(f"{label} grippers must be in [0, 1]; no clipping is permitted")


def _finite_math(values: np.ndarray) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("Candidate arithmetic produced nonfinite values")
    return values


def candidate_state(measured_yam_state) -> np.ndarray:
    """YAM SDK radians unchanged; each measured opening becomes declared closure 1-g."""
    result = _array(measured_yam_state, (YAM_DIM,), "Measured YAM state").astype(np.float64)
    _gripper_range(result, "Measured YAM state")
    result[list(GRIPPERS)] = 1 - result[list(GRIPPERS)]
    return _finite_math(result)


def candidate_action_chunk(measured_yam_state, future_absolute_commands) -> np.ndarray:
    """Every row is relative to ONE measured chunk anchor; grippers remain absolute."""
    state = candidate_state(measured_yam_state)
    result = _array(future_absolute_commands, (HORIZON, YAM_DIM), "YAM command chunk").astype(np.float64)
    _gripper_range(result, "YAM command chunk")
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, list(JOINTS)] -= state[list(JOINTS)]
        result[:, list(GRIPPERS)] = 1 - result[:, list(GRIPPERS)]
    return _finite_math(result)


def _reconstruct_unchecked(state: np.ndarray, transformed: np.ndarray) -> np.ndarray:
    result = transformed.astype(np.float64, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        result[:, list(JOINTS)] += state[list(JOINTS)]
        result[:, list(GRIPPERS)] = 1 - result[:, list(GRIPPERS)]
    return _finite_math(result)


def reconstruct_yam_chunk(measured_yam_state, joint_deltas_absolute_closure) -> np.ndarray:
    """Offline inverse, rejecting invalid closure rather than projecting an endpoint."""
    state = candidate_state(measured_yam_state)
    transformed = _array(joint_deltas_absolute_closure, (HORIZON, YAM_DIM), "Candidate action chunk")
    _gripper_range(transformed, "Candidate action chunk")
    return _reconstruct_unchecked(state, transformed)


@dataclass(frozen=True)
class CandidateQuantiles:
    """Explicit 14D deployment quantiles, never an upstream or learned YAM asset."""

    q01: Sequence[float]
    q99: Sequence[float]
    provenance: str
    corpus_sha256: str

    def __post_init__(self):
        low = _array(self.q01, (YAM_DIM,), "Candidate q01").astype(np.float64)
        high = _array(self.q99, (YAM_DIM,), "Candidate q99").astype(np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            spans = high - low
        if not np.isfinite(spans).all() or np.any(spans < MIN_QUANTILE_SPAN):
            raise ValueError("Candidate quantile spans must be finite and at least the declared 1e-4 floor")
        if not isinstance(self.provenance, str) or not self.provenance.strip() or len(self.provenance) > 2048:
            raise ValueError("Candidate quantiles require explicit bounded provenance")
        if not isinstance(self.corpus_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.corpus_sha256):
            raise ValueError("Candidate quantiles require a lowercase SHA-256 corpus identity")
        object.__setattr__(self, "q01", tuple(low.tolist()))
        object.__setattr__(self, "q99", tuple(high.tolist()))

    def as_dict(self) -> dict:
        return {"q01": list(self.q01), "q99": list(self.q99), "provenance": self.provenance,
                "corpus_sha256": self.corpus_sha256}


@dataclass(frozen=True)
class CandidateStatistics:
    state: CandidateQuantiles
    actions: CandidateQuantiles

    def __post_init__(self):
        if not isinstance(self.state, CandidateQuantiles) or not isinstance(self.actions, CandidateQuantiles):
            raise TypeError("Explicit state and action CandidateQuantiles are required")
        if self.state.corpus_sha256 != self.actions.corpus_sha256:
            raise ValueError("State and action statistics must identify the same paired corpus")

    def metadata(self) -> dict:
        data = {"candidate_id": CANDIDATE_ID, "state": self.state.as_dict(), "actions": self.actions.as_dict()}
        digest = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()
        return {**data, "candidate_statistics_sha256": digest, "channel_names": list(YAM_NAMES),
                "status": "UNQUALIFIED_ENGINEERING_HYPOTHESIS", "qualified_for_yam": False,
                "physical_ready": False, "hardware_tested": False, "model_rows_sent_to_robot": 0,
                "minimum_quantile_span": MIN_QUANTILE_SPAN, "native_epsilon": NATIVE_EPSILON,
                "joint_semantics": "unchanged YAM measured radians; every delta anchored to initial measured state",
                "gripper_semantics": "declared absolute closure 1-opening; not the Trossen linkage conversion",
                "unused_dimensions_assumption": "columns 14..31 excluded by this candidate, not proven YAM padding",
                "execution_frequency_established": False, "committed_horizon_established": False}


def _quantile_input(values, quantiles: CandidateQuantiles) -> np.ndarray:
    if not isinstance(quantiles, CandidateQuantiles):
        raise TypeError("Explicit candidate quantiles with provenance are required")
    result = np.asarray(values)
    if result.ndim not in (1, 2) or result.shape[-1] != YAM_DIM or not result.size:
        raise ValueError("Quantile input must be a nonempty 14D vector or row matrix")
    # Preserve the native arithmetic staging. In Unnormalize, (x + 1) / 2
    # runs in the model array's dtype BEFORE multiplying the float64 q span.
    # Upcasting a float32 model result first changes those rounded intermediates.
    return _array(result, result.shape, "Quantile input")


def normalize_14(values, quantiles: CandidateQuantiles) -> np.ndarray:
    """Match pinned native quantile arithmetic including epsilon; do not clip."""
    values = _quantile_input(values, quantiles)
    low, high = np.asarray(quantiles.q01), np.asarray(quantiles.q99)
    with np.errstate(over="ignore", invalid="ignore"):
        result = (values - low) / (high - low + NATIVE_EPSILON) * 2.0 - 1.0
    return _finite_math(result)


def unnormalize_14(values, quantiles: CandidateQuantiles) -> np.ndarray:
    values = _quantile_input(values, quantiles)
    low, high = np.asarray(quantiles.q01), np.asarray(quantiles.q99)
    with np.errstate(over="ignore", invalid="ignore"):
        result = (values + 1.0) / 2.0 * (high - low + NATIVE_EPSILON) + low
    return _finite_math(result)


def encode_state(measured_yam_state, statistics: CandidateStatistics) -> np.ndarray:
    """Return 14 normalized values: native tokenization must precede padding to 32."""
    if not isinstance(statistics, CandidateStatistics):
        raise TypeError("Explicit candidate state and action statistics are required")
    return normalize_14(candidate_state(measured_yam_state), statistics.state)


def decode_chunk(raw_normalized_model_chunk, measured_yam_state, statistics: CandidateStatistics) -> dict:
    """Preserve every model value and expose proposed arithmetic, NEVER dispatch.

    Out-of-range proposed grippers remain visible with a failed range audit, not
    silently clipped, discarded, or treated as qualifying physical commands.
    """
    if not isinstance(statistics, CandidateStatistics):
        raise TypeError("Explicit candidate state and action statistics are required")
    raw = _array(raw_normalized_model_chunk, (HORIZON, MODEL_DIM), "Raw normalized model chunk")
    state = candidate_state(measured_yam_state)
    transformed = unnormalize_14(raw[:, :YAM_DIM], statistics.actions)
    proposed = _reconstruct_unchecked(state, transformed)
    invalid_grippers = (proposed[:, list(GRIPPERS)] < 0) | (proposed[:, list(GRIPPERS)] > 1)
    return {**statistics.metadata(), "raw_normalized_model_chunk": raw,
            "measured_yam_state": np.array(measured_yam_state, copy=True), "candidate_state": state,
            "unused_normalized_model_dimensions": raw[:, YAM_DIM:].copy(),
            "unnormalized_joint_deltas_absolute_closure": transformed,
            "proposed_yam_absolute_commands": proposed,
            "proposed_gripper_range_passes": not bool(invalid_grippers.any()),
            "out_of_range_gripper_scalars": int(invalid_grippers.sum()),
            "joint_bounds_checked": False, "clipping_applied": False}


@dataclass(frozen=True)
class PairedEpisode:
    """Already-saved aligned measurements/commands; each row's command follows its state.

    ``sample_hz`` declares the recording timebase, not a validated execution rate.
    The caller must verify actual timestamps/paired-source provenance separately.
    No state-only convenience constructor or inferred action trajectory is supplied.
    """

    episode_id: str
    measured_states: np.ndarray
    absolute_commands: np.ndarray
    channel_names: tuple[str, ...]
    sample_hz: float

    def __post_init__(self):
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("Paired episode requires an explicit episode identity")
        if tuple(self.channel_names) != YAM_NAMES:
            raise ValueError("Paired episode channel names must match the exact YAM order")
        if (isinstance(self.sample_hz, bool)
                or not isinstance(self.sample_hz, (int, float, np.integer, np.floating))
                or not np.isfinite(self.sample_hz) or self.sample_hz <= 0):
            raise ValueError("Paired episode requires a finite positive declared sample rate")
        rows = np.asarray(self.measured_states).shape
        if len(rows) != 2 or rows[1] != YAM_DIM or rows[0] < HORIZON:
            raise ValueError("Paired episode must contain at least one complete 50-row window")
        for field in ("measured_states", "absolute_commands"):
            array = _array(getattr(self, field), rows, field).astype(np.float64)
            _finite_math(array)
            _gripper_range(array, field)
            array.flags.writeable = False
            object.__setattr__(self, field, array)
        object.__setattr__(self, "channel_names", YAM_NAMES)


def _paired_windows(episodes: Sequence[PairedEpisode]) -> tuple[np.ndarray, np.ndarray]:
    if not episodes or any(not isinstance(episode, PairedEpisode) for episode in episodes):
        raise ValueError("Explicit paired episodes are required; state-only fixtures cannot supply action statistics")
    if len({episode.episode_id for episode in episodes}) != len(episodes):
        raise ValueError("Paired episode identities must be unique")
    if len({episode.sample_hz for episode in episodes}) != 1:
        raise ValueError("Mixed episode timebases cannot define one candidate action distribution")
    count = sum(len(episode.measured_states) - HORIZON + 1 for episode in episodes)
    if count > MAX_WINDOWS:
        raise ValueError("Candidate corpus exceeds the bounded 20000 complete-window audit")
    states, chunks = [], []
    for episode in episodes:
        for start in range(len(episode.measured_states) - HORIZON + 1):
            state = episode.measured_states[start]
            states.append(candidate_state(state))
            chunks.append(candidate_action_chunk(state, episode.absolute_commands[start:start + HORIZON]))
    return np.stack(states), np.concatenate(chunks)


def estimate_statistics(episodes: Sequence[PairedEpisode], *, provenance: str,
                        corpus_sha256: str) -> CandidateStatistics:
    """Experimental exact NumPy quantiles, NOT upstream histogram estimates/training assets."""
    # Validate provenance before corpus work; never turn None into a plausible string.
    CandidateQuantiles(np.zeros(YAM_DIM), np.ones(YAM_DIM), provenance, corpus_sha256)
    states, actions = _paired_windows(episodes)

    def estimate(values, label):
        low, high = np.quantile(values, [0.01, 0.99], axis=0, method="linear")
        return CandidateQuantiles(low, high, f"{provenance}; {label}; recorded sample_hz={episodes[0].sample_hz}; "
                                  "NumPy linear 1%/99% quantiles",
                                  corpus_sha256)

    return CandidateStatistics(estimate(states, "complete-window measured anchors"),
                               estimate(actions, "all 50-row chunk-origin deltas and absolute closure"))


def audit_heldout(episodes: Sequence[PairedEpisode], statistics: CandidateStatistics) -> dict:
    """Support/round-trip diagnostics only; no automatic compatibility or readiness pass."""
    if not isinstance(statistics, CandidateStatistics):
        raise TypeError("Explicit candidate state and action statistics are required")
    states, actions = _paired_windows(episodes)

    def audit(values, quantiles):
        normalized = normalize_14(values, quantiles)
        restored = unnormalize_14(normalized, quantiles)
        low, high = np.asarray(quantiles.q01), np.asarray(quantiles.q99)
        return {"sample_count": len(values), "q99_minus_q01": (high - low).tolist(),
                "normalized_min": normalized.min(axis=0).tolist(),
                "normalized_max": normalized.max(axis=0).tolist(),
                "outside_normalized_unit_range_count": ((normalized < -1) | (normalized > 1)).sum(axis=0).tolist(),
                "outside_empirical_quantiles_count": ((values < low) | (values > high)).sum(axis=0).tolist(),
                "max_abs_round_trip_error": float(np.max(np.abs(values - restored)))}

    return {**statistics.metadata(), "declared_sample_hz": episodes[0].sample_hz,
            "timestamp_alignment_checked": False, "episode_ids": [episode.episode_id for episode in episodes],
            "complete_windows": len(states), "state_support": audit(states, statistics.state),
            "action_support": audit(actions, statistics.actions), "cross_episode_padding": False,
            "support_is_compatibility_proof": False}
