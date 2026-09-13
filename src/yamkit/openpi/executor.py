"""Synchronous OpenPI 25-of-50 FIFO with explicit YAM actuator adaptation.

Pure callbacks only: this module never constructs or imports hardware. The
caller owns acquisition, bounded RPC cancellation, fault release, healthy home
and post-release recording/upload. The native ALOHA client supplies the 50 Hz
maximum row rate and 25-row prefix. YAM-specific ordinary speed constraints may
insert declared linear substeps and dilate a row; no endpoint is silently lost.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from threading import Event

import numpy as np

from yamkit.inference.mapping import YAM_NAMES

from .interface import (
    COMMITTED_ROWS,
    CONTRACT_ID,
    EXECUTION_CONTRACT,
    MAX_COMMAND_DT,
    MODEL_ROWS,
    NOMINAL_HZ,
    STALE_COMMAND_S,
    OpenPiExecutionFault,
    executable_state,
    linear_transition,
    prepare_chunk,
    speed_vector,
)


def _summary(values: list[float]) -> dict:
    return {"sample_count": len(values),
            "p50": float(np.percentile(values, 50)) if values else None,
            "p95": float(np.percentile(values, 95)) if values else None,
            "max": max(values) if values else None}


def _target(values) -> dict:
    return dict(zip(YAM_NAMES, np.asarray(values).tolist(), strict=True))


class OpenPiYamExecutor:
    """Single-use engine. No automatic restart or retry is possible.

``predict(observation, timeout)`` returns a 50x14 array or
``{"chunk": array, "audit": {...}}``. Decoding must already have anchored every
joint delta to the measured state of that same request; the executor never
reanchors model outputs. ``observe`` returns ``{"state": measured14, ...}``.
``send(target, dispatch_check)`` MUST retain ordinary SDK speed limits, check
before each arm, and return the actual SDK receipt. Inexact receipts fault,
apart from explicitly counted floating-point roundoff (eight binary ulps).
    """

    def __init__(self, *, predict: Callable, observe: Callable, send: Callable,
                 validate_target: Callable, stop: Event, session_check: Callable = lambda: None,
                 clock: Callable = time.monotonic, wait: Callable | None = None,
                 event: Callable = lambda *_args, **_kwargs: None,
                 observe_policy_input: Callable | None = None,
                 max_joint_speed: float = 3.0, max_gripper_speed: float = 3.0,
                 rpc_timeout_s: float = 2.0):
        if (type(rpc_timeout_s) not in (int, float) or not math.isfinite(rpc_timeout_s)
                or not 0 < rpc_timeout_s <= 2):
            raise ValueError("OpenPI execution requires an RPC timeout in (0,2] seconds")
        self.predict, self.observe, self.send = predict, observe, send
        if observe_policy_input is not None and not callable(observe_policy_input):
            raise ValueError("Policy-input observation seam must be callable")
        self.observe_policy_input = observe if observe_policy_input is None else observe_policy_input
        self.validate_target, self.stop, self.session_check = validate_target, stop, session_check
        self.clock, self.wait, self.event = clock, wait or stop.wait, event
        self.speeds = speed_vector(max_joint_speed, max_gripper_speed)
        self.rpc_timeout_s = float(rpc_timeout_s)
        self.used = self.finished = self.stop_requested = False
        self.deadline = float("inf")
        self.started = self.ended = None
        self.last_sent = self.last_send_at = self.last_endpoint_at = None
        self.inference_calls = self.predicted_rows = self.admitted_rows = self.completed_rows = 0
        self.completed_chunks = self.attempted_points = self.completed_points = 0
        self.unknown_partial_dispatches = self.faults = self.invalid_chunks = 0
        self.observations = self.inserted_points = self.roundoff_receipts = 0
        self.policy_input_observations = 0
        self.maximum_receipt_roundoff = 0.0
        self.modified_commands = self.coherence_violations = 0
        self.projected_gripper_values = self.executed_projected_gripper_values = 0
        self.maximum_gripper_projection = 0.0
        self.stopped_rpc_exceptions = self.phase_tail_requests_avoided = 0
        self.late_response_rows_discarded = 0
        self.rpc_latencies, self.observation_ages, self.endpoint_times = [], [], []
        self.chunks, self.row_records = [], []

    def check(self) -> bool:
        if self.stop.is_set():
            self.stop_requested = True
            return False
        self.session_check()
        return self.clock() < self.deadline

    def dispatch_check(self) -> None:
        if not self.check():
            raise OpenPiExecutionFault("OpenPI dispatch cancelled by Stop or the bounded deadline")

    def _wait_until(self, until: float) -> bool:
        while self.check():
            remaining = min(until, self.deadline) - self.clock()
            if remaining <= 0:
                return self.check()
            self.wait(min(remaining, MAX_COMMAND_DT))
        return False

    def _observe(self, *, policy_input=False):
        if not self.check():
            return None, None
        observation = (self.observe_policy_input if policy_input else self.observe)()
        self.observations += 1
        self.policy_input_observations += int(policy_input)
        if not isinstance(observation, Mapping) or "state" not in observation:
            raise OpenPiExecutionFault("OpenPI observation requires explicit measured state")
        state = executable_state(observation["state"])
        self.validate_target(_target(state))
        return observation, state

    def _receipt(self, requested, sent) -> np.ndarray:
        if (not isinstance(sent, Mapping) or set(sent) != set(YAM_NAMES)
                or any(type(sent[name]) not in (int, float) for name in YAM_NAMES)):
            raise OpenPiExecutionFault("OpenPI SDK receipt is missing exact named real scalar targets")
        actual = executable_state([sent[name] for name in YAM_NAMES])
        error = np.abs(actual - requested)
        tolerance = 8 * np.spacing(np.maximum(1.0, np.maximum(np.abs(actual), np.abs(requested))))
        if np.any(error > tolerance):
            self.modified_commands += 1
            self.coherence_violations += 1
            raise OpenPiExecutionFault("Ordinary SDK altered an OpenPI planned point; endpoint delivery is not exact")
        if np.any(error):
            self.roundoff_receipts += 1
            self.maximum_receipt_roundoff = max(self.maximum_receipt_roundoff, float(np.max(error)))
        return actual

    def _execute_row(self, row, raw_row, *, chunk_index, row_index, observed_at, conversions) -> bool:
        row_started = self.clock()
        _, measured = self._observe()
        if measured is None or not self.check():
            return False
        measured_at = self.clock()
        stale = self.last_send_at is None or self.clock() - self.last_send_at > STALE_COMMAND_S
        origin = measured if stale else self.last_sent
        points = linear_transition(origin, row, self.speeds)
        # Validate every point in this row before its first send; all endpoints
        # were already checked before any command in the entire prefix.
        for point in points:
            self.validate_target(_target(point))
        record = {"chunk_index": chunk_index, "row_index": row_index,
                  "nominal_model_time_s": row_index / NOMINAL_HZ,
                  "row_started_monotonic_s": row_started,
                  "transition_origin": "fresh_measured" if stale else "last_sdk_receipt",
                  "origin": origin.tolist(), "raw_requested": raw_row.tolist(),
                  "requested": row.tolist(), "planned_points": len(points), "completed_points": 0,
                  "inserted_points": len(points) - 1,
                  "planned_minimum_row_duration_s": max(1 / NOMINAL_HZ, len(points) * MAX_COMMAND_DT),
                  "endpoint_completed": False}
        self.row_records.append(record)
        self.event("row_transition_planned", **record, points=points.tolist())
        for index, point in enumerate(points):
            is_endpoint = index == len(points) - 1
            ready_at = self.clock()
            if self.last_send_at is not None:
                ready_at = max(ready_at, self.last_send_at + MAX_COMMAND_DT)
            if is_endpoint and self.last_endpoint_at is not None:
                ready_at = max(ready_at, self.last_endpoint_at + 1 / NOMINAL_HZ)
            if not self._wait_until(ready_at):
                return False
            if (self.clock() - measured_at > STALE_COMMAND_S and index == 0
                    or self.last_send_at is not None and (index or not stale)
                    and self.clock() - self.last_send_at > STALE_COMMAND_S):
                raise OpenPiExecutionFault("An OpenPI transition stalled beyond the ordinary SDK stale reset")
            self.dispatch_check()
            sent_at = self.clock()
            self.attempted_points += 1
            target = _target(point)
            self.event("dispatch_attempt", chunk_index=chunk_index, row_index=row_index,
                       point_index=index, endpoint=is_endpoint, monotonic_s=sent_at, requested=target)
            try:
                sent = self.send(dict(target), self.dispatch_check)
            except BaseException:
                self.unknown_partial_dispatches += 1
                raise
            self.last_sent = self._receipt(point, sent)
            # Use callback completion, not invocation, to guarantee at least
            # .01 seconds before the next callback even for slow two-arm I/O.
            self.last_send_at = self.clock()
            self.completed_points += 1
            record["completed_points"] += 1
            self.inserted_points += int(not is_endpoint)
            self.observation_ages.append(sent_at - observed_at)
            self.event("dispatch", chunk_index=chunk_index, row_index=row_index, point_index=index,
                       endpoint=is_endpoint, monotonic_s=sent_at, requested=target, sent=dict(sent))
            if is_endpoint:
                self.last_endpoint_at = self.clock()
                self.endpoint_times.append(sent_at)
                self.completed_rows += 1
                self.chunks[chunk_index]["completed_rows"] += 1
                record["endpoint_completed"] = True
                self.executed_projected_gripper_values += sum(value["row_index"] == row_index
                                                              for value in conversions)
        finished_at = max(row_started + 1 / NOMINAL_HZ, self.last_send_at + MAX_COMMAND_DT)
        running = self._wait_until(finished_at)
        record["row_ended_monotonic_s"] = self.clock()
        record["actual_row_duration_s"] = self.clock() - row_started
        record["time_dilation_s"] = max(0.0, record["actual_row_duration_s"] - 1 / NOMINAL_HZ)
        self.event("row_completed", **record)
        return running

    def run(self, *, duration_s: float, max_chunks: int | None = None) -> dict:
        if self.used:
            raise OpenPiExecutionFault("An OpenPI executor is single use; automatic retries are forbidden")
        if (type(duration_s) not in (int, float) or not math.isfinite(duration_s) or not 0 < duration_s <= 90
                or max_chunks is not None and (type(max_chunks) is not int or not 1 <= max_chunks <= 180)):
            raise ValueError("OpenPI execution requires duration in (0,90] and optional bounded chunk count")
        self.used = True
        self.started = self.clock()
        self.deadline = self.started + duration_s
        try:
            while self.check() and (max_chunks is None or self.completed_chunks < max_chunks):
                remaining = self.deadline - self.clock()
                if self.completed_chunks and remaining < self.rpc_timeout_s:
                    self.phase_tail_requests_avoided += 1
                    self.event("phase_tail_wait", remaining_phase_s=remaining)
                    while self.check():
                        self._observe()
                        if not self._wait_until(self.clock() + 1 / NOMINAL_HZ):
                            break
                    break
                observed_at = self.clock()
                # The collector can retain this exact model-input RGB triplet
                # independently of its ordinary camera-rate video sampling.
                # No new observation or hardware read is added by this seam.
                observation, _ = self._observe(policy_input=True)
                if observation is None or not self.check():
                    break
                remaining = self.deadline - self.clock()
                requested_at = self.clock()
                timeout = min(self.rpc_timeout_s, remaining)
                self.inference_calls += 1
                try:
                    result = self.predict(observation, timeout)
                except Exception:
                    if not self.stop.is_set():
                        raise
                    self.stop_requested = True
                    self.stopped_rpc_exceptions += 1
                    break
                elapsed = self.clock() - requested_at
                self.rpc_latencies.append(elapsed)
                if not self.check():
                    values = result.get("chunk") if isinstance(result, Mapping) else result
                    if np.asarray(values).shape == (MODEL_ROWS, 14):
                        self.late_response_rows_discarded += MODEL_ROWS
                    self.event("late_response_discarded", rows=self.late_response_rows_discarded,
                               stop_requested=self.stop_requested, rpc_latency_s=elapsed)
                    break
                if elapsed >= timeout:
                    raise OpenPiExecutionFault("OpenPI inference exceeded its bounded request deadline")
                try:
                    values = result["chunk"] if isinstance(result, Mapping) else result
                    prepared = prepare_chunk(values)
                    self.predicted_rows += MODEL_ROWS
                    for row in prepared["rows"][:COMMITTED_ROWS]:
                        self.validate_target(_target(row))
                except Exception:
                    self.invalid_chunks += 1
                    raise
                self.admitted_rows += COMMITTED_ROWS
                conversions = prepared["gripper_conversions"]
                self.projected_gripper_values += len(conversions)
                self.maximum_gripper_projection = max(self.maximum_gripper_projection,
                                                      max((abs(v["delta"]) for v in conversions), default=0.0))
                chunk_index = len(self.chunks)
                record = {"chunk_index": chunk_index, "predicted_rows": MODEL_ROWS,
                          "committed_prefix_rows": COMMITTED_ROWS, "completed_rows": 0,
                          "intended_unused_rows": MODEL_ROWS - COMMITTED_ROWS,
                          "observation_monotonic_s": observed_at, "rpc_latency_s": elapsed,
                          "policy_observation_index": self.observations - 1}
                self.chunks.append(record)
                self.event("chunk_admitted", **record,
                           raw_rows=prepared["raw_rows"].tolist(), rows=prepared["rows"].tolist(),
                           gripper_conversions=conversions, action_transform=prepared["action_transform"],
                           audit=dict(result.get("audit", {})) if isinstance(result, Mapping) else {})
                for index in range(COMMITTED_ROWS):
                    if not self.check():
                        break
                    running = self._execute_row(prepared["rows"][index], prepared["raw_rows"][index],
                                                chunk_index=chunk_index, row_index=index,
                                                observed_at=observed_at, conversions=conversions)
                    if not running:
                        break
                if record["completed_rows"] == COMMITTED_ROWS:
                    self.completed_chunks += 1
                else:
                    break
            self.finished = True
        except BaseException:
            self.faults += 1
            raise
        finally:
            self.ended = self.clock()
            self.stop_requested = self.stop_requested or self.stop.is_set()
        return self.metrics()

    def metrics(self) -> dict:
        intervals = np.diff(self.endpoint_times)
        partial = self.admitted_rows - self.completed_rows
        return {"controller_mode": CONTRACT_ID, "execution_contract": dict(EXECUTION_CONTRACT),
                "predicted_rows": self.predicted_rows, "admitted_rows": self.admitted_rows,
                "completed_rows": self.completed_rows, "completed_chunks": self.completed_chunks,
                "admitted_chunks": len(self.chunks),
                "intended_unused_rows": len(self.chunks) * (MODEL_ROWS - COMMITTED_ROWS),
                "uncompleted_committed_rows": partial,
                "rows_discarded_on_stop_or_deadline": partial if self.finished else 0,
                "rows_discarded_on_fault": partial if self.faults else 0,
                "prefix_dropped_rows": 0, "reordered_rows": 0,
                "attempted_points": self.attempted_points, "completed_points": self.completed_points,
                "inserted_transition_points": self.inserted_points,
                "unknown_partial_dispatches": self.unknown_partial_dispatches,
                "modified_commands": self.modified_commands, "coherence_violations": self.coherence_violations,
                "roundoff_receipts": self.roundoff_receipts,
                "maximum_receipt_roundoff": self.maximum_receipt_roundoff,
                "projected_gripper_values": self.projected_gripper_values,
                "executed_projected_gripper_values": self.executed_projected_gripper_values,
                "maximum_gripper_projection": self.maximum_gripper_projection,
                "inference_calls": self.inference_calls, "observations": self.observations,
                "policy_input_observations": self.policy_input_observations,
                "faults": self.faults, "invalid_chunks": self.invalid_chunks,
                "stop_requested": self.stop_requested, "stopped_rpc_exceptions": self.stopped_rpc_exceptions,
                "late_response_rows_discarded": self.late_response_rows_discarded,
                "phase_tail_requests_avoided": self.phase_tail_requests_avoided,
                "rpc_latency_s": _summary(self.rpc_latencies),
                "observation_age_s": _summary(self.observation_ages),
                "endpoint_interval_s": _summary(intervals.tolist()),
                "maximum_row_time_dilation_s": max((row.get("time_dilation_s", 0.0)
                                                    for row in self.row_records), default=0.0),
                "chunks": list(self.chunks), "rows": list(self.row_records)}
