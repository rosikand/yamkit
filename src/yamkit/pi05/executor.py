"""Synchronous native PI05 FIFO execution with explicit I/O seams.

This module imports no hardware. Callers own resource acquisition and release.
No interpolation, rate catch-up, prefix drops, overlap merges or target shaping
are performed. Stop can discard an unexecuted tail, which is reported explicitly.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from threading import Event

import numpy as np

from yamkit.inference.mapping import YAM_NAMES

from .contract import CONTRACT_ID, PROFILE


class Pi05ExecutionFault(ValueError):
    """A native chunk, command, observation or execution lifecycle is invalid."""


def finite_rows(chunk) -> np.ndarray:
    values = np.asarray(chunk)
    if values.shape != (30, 14) or values.dtype.kind not in "fiu" or not np.isfinite(values).all():
        raise Pi05ExecutionFault("π0.5 requires exactly 30×14 finite native action values")
    rows = values.astype(np.float64, copy=True)
    if np.any(rows[:, [6, 13]] < 0) or np.any(rows[:, [6, 13]] > 1):
        raise Pi05ExecutionFault("π0.5 gripper output is outside [0,1]; no automatic clipping is allowed")
    return rows


def latency_summary(samples: list[float]) -> dict:
    return {"sample_count": len(samples), "p50": float(np.percentile(samples, 50)) if samples else None,
            "p95": float(np.percentile(samples, 95)) if samples else None,
            "max": max(samples) if samples else None}


class Pi05ReferenceExecutor:
    """Single-use FIFO engine; all external operations are explicit callbacks.

    ``predict`` returns postprocessed robot-unit rows; ``send`` returns the actual
    target mapping. ``validate_target`` is a passive whole-chunk bounds check.
    Measured state/cameras are sampled at each native policy chunk boundary.
    The physical adapter also validates measured state immediately before sends.
    """

    def __init__(self, *, predict: Callable, observe: Callable, send: Callable,
                 validate_target: Callable, stop: Event, session_check: Callable = lambda: None,
                 clock: Callable = time.monotonic, wait: Callable | None = None,
                 event: Callable = lambda *_args, **_kwargs: None, rpc_timeout_s: float = 2.0):
        if (type(rpc_timeout_s) not in (float, int) or not math.isfinite(rpc_timeout_s)
                or not 0 < rpc_timeout_s <= 2.0):
            raise ValueError("π0.5 reference requires a finite RPC timeout of at most two seconds")
        self.predict, self.observe, self.send = predict, observe, send
        self.validate_target, self.stop, self.session_check = validate_target, stop, session_check
        self.clock, self.wait, self.event = clock, wait or stop.wait, event
        self.rpc_timeout_s = rpc_timeout_s
        self.used = self.finished = False
        self.predicted_rows = self.completed_rows = self.completed_chunks = self.modified_commands = 0
        self.coherence_violations = self.dropped_rows = self.inference_calls = self.faults = 0
        self.rpc_latencies, self.observation_ages, self.dispatch_times, self.chunks = [], [], [], []
        self.started = self.ended = None
        self.deadline = float("inf")
        self.stop_requested = False

    def check(self) -> bool:
        self.session_check()
        if self.stop.is_set():
            self.stop_requested = True
            return False
        return self.clock() < self.deadline

    def dispatch_check(self) -> None:
        if not self.check():
            raise Pi05ExecutionFault("π0.5 dispatch was stopped or its bounded session ended")

    def _wait_until(self, deadline: float) -> bool:
        while self.check():
            remaining = min(deadline, self.deadline) - self.clock()
            if remaining <= 0:
                return self.check()
            self.wait(min(remaining, 0.01))
        return False

    def run(self, *, duration_s: float, max_chunks: int | None = None) -> dict:
        if self.used:
            raise Pi05ExecutionFault("A used π0.5 executor cannot resume; create a newly approved session")
        if (type(duration_s) not in (float, int) or not math.isfinite(duration_s) or not 0 < duration_s <= 90
                or max_chunks is not None and (type(max_chunks) is not int or not 1 <= max_chunks <= 90)):
            raise ValueError("π0.5 execution requires 0–90 seconds and a bounded optional chunk count")
        self.used = True
        self.started = self.clock()
        self.deadline = self.started + duration_s
        try:
            while self.check() and (max_chunks is None or self.completed_chunks < max_chunks):
                observed_at = self.clock()
                observation = self.observe()
                if not self.check():
                    break
                remaining = self.deadline - self.clock()
                if remaining < 0.01:
                    break
                requested_at = self.clock()
                self.inference_calls += 1
                result = self.predict(observation, min(self.rpc_timeout_s, remaining))
                returned_at = self.clock()
                elapsed = returned_at - requested_at
                self.rpc_latencies.append(elapsed)
                rows = finite_rows(result)
                self.predicted_rows += len(rows)
                if elapsed >= self.rpc_timeout_s:
                    raise Pi05ExecutionFault("π0.5 inference exceeded its request deadline")
                if not self.check():
                    break  # In particular, never dispatch a late response after Stop.
                targets = [dict(zip(YAM_NAMES, row.tolist(), strict=True)) for row in rows]
                for target in targets:
                    self.validate_target(target)
                chunk_index = len(self.chunks)
                record = {"index": chunk_index, "predicted_rows": 30, "completed_rows": 0,
                          "observation_time": observed_at, "rpc_latency_s": elapsed}
                self.chunks.append(record)
                self.event("chunk_admitted", **record, rows=rows.tolist())
                for index, target in enumerate(targets):
                    if not self.check():
                        break
                    tick_started = self.clock()
                    sent = self.send(dict(target), self.dispatch_check)
                    self.completed_rows += 1
                    record["completed_rows"] += 1
                    if not isinstance(sent, Mapping) or set(sent) != set(target) or any(
                            type(sent[name]) not in (int, float) or not math.isfinite(sent[name])
                            or sent[name] != target[name] for name in target):
                        self.modified_commands += 1
                        self.coherence_violations += 1
                        raise Pi05ExecutionFault("π0.5 SDK target differs from the native row; stopping")
                    self.dispatch_times.append(tick_started)
                    self.observation_ages.append(tick_started - observed_at)
                    self.event("dispatch", chunk_index=chunk_index, row_index=index,
                               monotonic_s=tick_started, requested=target, sent=dict(sent))
                    # Native 30 Hz FIFO rows: one point, then remaining per-tick
                    # period. An overrun never creates a burst of catch-up rows.
                    if not self._wait_until(tick_started + 1 / PROFILE.fps):
                        break
                if record["completed_rows"] == 30:
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
            self.dropped_rows = self.predicted_rows - self.completed_rows
        return self.metrics()

    def metrics(self) -> dict:
        intervals = np.diff(self.dispatch_times)
        return {"controller_mode": CONTRACT_ID, "predicted_rows": self.predicted_rows,
                "completed_rows": self.completed_rows, "dropped_rows": self.dropped_rows,
                "uncompleted_rows": self.predicted_rows - self.completed_rows,
                "prefix_dropped_rows": 0, "reordered_rows": 0,
                "modified_commands": self.modified_commands, "coherence_violations": self.coherence_violations,
                "completed_chunks": self.completed_chunks, "admitted_chunks": len(self.chunks),
                "inference_calls": self.inference_calls, "faults": self.faults,
                "stop_requested": self.stop_requested, "interpolation_points": 0,
                "execution_rate_hz": (float(1 / np.median(intervals)) if len(intervals) and np.median(intervals) > 0
                                      else None),
                "effective_rows_per_second": (self.completed_rows / (self.ended - self.started)
                                              if self.ended is not None and self.ended > self.started else None),
                "rpc_latency_s": latency_summary(self.rpc_latencies),
                "observation_age_s": latency_summary(self.observation_ages),
                "chunks": list(self.chunks)}
