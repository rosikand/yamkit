"""Saved real RGB/state observations plus fake arms; never open physical devices."""

from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from threading import Event, Timer

import numpy as np

from yamkit.inference.mapping import YAM_NAMES
from yamkit.inference.protocol import PROTOCOL_VERSION, encode_image

from .contract import CONTRACT, PROFILE, build_id
from .executor import Pi05ExecutionFault, Pi05ReferenceExecutor, finite_rows, latency_summary
from .transport import validate_readiness


def load_saved_observation(path: Path) -> dict:
    """Inert NPZ archive: state[14], top/left_wrist/right_wrist HWC uint8 RGB.

    The caller must export these from an existing recording, never acquire them
    during qualification. No pickle, video decoder, camera or robot is involved.
    """
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"state", *PROFILE.image_keys}:
            raise ValueError("Saved π0.5 observation requires state and all three named RGB cameras")
        result = {name: np.array(saved[name], copy=True) for name in saved.files}
    state = result["state"]
    if state.shape != (14,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
        raise ValueError("Saved YAM observation needs exactly 14 finite state values")
    for name in PROFILE.image_keys:
        encode_image(result[name])  # Pure envelope and RGB validation; no device access.
    return result


def observation_schema(observations: list[dict]) -> dict:
    """Bind actual recorded payloads, never infer their shape from the rig file."""
    schemas = []
    for observation in observations:
        if set(observation) != {"state", *PROFILE.image_keys}:
            raise ValueError("Saved π0.5 observation has missing or extra RGB/state fields")
        state = np.asarray(observation["state"])
        if state.shape != (14,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
            raise ValueError("Saved π0.5 state must contain exactly fourteen finite values")
        images = {}
        for name in PROFILE.image_keys:
            encoded = encode_image(observation[name])
            images[name] = [encoded["height"], encoded["width"], 3]
        schemas.append({"state_names": list(YAM_NAMES), "images": images, "image_encoding": "rgb8", "crop": "none"})
    if not schemas or any(schema != schemas[0] for schema in schemas):
        raise ValueError("All saved π0.5 observations must have the same exact RGB schema")
    return schemas[0]


def make_request(observation: dict, *, task: str, session_id: str, sequence_id: int,
                 timeout_s: float, mode: str = "robot") -> dict:
    names = PROFILE.native_image_keys if mode == "native_fixture" else PROFILE.image_keys
    return {"protocol_version": PROTOCOL_VERSION, "profile": PROFILE.id, "model_revision": PROFILE.revision,
            "session_id": session_id, "sequence_id": sequence_id, "observation_time": time.monotonic(),
            "observation_age_s": 0.0, "timeout_s": timeout_s, "task": task,
            "state": np.asarray(observation["state"], dtype=np.float64).tolist(),
            "state_names": list(YAM_NAMES), "images": {
                out: encode_image(observation[source])
                for source, out in zip(PROFILE.image_keys, names, strict=True)},
            "mode": mode, "crop": "none", "continuation": None, "execution_mode": "eager"}


class FakeArms:
    """Literal target receipt; not a dynamics, tracking or safety simulation."""

    def __init__(self, observation: dict):
        self.observation = observation
        self.state = np.asarray(observation["state"], dtype=np.float64).copy()
        self.sent = []
        self.released = False

    def observe(self):
        if self.released:
            raise RuntimeError("Fake arms are released")
        return {**self.observation, "state": self.state.copy()}

    def send(self, target, check):
        check()
        if self.released:
            raise RuntimeError("Fake arms are released")
        self.sent.append(dict(target))
        self.state = np.array([target[name] for name in YAM_NAMES])
        return dict(target)

    def release(self):
        self.released = True


_CONTROLLED_EXECUTION_REASONS = frozenset({
    "π0.5 requires exactly 30×14 finite native action values",
    "π0.5 gripper output is outside [0,1]; no automatic clipping is allowed",
    "π0.5 dispatch was stopped or its bounded session ended",
    "π0.5 inference exceeded its request deadline",
    "π0.5 SDK target differs from the native row; stopping",
})


def _failed_chunk_evidence(chunk) -> dict:
    """Retain bounded numeric output, not arbitrary response fields or strings.

    Called only after failure, outside measured inference/execution. Finite
    values remain exact; nonfinite values use explicit JSON-safe markers, not
    clipping or invented finite actions. Unsupported values are never repr'd.
    """
    if type(chunk) is np.ndarray:
        shape = list(chunk.shape)
        if chunk.ndim != 2 or chunk.dtype.kind not in "fiu":
            return {"shape": shape, "raw_chunk": None, "unsupported_values_omitted": True}
        truncated = chunk.shape[0] > 30 or chunk.shape[1] > 14
        rows = chunk[:30, :14].tolist()
    elif type(chunk) in (list, tuple):
        widths = [len(row) if type(row) in (list, tuple) else None for row in chunk[:30]]
        shape = [len(chunk), widths[0] if widths and all(width == widths[0] for width in widths) else None]
        truncated = len(chunk) > 30 or any(width is not None and width > 14 for width in widths)
        rows = chunk[:30]
    else:
        return {"shape": None, "raw_chunk": None, "unsupported_values_omitted": True}
    raw, nonfinite, violations, columns = [], [], [], [[] for _ in YAM_NAMES]
    omitted = False
    for row_index, row in enumerate(rows):
        if type(row) not in (list, tuple):
            raw.append(None)
            omitted = True
            continue
        values = []
        for column_index, value in enumerate(row[:14]):
            if type(value) not in (int, float) or type(value) is int and value.bit_length() > 1023:
                values.append(None)
                omitted = True
                continue
            if not math.isfinite(value):
                values.append(None)
                nonfinite.append({"row_index": row_index, "column_index": column_index,
                                  "kind": "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"})
                continue
            values.append(value)
            columns[column_index].append(value)
            if column_index in (6, 13) and not 0 <= value <= 1:
                violations.append({"row_index": row_index, "column_index": column_index,
                                   "name": YAM_NAMES[column_index], "value": value, "minimum": 0.0, "maximum": 1.0})
        raw.append(values)
    return {"shape": shape, "raw_chunk": raw, "truncated": truncated,
            "unsupported_values_omitted": omitted, "nonfinite_values": nonfinite,
            "gripper_bound_violations": violations,
            "column_ranges": [{"column_index": index, "name": name, "finite_count": len(values),
                               "minimum": min(values) if values else None, "maximum": max(values) if values else None}
                              for index, (name, values) in enumerate(zip(YAM_NAMES, columns, strict=True))]}


def collect_qualification(transport, *, observations: list[dict], task: str, requests: int = 50,
                          validate_target=None, rig_path: Path | None = None,
                          event=lambda *_args, **_kwargs: None) -> dict:
    """Run real GPU calls, native FIFO timing and Stop with explicit fake arms.

    ``validate_target`` should be the same passive configured YAM bound check
    used for physical admission. Without it, schema/fake results are useful but
    the result cannot be marked qualified for physical execution.
    """
    if type(requests) is not int or not 1 <= requests <= 50 or not observations:
        raise ValueError("Qualification needs 1–50 requests and saved real observations")
    if not isinstance(task, str) or not task.strip() or len(task) > 2048:
        raise ValueError("An explicit task is required")
    schema = observation_schema(observations)
    robot_host = None
    if rig_path is not None:
        from .admission import passive_target_validator, rig_binding

        robot_host = rig_binding(rig_path)
        if schema != robot_host["observation_schema"]:
            raise ValueError("Saved π0.5 RGB shapes differ from the rig; qualify the actual full camera payload")
        validate_target = passive_target_validator(rig_path)
    metadata = transport.ready(15.0)
    validate_readiness(metadata)
    session_id, sequence = str(uuid.uuid4()), 0
    direct, direct_chunks, errors = [], [], []
    report = {"version": 1, "profile": PROFILE.id, "controller_mode": CONTRACT["id"],
              "model_revision": PROFILE.revision, "pi05_build_id": build_id(),
              "instance_id": metadata["instance_id"], "task": task, "created_at": time.time(),
              "expires_at": metadata.get("http_session_expires_at"), "hardware_tested": False,
              "observation_source": "caller-supplied saved real recording; no new capture",
              "observation_schema": schema,
              "requested_warm_samples": requests, "bounds_checked": validate_target is not None,
              "qualified": False, "reasons": errors, "runtime": metadata}
    if robot_host is not None:
        report["robot_host"] = robot_host
    validator = validate_target or (lambda _: None)
    phase, sample_index, observation_index = "cold_warmup", None, 0
    request_sequence, validation_row_index, last_chunk = None, None, None
    response_received = False

    def predict(observation, timeout):
        nonlocal sequence, request_sequence, last_chunk, sample_index, response_received
        # Keep a reference only; diagnostic copying/range scans occur AFTER a
        # failure, never inside measured model or FIFO timing. Clear first so a
        # failed RPC cannot be attributed to the preceding successful chunk.
        last_chunk = None
        response_received = False
        request_sequence = sequence
        if phase == "integrated":
            sample_index = 0 if sample_index is None else sample_index + 1
        request = make_request(observation, task=task, session_id=session_id,
                               sequence_id=sequence, timeout_s=timeout)
        sequence += 1
        last_chunk = transport.predict_chunk(request, timeout)["chunk"]
        response_received = True
        return last_chunk

    fake, engine, stop_fake, stop_engine = None, None, None, None
    try:
        # One excluded cold/native warm, then the requested direct warm samples.
        predict(observations[0], 120.0)
        for index in range(requests):
            phase, sample_index, observation_index = "direct_warm", index, index % len(observations)
            validation_row_index = None
            began = time.monotonic()
            chunk = finite_rows(predict(observations[index % len(observations)], 2.0))
            direct.append(time.monotonic() - began)
            for validation_row_index, row in enumerate(chunk):
                validator(dict(zip(YAM_NAMES, row.tolist(), strict=True)))
            validation_row_index = None
            direct_chunks.append(chunk.tolist())
            event("direct_sample", completed=index + 1, total=requests)
        phase, sample_index, observation_index = "integrated", None, 0
        last_chunk, request_sequence = None, None
        response_received = False
        fake = FakeArms(observations[0])
        engine = Pi05ReferenceExecutor(predict=predict, observe=fake.observe, send=fake.send,
                                      validate_target=validator, stop=Event(),
                                      session_check=transport.ensure_session_active, event=event)
        engine.run(duration_s=90, max_chunks=requests)
        fake.release()
        result = engine.metrics()
        report["integrated"] = result
        if result["completed_chunks"] != requests or result["completed_rows"] != requests * 30:
            errors.append("Integrated fake execution did not complete all requested native chunks")
        if any(result[key] for key in ("dropped_rows", "modified_commands", "coherence_violations", "faults",
                                      "unknown_partial_dispatches")):
            errors.append("Native fake execution lost or modified rows, or faulted")
        # Exercise the SAME executor with Stop while its real RPC is in flight.
        phase, sample_index, observation_index = "stop_proof", 0, 0
        last_chunk, request_sequence = None, None
        response_received = False
        stop = Event()
        stop_fake = FakeArms(observations[0])
        proof = {"stop_requested_during_inflight_rpc": False}

        def stop_predict(observation, timeout):
            in_flight = Event()
            in_flight.set()

            def request_stop():
                proof["stop_requested_during_inflight_rpc"] = in_flight.is_set()
                stop.set()

            timer = Timer(0.01, request_stop)
            timer.daemon = True
            timer.start()
            try:
                return predict(observation, timeout)
            finally:
                in_flight.clear()
                timer.cancel()

        stop_engine = Pi05ReferenceExecutor(predict=stop_predict, observe=stop_fake.observe,
                                           send=stop_fake.send, validate_target=validator, stop=stop,
                                           session_check=transport.ensure_session_active)
        stop_engine.run(duration_s=5, max_chunks=1)
        stop_fake.release()
        proof.update(commands_after_stop=len(stop_fake.sent), all_fake_robots_released=stop_fake.released,
                     metrics=stop_engine.metrics())
        report["stop_proof"] = proof
        if not proof["stop_requested_during_inflight_rpc"] or proof["commands_after_stop"]:
            errors.append("Stop during the real RPC was not proven to reject all late commands")
        # This serial native contract has a finite 2 s RPC bound; it does not
        # drop rows to hide latency. Report its separate 1 s nominal chunk.
        summary = latency_summary(direct)
        if summary["p95"] is None or summary["p95"] > 1.6:
            errors.append("Warm RPC p95 exceeds the 1.6 s admission budget (20% margin on 2 s RPC)")
        if validate_target is None:
            errors.append("Configured physical YAM joint bounds were not supplied")
        if rig_path is None:
            errors.append("The robot host and rig were not bound to this qualification")
        if requests < 50:
            errors.append("Physical admission requires the complete 50-sample qualification")
        phase, sample_index, observation_index = "final_readiness", None, None
        last_chunk, request_sequence = None, None
        response_received = False
        current = transport.ready(15.0)
        if current["instance_id"] != metadata["instance_id"]:
            errors.append("Model instance changed during qualification")
    except Exception as exc:  # noqa: BLE001 — sanitized failure category, never bearer or model input
        reason = (exc.args[0] if type(exc) is Pi05ExecutionFault and len(exc.args) == 1
                  and type(exc.args[0]) is str and exc.args[0] in _CONTROLLED_EXECUTION_REASONS else None)
        errors.append(f"Software qualification failed: {type(exc).__name__}" + (f": {reason}" if reason else ""))
        report["failure"] = {"phase": phase, "sample_index": sample_index, "observation_index": observation_index,
                             "indices_are_zero_based": True, "request_sequence_id": request_sequence,
                             "validation_row_index": validation_row_index, "error_type": type(exc).__name__,
                             "reason": reason, "native_response_received": response_received,
                             "native_response": _failed_chunk_evidence(last_chunk) if response_received else None}
    finally:
        for robot in (fake, stop_fake):
            if robot is not None:
                robot.release()
        report.update(completed_warm_samples=len(direct), direct_warm_round_trip_s=latency_summary(direct),
                      completed_at=time.time(), all_fake_robots_released=all(
                          robot is None or robot.released for robot in (fake, stop_fake)),
                      nominal_chunk_horizon_s=1.0, rpc_timeout_s=2.0, maximum_qualifying_rpc_p95_s=1.6)
        if engine is not None:
            report["integrated"] = engine.metrics()
        report["qualified"] = not errors
        # Serialized full model rows are retained for exact offline FIFO replay.
        report["direct_chunks"] = direct_chunks
    return report


def save_qualification(report: dict, path: Path, *, observation_paths: list[Path]) -> None:
    """Save a new sanitized report, never overwrite prior qualification evidence."""
    from yamkit.paths import ROOT

    target = path.resolve()
    target.relative_to(Path(ROOT).resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    report = {**report, "saved_observations": [
        {"path": str(item), "sha256": hashlib.sha256(item.read_bytes()).hexdigest()}
        for item in observation_paths]}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
