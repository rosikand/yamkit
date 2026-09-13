"""Actual service, retained observations and explicitly fake YAM command receipts."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from yamkit.inference.mapping import YAM_NAMES

from .admission import (
    adapter_build_id,
    complete_execution_proof,
    rig_contract,
    service_binding,
    stopped_execution_proof,
    target_validator,
)
from .contract import SAVED_IMAGE_MAP
from .executor import OpenPiYamExecutor
from .interface import CONTRACT_ID, prepare_chunk
from .service import observation_request
from .yam_candidate import decode_chunk


def load_saved(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024 * 1024:
        raise ValueError("OpenPI qualification requires bounded existing repository-local observations")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {"state", *SAVED_IMAGE_MAP}:
            raise ValueError("Saved OpenPI observation schema differs")
        result = {key: saved[key].copy() for key in saved.files}
    request(result, "schema validation")
    return result


def request(observation, task):
    return observation_request({target: observation[source] for source, target in SAVED_IMAGE_MAP.items()},
                               observation["state"], task)


def decoded_response(response, observation, statistics):
    if not np.array_equal(np.asarray(response["state"]), observation["state"]):
        raise ValueError("OpenPI response must use its exact measured request anchor")
    audit = decode_chunk(response["raw_normalized_chunk"], observation["state"], statistics)
    return {"chunk": audit["proposed_yam_absolute_commands"],
            "raw_normalized_chunk": response["raw_normalized_chunk"], "audit": {
                "measured_yam_state": audit["measured_yam_state"].tolist(),
                "candidate_state": audit["candidate_state"].tolist(),
                "unnormalized_joint_deltas_absolute_closure": audit["unnormalized_joint_deltas_absolute_closure"].tolist(),
                "unused_normalized_model_dimensions": audit["unused_normalized_model_dimensions"].tolist(),
                "statistics_sha256": audit["candidate_statistics_sha256"],
                "clipping_applied_to_model_or_joints": False}}


class SavedFake:
    """Perfect target receipt with saved RGB; not physical tracking or task evidence."""

    def __init__(self, observations, validate):
        self.observations, self.validate = observations, validate
        self.state = observations[0]["state"].copy()
        self.index, self.sent = 0, []
        self.released = False

    def observe(self):
        if self.released:
            raise RuntimeError("Fake observation after release")
        value = self.observations[self.index % len(self.observations)]
        self.index += 1
        return {**value, "state": self.state.copy()}

    def send(self, target, check):
        check()
        if self.released:
            raise RuntimeError("Fake command after release")
        self.validate(target)
        self.sent.append(dict(target))
        self.state = np.array([target[name] for name in YAM_NAMES])
        return dict(target)

    def release(self):
        self.released = True


def latency(samples):
    return {"sample_count": len(samples), "p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)), "max": max(samples)}


def collect(transport, *, statistics, observations, task, rig_path, directory, progress=lambda _value: None):
    """50 saved-real direct calls, 50 full fake committed prefixes, actual in-flight Stop."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    binding = rig_contract(rig_path)
    validate = target_validator(binding)
    metadata = transport.ready()
    transport.warm(request(observations[0], task))
    metadata = transport.ready()
    report = {"profile": "pi05-base", "qualified": False, "reasons": [], "hardware_tested": False,
              "controller_mode": CONTRACT_ID, "adapter_build_id": adapter_build_id(),
              "statistics_sha256": statistics.metadata()["candidate_statistics_sha256"],
              "service_identity": service_binding(metadata), "expires_at": metadata["session_expires_at"],
              "task": task, "robot_host": binding, "completed_warm_samples": 0,
              "bounds_checked": True, "direct_chunks_validated": 0, "source_observation_count": len(observations),
              "fake_scope": "Saved real direct inputs; integrated perfect target tracking with saved RGB, not robot dynamics"}
    samples = []
    try:
        for index in range(50):
            obs = observations[index % len(observations)]
            started = time.monotonic()
            response = transport.predict_chunk(request(obs, task))
            samples.append(time.monotonic() - started)
            # Retain the exact native tensor even if decoding rejects it.
            decoded = None
            try:
                decoded = decoded_response(response, obs, statistics)
            finally:
                with (directory / f"direct-{index:03d}.npz").open("xb") as stream:
                    np.savez_compressed(stream, state=obs["state"], raw_normalized_chunk=response["raw_normalized_chunk"],
                                        **({"proposed_yam_chunk": decoded["chunk"]} if decoded else {}),
                                        **{name: obs[name] for name in SAVED_IMAGE_MAP})
            prepared = prepare_chunk(decoded["chunk"])
            for row in prepared["rows"][:25]:
                validate(dict(zip(YAM_NAMES, row.tolist(), strict=True)))
            (directory / f"direct-{index:03d}-audit.json").write_text(json.dumps({
                "decode": decoded["audit"], "gripper_conversions": prepared["gripper_conversions"]}, allow_nan=False))
            report["direct_chunks_validated"] += 1
            report["completed_warm_samples"] += 1
        report["direct_warm_round_trip_s"] = latency(samples)
        progress("Checking 50 committed prefixes with saved images and explicit fake arms")
        fake, stop = SavedFake(observations, validate), threading.Event()
        integrated_replies, events = [], []

        def predict(obs, timeout):
            response = transport.predict_chunk(request(obs, task), timeout_s=timeout)
            retained = {"raw_normalized_chunk": response["raw_normalized_chunk"]}
            integrated_replies.append((obs, retained))
            decoded = decoded_response(response, obs, statistics)
            retained["chunk"] = decoded["chunk"]
            return decoded

        engine = OpenPiYamExecutor(predict=predict, observe=fake.observe, send=fake.send,
                                  validate_target=validate, stop=stop, session_check=transport.ensure_session_active,
                                  max_joint_speed=binding["max_joint_speed"], max_gripper_speed=binding["max_gripper_speed"],
                                  event=lambda kind, **data: events.append({"kind": kind, **data}))
        try:
            report["integrated"] = engine.run(duration_s=90, max_chunks=50)
        finally:
            report["integrated"] = engine.metrics()
            fake.release()
            for index, (obs, decoded) in enumerate(integrated_replies):
                with (directory / f"integrated-{index:03d}.npz").open("xb") as stream:
                    np.savez_compressed(stream, state=obs["state"], raw_normalized_chunk=decoded["raw_normalized_chunk"],
                                        **({"proposed_yam_chunk": decoded["chunk"]} if "chunk" in decoded else {}),
                                        **{name: obs[name] for name in SAVED_IMAGE_MAP})
            (directory / "integrated-events.json").write_text(json.dumps(events, allow_nan=False))
        # Observe this exact request's completed wire send before cancelling.
        # The server may finish unused; no late result can reach either arm.
        stop = threading.Event()
        stopped = SavedFake(observations, validate)
        issued, after, wire_sent, cancelled_before_return = [], [], [], []
        returned = threading.Event()

        def cancel():
            if not transport.request_sent.wait(3):
                return
            wire_sent.append(True)
            cancelled_before_return.append(not returned.is_set())
            stop.set()
            transport.cancel()

        def stopping_predict(obs, timeout):
            issued.append(True)
            payload = request(obs, task)
            transport.request_sent.clear()
            watcher = threading.Thread(target=cancel, daemon=True)
            watcher.start()
            try:
                value = transport.predict_chunk(payload, timeout_s=timeout)
                returned.set()
                return decoded_response(value, obs, statistics)
            finally:
                returned.set()
                watcher.join(timeout=4)

        def stopping_send(target, check):
            if stop.is_set():
                after.append(True)
            return stopped.send(target, check)

        stopped_engine = OpenPiYamExecutor(predict=stopping_predict, observe=stopped.observe, send=stopping_send,
                                          validate_target=validate, stop=stop,
                                          max_joint_speed=binding["max_joint_speed"], max_gripper_speed=binding["max_gripper_speed"])
        try:
            stopped_engine.run(duration_s=3)
        finally:
            stopped.release()
        report["stop_proof"] = {"stop_requested_during_inflight_rpc": bool(issued and wire_sent and stop.is_set()),
                                "cancelled_before_transport_return": bool(cancelled_before_return and cancelled_before_return[0]),
                                "commands_after_stop": len(after), "total_fake_commands": len(stopped.sent),
                                "all_fake_robots_released": fake.released and stopped.released,
                                "execution": stopped_engine.metrics()}
        if not complete_execution_proof(report["integrated"]):
            report["reasons"].append("integrated_execution_proof")
        if report["direct_warm_round_trip_s"]["p95"] > 1.6:
            report["reasons"].append("synchronous_rpc_deadline_margin")
        if after or stopped.sent:
            report["reasons"].append("commands_during_stop_probe")
        if not report["stop_proof"]["cancelled_before_transport_return"]:
            report["reasons"].append("stop_not_observed_during_rpc")
        if not stopped_execution_proof(report["stop_proof"]["execution"]):
            report["reasons"].append("stop_execution_proof")
        report["qualified"] = not report["reasons"]
    except Exception as exc:  # noqa: BLE001 — no private transport or SDK payload in report
        report["reasons"].append("qualification_" + type(exc).__name__)
    finally:
        report["completed_at"] = time.time()
        (directory / "qualification.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
