"""Official frozen OpenPI recordings with an explicit experimental YAM interface.

Only already acquired observations enter this module. It imports no robot,
camera, CAN or hardware factory. The frozen MolmoAct2 collector and controller
are never invoked; its pure timestamp/video helpers are reused unchanged.
"""

from __future__ import annotations

import copy
import html
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..inference.mapping import YAM_NAMES
from ..paths import ROOT
from ..rollout_artifacts import (
    _SENSITIVE_KEY,
    ARTIFACTS,
    CAMERAS,
    REQUIRED,
    SCHEMA_VERSION,
    _copy,
    _hash,
    _json,
    _safe_path,
    upload_rollout,
    validate_bundle,
)
from ..rollout_artifacts import (
    sanitize as _shared_sanitize,
)
from .contract import CHECKPOINT, UPSTREAM_REVISION
from .interface import CONTRACT_ID, EXECUTION_CONTRACT

CONTRACT = EXECUTION_CONTRACT
POLICY_ID = "pi05-base"
POLICY_LABEL = "official frozen pi05_base + documented experimental YAM adapter"
MAX_DURATION_S = 60
CAPTURE_VIDEO_HZ = 30
CAPTURE_REQUEST_HZ = 2
MAX_FRAME_CAPACITY = MAX_DURATION_S * (CAPTURE_VIDEO_HZ + CAPTURE_REQUEST_HZ) + 5


def sanitize(value, *, secrets=()):
    """Retain strictly typed actuator proof while sharing credential filtering.

    The shared filter removes ``endpoint`` keys because those usually mean a
    private service URL. These exact numeric/boolean fields are actuator proof;
    string values never receive an exemption.
    """
    if isinstance(value, dict):
        result = _shared_sanitize(value, secrets=secrets)
        for key, item in value.items():
            if key in result:
                result[key] = sanitize(item, secrets=secrets)
            elif key in ("endpoint", "endpoint_completed") and type(item) is bool:
                result[key] = item
            elif (key == "endpoint_interval_s" and isinstance(item, dict)
                  and set(item) <= {"sample_count", "p50", "p95", "max"}
                  and all(v is None or type(v) in (int, float) and math.isfinite(v) for v in item.values())):
                result[key] = dict(item)
        return result
    if isinstance(value, list):
        return [sanitize(item, secrets=secrets) for item in value]
    return _shared_sanitize(value, secrets=secrets)


@lru_cache(maxsize=1)
def _trace_tools():
    path = Path(ROOT) / "scripts/trace_rollout.py"
    spec = importlib.util.spec_from_file_location("_yamkit_openpi_video_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # This unregistered module instance belongs exclusively to OpenPI export;
    # neither frozen source nor MA2/pi05_yam helper instances are modified.
    # video_timeline bounds len(frames) by MAX_DURATION_S * VIDEO_FPS + 3.
    # Configure that pure frame-count bound to this collector's 1,925 slots;
    # the actual policy duration remains independently bounded at 60 seconds.
    module.VIDEO_FPS = CAPTURE_VIDEO_HZ
    module.MAX_DURATION_S = (MAX_FRAME_CAPACITY - 3) / CAPTURE_VIDEO_HZ
    return module


def repository_path(path):
    selected = _safe_path(Path(path))
    selected.relative_to(Path(ROOT).resolve())
    return selected


def _write(path, value):
    repository_path(path).write_text(json.dumps(sanitize(value), indent=2, allow_nan=False) + "\n")


def rollout_phase(phase):
    if phase not in ("running", "returning_home", "releasing", "released"):
        raise ValueError("Unexpected native rollout display phase")
    try:
        print("[yamkit-rollout] " + phase, file=sys.stderr, flush=True)
    except OSError:
        pass  # Display failure cannot interrupt resource release or motor control.


def _finalize_outcome(directory, summary, report, meta, *, upload_pending=False):
    """Keep completed control/release evidence distinct from the requested pipeline."""
    execution_status = report["status"]
    if summary["resources_released"] is not True:
        error = "release_not_confirmed"
    elif summary["status"] != "TRACE_SAVED":
        error = "recording_export_failed"
    elif "upload" in summary and summary["upload"].get("status") not in ("uploaded", "already_uploaded"):
        error = "upload_failed"
    else:
        error = None
    exit_status = (1 if error or execution_status not in ("completed", "stopped")
                   else 130 if execution_status == "stopped" else 0)
    outcome = {"execution_status": execution_status, "exit_status": exit_status,
               "pipeline_complete": exit_status == 0 and not upload_pending,
               "postprocess_error": error, "upload_pending": upload_pending}
    summary.update(outcome)
    report.update(outcome)
    meta.update(outcome, returncode=exit_status,
                status="stopped" if execution_status == "stopped" else "success" if exit_status == 0 else "failed",
                resources_released=summary["resources_released"])
    _write(directory / "summary.json", summary)
    _write(directory / "report.json", report)
    _write(directory / "metrics.json", {"controller_mode": CONTRACT_ID,
                                        "openpi_rollout": report,
                                        "openpi_execution": report.get("execution", {})})
    _write(directory / "meta.json", meta)
    repository_path(directory / "log.txt").write_text(
        "Official OpenPI π0.5 + experimental YAM lifecycle: " + execution_status + "; pipeline: "
        + ("upload pending" if upload_pending else "complete" if outcome["pipeline_complete"] else "incomplete") + "\n")


class OpenPiCapture:
    """Capture at existing observation seams; serialize only after release.

    Full RGB slots are prefaulted before robot construction. This separate base-
    model pool records at most one RGB triplet in each phase-aligned 30 Hz bin,
    plus EVERY marked policy-input observation (at most two requests per second
    because 25 endpoints complete at no more than 50 Hz), and five edge slots.
    All measured-state observations and original timestamps remain in the event
    ledger; omitted non-input RGB ticks are explicitly counted, not lost frames.
    The 512 MiB headroom and strict admission remain. Duration is bounded at
    60 seconds; MA2 and pi05_yam capture limits remain unchanged. No additional
    observation calls, encoders, disk writes or background work occur in motion.
    Trace errors are counted; instrumentation never alters an action or masks a
    control fault. Native responses remain distinct from admitted/executed rows.
    """

    def __init__(self, *, task, duration_s, capture_trace=False, clock=time.monotonic):
        if (type(duration_s) not in (int, float) or not math.isfinite(duration_s)
                or not 0 < duration_s <= MAX_DURATION_S or type(capture_trace) is not bool):
            raise ValueError("Native capture requires a bounded 0–60 second duration")
        self.task, self.duration_s, self.capture_trace, self.clock = task, duration_s, capture_trace, clock
        self.started = self.ended = None
        self.events, self.responses, self.frames = [], [], []
        self.frame_pool, self.memory_preflight = None, None
        self.frame_capacity = (math.ceil(duration_s * CAPTURE_VIDEO_HZ)
                               + math.ceil(duration_s * CAPTURE_REQUEST_HZ) + 5)
        self.counts = {"events_dropped": 0, "frames_dropped": 0, "chunks_dropped": 0,
                       "trace_errors": 0, "observation_frames_seen": 0,
                       "skipped_rgb_observations": 0, "policy_input_observations": 0,
                       "policy_input_rgb_frames": 0, "cadence_rgb_frames": 0,
                       "extra_policy_input_rgb_frames": 0}
        self.next_rgb_bin = 0
        self.policy_input_indices = set()
        self.frame_by_observation = {}
        self.error_types = set()

    def reserve(self):
        if not self.capture_trace:
            return
        tools = _trace_tools()
        needed = self.frame_capacity * tools.FRAME_TRIPLET_BYTES
        available = tools.available_memory_bytes()
        self.memory_preflight = {"available_bytes": available, "frame_bytes": needed,
                                 "headroom_bytes": tools.MEMORY_HEADROOM_BYTES,
                                 "required_bytes": needed + tools.MEMORY_HEADROOM_BYTES}
        if available < self.memory_preflight["required_bytes"]:
            raise MemoryError("Insufficient available memory for the complete official OpenPI π0.5 recording")
        self.frame_pool = np.empty((self.frame_capacity, 3, 480, 640, 3), dtype=np.uint8)
        self.frame_pool.fill(0)

    def safely(self, callback, *args, **kwargs):
        try:
            callback(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 — recording never changes control semantics
            self.counts["trace_errors"] += 1
            if len(self.error_types) < 16:
                self.error_types.add(type(exc).__name__)

    def start(self):
        self.started = self.clock()
        self.event("policy_phase_started")

    def end(self):
        if self.started is not None and self.ended is None:
            self.ended = self.clock()
            self.event("policy_phase_ended")

    def event(self, kind, **value):
        if len(self.events) >= 32768:
            self.counts["events_dropped"] += 1
            return
        self.events.append({"kind": kind, "monotonic_s": self.clock(), **value})

    def observation(self, observation, *, policy_input=False):
        if self.started is None or self.ended is not None:
            return
        if type(policy_input) is not bool:
            raise ValueError("Policy-input capture marking must be explicit boolean")
        observed_at = self.clock()
        index = self.counts["observation_frames_seen"]
        self.counts["observation_frames_seen"] += 1
        state = np.asarray(observation["state"])
        if state.shape != (14,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
            raise ValueError("Invalid captured measured-state vector")
        self.event("observation", monotonic_s=observed_at, observation_index=index,
                   positions=state.tolist(), policy_input=policy_input)
        if policy_input:
            self.policy_input_indices.add(index)
            self.counts["policy_input_observations"] += 1
        if not self.capture_trace:
            return
        # Use phase-aligned bins, not last-sample + 1/30: with 50 Hz incoming
        # observations the latter would unintentionally reduce video to 25 Hz.
        cadence_bin = math.floor((observed_at - self.started) * CAPTURE_VIDEO_HZ)
        cadence_due = cadence_bin >= self.next_rgb_bin
        if not cadence_due and not policy_input:
            self.counts["skipped_rgb_observations"] += 1
            return
        images = [observation[name] for name in CAMERAS]
        if (self.frame_pool is None or len(self.frames) >= self.frame_capacity
                or any(not isinstance(image, np.ndarray) or image.shape != (480, 640, 3)
                       or image.dtype != np.uint8 for image in images)):
            self.counts["frames_dropped"] += 1
            raise ValueError("Native recording requires reserved full 640×480 RGB triplets")
        slot = len(self.frames)
        for camera, image in enumerate(images):
            np.copyto(self.frame_pool[slot, camera], image)
        self.frames.append((observed_at, index))
        self.frame_by_observation[index] = slot
        if cadence_due:
            self.next_rgb_bin = cadence_bin + 1
            self.counts["cadence_rgb_frames"] += 1
        elif policy_input:
            self.counts["extra_policy_input_rgb_frames"] += 1
        if policy_input:
            self.counts["policy_input_rgb_frames"] += 1
        self.event("video_sample", frame_index=slot, observation_index=index,
                   observation_receipt_monotonic_s=observed_at, policy_input=policy_input,
                   recording_cadence_bin=cadence_bin,
                   sampling_reason=("cadence_and_policy_input" if cadence_due and policy_input
                                    else "cadence" if cadence_due else "policy_input"))

    def response(self, result, *, sequence_id, observation_index):
        if len(self.responses) >= 128:
            self.counts["chunks_dropped"] += 1
            return
        # Detach these small arrays from caller-owned storage. Raw normalized
        # model rows and decoded requests remain separate, including failures.
        self.responses.append({"sequence_id": sequence_id, "observation_index": observation_index,
                               "returned_monotonic_s": self.clock(), "result": copy.deepcopy(result),
                               "policy_input_marked": observation_index in self.policy_input_indices,
                               "rgb_frame_index": self.frame_by_observation.get(observation_index)})
        if self.capture_trace and (observation_index not in self.policy_input_indices
                                   or observation_index not in self.frame_by_observation):
            raise ValueError("Every recorded model response requires its exact marked input RGB")

    def _trace(self):
        responses = []
        for response in self.responses:
            record = {key: value for key, value in response.items() if key != "result"}
            supplied = response["result"]
            if not isinstance(supplied, dict):
                record["invalid_response_schema"] = True
                supplied = {}
            for field, shape, label in (("raw_normalized_chunk", (50, 32), "raw_normalized_rows"),
                                        ("chunk", (50, 14), "decoded_requested_rows")):
                try:
                    values = np.asarray(supplied.get(field))
                    if values.shape != shape or values.dtype.kind not in "fiu":
                        raise ValueError("Invalid official OpenPI response schema")
                    record[label + "_dtype"] = str(values.dtype)
                    record[label] = [[float(value) if np.isfinite(value) else None for value in row]
                                     for row in values]
                    record[label + "_nonfinite_values"] = int(np.count_nonzero(~np.isfinite(values)))
                except (TypeError, ValueError, OverflowError):
                    record[label] = None
                    record[label + "_invalid_numeric_schema"] = True
            record["audit"] = sanitize(supplied.get("audit", {}))
            responses.append(record)
        return {"schema": "yamkit_openpi_trace_v1", "controller_contract": CONTRACT,
                "policy_label": POLICY_LABEL,
                "events": self.events, "native_responses": responses,
                "chunks": [{"chunk_index": event["chunk_index"], "actions": event["rows"],
                            "raw_actions": event.get("raw_rows", event["rows"]),
                            "policy_observation_index": event.get("policy_observation_index"),
                            "observation_monotonic_s": event.get("observation_monotonic_s"),
                            "predicted_rows": event.get("predicted_rows"),
                            "committed_prefix_rows": event.get("committed_prefix_rows"),
                            "action_transform": event.get("action_transform"),
                            "gripper_conversions": event.get("gripper_conversions", []),
                            "audit": event.get("audit", {})}
                           for event in self.events if event["kind"] == "chunk_admitted"]}

    def finalize(self, directory, report, *, upload_repo_id=None, metadata=None):
        """Persist one completed lifecycle; never retry motion after any failure."""
        directory = repository_path(directory)
        if not directory.is_dir():
            raise ValueError("Native artifact directory must already exist")
        self.end()
        released = report.get("released") is True
        summary = {"status": "EXPORTING" if released else "EXPORT_SKIPPED_RESOURCES_OPEN",
                   "controller_mode": CONTRACT_ID, "task": self.task, "duration_s": self.duration_s,
                   "resources_released": released, "hardware_tested": report.get("hardware_tested") is True,
                   "synthetic_fixture": report.get("hardware_tested") is not True,
                   "qualification_evidence": False, "task_success": None,
                   "phase_started_monotonic_s": self.started, "phase_ended_monotonic_s": self.ended,
                   "action_names": list(YAM_NAMES), "camera_names": list(CAMERAS), "video_fps": CAPTURE_VIDEO_HZ,
                   "policy_label": POLICY_LABEL, "capture_capacity_hz": CAPTURE_VIDEO_HZ + CAPTURE_REQUEST_HZ,
                   "capture_scope": "30 Hz phase-bin RGB plus every policy input; all measured-state observations retained; no extra reads",
                   "rgb_sampling": {"cadence_hz": CAPTURE_VIDEO_HZ, "mandatory_policy_inputs": True,
                                    "maximum_request_hz": CAPTURE_REQUEST_HZ, "temporal_subsampling": True,
                                    "images_or_timestamps_synthesized": False,
                                    "omitted_rgb_counter": "skipped_rgb_observations"},
                   "frame_count": len(self.frames), "counts": self.counts,
                   "memory_preflight": self.memory_preflight, "capture_requested": self.capture_trace,
                   "trace_error_types": sorted(self.error_types), "video_export_errors": {},
                   "render_error_type": None, "report_available": False,
                   "overflow": bool(any(self.counts[key] for key in ("events_dropped", "frames_dropped", "chunks_dropped")))}
        report = {**report, "artifact_directory": str(directory), "capture": summary}
        # Safety evidence can be saved even when release failed; images, report
        # rendering, packaging and upload cannot start without confirmed release.
        _write(directory / "report.json", report)
        _write(directory / "summary.json", summary)
        _write(directory / "trace.json", self._trace())
        _write(directory / "metrics.json", {"controller_mode": CONTRACT_ID,
                                            "openpi_rollout": report,
                                            "openpi_execution": report.get("execution", {})})
        started_at = report.get("started_at", report.get("completed_at"))
        meta = {"id": directory.name, "kind": "rollout", "policy": POLICY_ID, "task": self.task,
                "status": ("success" if report.get("status") == "completed" else report.get("status", "failed")),
                "returncode": 0 if report.get("status") in ("completed", "stopped") else 1,
                "started_at": started_at, "ended_at": report.get("completed_at"), "active": False,
                "task_success": None, "hardware_tested": report.get("hardware_tested") is True,
                "duration_s": self.duration_s, "log_complete": False}
        _write(directory / "meta.json", meta)
        _write(directory / "run_metadata.json", {
            "model": {"id": "pi05_base", "checkpoint": CHECKPOINT, "runtime_revision": UPSTREAM_REVISION,
                      "label": POLICY_LABEL},
            "runtime": {"controller_contract": CONTRACT, "openpi_build_id": report.get("openpi_build_id"),
                        "instance_id": report.get("instance_id")},
            "capture": {"requested": self.capture_trace, "hardware_tested": report.get("hardware_tested") is True},
            "provenance": metadata or {}, "known_missing_data": [
                "Software/fake execution is not physical task success.",
                "Host receipt timestamps are not camera exposure timestamps.",
                "Non-policy RGB within an already populated 30 Hz phase bin is intentionally omitted; every state/timestamp is retained.",
                "Startup/home frames and complete process stdout/stderr are not captured."]})
        repository_path(directory / "log.txt").write_text("Official OpenPI π0.5 + experimental YAM lifecycle: " + str(meta["status"]) + "\n")
        if not released:
            _finalize_outcome(directory, summary, report, meta)
            return summary
        tools = _trace_tools()
        progress = tools.ExportProgress()
        try:
            with tools.wall_limit(tools.MAX_EXPORT_WALL_S):
                timeline = tools.video_timeline([at for at, _ in self.frames], self.started, self.ended,
                                                [index for _, index in self.frames])
                _write(directory / "frame_timestamps.json", [at for at, _ in self.frames])
                _write(directory / "video_timeline.json", timeline)
                if self.frames:
                    from lerobot.datasets.image_writer import write_image

                    for camera_index, camera in enumerate(CAMERAS):
                        try:
                            frames = directory / "frames" / camera
                            repository_path(frames).mkdir(parents=True, exist_ok=False)
                            for index in range(len(self.frames)):
                                write_image(self.frame_pool[index, camera_index], frames / f"frame-{index:06d}.png",
                                            compress_level=1)
                                progress.report("saving_frames", completed=index + 1, total=len(self.frames),
                                                unit="frames", camera=camera)
                            progress.report("encoding_videos", completed=camera_index, total=3,
                                            unit="videos", camera=camera, force=True)
                            tools.encode_timestamped_video((self.frame_pool[index, camera_index]
                                                            for index in range(len(self.frames))),
                                                           directory / f"{camera}.mp4", timeline)
                        except TimeoutError:
                            raise
                        except Exception as exc:  # noqa: BLE001 — retain originals after codec failures
                            summary["video_export_errors"][camera] = type(exc).__name__
                progress.report("rendering", force=True)
                _render_openpi_report(directory, summary, report)
                summary["report_available"] = True
            summary["status"] = ("TRACE_SAVED_WITH_EXPORT_ERRORS" if summary["video_export_errors"]
                                 or summary["trace_error_types"] or summary["overflow"] else "TRACE_SAVED")
        except Exception as exc:  # noqa: BLE001 — artifact failure never triggers physical work
            summary.update(status="EXPORT_FAILED", render_error_type=type(exc).__name__)
            _write(directory / "export-error.json", {"status": "EXPORT_FAILED", "error_type": type(exc).__name__,
                                                      "resources_released": True})
        finally:
            self.frame_pool = None
            # Packaging requires finalized local metadata. The immutable archive
            # may record upload_pending; the local history is updated after upload.
            _finalize_outcome(directory, summary, report, meta,
                              upload_pending=upload_repo_id is not None and summary["status"] == "TRACE_SAVED")
        if upload_repo_id is not None:
            if summary["status"] != "TRACE_SAVED":
                summary["upload"] = {"status": "failed", "repo_id": upload_repo_id}
                _write(directory / "hf-upload.json", {"status": "failed", "repo_id": upload_repo_id,
                                                       "error": "Export incomplete; local originals retained"})
            else:
                progress.report("uploading", force=True)
                try:
                    result = upload_rollout(package_openpi_rollout(directory), repo_id=upload_repo_id)
                    if not isinstance(result, dict) or result.get("status") not in ("uploaded", "already_uploaded"):
                        raise ValueError("Private upload did not confirm completion")
                    summary["upload"] = {key: result.get(key) for key in ("status", "repo_id", "revision")}
                except Exception as exc:  # noqa: BLE001 — no raw service diagnostics or automatic retry
                    summary["upload"] = {"status": "failed", "repo_id": upload_repo_id}
                    _write(directory / "hf-upload.json", {"status": "failed", "repo_id": upload_repo_id,
                                                           "error_type": type(exc).__name__,
                                                           "error": "Private upload failed; local originals retained"})
            _finalize_outcome(directory, summary, report, meta)
        progress.report("finalizing", force=True)
        return summary


def _render_openpi_report(directory, summary, report):
    """Native-only wording: never mislabel measured state as MA2 cached state."""
    videos = "".join('<figure><figcaption>' + camera + '</figcaption><video controls preload="metadata" src="'
                     + camera + '.mp4"></video><output>0.0 s</output></figure>'
                     for camera in CAMERAS if repository_path(directory / f"{camera}.mp4").is_file())
    execution = report.get("execution", {})
    metrics = html.escape(json.dumps(sanitize(execution), indent=2, allow_nan=False))
    task = html.escape(str(sanitize(summary["task"])))
    label = "Saved-observation / fake-arm software run" if summary["synthetic_fixture"] else "Supervised physical run"
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Official OpenPI π0.5 + experimental YAM rollout</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#111827}}
video{{width:100%;max-width:640px}}figure{{margin:16px 0}}output{{display:block;font-variant-numeric:tabular-nums}}pre{{overflow:auto}}
</style><h1>Official OpenPI π0.5 + experimental YAM rollout</h1><p>{label}. Process completion is not evidence of task success.</p><p>{task}</p>
<p>Measured 14D YAM state and chunk-origin joint deltas. Genuine native 50×32 predictions are decoded into 50×14 requests;
25 endpoints are committed at a maximum 50 Hz, and the remaining 25 rows are intentionally unused like the native ALOHA client.
Explicit non-learned YAM gripper endpoint conversion and speed-bounded linear transition substeps are recorded separately.
No model weights or native inference operations are changed; actuator time dilation is disclosed, not native ALOHA timing parity.
Requested and returned command targets are not proof of measured motion. See action-transform and incomplete-tail counters.</p>
<p>RGB records use a 30 Hz phase-bin cadence plus every exact model-input image triplet. All measured-state observations remain in the trace.
Omitted non-input RGB ticks are counted separately from capture errors. Selected images retain original host receipt timestamps and gaps;
no new images or timestamps are synthesized. These are not camera exposure timestamps.
Startup and return-home footage are not recorded. All saving and upload occur after confirmed release.</p>
<h2>Recording</h2>{videos or '<p>No camera recording was requested or available.</p>'}
<p><a href="summary.json">Recording summary</a> · <a href="trace.json">Raw predictions, transforms and command trace</a> ·
<a href="metrics.json">Lifecycle and full execution report</a> · <a href="video_timeline.json">Video timing</a></p>
<details><summary>Execution metrics</summary><pre>{metrics}</pre></details>
<script>for(const v of document.querySelectorAll('video')){{const o=v.nextElementSibling;
const tick=()=>{{o.textContent=v.currentTime.toFixed(1)+' s / '+(Number.isFinite(v.duration)?v.duration.toFixed(1):'—')+' s';}};
v.addEventListener('timeupdate',tick);v.addEventListener('loadedmetadata',tick);}}</script></html>'''
    repository_path(directory / "report.html").write_text(document)


def package_openpi_rollout(run_dir, *, trace_dir=None):
    """Native schema on the existing allowlisted, private-HF bundle contract."""
    run_dir = repository_path(run_dir)
    trace_dir = repository_path(trace_dir) if trace_dir is not None else run_dir
    summary, meta = _json(run_dir / "summary.json"), _json(run_dir / "meta.json")
    if (summary.get("controller_mode") != CONTRACT_ID or summary.get("resources_released") is not True
            or summary.get("status") != "TRACE_SAVED" or meta.get("active", False) is not False
            or meta.get("kind") != "rollout" or meta.get("status") not in ("success", "failed", "stopped")
            or type(meta.get("ended_at")) not in (int, float) or not math.isfinite(meta["ended_at"])
            or type(meta.get("returncode")) is not int):
        raise ValueError("Only released and finalized official OpenPI π0.5 recordings can be packaged")
    bundle = repository_path(run_dir / "bundle")
    if bundle.exists():
        validate_bundle(bundle)
        return bundle
    from huggingface_hub import get_token

    token = get_token()
    secrets = tuple({value for key, value in os.environ.items() if _SENSITIVE_KEY.search(key) and len(value) >= 4}
                    | ({token} if token else set()))
    with tempfile.TemporaryDirectory(prefix=".openpi-bundle-", dir=run_dir) as temporary:
        stage = Path(temporary)
        missing = []
        for name in ARTIFACTS:
            source = repository_path(run_dir / name)
            if not source.is_file():
                source = repository_path(trace_dir / name)
            if not source.is_file():
                if name in REQUIRED:
                    missing.append("Missing artifact: " + name)
                continue
            if source.suffix == ".json":
                _write(stage / name, sanitize(_json(source), secrets=secrets))
            elif source.suffix in (".txt", ".html"):
                from ..rollout_artifacts import _read, sanitize_text

                (stage / name).write_text(sanitize_text(_read(source).decode(), secrets=secrets))
            else:
                _copy(source, stage / name)
        frame_counts = {}
        for camera in CAMERAS:
            frames = repository_path(trace_dir / "frames" / camera)
            selected = sorted(frames.glob("frame-*.png")) if frames.is_dir() else []
            expected = [f"frame-{index:06d}.png" for index in range(summary["frame_count"])]
            if [path.name for path in selected] != expected:
                raise ValueError("Native original frame count or order differs from its finalized recording")
            frame_counts[camera] = len(selected)
            if selected:
                (stage / "frames" / camera).mkdir(parents=True)
                for source in selected:
                    _copy(source, stage / "frames" / camera / source.name)
        (stage / "README.md").write_text(
            "# Official OpenPI π0.5 + experimental YAM diagnostic recording\n\nOpen `report.html` for timestamped camera playback.\n"
            "`metrics.json.openpi_rollout` is the full lifecycle report; execution is not task success.\n"
            "`trace.json.native_responses` preserves raw native 50x32 and decoded 50x14 requests; `chunks` and `events`\n"
            "retain endpoint conversions, actuator transition substeps and explicit chunk/row dispatch joins.\n"
            "Measured chunk-origin state is used, not MA2 cached command state. `video_timeline.json` preserves original observation receipt gaps; original\n"
            "RGB PNGs remain under `frames/`. RGB is sampled at a 30 Hz phase-bin cadence plus every marked policy input;\n"
            "every measured-state observation remains in the trace, and omitted non-input RGB ticks are counted explicitly.\n"
            "Each native response joins its exact input observation and `rgb_frame_index`. Saved/fake evidence never proves physical success.\n"
            "All artifacts were finalized after confirmed release. Private upload preserves local originals.\n")
        manifest = {"schema_version": SCHEMA_VERSION, "run_id": meta["id"], "sanitized": True,
                    "missing_data": missing, "frame_counts": frame_counts, "files": {}}
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                manifest["files"][path.relative_to(stage).as_posix()] = {
                    "sha256": _hash(path), "size_bytes": path.stat().st_size}
        _write(stage / "manifest.json", manifest)
        validate_bundle(stage)
        stage.rename(bundle)
    return bundle
