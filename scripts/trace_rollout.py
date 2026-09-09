"""Plan by default; an explicitly approved run traces the unchanged physical rollout.

This helper does not provision cloud compute, bypass qualification, or open extra
cameras/arms. --run DOES energize/home both followers and run policy actions.
All image persistence/encoding and full metrics export happen after robot release.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

TASK = "put the red cube into the black container"
CAMERAS = ("top", "left_wrist", "right_wrist")
ACTION_NAMES = tuple(f"{side}_{joint}.pos" for side in ("left", "right")
                     for joint in (*[f"joint_{i}" for i in range(1, 7)], "gripper"))
VIDEO_FPS = 30
MAX_DURATION_S = 30
VIDEO_TIME_BASE = Fraction(1, 1_000_000)
FRAME_TRIPLET_BYTES = 3 * 480 * 640 * 3
MEMORY_HEADROOM_BYTES = 512 * 1024 * 1024
MAX_EVENTS = 8192  # Covers 30 seconds of observation, shaping and both arm-send events.
MAX_CHUNKS = 128
MAX_ROLLOUT_WALL_S = 120
MAX_EXPORT_WALL_S = 90


def plan(duration=5, controller_mode="async"):
    return {"status": "PLAN_ONLY", "hardware_opened": False, "task": TASK,
            "duration_s": duration, "controller_mode": controller_mode,
            "maximum_rollout_wall_s": MAX_ROLLOUT_WALL_S,
            "maximum_export_wall_s": MAX_EXPORT_WALL_S,
            "video_fps": VIDEO_FPS, "max_frame_triplets": duration * VIDEO_FPS + 3,
            "max_frame_bytes": (duration * VIDEO_FPS + 3) * FRAME_TRIPLET_BYTES,
            "memory_headroom_bytes": MEMORY_HEADROOM_BYTES,
            "max_events": MAX_EVENTS, "max_chunks": MAX_CHUNKS,
            "observations": "Existing robot observations; local receipt timestamps, not camera exposure",
            "commands": "Requested and completed per-side post-clamp targets; failed sends are retained",
            "production_guards_unchanged": True,
            "guard_scope": "Capture leaves the selected controller unchanged; reference explicitly disables policy speed/acceleration shaping",
            "policy_speed_clamp_enabled": controller_mode != "reference", "qualification_evidence": False,
            "video_encoding": "Exact RGB PNGs plus near-lossless H.264 with observation-timestamp PTS, after release",
            "run_effects": "Connect cameras, energize/home both followers, run policy phase, return home after healthy duration completion, then release; Stop or a fault aborts movement and releases",
            "capture_scope": "Policy phase only; startup and return-home movement are not recorded"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--duration", type=int, choices=(5, 10, 20, MAX_DURATION_S), default=5)
    parser.add_argument("--modal-app")
    parser.add_argument("--backend", choices=("modal", "external"), default="modal")
    parser.add_argument("--external-service")
    parser.add_argument("--controller-mode", choices=("async", "reference"), default="async")
    parser.add_argument("--confirm-supervised", action="store_true")
    parser.add_argument("--rig", type=Path, help="Selected rig inside this repository; normal CLI validation still applies")
    parser.add_argument("--output-dir", type=Path, help="New directory inside this repository's .context/rollout-traces")
    args = parser.parse_args(argv)
    if args.backend == "external":
        if (args.modal_app or not args.external_service
                or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", args.external_service) is None):
            parser.error("external backend requires an exact --external-service without --modal-app")
    elif args.external_service:
        parser.error("--external-service requires --backend external")
    if args.run and (not args.confirm_supervised or (args.backend == "modal" and (
            not args.modal_app or re.fullmatch(r"yamkit-vla-[a-z0-9-]{1,80}", args.modal_app) is None))):
        parser.error("--run requires an exact owned service and --confirm-supervised")
    return args


def available_memory_bytes(meminfo=Path('/proc/meminfo'), cgroup_root=Path('/sys/fs/cgroup'),
                           process_cgroup=Path('/proc/self/cgroup')):
    """Read Linux host and applicable cgroup limits; no allocations or device calls."""
    lines = meminfo.read_text().splitlines()
    candidates = [int(line.split()[1]) * 1024 for line in lines if line.startswith('MemAvailable:')]
    if len(candidates) != 1 or candidates[0] <= 0:
        raise ValueError('Linux MemAvailable is required before reserving trace frames')
    if process_cgroup.is_file():
        for line in process_cgroup.read_text().splitlines():
            _, controllers, relative = line.split(':', 2)
            if controllers == '':
                base, limit_name, used_name = cgroup_root, 'memory.max', 'memory.current'
            elif 'memory' in controllers.split(','):
                base, limit_name, used_name = cgroup_root / 'memory', 'memory.limit_in_bytes', 'memory.usage_in_bytes'
            else:
                continue
            current = base / relative.lstrip('/')
            if '..' in current.parts:
                continue
            while current.is_relative_to(base):
                limit, used = current / limit_name, current / used_name
                if limit.is_file() and used.is_file():
                    value = limit.read_text().strip()
                    if value != 'max':
                        candidates.append(max(0, int(value) - int(used.read_text().strip())))
                if current == base:
                    break
                current = current.parent
    return min(candidates)


class Collector:
    def __init__(self, duration, *, clock=time.monotonic, controller_mode="async"):
        self.duration, self.clock = duration, clock
        self.controller_mode = controller_mode
        self.events, self.frames, self.chunks, self.robots = [], [], [], []
        self.metrics = None
        self.rollout_error = None
        self.active = False
        self.phase_started = self.phase_ended = None
        self.frame_capacity = duration * VIDEO_FPS + 3
        self.frame_pool = None
        self.frame_observation_indices = []
        self.memory_preflight = None
        self.lock = threading.RLock()
        self.counts = {"events_dropped": 0, "frames_dropped": 0, "chunks_dropped": 0,
                       "observation_frames_seen": 0, "trace_errors": 0, "reference_row_observations": 0}
        self.trace_error_types = set()

    def safely(self, operation, *args, **kwargs):
        """Instrumentation failures cannot change command results or mask faults."""
        try:
            return operation(*args, **kwargs)
        except TimeoutError:
            raise  # Preserve the helper's absolute wall alarm and normal runner cleanup.
        except Exception as exc:  # noqa: BLE001 — trace failures are counted, never propagated into control.
            with self.lock:
                self.counts["trace_errors"] += 1
                if len(self.trace_error_types) < 16:
                    self.trace_error_types.add(type(exc).__name__)
            return None

    def event(self, kind, **values):
        with self.lock:
            if len(self.events) >= MAX_EVENTS:
                self.counts["events_dropped"] += 1
                return
            self.events.append({"kind": kind, "monotonic_s": self.clock(), **values})

    def row_observation(self, observation):
        with self.lock:
            self.counts["reference_row_observations"] += 1
            self.event("reference_row_observation",
                       positions={name: float(observation[name]) for name in ACTION_NAMES},
                       rgb_retained=False,
                       purpose="unused row-anchor camera read; policy uses post-step observation")

    def start_phase(self):
        self.phase_started = self.clock()
        self.active = True
        self.event("policy_phase_started")

    def reserve_frames(self):
        import numpy as np

        needed = self.frame_capacity * FRAME_TRIPLET_BYTES
        self.memory_preflight = {"available_bytes": available_memory_bytes(), "frame_bytes": needed,
                                 "headroom_bytes": MEMORY_HEADROOM_BYTES}
        if self.memory_preflight['available_bytes'] < needed + MEMORY_HEADROOM_BYTES:
            raise MemoryError('Insufficient available memory for all 30 Hz observation frames')
        # Prefault the complete bounded pool before the normal CLI can connect
        # hardware. Observation callbacks copy into these slots without another
        # retained RGB allocation or any compression/disk work.
        self.frame_pool = np.empty((self.frame_capacity, 3, 480, 640, 3), dtype=np.uint8)
        self.frame_pool.fill(0)

    def end_phase(self):
        self.active = False
        self.phase_ended = self.clock()
        self.event("policy_phase_ended")

    def observation(self, obs):
        if not self.active:
            return
        import numpy as np

        started = self.clock()
        observation_index = self.counts['observation_frames_seen']
        self.counts['observation_frames_seen'] += 1
        self.event("observation", observation_index=observation_index,
                   positions=[float(obs[key]) for key in ACTION_NAMES])
        if len(self.frames) >= self.frame_capacity:
            self.counts["frames_dropped"] += 1
            return
        images = [obs[name] for name in CAMERAS]
        if any(not isinstance(frame, np.ndarray) or frame.shape != (480, 640, 3)
               or frame.dtype != np.uint8 for frame in images):
            self.counts["frames_dropped"] += 1
            raise ValueError("Unexpected trace image shape or dtype")
        # Only bounded copies here; no compression, disk I/O, extra reads or video workers.
        if self.frame_pool is None:  # Hardware-free collectors/tests can remain lazily allocated.
            copied = tuple(frame.copy() for frame in images)
        else:
            copied = tuple(self.frame_pool[len(self.frames), index] for index in range(len(CAMERAS)))
            for destination, frame in zip(copied, images, strict=True):
                np.copyto(destination, frame)
        self.frames.append((started, copied))
        self.frame_observation_indices.append(observation_index)
        self.event("video_sample", frame_index=len(self.frames) - 1,
                   observation_index=observation_index,
                   observation_receipt_monotonic_s=started, copy_s=self.clock() - started)

    def capture_chunk(self, result, observation_time, policy_state=None):
        if not self.active:
            return
        if len(self.chunks) >= MAX_CHUNKS:
            self.counts["chunks_dropped"] += 1
            return
        if tuple(result.shape) != (1, 30, 14) or result.device.type != "cpu":
            raise ValueError("Trace requires the unchanged CPU 30x14 policy boundary")
        self.chunks.append({"chunk_index": len(self.chunks), "returned_monotonic_s": self.clock(),
                            "observation_monotonic_s": observation_time,
                            "policy_state": (policy_state.detach().cpu().numpy().copy()[0]
                                             if policy_state is not None else None),
                            "actions": result.detach().numpy().copy()[0]})

    def register_robot(self, robot):
        if all(robot is not existing for existing in self.robots):
            self.robots.append(robot)

    def released(self):
        return all(all(h.arm is None for h in robot._sides.values())
                   and not robot._opened_cameras and robot._camera_lease is None
                   for robot in self.robots)


@contextmanager
def install_hooks(collector):
    """Scoped observation hooks; every production operation is called exactly once."""
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot_robot_yamkit.yam_follower import BiYamFollower

    from yamkit import cli
    from yamkit.arm import YamArm
    from yamkit.reference_rollout import ReferenceRemoteInferenceEngine
    from yamkit.reference_strategy import ReferenceStrategy
    from yamkit.remote_policy.modeling_yamkit_remote import YamkitRemotePolicy
    from yamkit.remote_rollout import InvalidatableActionQueue, UnguidedRemoteInferenceEngine

    original_run = BaseStrategy.run
    original_reference_run = ReferenceStrategy.run
    original_connect = BiYamFollower.connect
    original_observation = BiYamFollower.get_observation
    original_send = YamArm.command
    original_predict = YamkitRemotePolicy.predict_action_chunk
    original_merge = UnguidedRemoteInferenceEngine._record_merge
    original_get = InvalidatableActionQueue.get
    original_reference_event = ReferenceRemoteInferenceEngine.trace_event

    def run(strategy, ctx):
        collector.safely(collector.start_phase)
        try:
            return original_run(strategy, ctx)
        finally:
            collector.safely(collector.end_phase)

    def connect(robot, *args, **kwargs):
        collector.register_robot(robot)
        return original_connect(robot, *args, **kwargs)

    def observation(robot):
        result = original_observation(robot)
        if getattr(robot, "_reference_observation_role", None) == "row_anchor":
            if collector.active:
                collector.safely(collector.row_observation, result)
        else:
            collector.safely(collector.observation, result)
        return result

    def run_reference(strategy, ctx):
        collector.safely(collector.start_phase)
        try:
            return original_reference_run(strategy, ctx)
        finally:
            collector.safely(collector.end_phase)

    def send(arm, q, gripper=None, *, limit_speed=True):
        active = collector.active
        if active:
            collector.safely(lambda: collector.event("send_start", arm=arm.name,
                requested={**{f"joint_{i + 1}.pos": float(value) for i, value in enumerate(q)},
                           **({"gripper.pos": float(gripper)} if gripper is not None else {})},
                speed_clamp_enabled=limit_speed))
        try:
            result = original_send(arm, q, gripper, limit_speed=limit_speed)
        except BaseException as exc:
            if active:
                collector.safely(collector.event, "send_error", arm=arm.name,
                                 error_type=type(exc).__name__, partial_dispatch_possible=True)
            raise
        if active:
            collector.safely(lambda: collector.event("send_end", arm=arm.name,
                postclamp={f"{name}.pos": float(value) for name, value in zip(
                    [*[f"joint_{i + 1}" for i in range(6)], "gripper"], result, strict=False)},
                speed_clamp_enabled=limit_speed))
        return result

    def predict(policy, *args, **kwargs):
        observation_time = policy._observation_time
        batch = args[0] if args else kwargs.get("batch", {})
        result = original_predict(policy, *args, **kwargs)
        collector.safely(collector.capture_chunk, result, observation_time, batch.get("observation.state"))
        return result

    def merge(engine, metrics):
        result = original_merge(engine, metrics)
        collector.safely(collector.event, "chunk_merge", **metrics)
        return result

    def get(queue):
        result = original_get(queue)
        if result is not None and collector.active:
            collector.safely(collector.event, "action_dequeued", deadline_monotonic_s=queue.last_action_deadline)
        return result

    def result(metrics):
        # CLI invokes this only after run_remote_rollout's complete cleanup.
        collector.metrics = metrics

    def reference_event(engine, kind, **values):
        result = original_reference_event(engine, kind, **values)
        if collector.active:
            collector.safely(collector.event, kind, **values)
        return result

    with ExitStack() as stack:
        for target, name, replacement in (
            (BaseStrategy, "run", run), (ReferenceStrategy, "run", run_reference),
            (BiYamFollower, "connect", connect),
            (BiYamFollower, "get_observation", observation), (YamArm, "command", send),
            (YamkitRemotePolicy, "predict_action_chunk", predict),
            (UnguidedRemoteInferenceEngine, "_record_merge", merge),
            (InvalidatableActionQueue, "get", get), (cli, "_print_inference_result", result),
            (ReferenceRemoteInferenceEngine, "trace_event", reference_event),
        ):
            stack.enter_context(patch.object(target, name, replacement))
        yield


def write_json(path, value):
    def array(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        raise TypeError("Trace export contains an unsupported value")

    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".trace-json-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(value, default=array, allow_nan=False) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def wall_limit(seconds):
    def expired(*args):
        raise TimeoutError("Bounded trace phase expired")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def video_timeline(timestamps, phase_start, phase_end, observation_indices=None):
    """One presentation entry per original observation; never resample across gaps."""
    if not timestamps:
        return {"nominal_fps": VIDEO_FPS, "time_base": [1, 1_000_000], "frames": [],
                "origin_monotonic_s": None, "policy_phase_offset_s": None, "duration_s": 0.0}
    if (len(timestamps) > MAX_DURATION_S * VIDEO_FPS + 3
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in [phase_start, phase_end, *timestamps])
            or phase_end <= phase_start or phase_end - phase_start > MAX_ROLLOUT_WALL_S
            or timestamps[0] < phase_start or timestamps[-1] >= phase_end):
        raise ValueError('Video timestamps must lie within the bounded recorded policy phase')
    origin = timestamps[0]
    points = [round((at - origin) / VIDEO_TIME_BASE) for at in timestamps]
    end = round((phase_end - origin) / VIDEO_TIME_BASE)
    if any(right <= left for left, right in pairwise(points)) or end <= points[-1]:
        raise ValueError('Video timestamps must be strictly increasing at microsecond precision')
    indices = list(range(len(points))) if observation_indices is None else observation_indices
    if (len(indices) != len(points) or any(type(index) is not int or index < 0 for index in indices)
            or any(right <= left for left, right in pairwise(indices))):
        raise ValueError('Video observation indices must preserve their original order')
    return {"nominal_fps": VIDEO_FPS, "time_base": [1, 1_000_000], "origin_monotonic_s": origin,
            "policy_phase_offset_s": origin - phase_start, "duration_s": end * float(VIDEO_TIME_BASE),
            "timestamp_basis": "Original observation receipt; not camera exposure",
            "gap_behavior": "Previous image remains displayed until the next original observation; no invented frames",
            "frames": [{"source_index": index, "observation_index": indices[index],
                        "receipt_monotonic_s": timestamps[index], "pts": point,
                        "duration_ticks": (points[index + 1] if index + 1 < len(points) else end) - point}
                       for index, point in enumerate(points)]}


def encode_timestamped_video(images, video_path, timeline):
    """Use LeRobot's codec configuration and PyAV for the missing variable-PTS API.

    LeRobot encode_video_frames and StreamingVideoEncoder use frame count / FPS,
    with no caller-supplied timestamps. Keep their H.264 configuration but assign
    original PTS and durations here so irregular observations cannot speed up.
    """
    import av
    from lerobot.configs.video import RGBEncoderConfig

    rows = timeline['frames']
    if not rows:
        raise ValueError('Video encoding requires recorded observation frames')
    configuration = RGBEncoderConfig(vcodec='h264', crf=12, preset='veryfast')
    iterator = iter(images)
    with av.open(str(video_path), 'w') as output:
        stream = output.add_stream(configuration.vcodec, rate=VIDEO_FPS,
                                   options=configuration.get_codec_options(1, as_strings=True))
        stream.pix_fmt = configuration.pix_fmt
        stream.time_base = stream.codec_context.time_base = VIDEO_TIME_BASE
        stream.codec_context.max_b_frames = 0
        durations = {row['pts']: row['duration_ticks'] for row in rows}
        encoded_points = set()

        def mux(packets):
            for packet in packets:
                # With B-frames disabled there is one encoded access unit per
                # input frame; reject unexpected timestamp conversion explicitly.
                point = round(packet.pts * packet.time_base / VIDEO_TIME_BASE)
                if point not in durations or point in encoded_points:
                    raise ValueError('Encoder returned an unknown observation timestamp')
                encoded_points.add(point)
                packet.duration = round(durations[point] * VIDEO_TIME_BASE / packet.time_base)
                output.mux(packet)

        for index, row in enumerate(rows):
            try:
                image = next(iterator)
            except StopIteration:
                raise ValueError('Video frame count differs from its observation timeline') from None
            if index == 0:
                stream.height, stream.width = image.shape[:2]
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts, frame.time_base = row['pts'], VIDEO_TIME_BASE
            mux(stream.encode(frame))
        if next(iterator, None) is not None:
            raise ValueError('Video contains more images than its observation timeline')
        mux(stream.encode())
        if encoded_points != set(durations):
            raise ValueError('Encoder did not preserve every observation frame')


def export(collector, outdir):
    """Only call after rollout unwound; no image writes until resources are released."""
    released = collector.released()
    summary = {"status": "EXPORTING", "qualification_evidence": False,
               "instrumented_rollout": True, "task": TASK, "duration_s": collector.duration,
               "resources_released": released, "production_guards_unchanged": True,
               "guard_scope": "Capture leaves the selected controller unchanged; reference explicitly disables policy speed/acceleration shaping",
               "policy_speed_clamp_enabled": collector.controller_mode != "reference",
               "auxiliary_observation_scope": "Reference row-anchor reads retain timestamps and measured state, not unused RGB; all policy-selected/post-step RGB observations are captured",
               "rollout_error": collector.rollout_error,
               "phase_started_monotonic_s": collector.phase_started,
               "phase_ended_monotonic_s": collector.phase_ended,
               "action_names": ACTION_NAMES, "camera_names": CAMERAS,
               "timestamps": "Host monotonic receipt/dispatch times; camera exposure unobserved",
               "counts": collector.counts, "trace_error_types": sorted(collector.trace_error_types),
               "memory_preflight": collector.memory_preflight,
               "overflow": any(collector.counts[key] for key in ("events_dropped", "frames_dropped", "chunks_dropped")),
               "frame_count": len(collector.frames), "video_fps": VIDEO_FPS,
               "observation_frames_missing": max(0, collector.counts['observation_frames_seen'] - len(collector.frames)),
               "video_timing": "Every captured observation has its original variable presentation timestamp; video_timeline.json preserves gaps and the policy-phase offset",
               "video_quality": "Near-lossless H.264 CRF 12, yuv420p; exact original RGB PNGs retained",
               "capture_scope": "Policy phase only; startup and return-home movement are not recorded",
               "full_metrics_available": collector.metrics is not None, "video_export_errors": {},
               "controller_mode": (collector.metrics or {}).get("controller_mode", collector.controller_mode),
               "report_available": False, "render_error_type": None}
    write_json(outdir / "trace.json", {"events": collector.events, "chunks": collector.chunks})
    write_json(outdir / "frame_timestamps.json", [at for at, _ in collector.frames])
    if collector.metrics is not None:
        write_json(outdir / "metrics.json", collector.metrics)
    write_json(outdir / "summary.json", summary)
    if not released:
        summary["status"] = "EXPORT_SKIPPED_RESOURCES_OPEN"
        write_json(outdir / "summary.json", summary)
        return summary
    from lerobot.datasets.image_writer import write_image

    timeline = video_timeline([at for at, _ in collector.frames], collector.phase_started, collector.phase_ended,
                              collector.frame_observation_indices or None)
    write_json(outdir / "video_timeline.json", timeline)
    for camera_index, name in enumerate(CAMERAS if collector.frames else ()):
        try:
            directory = outdir / "frames" / name
            directory.mkdir(parents=True)
            for index, (_, images) in enumerate(collector.frames):
                write_image(images[camera_index], directory / f"frame-{index:06d}.png", compress_level=1)
            encode_timestamped_video((images[camera_index] for _, images in collector.frames),
                                     outdir / f"{name}.mp4", timeline)
        except TimeoutError:
            raise  # The export alarm is absolute; never continue into another encoder after it fires.
        except Exception as exc:  # noqa: BLE001 — artifacts survive an encoder error after hardware release.
            summary["video_export_errors"][name] = type(exc).__name__
    write_json(outdir / "summary.json", summary)
    try:
        render_report(outdir)
        summary["report_available"] = True
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001 — rendering cannot alter the completed arm run or its saved evidence.
        summary["render_error_type"] = type(exc).__name__
    summary["status"] = ("TRACE_SAVED_WITH_EXPORT_ERRORS"
                         if summary["video_export_errors"] or summary["render_error_type"] else "TRACE_SAVED")
    write_json(outdir / "summary.json", summary)
    return summary


def render_report(outdir):
    # A fresh, bounded process avoids inherited Matplotlib caches and globals.
    completed = subprocess.run([sys.executable, str(Path(__file__).with_name("render_rollout_trace.py")),
                                str(outdir)], capture_output=True, timeout=45, check=False)
    if completed.returncode or not (outdir / "report.html").is_file():
        raise RuntimeError("Saved trace report rendering failed")


def rollout_arguments(args):
    result = ["rollout", "--policy", "molmoact2", "--backend", args.backend, "--call-mode", "http",
            "--execution-mode", "cuda_graph10", "--task", TASK, "--arms", "left_follower",
            "--arms", "right_follower", "--duration", str(args.duration),
            "--accept-mapping", "--confirm-supervised"]
    if getattr(args, "controller_mode", "async") != "async":
        result.extend(["--controller-mode", args.controller_mode])
    if args.backend == "external":
        result.extend(["--external-service", args.external_service])
    else:
        result.extend(["--modal-app", args.modal_app])
    if args.rig is not None:
        result.extend(["--rig", str(args.rig)])
    return result


def artifact_directory(root, requested, stamp):
    root = root.resolve()
    base = (root / ".context" / "rollout-traces").resolve()
    selected = (requested if requested is not None else base / stamp).resolve()
    if (not base.is_relative_to(root) or not selected.is_relative_to(base)
            or selected == base or selected.exists()):
        raise ValueError("Trace output must be a new directory inside this repository's .context/rollout-traces")
    return selected


def execute(args):
    from yamkit import cli
    from yamkit.paths import ROOT

    # LeRobot's import-time logging.debug can install a default WARNING handler
    # before Typer's callback, making its later basicConfig(INFO) a no-op. Set up
    # the normal CLI filters before hook imports and explicitly select this
    # helper's normal INFO verbosity without replacing inherited handlers.
    cli._setup_logging(verbose=False)
    logging.getLogger().setLevel(logging.INFO)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    outdir = artifact_directory(ROOT, args.output_dir, stamp)
    if args.rig is not None:
        args.rig = args.rig.resolve()
        if not args.rig.is_relative_to(ROOT.resolve()) or not args.rig.is_file():
            raise ValueError("Selected rig must be an existing file inside this repository")
    outdir.mkdir(parents=True, exist_ok=False)
    collector = Collector(args.duration, controller_mode=args.controller_mode)
    write_json(outdir / "plan.json", {**plan(args.duration, args.controller_mode), "argv": rollout_arguments(args)})
    status, error_type = 0, None
    try:
        with install_hooks(collector), wall_limit(MAX_ROLLOUT_WALL_S):
            collector.reserve_frames()
            cli.app(args=rollout_arguments(args), standalone_mode=False)
    except BaseException as exc:  # noqa: BLE001 — preserve partial trace after normal runner cleanup.
        from yamkit.rollout_artifacts import sanitize_text

        # The runner attaches complete metrics after fault cleanup. Exceptions
        # other than RemoteFault do not pass through the CLI's result printer.
        attached_metrics = getattr(exc, "metrics", None)
        if isinstance(attached_metrics, dict) and (attached_metrics or collector.metrics is None):
            collector.metrics = attached_metrics
        status, error_type = 1, type(exc).__name__
        collector.rollout_error = {"type": error_type, "message": sanitize_text(str(exc))[:2048]}
        logging.getLogger(__name__).error(
            "Rollout failed before artifact export:\n%s",
            sanitize_text("".join(traceback.format_exception(exc)))[-12000:])
    try:
        with wall_limit(MAX_EXPORT_WALL_S):
            summary = export(collector, outdir)
    except BaseException as exc:  # noqa: BLE001 — export errors cannot trigger another robot operation.
        try:
            summary = json.loads((outdir / "summary.json").read_text())
        except (OSError, ValueError):
            summary = {}
        summary.update(status="EXPORT_FAILED", error_type=type(exc).__name__,
                       controller_mode=args.controller_mode,
                       resources_released=collector.released(), rollout_error=collector.rollout_error)
        write_json(outdir / "summary.json", summary)
        write_json(outdir / "export-error.json", summary)
        status = 1
    if not summary["resources_released"]:
        status = 1
    print(json.dumps({"trace_directory": str(outdir), "rollout_error_type": error_type,
                      "resources_released": summary["resources_released"], "exit_status": status}), flush=True)
    return status


def main(argv=None):
    args = parse_args(argv)
    if not args.run:
        print(json.dumps(plan(args.duration, args.controller_mode)), flush=True)
        return 0
    return execute(args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    raise SystemExit(main())
