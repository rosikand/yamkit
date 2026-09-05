"""Narrow unguided async adapter around LeRobot 0.6.1's actual rollout components.

The upstream factory gates *all* RTCInferenceConfig instances on guidance support,
even with guidance disabled. Build its sync context normally, then substitute an
RTCInferenceEngine subclass with guidance disabled. Its worker and the upstream
strategy/action-dispatch loop are reused unchanged. No installed files are patched.
"""

from __future__ import annotations

import math
import time
from collections import deque
from threading import Event, RLock

import torch
from lerobot.policies.rtc import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.rollout import build_rollout_context, create_strategy
from lerobot.rollout.configs import BaseStrategyConfig
from lerobot.rollout.inference import SyncInferenceConfig
from lerobot.rollout.inference.rtc import RTCInferenceEngine
from lerobot.rollout.robot_wrapper import ThreadSafeRobot

from yamkit.inference.client import RemoteFault
from yamkit.remote_policy import YamkitRemoteConfig


class InvalidatableActionQueue(ActionQueue):
    """Upstream action queue with a finite size and permanent invalidation."""

    def __init__(self, *, max_steps: int, max_age_s: float, observation_time=None, on_fault=None,
                 on_depth=None, fps: float = 30.0, on_merge=None):
        super().__init__(RTCConfig(enabled=False))
        self.lock = RLock()
        self.max_steps = max_steps
        self.max_age_s = max_age_s
        self.fps = fps
        self.valid = True
        self.inserted_at = None
        self._observation_time = observation_time or time.monotonic
        self._deadlines = []
        self.last_action_deadline = None
        self._on_fault = on_fault
        self._on_depth = on_depth
        self._on_merge = on_merge
        self.expired_prefix_dropped = 0
        self.overlap_prefix_dropped = 0
        self.expired_chunks = 0
        self.expired_queued_actions = 0
        self.redundant_chunks = 0

    def timing_snapshot(self, now=None):
        """Report actual queued deadlines, separately from depth divided by FPS."""
        with self.lock:
            now = time.monotonic() if now is None else now
            deadlines = self._deadlines[self.last_index:]
            return {"depth": self.qsize(),
                    "deadline_horizon_s": max(0.0, deadlines[-1] - now) if deadlines else 0.0,
                    "next_action_margin_s": deadlines[0] - now if deadlines else None}

    def invalidate(self):
        with self.lock:
            self.valid = False
            self.queue = self.original_queue = None
            self.last_index = 0
            self._deadlines.clear()

    def merge(self, original_actions, processed_actions, real_delay, action_index_before_inference=None):
        try:
            with self.lock:
                if not self.valid:
                    return
                now = time.monotonic()
                observation_time = self._observation_time()
                remaining = 0 if self.queue is None else len(self.queue) - self.last_index
                if not torch.isfinite(processed_actions).all():
                    raise RemoteFault("Nonfinite remote action queue")
                if len(processed_actions) > self.max_steps:
                    raise RemoteFault("Remote action queue capacity exceeded")
                # A chunk starts at its observation, not at RPC completion.
                # Preserve queued commands and skip the corresponding overlapping
                # future prefix too; appending it would shift old targets later.
                elapsed_steps = max(real_delay, math.ceil(max(0.0, now - observation_time) * self.fps))
                expired = min(len(processed_actions), elapsed_steps)
                overlap = min(len(processed_actions) - expired, remaining)
                dropped = expired + overlap
                self.expired_prefix_dropped += expired
                self.overlap_prefix_dropped += overlap
                available = len(processed_actions) - dropped
                old_deadline = self._deadlines[-1] if remaining else now
                new_deadline = min(observation_time + self.max_age_s,
                                   observation_time + len(processed_actions) / self.fps) if available else now
                if self._on_merge is not None:
                    self._on_merge({"queue_depth_at_merge": remaining,
                                    "queue_deadline_horizon_at_merge_s": max(0.0, old_deadline - now),
                                    "expired_prefix_dropped": expired,
                                    "overlap_prefix_dropped": overlap,
                                    "accepted_steps": available,
                                    "remaining_valid_action_horizon_s": max(0.0, min(
                                        len(processed_actions) / self.fps, self.max_age_s)
                                        - (now - observation_time)),
                                    "queue_horizon_after_merge_s": (remaining + available) / self.fps,
                                    "queue_deadline_horizon_after_merge_s": max(0.0,
                                        max(old_deadline, new_deadline) - now)})
                if observation_time + self.max_age_s <= now or expired == len(processed_actions):
                    self.expired_chunks += 1
                    raise RemoteFault("Remote chunk expired: no valid future actions remain")
                if available == 0:
                    # An immediate prediction can finish before the main thread
                    # consumes its next tick. Its entirely overlapping fresh
                    # prefix adds nothing; existing deadlines remain untouched.
                    self.redundant_chunks += 1
                    return
                if remaining + available > self.max_steps:
                    raise RemoteFault("Remote action queue capacity exceeded")
                self._deadlines = self._deadlines[self.last_index:] + [
                    min(observation_time + self.max_age_s, observation_time + (i + 1) / self.fps)
                    for i in range(dropped, len(processed_actions))]
                # This calls the upstream append operation under its own expected lock.
                self._append_actions_queue(original_actions[dropped:], processed_actions[dropped:])
                self.inserted_at = now
                if self._on_depth is not None:
                    self._on_depth(len(self.queue) - self.last_index)
        except Exception:
            # Call outside the queue lock: fault handling invalidates this queue.
            if self._on_fault is not None:
                self._on_fault()
            raise

    def get(self):
        with self.lock:
            if not self.valid:
                return None
            if self.last_index < len(self._deadlines) and time.monotonic() >= self._deadlines[self.last_index]:
                self.expired_queued_actions += 1
                raise RemoteFault("Queued remote actions expired")
            self.last_action_deadline = self._deadlines[self.last_index] if self.last_index < len(self._deadlines) else None
            return super().get()


class _ObservationSlot(dict):
    """Timestamp the exact snapshot read by the inherited upstream worker."""

    def __init__(self, policy, robot_type):
        super().__init__(obs=None, robot_type=robot_type)
        self.policy = policy
        self.timestamp = None

    def get(self, key, default=None):
        value = super().get(key, default)
        if key == "obs" and value is not None:
            if self.timestamp == self.policy._last_requested_observation_time:
                return None  # Exactly one fresh observation per serial prediction.
            self.policy._observation_time = self.timestamp
            self.policy._observation_selected_time = time.monotonic()
        return value


class UnguidedRemoteInferenceEngine(RTCInferenceEngine):
    """Keep RPC waits in the upstream background worker; fail closed on underrun."""

    def __init__(self, *, policy, preprocessor, postprocessor, robot_wrapper, hw_features, task, fps,
                 shutdown_event):
        threshold = policy.config.prediction_queue_threshold
        if threshold is None:
            threshold = policy.profile.chunk_size
        # Prefetch timing must not enlarge the established queue safety bound.
        self.max_steps = policy.profile.chunk_size + max(1, policy.profile.chunk_size // 2)
        self.max_age_s = policy.config.max_observation_age_s
        self.startup_timeout_s = policy.config.request_timeout_s
        self.underruns = 0
        self.peak_queue_depth = 0
        self.last_queue_depth_before_stop = 0
        self.executed_actions = 0
        self.dequeued_actions = 0
        self.minimum_execution_queue_depth = None
        self.minimum_dispatch_margin_s = None
        self.expired_before_dispatch = 0
        self.stop_detected_at = None
        self.robot_released_at = None
        self.predictions = deque(maxlen=1000)
        self._ever_had_action = False
        self._started_at = None
        super().__init__(policy, preprocessor, postprocessor, robot_wrapper, RTCConfig(enabled=False),
                         hw_features, task, fps, "cpu", use_torch_compile=False,
                         rtc_queue_threshold=threshold, shutdown_event=shutdown_event)
        policy.on_fault = self._fault
        policy.on_prediction_start = self._prediction_start
        policy.on_prediction_end = self._prediction_end

    def _prediction_start(self):
        now = time.monotonic()
        queue = self.action_queue.timing_snapshot(now)
        depth = queue["depth"]
        event = {"prediction_started_monotonic_s": now,
                 "observation_timestamp_monotonic_s": self._policy._observation_time,
                 "observation_age_at_start_s": now - self._policy._observation_time,
                 "observation_processing_s": now - self._policy._observation_selected_time,
                 "queue_depth_at_start": depth, "queue_horizon_at_start_s": depth / self._fps,
                 "queue_deadline_horizon_at_start_s": queue["deadline_horizon_s"],
                 "next_action_margin_at_start_s": queue["next_action_margin_s"],
                 "prefetch_threshold_steps": self._rtc_queue_threshold,
                 "executed_actions_at_start": self.executed_actions,
                 "expired_prefix_dropped": 0, "overlap_prefix_dropped": 0, "accepted_steps": 0}
        self.predictions.append(event)
        return event

    def _prediction_end(self, event, error):
        now = time.monotonic()
        age = now - event["observation_timestamp_monotonic_s"]
        queue = self.action_queue.timing_snapshot(now)
        event.update(prediction_s=now - event["prediction_started_monotonic_s"],
                     observation_age_at_return_s=age,
                     remaining_valid_action_horizon_s=max(0.0, min(
                         self._policy.profile.chunk_size / self._fps, self.max_age_s) - age),
                     queue_depth_at_return=queue["depth"],
                     queue_deadline_horizon_at_return_s=queue["deadline_horizon_s"],
                     next_action_margin_at_return_s=queue["next_action_margin_s"],
                     actions_executed_during_prediction=self.executed_actions - event["executed_actions_at_start"],
                     error=error)
        event.update(self._policy._last_prediction_timing)

    def _record_merge(self, metrics):
        if self.predictions:
            self.predictions[-1].update(metrics)

    def record_execution(self):
        # Called only after the canonical Robot.send_action completed successfully.
        self.executed_actions += 1
        depth = self.action_queue.qsize()
        if self.minimum_execution_queue_depth is None or depth < self.minimum_execution_queue_depth:
            self.minimum_execution_queue_depth = depth

    def record_dispatch(self, margin_s):
        if margin_s is not None:
            if self.minimum_dispatch_margin_s is None or margin_s < self.minimum_dispatch_margin_s:
                self.minimum_dispatch_margin_s = margin_s
            if margin_s <= 0:
                self.expired_before_dispatch += 1

    def _new_queue(self):
        return InvalidatableActionQueue(max_steps=self.max_steps, max_age_s=self.max_age_s,
                                        observation_time=lambda: self._policy._observation_time
                                        or time.monotonic(), on_fault=self._fault, on_depth=self._record_depth,
                                        fps=self._fps, on_merge=self._record_merge)

    def _record_depth(self, depth):
        self.peak_queue_depth = max(self.peak_queue_depth, depth)
        if self._policy.session.samples:
            self._policy.session.samples[-1]["queue_depth"] = depth

    def start(self):
        super().start()
        # Worker starts paused. Install guards before resume can produce anything.
        self._action_queue = self._new_queue()
        self._obs_holder = _ObservationSlot(self._policy, self._robot.robot_type)
        self._started_at = time.monotonic()

    def notify_observation(self, obs):
        with self._obs_lock:
            self._obs_holder["obs"] = obs
            self._obs_holder.timestamp = time.monotonic()

    def _fault(self):
        if self.stop_detected_at is None:
            self.stop_detected_at = time.monotonic()
        self._rtc_error.set()
        self._shutdown_event.set()
        self._policy_active.clear()
        if self._action_queue is not None:
            self.last_queue_depth_before_stop = self._action_queue.qsize()
            self._action_queue.invalidate()
        self._policy.close()
        if self._global_shutdown_event is not None:
            self._global_shutdown_event.set()

    def get_action(self, obs_frame):
        if self.failed:
            raise RemoteFault("Remote inference failed; local execution stopped")
        if self._shutdown_event.is_set() or (self._global_shutdown_event is not None
                                             and self._global_shutdown_event.is_set()):
            raise RemoteFault("Local execution stopped")
        try:
            result = super().get_action(obs_frame)
            if result is None:
                if self._ever_had_action or (self._started_at is not None and
                                             time.monotonic() - self._started_at > self.startup_timeout_s):
                    self.underruns += 1
                    raise RemoteFault("Remote action queue underrun; no replay or CPU takeover")
                return None
            self._ever_had_action = True
            self.dequeued_actions += 1
            return result
        except RemoteFault:
            self._fault()
            raise

    def pause(self):
        super().pause()
        if self._action_queue is not None:
            self._action_queue.invalidate()
            self._action_queue = self._new_queue()
        self._policy.close()

    def resume(self):
        if self._policy.session._closed:
            self.reset()
        super().resume()

    def reset(self):
        # The worker may hold the old queue while RPC is in flight. Permanently
        # invalidating that object prevents a late merge after reset/resume.
        if self._action_queue is not None:
            self._action_queue.invalidate()
            self._action_queue = self._new_queue()
        super().reset()
        self._ever_had_action = False
        self._started_at = time.monotonic()

    def invalidate(self):
        if self.stop_detected_at is None:
            self.stop_detected_at = time.monotonic()
        self._shutdown_event.set()
        self._policy_active.clear()
        self._policy.close()
        if self._action_queue is not None:
            if self._action_queue.valid:
                self.last_queue_depth_before_stop = self._action_queue.qsize()
            self._action_queue.invalidate()

    def stop(self):
        self.invalidate()
        super().stop()


def validate_remote_rollout(cfg):
    """All static validation happens before the upstream context can connect arms."""
    from lerobot.robots import make_robot_from_config

    from yamkit.inference.profiles import get_profile

    if not isinstance(cfg.policy, YamkitRemoteConfig):
        raise TypeError("Remote rollout requires a registered YamkitRemoteConfig")
    if (not isinstance(cfg.inference, SyncInferenceConfig) or cfg.use_torch_compile or cfg.policy.use_peft
            or cfg.interpolation_multiplier != 1 or cfg.device != "cpu"):
        raise ValueError("Remote rollout supports unguided async, CPU proxy, no compile or interpolation")
    if not isinstance(cfg.strategy, BaseStrategyConfig) or cfg.teleop is not None or cfg.dataset is not None:
        raise ValueError("Remote rollout currently supports the upstream base strategy only")
    if cfg.rename_map:
        raise ValueError("Camera rename_map is applied once by the saved server processor")
    if cfg.return_to_initial_position:
        raise ValueError("Remote rollout requires return_to_initial_position=False for safe fault release")
    profile = get_profile(cfg.policy.profile)
    if not profile.mapping_verified:
        raise ValueError("Physical YAM mapping is unverified; use hardware-free native fixture checks")
    if abs(cfg.fps - profile.fps) > 1e-6:
        raise ValueError("Rollout FPS must match the checkpoint's validated action cadence")
    robot = make_robot_from_config(cfg.robot)  # schema inspection only; no connect
    if not hasattr(robot, "disconnect_no_home"):
        raise ValueError("Remote rollout requires a YAM follower with explicit no-home cleanup")
    if list(robot.action_features) != list(profile.action_names):
        raise ValueError("Physical robot action names/order do not exactly match the profile")
    state_names = [k for k, v in robot.observation_features.items() if v is float and k.endswith(".pos")]
    if state_names != list(profile.state_names):
        raise ValueError("Physical robot state names/order do not exactly match the profile")
    if set(robot.cameras) != set(profile.image_keys):
        raise ValueError("Rig cameras must exactly match the profile's original camera names")
    for camera in robot.camera_configs.values():
        color = getattr(camera, "color_mode", "rgb")
        if str(getattr(color, "value", color)).lower() != "rgb":
            raise ValueError("Remote inference requires RGB camera configuration")
    handles = [robot._h] if hasattr(robot, "_h") else list(robot._sides.values())
    for side, handle in getattr(robot, "_sides", {}).items():
        if handle.spec.side != side:
            raise ValueError("The configured arm side must match its physically verified rig side")
    for handle in handles:
        if handle.spec.arm_type != "yam" or handle.spec.gripper != "linear_4310":
            raise ValueError("This physical profile requires standard YAM arms with LINEAR_4310 grippers")
        limits = handle.spec.gripper_limits
        if handle.spec.has_motor_gripper and (limits is None or len(limits) != 2
                or not all(type(x) in (int, float) and math.isfinite(x) for x in limits)
                or limits[0] == limits[1]):
            raise ValueError(f"Valid saved gripper calibration required before activating {handle.spec.name}")
    from yamkit.inference.performance import require_physical_modal_rollout
    from yamkit.inference.qualification import current_settings

    dimensions = {(camera.height, camera.width) for camera in robot.camera_configs.values()}
    if len(dimensions) != 1:
        raise ValueError("Qualification requires exact, equal camera dimensions")
    cfg.policy.set_image_shape(next(iter(dimensions)))

    def settings():
        if len(dimensions) != 1:
            raise ValueError("Qualification requires exact, equal camera dimensions")
        return current_settings(cfg.policy, image_hw=next(iter(dimensions)))

    require_physical_modal_rollout(settings, supervised_confirmed=cfg.policy.supervised_confirmed,
                                   mapping_accepted=cfg.policy.mapping_accepted)


class _StoppableRobot(ThreadSafeRobot):
    """The upstream dispatch loop also checks Stop immediately before sending."""

    def __init__(self, robot, shutdown_event):
        super().__init__(robot)
        self.shutdown_event = shutdown_event
        self.on_action = None
        self.action_deadline = None
        self.on_fault = None
        self.on_dispatch = None

    def send_action(self, action):
        with self._lock:
            if self.shutdown_event.is_set():
                raise RemoteFault("Local execution stopped before action dispatch")
            deadline = self.action_deadline() if self.action_deadline is not None else None
            margin_s = deadline - time.monotonic() if deadline is not None else None
            if margin_s is not None and margin_s <= 0:
                if self.on_dispatch is not None:
                    self.on_dispatch(margin_s)
                if self.on_fault is not None:
                    self.on_fault()
                raise RemoteFault("Remote action expired before hardware dispatch")
            result = self.inner.send_action(action)
            if self.on_dispatch is not None:
                self.on_dispatch(margin_s)
            if self.on_action is not None:
                self.on_action()
            return result


def run_remote_rollout(cfg, *, shutdown_event: Event | None = None):
    """Reuse upstream context, strategy and robot execution; remote-specific release only."""
    validate_remote_rollout(cfg)
    if shutdown_event is None:
        from lerobot.utils.process import ProcessSignalHandler

        signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
        shutdown_event = signal_handler.shutdown_event
    ctx = None
    engine = None
    fault = None
    cfg.policy._session_shutdown_event = shutdown_event
    cfg.robot._session_shutdown_event = shutdown_event
    try:
        from yamkit.inference.qualification import validated_runner_context

        with validated_runner_context():
            ctx = build_rollout_context(cfg, shutdown_event)
        ctx.hardware.robot_wrapper = _StoppableRobot(ctx.hardware.robot_wrapper.inner, shutdown_event)
        engine = UnguidedRemoteInferenceEngine(
            policy=ctx.policy.policy, preprocessor=ctx.policy.preprocessor, postprocessor=ctx.policy.postprocessor,
            robot_wrapper=ctx.hardware.robot_wrapper, hw_features=ctx.data.hw_features,
            task=cfg.task, fps=cfg.fps, shutdown_event=shutdown_event)
        ctx.hardware.robot_wrapper.on_action = engine.record_execution
        ctx.hardware.robot_wrapper.action_deadline = lambda: engine.action_queue.last_action_deadline
        ctx.hardware.robot_wrapper.on_fault = engine._fault
        ctx.hardware.robot_wrapper.on_dispatch = engine.record_dispatch
        ctx.policy.inference = engine
        strategy = create_strategy(cfg.strategy)
        strategy.setup(ctx)
        strategy.run(ctx)
        if engine.failed:
            raise RemoteFault("Remote rollout stopped after an inference fault")
    except RemoteFault as exc:
        fault = exc
        raise
    finally:
        shutdown_event.set()
        if engine is not None:
            engine.invalidate()
        elif ctx is not None:
            ctx.policy.policy.close()
        robot = getattr(cfg.robot, "_runtime_robot", None)
        try:
            if robot is not None:
                robot.disconnect_no_home()
                if engine is not None:
                    engine.robot_released_at = time.monotonic()
        finally:
            if engine is not None:
                engine.stop()
                if fault is not None:
                    # Include cancelled in-flight attempts and final queue state.
                    fault.metrics = _rollout_metrics(ctx, engine)
    return _rollout_metrics(ctx, engine)


def _rollout_metrics(ctx, engine):
    return {"inference": "unguided_async", "failed": engine.failed, "underruns": engine.underruns,
            "queue_depth": engine.action_queue.qsize(), "peak_queue_depth": engine.peak_queue_depth,
            "last_queue_depth_before_stop": engine.last_queue_depth_before_stop,
            "executed_actions": engine.executed_actions,
            "dequeued_actions": engine.dequeued_actions,
            "minimum_execution_queue_depth": engine.minimum_execution_queue_depth,
            "minimum_dispatch_margin_s": engine.minimum_dispatch_margin_s,
            "expired_before_dispatch": engine.expired_before_dispatch,
            "expired_queued_actions": engine.action_queue.expired_queued_actions,
            "stop_to_robot_release_s": (engine.robot_released_at - engine.stop_detected_at)
            if engine.robot_released_at is not None and engine.stop_detected_at is not None else None,
            "stop_timing_basis": "local stop/fault detection to completed no-home robot release",
            "prediction_samples": [dict(event) for event in engine.predictions],
            "readiness_s": ctx.policy.policy.readiness_s,
            "readiness_model_warmup_s": ctx.policy.policy.warmup_s,
            "prefetch_threshold_steps": engine._rtc_queue_threshold,
            "expired_prefix_dropped": engine.action_queue.expired_prefix_dropped,
            "overlap_prefix_dropped": engine.action_queue.overlap_prefix_dropped,
            "expired_chunks": engine.action_queue.expired_chunks,
            "redundant_chunks": engine.action_queue.redundant_chunks,
            **ctx.policy.policy.session.metrics()}
