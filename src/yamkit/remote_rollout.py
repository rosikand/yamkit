"""Narrow unguided async adapter around LeRobot 0.6.1's actual rollout components.

The upstream factory gates *all* RTCInferenceConfig instances on guidance support,
even with guidance disabled. Build its sync context normally, then substitute an
RTCInferenceEngine subclass with guidance disabled. Its worker and the upstream
strategy/action-dispatch loop are reused unchanged. No installed files are patched.
"""

from __future__ import annotations

import math
import time
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
                 on_depth=None):
        super().__init__(RTCConfig(enabled=False))
        self.lock = RLock()
        self.max_steps = max_steps
        self.max_age_s = max_age_s
        self.valid = True
        self.inserted_at = None
        self._observation_time = observation_time or time.monotonic
        self._deadlines = []
        self._on_fault = on_fault
        self._on_depth = on_depth

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
                remaining = 0 if self.queue is None else len(self.queue) - self.last_index
                if remaining + len(processed_actions) > self.max_steps:
                    raise RemoteFault("Remote action queue capacity exceeded")
                if not torch.isfinite(processed_actions).all():
                    raise RemoteFault("Nonfinite remote action queue")
                self._deadlines = self._deadlines[self.last_index:] + [
                    self._observation_time() + self.max_age_s] * len(processed_actions)
                # This calls the upstream append operation under its own expected lock.
                self._append_actions_queue(original_actions, processed_actions)
                self.inserted_at = time.monotonic()
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
            if self.last_index < len(self._deadlines) and time.monotonic() > self._deadlines[self.last_index]:
                raise RemoteFault("Queued remote actions expired")
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
            self.policy._observation_time = self.timestamp
        return value


class UnguidedRemoteInferenceEngine(RTCInferenceEngine):
    """Keep RPC waits in the upstream background worker; fail closed on underrun."""

    def __init__(self, *, policy, preprocessor, postprocessor, robot_wrapper, hw_features, task, fps,
                 shutdown_event):
        threshold = max(1, policy.profile.chunk_size // 2)
        self.max_steps = policy.profile.chunk_size + threshold
        self.max_age_s = policy.config.max_observation_age_s
        self.startup_timeout_s = policy.config.request_timeout_s
        self.underruns = 0
        self.peak_queue_depth = 0
        self.last_queue_depth_before_stop = 0
        self._ever_had_action = False
        self._started_at = None
        super().__init__(policy, preprocessor, postprocessor, robot_wrapper, RTCConfig(enabled=False),
                         hw_features, task, fps, "cpu", use_torch_compile=False,
                         rtc_queue_threshold=threshold, shutdown_event=shutdown_event)
        policy.on_fault = self._fault

    def _new_queue(self):
        return InvalidatableActionQueue(max_steps=self.max_steps, max_age_s=self.max_age_s,
                                        observation_time=lambda: self._policy._observation_time
                                        or time.monotonic(), on_fault=self._fault, on_depth=self._record_depth)

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


class _StoppableRobot(ThreadSafeRobot):
    """The upstream dispatch loop also checks Stop immediately before sending."""

    def __init__(self, robot, shutdown_event):
        super().__init__(robot)
        self.shutdown_event = shutdown_event

    def send_action(self, action):
        with self._lock:
            if self.shutdown_event.is_set():
                raise RemoteFault("Local execution stopped before action dispatch")
            return self.inner.send_action(action)


def run_remote_rollout(cfg, *, shutdown_event: Event | None = None):
    """Reuse upstream context, strategy and robot execution; remote-specific release only."""
    validate_remote_rollout(cfg)
    if shutdown_event is None:
        from lerobot.utils.process import ProcessSignalHandler

        signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
        shutdown_event = signal_handler.shutdown_event
    ctx = None
    engine = None
    cfg.policy._session_shutdown_event = shutdown_event
    cfg.robot._session_shutdown_event = shutdown_event
    try:
        ctx = build_rollout_context(cfg, shutdown_event)
        ctx.hardware.robot_wrapper = _StoppableRobot(ctx.hardware.robot_wrapper.inner, shutdown_event)
        engine = UnguidedRemoteInferenceEngine(
            policy=ctx.policy.policy, preprocessor=ctx.policy.preprocessor, postprocessor=ctx.policy.postprocessor,
            robot_wrapper=ctx.hardware.robot_wrapper, hw_features=ctx.data.hw_features,
            task=cfg.task, fps=cfg.fps, shutdown_event=shutdown_event)
        ctx.policy.inference = engine
        strategy = create_strategy(cfg.strategy)
        strategy.setup(ctx)
        strategy.run(ctx)
        if engine.failed:
            raise RemoteFault("Remote rollout stopped after an inference fault")
    except RemoteFault as exc:
        if engine is not None:
            exc.metrics = _rollout_metrics(ctx, engine)
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
        finally:
            if engine is not None:
                engine.stop()
    return _rollout_metrics(ctx, engine)


def _rollout_metrics(ctx, engine):
    return {"inference": "unguided_async", "failed": engine.failed, "underruns": engine.underruns,
            "queue_depth": engine.action_queue.qsize(), "peak_queue_depth": engine.peak_queue_depth,
            "last_queue_depth_before_stop": engine.last_queue_depth_before_stop,
            **ctx.policy.policy.session.metrics()}
