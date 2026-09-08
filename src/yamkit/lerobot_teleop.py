"""YAM operator processor injected into the unchanged pinned LeRobot entry points.

``python -m yamkit.lerobot_teleop record|teleoperate <LeRobot flags>`` is used by
yamkit's thin CLI wrappers. There is no alternate recorder or acquisition loop.
"""

import inspect
import json
import logging
import os
import signal
import sys
import threading
import time
from contextlib import contextmanager

from lerobot.configs import parser
from lerobot.lerobot_types import TransitionKey
from lerobot.processor import ProcessorStep, RobotProcessorPipeline
from lerobot.processor.converters import robot_action_observation_to_transition, transition_to_robot_action
from lerobot.scripts import lerobot_record, lerobot_teleoperate
from lerobot.utils.import_utils import register_third_party_plugins

from .arm import check_joint_bounds
from .config import RigConfig
from .teleop_control import GatedAction, LeaderAction, PairGate, action_vector, vector_action
from .validation import finite_scalar, vendor_joint_limits

log = logging.getLogger(__name__)
# Pinned LeRobot's image-writer decorator hides record_loop's signature. Declare
# its public call layout so positional/keyword calls bind before any preparation;
# execution still goes through the current upstream and Stop adapters.
_RECORD_LOOP_SIGNATURE = inspect.Signature([
    inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default)
    for name, default in {
        **dict.fromkeys(("robot", "events", "fps", "teleop_action_processor",
                         "robot_action_processor", "robot_observation_processor"), inspect.Parameter.empty),
        "dataset": None, "teleop": None, "control_time_s": None, "single_task": None,
        "display_data": False, "display_mode": "rerun", "display_compressed_images": False,
    }.items()
])


class OperatorStep(ProcessorStep):
    def __init__(self, robot_config, teleop_config, fps):
        self.period = 1 / finite_scalar(fps, "operator FPS", positive=True)
        if (robot_config.type, teleop_config.type) not in (
            ("yam_follower", "yam_leader"), ("bi_yam_follower", "bi_yam_leader"),
        ):
            raise ValueError("operator parity requires matching YAM follower and leader plugins")
        self.rig = RigConfig.load(robot_config.rig)
        if RigConfig.load(teleop_config.rig).to_dict() != self.rig.to_dict():
            raise ValueError("leader and follower must use the same rig configuration")
        if errors := self.rig.validate():
            raise ValueError("invalid rig: " + "; ".join(errors))
        if self.rig.control.bilateral_kp:
            raise ValueError("recording/LeRobot teleoperation does not support bilateral feedback; "
                             "set control.bilateral_kp to 0 or use yamkit teleop")
        self.sides = {"": robot_config.arm} if robot_config.type == "yam_follower" else {
            "left_": robot_config.left, "right_": robot_config.right,
        }
        leaders = {"": teleop_config.arm} if teleop_config.type == "yam_leader" else {
            "left_": teleop_config.left, "right_": teleop_config.right,
        }
        for prefix, follower in self.sides.items():
            if not any(pair.follower == follower and pair.leader == leaders[prefix] for pair in self.rig.pairs):
                raise ValueError("selected leader/follower order does not match rig.pairs")
        self.pair_names = {prefix: f"{leaders[prefix]}->{follower}" for prefix, follower in self.sides.items()}
        ctrl = self.rig.control
        self.joint_speed = ctrl.max_joint_speed if robot_config.max_joint_speed is None else robot_config.max_joint_speed
        self.gripper_speed = ctrl.max_gripper_speed if robot_config.max_gripper_speed is None else robot_config.max_gripper_speed
        finite_scalar(self.joint_speed, "joint speed", positive=True)
        finite_scalar(self.gripper_speed, "gripper speed", positive=True)
        if not isinstance(teleop_config.auto_engage, bool):
            raise TypeError("auto_engage must be a boolean")
        self.auto_engage = teleop_config.auto_engage
        self.gates = {prefix: PairGate() for prefix in self.sides}
        self._reported_phase = None
        self._on_ready = None

    @property
    def ready(self):
        """All pairs synchronized, as acknowledged by the follower's send path."""
        return self._reported_phase == "ready"

    def __call__(self, transition):
        raw, obs = transition[TransitionKey.ACTION], transition[TransitionKey.OBSERVATION]
        if not isinstance(raw, LeaderAction) or set(raw.buttons) != set(self.sides):
            raise ValueError("YAM operator action is missing teaching-handle button metadata")
        now = time.monotonic()
        pending, output, captures, transitions = {}, {}, [], []
        for prefix, name in self.sides.items():
            spec = self.rig.arm(name)
            gripper = spec.has_motor_gripper
            buttons = raw.buttons[prefix]
            index = self.rig.control.engage_button
            pressed = bool(buttons[index]) if buttons and len(buttons) > index else False
            leader = action_vector(raw, prefix, gripper=gripper)
            # The upstream recorder only reaches this processor after both plugins
            # have connected and completed their configured startup home moves.
            # Engage once per session, never again at an episode/reset boundary or
            # after the operator deliberately pauses with the handle button.
            automatic = self.auto_engage and self.gates[prefix].previous_t is None
            gate, command, capture = self.gates[prefix].advance(
                leader, action_vector(obs, prefix, gripper=gripper),
                pressed=pressed, now=now, period=self.period, sync_seconds=self.rig.control.sync_seconds,
                joint_speed=self.joint_speed, gripper_speed=self.gripper_speed,
                engage=True if automatic else None,
            )
            if gate.engaged:
                # Validate the full requested pose before interpolation can conceal
                # an out-of-bounds leader target behind a small valid first step.
                check_joint_bounds(leader[:6] - (spec.joint_offsets or [0.0] * 6),
                                   vendor_joint_limits(spec.arm_type, spec.gripper), "operator target")
            pending[prefix] = gate
            if gate.engaged != self.gates[prefix].engaged:
                transitions.append((prefix, automatic))
            output.update({prefix + key: value for key, value in vector_action(command).items()})
            if capture:
                captures.append(prefix)
        self.gates = pending  # malformed second-side input never consumes the first button edge
        def latch_sent_holds(sent):
            for prefix in captures:
                hold = action_vector(sent, prefix, gripper=self.rig.arm(self.sides[prefix]).has_motor_gripper)
                self.gates[prefix] = self.gates[prefix].acknowledge_hold(
                    hold, joint_speed=self.joint_speed, gripper_speed=self.gripper_speed)
            # A processed action is only an intent until the follower acknowledges
            # the sent command. Keep operator cues after that same boundary.
            for prefix, automatic in transitions:
                gate = self.gates[prefix]
                if gate.engaged:
                    source = "automatically" if automatic else f"via button {self.rig.control.engage_button}"
                    log.info("[%s] engaged %s: follower synchronizing for at least %.2f s",
                             self.pair_names[prefix], source, gate.duration)
                else:
                    log.info("[%s] disengaged via button %d: follower holding measured pose",
                             self.pair_names[prefix], self.rig.control.engage_button)
            transitions.clear()  # repeated acknowledgment cannot repeat a button edge
            if not all(gate.engaged for gate in self.gates.values()):
                phase = "holding"
            elif any(gate.syncing for gate in self.gates.values()):
                phase = "synchronizing"
            else:
                phase = "ready"
            if phase != self._reported_phase:
                log.info("[yamkit-operator] %s", phase)
                self._reported_phase = phase
            if self.ready and self._on_ready is not None:
                self._on_ready()

        return {**transition, TransitionKey.ACTION: GatedAction(output, capture_hold=captures,
                                                               on_sent=latch_sent_holds)}

    def transform_features(self, features):
        return features  # preserve all existing dataset names, order, shapes and units


def make_teleop_processor(robot_config, teleop_config, fps):
    return RobotProcessorPipeline(
        steps=[OperatorStep(robot_config, teleop_config, fps)],
        to_transition=robot_action_observation_to_transition, to_output=transition_to_robot_action,
    )


@contextmanager
def release_after_upstream(cfg):
    """A no-home safety net if upstream finalization/partial startup skips its cleanup."""
    try:
        yield
    finally:
        failed = sys.exc_info()[0] is not None
        errors = []
        for config, attr in ((cfg.robot, "_runtime_robot"), (cfg.teleop, "_runtime_teleop")):
            device = getattr(config, attr, None)
            if device is not None:
                try:
                    device.disconnect(home=False)
                except BaseException as exc:  # noqa: BLE001 — every remaining arm/camera must be attempted
                    errors.append(exc)
        for error in errors:
            log.error("operator cleanup failure: %s", error)
        if errors and not failed:
            raise errors[0]


@contextmanager
def record_stop_events():
    """Keep CLI interrupts and add a targeted, cooperative first dashboard Stop.

    The dashboard learns this process's PID only after acquisition is available.
    SIGUSR1 then updates LeRobot's own events, including between loops while an
    episode saves. It never interrupts an encoder or replaces the upstream loop.
    CLI Ctrl-C retains its existing behavior; another dashboard Stop uses SIGINT.
    """
    original = lerobot_record.record_loop
    original_save = lerobot_record.LeRobotDataset.save_episode
    owns_control = not getattr(original, "_yamkit_record_stop", False)
    session = os.environ.get("YAMKIT_PREVIEW_SESSION", "")
    ui_enabled = bool(session and os.environ.get("YAMKIT_PREVIEW_TOKEN") and hasattr(signal, "SIGUSR1"))
    registered = requested = False
    current_events = None
    previous_ui_signal = None

    def announce(event):
        print("@yamkit-record-stop/1 " + json.dumps(
            {"v": 1, "session": session, "event": event, "pid": os.getpid()}, separators=(",", ":")),
            flush=True)

    def ui_stop(signum, frame):
        nonlocal requested
        requested = True
        if current_events is not None:
            current_events["exit_early"] = current_events["stop_recording"] = True

    def saving_episode(dataset, *args, **kwargs):
        log.info("Saving episode %d: encoding videos; followers hold their last command. "
                 "Dashboard Stop waits for saving; Ctrl-C interrupts saving.",
                 dataset.num_episodes)
        return original_save(dataset, *args, **kwargs)

    saving_episode._yamkit_saving_phase = True

    def stoppable_loop(*args, **kwargs):
        nonlocal registered, current_events, previous_ui_signal
        if not owns_control or threading.current_thread() is not threading.main_thread():
            return original(*args, **kwargs)
        events = kwargs["events"] if "events" in kwargs else args[1]
        current_events = events  # retain through reset, encoding and finalization
        previous = signal.getsignal(signal.SIGINT)

        def stop(signum, frame):
            signal.signal(signal.SIGINT, previous)
            events["exit_early"] = events["stop_recording"] = True

        if requested:
            events["exit_early"] = events["stop_recording"] = True
        elif not ui_enabled:
            signal.signal(signal.SIGINT, stop)
        try:
            if ui_enabled and not registered:
                previous_ui_signal = signal.getsignal(signal.SIGUSR1)
                signal.signal(signal.SIGUSR1, ui_stop)
                registered = True
                # UI SIGINT is always immediate cancellation. If SIGINT and
                # SIGUSR1 arrive together, Python may deliver SIGINT first;
                # never consume that second Stop as a graceful CLI interrupt.
                announce("ready")  # handler and upstream events exist before the PID is published
            result = original(*args, **kwargs)
        finally:
            signal.signal(signal.SIGINT, previous)
        dataset = kwargs.get("dataset", args[6] if len(args) > 6 else None)
        if events["stop_recording"] and dataset is not None and not dataset.has_pending_frames():
            # Upstream would save an empty episode. Cancel before that save; its
            # VideoEncodingManager and finally blocks retain all cleanup.
            raise KeyboardInterrupt("Recording stopped before any frames were captured for this episode")
        return result

    stoppable_loop._yamkit_record_stop = True
    lerobot_record.record_loop = stoppable_loop
    if not getattr(original_save, "_yamkit_saving_phase", False):
        lerobot_record.LeRobotDataset.save_episode = saving_episode
    try:
        yield
    finally:
        lerobot_record.record_loop = original
        lerobot_record.LeRobotDataset.save_episode = original_save
        if registered:
            # Parent stdout delivery is asynchronous. Ignore a late first Stop
            # after teardown instead of restoring SIGUSR1's terminating default.
            # This only applies to the dashboard's one-shot recorder; nested or
            # explicitly installed handlers are restored normally.
            signal.signal(signal.SIGUSR1, signal.SIG_IGN if previous_ui_signal == signal.SIG_DFL else previous_ui_signal)
            try:
                announce("release")
            except (OSError, ValueError):
                pass  # the manager also forgets registration when this process exits


@contextmanager
def record_ready_events(processor):
    """Run automatic engagement through LeRobot before its timed dataset loop.

    Preparation uses the very same upstream control loop without a dataset, like
    an environment reset. Cameras, safety clamps, button edges and Stop remain
    active, but no episode frames or time are consumed. The manual CLI still
    permits recording held poses before engagement.
    """
    operator = processor.steps[0]
    if not operator.auto_engage:
        yield
        return
    original_loop, original_say = lerobot_record.record_loop, lerobot_record.log_say
    if getattr(original_loop, "_yamkit_record_ready", False):
        yield
        return
    pending_announcement = None

    def delay_episode_announcement(message, *args, **kwargs):
        nonlocal pending_announcement
        if isinstance(message, str) and message.startswith("Recording episode "):
            episode = message.removeprefix("Recording episode ")
            if episode.isdecimal():
                pending_announcement = (message, args, kwargs)
                log.info("Preparing episode %s: waiting for operator readiness.", episode)
                return
        return original_say(message, *args, **kwargs)

    def ready_loop(*args, **kwargs):
        nonlocal pending_announcement
        bound = _RECORD_LOOP_SIGNATURE.bind(*args, **kwargs)
        dataset = bound.arguments.get("dataset")
        if dataset is None:
            return original_loop(*args, **kwargs)
        events = bound.arguments["events"]

        def cancelled():
            return events.get("stop_recording") or events.get("rerecord_episode")

        try:
            if not cancelled():
                # Refresh at least one observation/button sample per episode.
                # Cached readiness predates any time spent saving; a handle may
                # have been paused or moved while the control loop was stopped.
                preparation = dict(bound.arguments, dataset=None, control_time_s=float("inf"))
                previous_ready = operator._on_ready
                operator._on_ready = lambda: events.update(exit_early=True)
                try:
                    original_loop(**preparation)
                finally:
                    operator._on_ready = previous_ready
            if cancelled() or not operator.ready:
                raise KeyboardInterrupt
            if pending_announcement is not None:
                message, announcement_args, announcement_kwargs = pending_announcement
                pending_announcement = None
                original_say(message, *announcement_args, **announcement_kwargs)
            if cancelled():
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            # Preparation has no episode to save. SystemExit preserves the normal
            # upstream finally blocks while the YAM plugins release without a new
            # home move, matching cancellation during incomplete startup.
            log.info("Recording cancelled before acquisition; no frames captured for this episode.")
            raise SystemExit(130) from None
        return original_loop(*args, **kwargs)

    ready_loop._yamkit_record_ready = True
    lerobot_record.record_loop = ready_loop
    lerobot_record.log_say = delay_episode_announcement
    try:
        yield
    finally:
        lerobot_record.record_loop = original_loop
        lerobot_record.log_say = original_say


@parser.wrap()
def record(cfg: lerobot_record.RecordConfig):
    processor = make_teleop_processor(cfg.robot, cfg.teleop, cfg.dataset.fps)
    with release_after_upstream(cfg), record_stop_events(), record_ready_events(processor):
        return lerobot_record.record(cfg, teleop_action_processor=processor)


@parser.wrap()
def teleoperate(cfg: lerobot_teleoperate.TeleoperateConfig):
    processor = make_teleop_processor(cfg.robot, cfg.teleop, cfg.fps)
    # Unlike record(), pinned teleoperate() has no processor parameter. Restrict this
    # factory override to this process's existing entry point and restore it on exit.
    original = lerobot_teleoperate.make_default_processors
    lerobot_teleoperate.make_default_processors = lambda: (processor, *original()[1:])
    try:
        with release_after_upstream(cfg):
            return lerobot_teleoperate.teleoperate(cfg)
    finally:
        lerobot_teleoperate.make_default_processors = original


def main():
    command = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    if command not in ("record", "teleoperate"):
        raise SystemExit("expected record or teleoperate followed by LeRobot flags")
    register_third_party_plugins()
    (record if command == "record" else teleoperate)()


if __name__ == "__main__":
    main()
