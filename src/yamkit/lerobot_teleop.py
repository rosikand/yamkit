"""YAM operator processor injected into the unchanged pinned LeRobot entry points.

``python -m yamkit.lerobot_teleop record|teleoperate <LeRobot flags>`` is used by
yamkit's thin CLI wrappers. There is no alternate recorder or acquisition loop.
"""

import logging
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
        self.gates = {prefix: PairGate() for prefix in self.sides}

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
            gate, command, capture = self.gates[prefix].advance(
                leader, action_vector(obs, prefix, gripper=gripper),
                pressed=pressed, now=now, period=self.period, sync_seconds=self.rig.control.sync_seconds,
                joint_speed=self.joint_speed, gripper_speed=self.gripper_speed,
            )
            if gate.engaged:
                # Validate the full requested pose before interpolation can conceal
                # an out-of-bounds leader target behind a small valid first step.
                check_joint_bounds(leader[:6] - (spec.joint_offsets or [0.0] * 6),
                                   vendor_joint_limits(spec.arm_type, spec.gripper), "operator target")
            pending[prefix] = gate
            if gate.engaged != self.gates[prefix].engaged:
                transitions.append(prefix)
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
            for prefix in transitions:
                gate = self.gates[prefix]
                if gate.engaged:
                    log.info("[%s] engaged via button %d: follower synchronizing for at least %.2f s",
                             self.pair_names[prefix], self.rig.control.engage_button, gate.duration)
                else:
                    log.info("[%s] disengaged via button %d: follower holding measured pose",
                             self.pair_names[prefix], self.rig.control.engage_button)
            transitions.clear()  # repeated acknowledgment cannot repeat a button edge

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
    """Let one Ctrl-C end the existing acquisition loop and save its partial episode.

    Only the pinned upstream loop receives this handler. Startup, episode saving,
    finalization and homing retain their existing interrupt behavior. A second
    interrupt immediately uses the previous handler, including before the loop exits.
    """
    original = lerobot_record.record_loop

    def stoppable_loop(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            return original(*args, **kwargs)
        events = kwargs["events"] if "events" in kwargs else args[1]
        previous = signal.getsignal(signal.SIGINT)

        def stop(signum, frame):
            signal.signal(signal.SIGINT, previous)
            events["exit_early"] = events["stop_recording"] = True

        signal.signal(signal.SIGINT, stop)
        try:
            result = original(*args, **kwargs)
        finally:
            signal.signal(signal.SIGINT, previous)
        dataset = kwargs.get("dataset", args[6] if len(args) > 6 else None)
        if events["stop_recording"] and dataset is not None and not dataset.has_pending_frames():
            # Upstream would save an empty episode. Cancel before that save; its
            # VideoEncodingManager and finally blocks retain all cleanup.
            raise KeyboardInterrupt("Recording stopped before any frames were captured for this episode")
        return result

    lerobot_record.record_loop = stoppable_loop
    try:
        yield
    finally:
        lerobot_record.record_loop = original


@parser.wrap()
def record(cfg: lerobot_record.RecordConfig):
    processor = make_teleop_processor(cfg.robot, cfg.teleop, cfg.dataset.fps)
    with release_after_upstream(cfg), record_stop_events():
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
