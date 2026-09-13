"""Explicit software-only robot seams for real-service CLI validation.

Never selected by defaults or an environment variable. Saved frames and perfect
target receipts are not a dynamics simulator or evidence of manipulation success.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .inference.mapping import YAM_NAMES


class SavedRobot:
    """Native runner fake: no SDK, CAN, camera, ownership or device constructors."""

    def __init__(self, observations, validator):
        if not observations:
            raise ValueError("Fake execution requires existing saved observations")
        self.observations, self.validator = observations, validator
        self.state = np.asarray(observations[0]["state"], dtype=np.float64).copy()
        self.observation_index = 0
        self.connected = self.released = False
        self.sent = []

    def connect(self):
        self.connected = True

    def get_observation(self):
        if not self.connected or self.released:
            raise RuntimeError("Saved fake robot is not connected")
        sample = self.observations[self.observation_index % len(self.observations)]
        self.observation_index += 1
        return {**dict(zip(YAM_NAMES, self.state.tolist(), strict=True)),
                **{name: frame for name, frame in sample.items() if name != "state"}}

    def validate_action_target(self, target):
        self.validator(target)

    def send_reference_action(self, target, *, dispatch_check):
        dispatch_check()
        if not self.connected or self.released:
            raise RuntimeError("Saved fake robot is released")
        self.validator(target)
        self.sent.append(dict(target))
        self.state = np.array([target[name] for name in YAM_NAMES], dtype=np.float64)
        return dict(target)

    def disconnect(self, *, home=False):
        if home:
            raise ValueError("Fake cleanup must not request real homing")
        self.released = True
        self.connected = False


def saved_observations(target):
    from .backend_workflow import WorkflowError
    from .pi05.qualification import load_saved_observation, observation_schema

    paths = target.saved_observations
    if not paths:
        raise WorkflowError("--fake-hardware requires configured saved_observations; it never captures live frames")
    for path in paths:
        if not path.is_file() or not 0 < path.stat().st_size <= 32 * 1024 * 1024:
            raise WorkflowError("Saved fake inputs must be bounded existing repository-local NPZ files")
    result = [load_saved_observation(path) for path in paths]
    observation_schema(result)
    return result


@contextmanager
def forbid_device_io():
    """Defense in depth for an explicit fake process; TCP inference remains usable."""
    import socket

    import cv2

    class GuardedSocket(socket.socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family == getattr(socket, "AF_CAN", object()):
                raise RuntimeError("CAN is forbidden in --fake-hardware mode")
            super().__init__(family, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("Real camera construction is forbidden in --fake-hardware mode")

    with patch.object(socket, "socket", GuardedSocket), patch.object(cv2, "VideoCapture", forbidden):
        yield


@contextmanager
def molmo_fake_devices(observations):
    """Run the unchanged production MA2 plugin/executor against explicitly fake I/O."""
    from .arm import YamArm
    from .camera_ownership import CameraLease
    from .preview import NullPreview

    class SDK:
        def __init__(self):
            self.pos = np.zeros(7)
            self.commands = []
            self.closed = False

        def num_dofs(self): return 7

        def get_robot_info(self):
            return {"kp": np.full(7, 80.), "kd": np.full(7, 5.), "gripper_limits": np.array([0., 6.5])}

        def get_observations(self):
            if self.closed:
                raise RuntimeError("Fake SDK observation after release")
            return {"joint_pos": self.pos[:6], "joint_vel": np.zeros(6), "joint_eff": np.zeros(6),
                    "gripper_pos": self.pos[6:7]}

        def command_joint_pos(self, values):
            if self.closed:
                raise RuntimeError("Fake SDK command after release")
            values = np.asarray(values, dtype=float)
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError("Invalid fake SDK target")
            self.commands.append(values.copy())
            self.pos = values.copy()

        def update_kp_kd(self, *_args): pass
        def enter_gravity_comp_idle(self): pass
        def close(self): self.closed = True

    class Camera:
        is_connected = False

        def __init__(self, name): self.name, self.index = name, 0
        def connect(self): self.is_connected = True
        def disconnect(self): self.is_connected = False

        def read_latest(self):
            if not self.is_connected:
                raise RuntimeError("Fake camera observation after disconnect")
            frame = observations[self.index % len(observations)][self.name]
            self.index += 1
            return frame

    robots = []

    def connect(spec, channel, **kwargs):
        sdk = SDK()
        robots.append(sdk)
        return YamArm(spec, channel, sdk, max_joint_speed=kwargs["max_joint_speed"],
                      max_gripper_speed=kwargs["max_gripper_speed"])

    with ExitStack() as stack:
        stack.enter_context(forbid_device_io())
        stack.enter_context(patch("yamkit.arm.YamArm.connect", connect))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.resolve_channel", lambda spec: "FAKE-" + spec.name))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.make_cameras_from_configs",
                                  lambda configs: {name: Camera(name) for name in configs}))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.claim_from_env", lambda *_: CameraLease()))
        stack.enter_context(patch("lerobot_robot_yamkit.yam_follower.start_from_env", lambda *_a, **_k: NullPreview()))
        yield robots


def run_fake_pi05(selection, *, artifact_dir: Path, capture_trace=False, upload_repo_id=None):
    from .backend_workflow import load_target
    from .pi05.admission import passive_target_validator
    from .pi05_workflow import run_prepared_pi05

    target = load_target("lambda", "pi05-yam", config=selection.config)
    observations = saved_observations(target)
    robot = SavedRobot(observations, passive_target_validator(Path(selection.rig_path)))
    with forbid_device_io():
        return run_prepared_pi05(selection, confirm_supervised=False, accept_mapping=False,
                                 artifact_dir=artifact_dir, capture_trace=capture_trace,
                                 upload_repo_id=upload_repo_id, fake_robot=robot)
