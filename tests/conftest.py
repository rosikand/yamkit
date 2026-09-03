from typing import ClassVar

import numpy as np
import pytest

from yamkit.config import ArmSpec, ControlSpec, PairSpec, RigConfig


class FakeRobot:
    """Stand-in for i2rt.MotorChainRobot: enough surface for YamArm + plugins."""

    def __init__(self, n_dofs: int = 7, gripper: bool = True, handle: bool = False):
        self.n = n_dofs
        self.gripper = gripper
        self.pos = np.zeros(n_dofs)
        self.commands: list[np.ndarray] = []
        self.kp = np.full(n_dofs, 80.0)
        self.kd = np.full(n_dofs, 5.0)
        self.idle_calls = 0
        self.closed = False
        self.motor_chain = self
        self.encoder = None
        if handle:
            self.encoder = [type("Enc", (), {"position": 0.0, "io_inputs": [0, 0]})()]

    # -- i2rt API used by yamkit
    def num_dofs(self):
        return self.n

    def get_robot_info(self):
        return {"kp": self.kp, "kd": self.kd, "gripper_limits": np.array([0.0, 6.5]) if self.gripper else None}

    def get_observations(self):
        if self.gripper:
            return {"joint_pos": self.pos[:6], "joint_vel": np.zeros(6), "joint_eff": np.zeros(6), "gripper_pos": self.pos[6:7]}
        return {"joint_pos": self.pos[:6], "joint_vel": np.zeros(6), "joint_eff": np.zeros(6)}

    def command_joint_pos(self, q):
        q = np.asarray(q, dtype=float)
        assert q.shape == (self.n,)
        self.commands.append(q.copy())
        self.pos = q.copy()  # perfect tracking

    def update_kp_kd(self, kp, kd):
        self.kp, self.kd = np.asarray(kp), np.asarray(kd)

    def enter_gravity_comp_idle(self):
        self.idle_calls += 1

    def zero_torque_mode(self):
        self.kp = np.zeros(self.n)

    def get_same_bus_device_states(self):
        return self.encoder

    def close(self):
        self.closed = True


@pytest.fixture
def rig(tmp_path):
    arms = {
        "left_leader": ArmSpec(name="left_leader", role="leader", side="left", gripper="yam_teaching_handle", can_serial="AAA"),
        "left_follower": ArmSpec(name="left_follower", role="follower", side="left", gripper="linear_4310", can_serial="BBB"),
        "right_leader": ArmSpec(name="right_leader", role="leader", side="right", gripper="yam_teaching_handle", can_serial="CCC"),
        "right_follower": ArmSpec(name="right_follower", role="follower", side="right", gripper="linear_4310", can_serial="DDD"),
    }
    r = RigConfig(arms=arms, pairs=[PairSpec("left_leader", "left_follower"), PairSpec("right_leader", "right_follower")], control=ControlSpec())
    r.save(tmp_path / "rig.yaml")
    return r


@pytest.fixture
def fake_connect(monkeypatch):
    """Patch YamArm.connect + resolve_channel so plugins/teleop run without hardware."""
    from yamkit import arm as arm_mod

    class Robots(dict):
        presets: ClassVar[dict[str, np.ndarray]] = {}  # initial joint positions per arm name, applied at connect

    robots = Robots()

    def connect(spec, channel, **kw):
        r = FakeRobot(n_dofs=spec.n_dofs, gripper=spec.has_motor_gripper, handle=spec.has_handle)
        if spec.name in robots.presets:
            r.pos[: len(robots.presets[spec.name])] = robots.presets[spec.name]
        robots[spec.name] = r
        return arm_mod.YamArm(spec, channel, r, max_joint_speed=kw.get("max_joint_speed", 3.0), max_gripper_speed=kw.get("max_gripper_speed", 3.0))

    monkeypatch.setattr(arm_mod.YamArm, "connect", staticmethod(connect))
    monkeypatch.setattr(arm_mod, "HOME_MIN_S", 0.01)  # keep the fake arms' home moves short
    monkeypatch.setattr(arm_mod, "HOME_SETTLE_S", 0.0)
    monkeypatch.setattr(arm_mod, "resolve_channel", lambda spec: f"can_{spec.name}")
    for mod in ("lerobot_robot_yamkit.yam_follower", "lerobot_teleoperator_yamkit.yam_leader", "yamkit.teleop"):
        try:
            m = __import__(mod, fromlist=["x"])
            monkeypatch.setattr(m, "resolve_channel", lambda spec: f"can_{spec.name}", raising=False)
        except ImportError:
            pass
    return robots
