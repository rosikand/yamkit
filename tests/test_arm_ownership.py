"""YamArm lifecycle integration using the fake SDK and actual Linux flock."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from i2rt.robots import get_robot

from yamkit import arm as arm_module
from yamkit import ownership
from yamkit.arm import YamArm
from yamkit.config import ArmSpec

from .conftest import FakeRobot
from .test_ownership import _probe


@pytest.fixture
def sdk(tmp_path, monkeypatch):
    locks = tmp_path / "locks"
    monkeypatch.setattr(ownership, "LOCK_DIR", locks)
    monkeypatch.setattr(ownership, "adapter_identity", lambda channel: f"usb-serial:{channel}")
    monkeypatch.setattr(arm_module.time, "sleep", lambda seconds: None)
    env = SimpleNamespace(
        locks=locks,
        robot=FakeRobot(),
        spec=ArmSpec(name="f", role="follower", can_serial="A"),
        opens=[],
    )

    def factory(**kwargs):
        env.opens.append(kwargs["channel"])
        return env.robot

    monkeypatch.setattr(get_robot, "get_yam_robot", factory)
    yield env
    # Fakes have no background transmitter; release deliberately retained failure leases.
    for lease in list(ownership._leases):
        if lease.path.parent == locks:
            lease.release()


@pytest.mark.parametrize(
    "option,value", [("max_joint_speed", 0), ("max_gripper_speed", np.nan), ("encoder_timeout_s", -1)]
)
def test_invalid_connect_options_do_not_acquire_or_activate(sdk, option, value):
    with pytest.raises(ValueError):
        YamArm.connect(sdk.spec, "A", **{option: value})
    assert sdk.opens == []
    assert not sdk.locks.exists()


def test_mutated_invalid_spec_does_not_acquire_or_activate(sdk):
    sdk.spec.joint_offsets = [np.nan] * 6
    with pytest.raises(ValueError):
        YamArm.connect(sdk.spec, "A")
    assert sdk.opens == []
    assert not sdk.locks.exists()


def test_lease_is_acquired_before_sdk_activation_and_until_sdk_close(sdk, monkeypatch):
    def factory(**kwargs):
        assert _probe(sdk.locks) == "busy"
        return sdk.robot

    def close():
        assert _probe(sdk.locks) == "busy"
        sdk.robot.closed = True

    monkeypatch.setattr(get_robot, "get_yam_robot", factory)
    monkeypatch.setattr(sdk.robot, "close", close)
    arm = YamArm.connect(sdk.spec, "A")
    assert _probe(sdk.locks) == "busy"
    arm.close(settle_s=0)
    arm.close(settle_s=0)
    assert sdk.robot.closed
    assert _probe(sdk.locks) == "free"


def test_second_open_fails_before_sdk_activation(sdk):
    arm = YamArm.connect(sdk.spec, "A")
    alias = ArmSpec(name="other-rig-name", role="follower", can_iface="A")
    with pytest.raises(RuntimeError, match="already owned"):
        YamArm.connect(alias, "A")
    assert sdk.opens == ["A"]
    arm.close(settle_s=0)


def test_wrapper_validation_failure_closes_sdk_before_releasing(sdk, monkeypatch):
    sdk.robot.n = 6  # mismatches follower's seven motor DOFs

    def close():
        assert _probe(sdk.locks) == "busy"
        sdk.robot.closed = True

    monkeypatch.setattr(sdk.robot, "close", close)
    with pytest.raises(ValueError, match="DOFs"):
        YamArm.connect(sdk.spec, "A")
    assert sdk.robot.closed
    assert _probe(sdk.locks) == "free"


@pytest.mark.parametrize("position", [np.nan, 100.0])
def test_invalid_initial_measured_state_closes_sdk_and_releases(sdk, position):
    sdk.robot.pos[0] = position
    with pytest.raises(ValueError):
        YamArm.connect(sdk.spec, "A")
    assert sdk.robot.closed
    assert sdk.robot.commands == []
    assert _probe(sdk.locks) == "free"


def test_encoder_wait_cancellation_closes_partial_connection(sdk, monkeypatch):
    sdk.spec = ArmSpec(name="l", role="leader", gripper="yam_teaching_handle", can_serial="A")
    sdk.robot = FakeRobot(6, gripper=False, handle=False)

    def sleep(seconds):
        if seconds == 0.02:
            raise KeyboardInterrupt("encoder cancelled")

    monkeypatch.setattr(arm_module.time, "sleep", sleep)
    with pytest.raises(KeyboardInterrupt, match="encoder cancelled"):
        YamArm.connect(sdk.spec, "A")
    assert sdk.robot.closed
    assert _probe(sdk.locks) == "free"


def test_encoder_timeout_closes_partial_connection(sdk, monkeypatch):
    sdk.spec = ArmSpec(name="l", role="leader", gripper="yam_teaching_handle", can_serial="A")
    sdk.robot = FakeRobot(6, gripper=False, handle=False)
    ticks = iter([0.0, 10.0])
    monkeypatch.setattr(arm_module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="encoder never reported"):
        YamArm.connect(sdk.spec, "A", encoder_timeout_s=1.0)
    assert sdk.robot.closed
    # subprocess.run uses time.monotonic internally, so restore the real clock first.
    monkeypatch.undo()
    assert _probe(sdk.locks) == "free"


@pytest.mark.parametrize("cleanup_failed", [False, True])
def test_sdk_factory_error_retains_lease_only_if_cleanup_failed(sdk, monkeypatch, cleanup_failed):
    error = KeyboardInterrupt("SDK construction cancelled")
    if cleanup_failed:
        error._yamkit_cleanup_failed = True

    def factory(**kwargs):
        raise error

    monkeypatch.setattr(get_robot, "get_yam_robot", factory)
    with pytest.raises(KeyboardInterrupt, match="SDK construction cancelled"):
        YamArm.connect(sdk.spec, "A")
    assert _probe(sdk.locks) == ("busy" if cleanup_failed else "free")


def test_sdk_close_failure_retains_lease_and_close_can_be_retried(sdk, monkeypatch):
    arm = YamArm.connect(sdk.spec, "A")
    close = sdk.robot.close

    def fail():
        raise RuntimeError("transmitter still active")

    monkeypatch.setattr(sdk.robot, "close", fail)
    with pytest.raises(RuntimeError, match="transmitter still active"):
        arm.close(settle_s=0)
    assert _probe(sdk.locks) == "busy"
    assert not arm._closed
    monkeypatch.setattr(sdk.robot, "close", close)
    arm.close(settle_s=0)
    assert _probe(sdk.locks) == "free"


def test_idle_failure_still_closes_sdk_and_releases(sdk, monkeypatch):
    arm = YamArm.connect(sdk.spec, "A")

    def fail():
        raise KeyboardInterrupt("idle cancelled")

    monkeypatch.setattr(sdk.robot, "enter_gravity_comp_idle", fail)
    with pytest.raises(KeyboardInterrupt, match="idle cancelled"):
        arm.close(settle_s=0)
    assert sdk.robot.closed
    assert _probe(sdk.locks) == "free"
    arm.close(settle_s=0)


def test_inherited_arm_cannot_touch_hardware_or_release_parent(sdk):
    arm = YamArm.connect(sdk.spec, "A")
    read_fd, write_fd = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        actions = [
            arm.read,
            lambda: arm.command(np.zeros(6)),
            lambda: arm.set_gains(np.ones(7), np.ones(7)),
            arm.gravity_idle,
            arm.close,
            lambda: arm.robot,
        ]
        rejected = 0
        try:
            for action in actions:
                try:
                    action()
                except RuntimeError as error:
                    if "inherited" in str(error):
                        rejected += 1
            os.write(write_fd, str(rejected).encode())
        finally:
            os._exit(0)
    os.close(write_fd)
    try:
        assert os.read(read_fd, 8) == b"6"
        assert os.waitpid(child, 0)[1] == 0
        assert _probe(sdk.locks) == "busy"
        assert not sdk.robot.closed and sdk.robot.commands == []
        arm.close(settle_s=0)
        assert _probe(sdk.locks) == "free"
    finally:
        os.close(read_fd)
