"""Guards for the local patches listed in third_party/i2rt.VERSION."""

import struct
import threading
from types import SimpleNamespace

import can
import numpy as np
import pytest
from i2rt.motor_drivers.dm_driver import PassiveEncoderReader


def _parse(counts: int):
    data = struct.pack("!B H h B", 0, counts, 0, 0)
    return PassiveEncoderReader._parse_encoder_message(None, can.Message(arbitration_id=0x50F, data=data))


def test_encoder_counts_wrap_negative():
    pos, _, _ = _parse(4091)  # -5 counts
    assert abs(pos - (-5 * 2 * 3.141592653589793 / 4096)) < 1e-6
    pos, _, _ = _parse(20)
    assert pos > 0


def test_wrap_correction_uses_joint_limits():
    """A base parked at +183° that reads -177° (inside ±π, but past the -150° stop) is unwrapped."""
    import numpy as np
    from i2rt.robots.get_robot import wrap_correction

    two_pi = 2 * np.pi
    base = np.array([-2.618 - 0.15, 3.1416 + 0.15])  # YAM joint 1 limits with the SDK's safety buffer
    assert wrap_correction(-3.0885, base) == two_pi  # -177° -> +183°, which is inside the limits
    assert wrap_correction(3.19, base) == 0.0  # +183° already fine
    assert wrap_correction(-2.5, base) == 0.0  # legal reading, untouched
    elbow = np.array([0.0 - 0.15, 3.1416 + 0.15])
    assert wrap_correction(-3.0, elbow) == two_pi  # -172° on a 0..180° joint can only be +188° (inside the buffered limit)
    assert wrap_correction(-1.0, elbow) == 0.0  # below the limit and shifting a turn does not help: leave it (the SDK then reports the violation)
    # no limits known (e.g. the gripper motor): original ±π behaviour
    assert wrap_correction(-3.3) == two_pi and wrap_correction(3.3) == -two_pi and wrap_correction(3.0) == 0.0


@pytest.mark.parametrize("arm_name", ["yam", "yam_pro", "yam_ultra", "yam_ultra_2", "big_yam"])
@pytest.mark.parametrize("gripper_name", ["linear_4310", "yam_teaching_handle", "no_gripper"])
def test_preflight_bounds_equal_original_sdk_ranges(arm_name, gripper_name):
    from i2rt.robots.get_robot import _load_joint_limits_from_xml, get_yam_joint_limits
    from i2rt.robots.utils import ArmType, GripperType, _load_arm_config

    arm, gripper = ArmType.from_string_name(arm_name), GripperType.from_string_name(gripper_name)
    original = _load_joint_limits_from_xml(arm.get_xml_path(), gripper.get_xml_path())
    original = original[:len(_load_arm_config(arm).motor_list)]
    original[:, 0] -= 0.15
    original[:, 1] += 0.15
    np.testing.assert_array_equal(get_yam_joint_limits(arm, gripper), original)
    assert original.shape == (6, 2)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_can_socket_initialization_failure_closes_bus(monkeypatch, error_type):
    from i2rt.motor_drivers.can_interface import CanInterface

    closed = []

    class Bus:
        @property
        def state(self):
            raise error_type("state failed")

        def shutdown(self):
            closed.append(True)

    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: Bus())
    with pytest.raises(error_type, match="state failed"):
        CanInterface()
    assert closed == [True]


def test_can_close_attempts_bus_after_notifier_failure(monkeypatch):
    from i2rt.motor_drivers.can_interface import CanInterface

    closed = []
    bus = SimpleNamespace(state=None, shutdown=lambda: closed.append("bus"))
    monkeypatch.setattr(can.interface, "Bus", lambda **kwargs: bus)
    interface = CanInterface()

    def stop():
        closed.append("notifier")
        raise KeyboardInterrupt("stop failed")

    interface.notifier = SimpleNamespace(stop=stop)
    with pytest.raises(KeyboardInterrupt, match="stop failed"):
        interface.close()
    assert closed == ["notifier", "bus"]
    interface.notifier = None
    interface.close()
    interface.close()
    assert closed == ["notifier", "bus", "bus"]


@pytest.fixture
def fake_chain_socket(monkeypatch):
    from i2rt.motor_drivers import dm_driver

    sockets = []

    class Socket:
        def __init__(self, **kwargs):
            self.closes = 0
            sockets.append(self)

        def close(self):
            self.closes += 1

    monkeypatch.setattr(dm_driver, "run_startup_checks", lambda *a, **kw: None)
    monkeypatch.setattr(dm_driver, "DMSingleMotorCanInterface", Socket)
    return sockets


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_partial_motor_enable_closes_socket(monkeypatch, fake_chain_socket, error_type):
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface

    def fail(self):
        raise error_type("partial enable failed")

    monkeypatch.setattr(DMChainCanInterface, "_motor_on", fail)
    with pytest.raises(error_type, match="partial enable failed"):
        DMChainCanInterface([(1, "DM4310")], [0.0], [1.0], channel="can-fake", start_thread=False)
    assert len(fake_chain_socket) == 1
    assert fake_chain_socket[0].closes == 1


def test_chain_close_joins_transmitter_before_socket(monkeypatch, fake_chain_socket):
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface

    started = threading.Event()

    def enable(self):
        self.state = [SimpleNamespace(torque=0.0)]
        self.running = True

    def worker(self):
        started.set()
        while self.running:
            threading.Event().wait(0.001)

    monkeypatch.setattr(DMChainCanInterface, "_motor_on", enable)
    monkeypatch.setattr(DMChainCanInterface, "_set_torques_and_update_state", worker)
    chain = DMChainCanInterface([(1, "DM4310")], [0.0], [1.0], channel="can-fake")
    assert started.wait(1)

    def close_socket():
        assert not chain._control_thread.is_alive()
        fake_chain_socket[0].closes += 1

    fake_chain_socket[0].close = close_socket
    chain.close()
    chain.close()
    assert not chain._control_thread.is_alive()
    assert fake_chain_socket[0].closes == 1


def test_chain_close_attempts_socket_when_join_is_cancelled():
    from i2rt.motor_drivers.dm_driver import DMChainCanInterface

    calls = []
    chain = DMChainCanInterface.__new__(DMChainCanInterface)
    chain._closed = False

    def join(**kwargs):
        raise KeyboardInterrupt("join cancelled")

    chain._control_thread = SimpleNamespace(is_alive=lambda: True, join=join)
    chain.motor_interface = SimpleNamespace(close=lambda: calls.append("socket"))
    with pytest.raises(KeyboardInterrupt, match="join cancelled"):
        chain.close()
    assert calls == ["socket"]
    assert not chain.running


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_gripper_calibration_failure_closes_active_chain(monkeypatch, error_type):
    from i2rt.robots import motor_chain_robot

    class Chain:
        def __len__(self):
            return 7

        def close(self):
            self.closed = True

    chain = Chain()

    def calibration(**kwargs):
        raise error_type("calibration failed")

    monkeypatch.setattr(motor_chain_robot, "detect_gripper_limits", calibration)
    with pytest.raises(error_type, match="calibration failed"):
        motor_chain_robot.MotorChainRobot(chain, gripper_index=6, enable_gripper_calibration=True)
    assert chain.closed


def test_constructor_preserves_cancellation_and_reports_incomplete_cleanup(monkeypatch):
    from i2rt.robots import motor_chain_robot

    class Chain:
        def __len__(self):
            return 7

        def close(self):
            raise RuntimeError("socket close failed")

    def calibration(**kwargs):
        raise KeyboardInterrupt("calibration cancelled")

    monkeypatch.setattr(motor_chain_robot, "detect_gripper_limits", calibration)
    with pytest.raises(KeyboardInterrupt, match="calibration cancelled") as error:
        motor_chain_robot.MotorChainRobot(Chain(), gripper_index=6, enable_gripper_calibration=True)
    assert error.value._yamkit_cleanup_failed
    assert "socket close failed" in error.value.__notes__[0]


def test_failure_after_robot_server_start_joins_server_and_closes_chain(monkeypatch):
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    servers = []

    class Chain:
        def __len__(self):
            return 6

        def read_states(self):
            return None  # conversion is replaced below; no hardware accessed

        def close(self):
            self.closed = True

    def worker(self):
        servers.append(threading.current_thread())
        self._stop_event.wait(5)

    def fail_command(self, position):
        raise KeyboardInterrupt("first command cancelled")

    monkeypatch.setattr(MotorChainRobot, "start_server", worker)
    monkeypatch.setattr(MotorChainRobot, "command_joint_pos", fail_command)
    monkeypatch.setattr(
        MotorChainRobot, "_motor_state_to_joint_state", lambda self, states: SimpleNamespace(pos=np.zeros(6))
    )
    chain = Chain()
    with pytest.raises(KeyboardInterrupt, match="first command cancelled"):
        MotorChainRobot(
            chain, use_gravity_comp=False, zero_gravity_mode=False,
            joint_limits=np.tile([-1.0, 1.0], (6, 1)),
        )
    assert chain.closed
    assert len(servers) == 1 and not servers[0].is_alive()


def test_factory_failure_after_chain_open_closes_chain(monkeypatch):
    from i2rt.robots import get_robot

    class Chain:
        closed = False

        def read_states(self):
            raise KeyboardInterrupt("initial feedback cancelled")

        def close(self):
            self.closed = True

    chain = Chain()
    monkeypatch.setattr(get_robot, "DMChainCanInterface", lambda *a, **kw: chain)
    monkeypatch.setattr(get_robot, "combine_arm_and_gripper_xml", lambda *a, **kw: "unused")
    with pytest.raises(KeyboardInterrupt, match="initial feedback cancelled"):
        get_robot.get_yam_robot()
    assert chain.closed


def test_robot_cleanup_attempts_recorder_after_chain_failure():
    from i2rt.robots.motor_chain_robot import MotorChainRobot

    calls = []
    robot = MotorChainRobot.__new__(MotorChainRobot)
    robot._closed = False
    robot._stop_event = threading.Event()
    robot._server_thread = None

    def close_chain():
        calls.append("chain")
        raise RuntimeError("chain close failed")

    robot.motor_chain = SimpleNamespace(close=close_chain)
    robot.stop_mcap_recording = lambda: calls.append("recorder")
    with pytest.raises(RuntimeError, match="chain close failed"):
        robot.close()
    assert calls == ["chain", "recorder"]
    assert robot._stop_event.is_set()
    robot.motor_chain.close = lambda: calls.append("chain retry")
    robot.close()
    robot.close()
    assert calls == ["chain", "recorder", "chain retry", "recorder"]
