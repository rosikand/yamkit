from yamkit.can import CanIface
from yamkit.discovery import ChannelProbe, MotorProbe, suggest_rig


def _iface(name, serial):
    return CanIface(name, True, 1000000, serial, "CANable", "x", "3-1", 0, 0, 0)


def test_classification():
    arm = [MotorProbe(i, 40.0 if i <= 3 else 10.0) for i in range(1, 7)]
    assert ChannelProbe("can0", arm + [MotorProbe(7, 10.0)]).classification == "follower"
    assert ChannelProbe("can1", arm, ["dev1:v2.4.0"]).classification == "leader"
    assert ChannelProbe("can2", arm).classification == "arm_no_gripper"
    assert ChannelProbe("can3").classification == "empty"
    assert ChannelProbe("can3", error="down").classification == "error"
    assert ChannelProbe("can4", arm[:3]).classification == "partial"
    assert MotorProbe(1, 40.0).motor_type == "DM4340"
    bad = ChannelProbe("can5", [MotorProbe(1, 10.0)] + arm[1:])
    assert bad.type_mismatches and "motor 1" in bad.type_mismatches[0]


def test_suggest_rig_two_pairs():
    arm = [MotorProbe(i, 40.0 if i <= 3 else 10.0) for i in range(1, 7)]
    probes = [
        ChannelProbe("can0", arm + [MotorProbe(7, 10.0)]),
        ChannelProbe("can1", arm, ["dev1:v2.4.0"]),
        ChannelProbe("can2", arm + [MotorProbe(7, 10.0)]),
        ChannelProbe("can3", arm, ["dev1:v2.4.0"]),
    ]
    ifaces = [_iface(f"can{i}", f"S{i}") for i in range(4)]
    rig = suggest_rig(probes, ifaces)
    assert set(rig.arms) == {"left_leader", "left_follower", "right_leader", "right_follower"}
    assert rig.arm("left_follower").can_serial == "S0" and rig.arm("left_leader").can_serial == "S1"
    assert rig.arm("left_leader").gripper == "yam_teaching_handle"
    assert [(p.leader, p.follower) for p in rig.pairs] == [("left_leader", "left_follower"), ("right_leader", "right_follower")]
    assert rig.validate() == []


def test_read_register_matches_on_echo_not_arbitration_id():
    import struct

    from yamkit.discovery import read_register_float

    class Msg:
        def __init__(self, arb, data):
            self.arbitration_id, self.data = arb, bytearray(data)

    class FakeBus:
        channel_info = "fake"

        def __init__(self):
            self.queue = []

        def send(self, m):
            mid, reg = m.data[0], m.data[3]
            # a stale foreign frame first, then the real reply from master id 0x10+id
            self.queue = [Msg(0x7FF, [9, 0, 0x33, 1, 0, 0, 0, 0]), Msg(0x10 + mid, [mid, 0, 0x33, reg, *struct.pack("<f", 40.0)])]

        def recv(self, timeout=0):
            return self.queue.pop(0) if self.queue else None

    assert read_register_float(FakeBus(), 1, 20) == 40.0
    assert read_register_float(FakeBus.__new__(FakeBus).__class__(), 3, 20) == 40.0
