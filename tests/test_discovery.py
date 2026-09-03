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


def _two_pair_probes():
    arm = [MotorProbe(i, 40.0 if i <= 3 else 10.0) for i in range(1, 7)]
    return {
        "F0": ChannelProbe("can0", arm + [MotorProbe(7, 10.0)]),
        "L1": ChannelProbe("can1", arm, ["dev1:v2.4.0"]),
        "F2": ChannelProbe("can2", arm + [MotorProbe(7, 10.0)]),
        "L3": ChannelProbe("can3", arm, ["dev1:v2.4.0"]),
    }


def test_rediscovery_keeps_names_by_serial_when_bus_order_changes():
    """After a reboot can0..can3 can be renumbered; a verified left/right must not flip."""
    pr = _two_pair_probes()
    ifaces = [_iface(f"can{i}", s) for i, s in enumerate(["S0", "S1", "S2", "S3"])]
    first = suggest_rig(list(pr.values()), ifaces)
    # user verified: what discovery called left is really right → swapped + calibrated + rest pose
    first.arm("left_follower").can_serial, first.arm("right_follower").can_serial = "S2", "S0"
    first.arm("left_follower").gripper_limits = [6.4, 1.2]
    first.arm("left_leader").rest_pose = [0.0] * 6
    # next boot: interfaces renumbered, probes come back in a different order
    shuffled = [pr["L3"], pr["F2"], pr["L1"], pr["F0"]]
    ifaces2 = [_iface(p.iface, {"can0": "S0", "can1": "S1", "can2": "S2", "can3": "S3"}[p.iface]) for p in shuffled]
    again = suggest_rig(shuffled, ifaces2, existing=first)
    assert again.arm("left_follower").can_serial == "S2" and again.arm("right_follower").can_serial == "S0"
    assert again.arm("left_follower").gripper_limits == [6.4, 1.2]
    assert again.arm("left_leader").rest_pose == [0.0] * 6
    assert list(again.arms) == ["left_leader", "left_follower", "right_leader", "right_follower"]
    assert [(p.leader, p.follower) for p in again.pairs] == [("left_leader", "left_follower"), ("right_leader", "right_follower")]
    assert again.validate() == []


def test_rediscovery_keeps_absent_arm_and_names_new_adapter():
    from yamkit.discovery import absent_arms

    pr = _two_pair_probes()
    ifaces = [_iface(f"can{i}", s) for i, s in enumerate(["S0", "S1", "S2", "S3"])]
    first = suggest_rig(list(pr.values()), ifaces)
    # right follower's adapter unplugged, a brand-new follower adapter appears
    probes = [pr["F0"], pr["L1"], pr["L3"], ChannelProbe("can4", pr["F2"].motors)]
    ifaces2 = [_iface("can0", "S0"), _iface("can1", "S1"), _iface("can3", "S3"), _iface("can4", "NEW")]
    again = suggest_rig(probes, ifaces2, existing=first)
    assert again.arm("right_follower").can_serial == "S2"  # kept, not re-assigned to the new adapter
    assert [a.name for a in absent_arms(again, ifaces2)] == ["right_follower"]
    new = [a for a in again.arms.values() if a.can_serial == "NEW"]
    assert len(new) == 1 and new[0].role == "follower" and new[0].name == "third_follower"
    assert "verify" in new[0].notes
    assert ("right_leader", "right_follower") in [(p.leader, p.follower) for p in again.pairs]


def test_suggest_rig_cameras_argument():
    pr = _two_pair_probes()
    ifaces = [_iface(f"can{i}", s) for i, s in enumerate(["S0", "S1", "S2", "S3"])]
    existing = suggest_rig(list(pr.values()), ifaces)
    existing.cameras = {"top": {"type": "opencv", "index_or_path": "/dev/video0"}}
    assert suggest_rig(list(pr.values()), ifaces, existing).cameras == existing.cameras  # untouched by default
    new = {"left_wrist": {"type": "opencv", "index_or_path": "/dev/video4"}}
    assert suggest_rig(list(pr.values()), ifaces, existing, cameras=new).cameras == new
