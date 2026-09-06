import pytest

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
    first.arm("left_leader").joint_offsets = [0.129, 0, 0, 0, 0, 0]
    # next boot: interfaces renumbered, probes come back in a different order
    shuffled = [pr["L3"], pr["F2"], pr["L1"], pr["F0"]]
    ifaces2 = [_iface(p.iface, {"can0": "S0", "can1": "S1", "can2": "S2", "can3": "S3"}[p.iface]) for p in shuffled]
    again = suggest_rig(shuffled, ifaces2, existing=first)
    assert again.arm("left_follower").can_serial == "S2" and again.arm("right_follower").can_serial == "S0"
    assert again.arm("left_follower").gripper_limits == [6.4, 1.2]
    assert again.arm("left_leader").rest_pose == [0.0] * 6
    assert again.arm("left_leader").joint_offsets == [0.129, 0, 0, 0, 0, 0]  # alignment survives rediscovery
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


def test_rediscovery_preserves_hub_preferences(tmp_path):
    from yamkit.config import RigConfig

    pr = _two_pair_probes()
    ifaces = [_iface(f"can{i}", f"S{i}") for i in range(4)]
    existing = suggest_rig(list(pr.values()), ifaces)
    existing.hub.username = "rig-owner"
    existing.hub.private = False
    existing.hub.datasets = "both"
    existing.control.max_joint_speed = 0.4
    existing.save(tmp_path / "rig.yaml")

    again = suggest_rig(list(pr.values()), ifaces, existing)
    again.save()
    saved = RigConfig.load(existing.path)
    assert saved.hub == existing.hub
    assert saved.control == existing.control


def test_rediscovery_without_serials_keeps_verified_identity():
    pr = _two_pair_probes()
    ifaces = [_iface(f"can{i}", None) for i in range(4)]
    existing = suggest_rig(list(pr.values()), ifaces)
    # The interface is the only available identity; a verified physical swap must survive.
    existing.arm("left_follower").can_iface, existing.arm("right_follower").can_iface = "can2", "can0"
    existing.arm("left_follower").gripper_limits = [6.4, 1.2]
    existing.arm("left_leader").joint_offsets = [0.1, 0, 0, 0, 0, 0]
    existing.arm("left_leader").rest_pose = [0.2] * 6
    expected_pairs = [(p.leader, p.follower) for p in existing.pairs]

    for _ in range(2):
        existing = suggest_rig(list(reversed(pr.values())), ifaces, existing)
        assert set(existing.arms) == {"left_leader", "left_follower", "right_leader", "right_follower"}
        assert existing.arm("left_follower").can_iface == "can2"
        assert existing.arm("right_follower").can_iface == "can0"
        assert existing.arm("left_follower").gripper_limits == [6.4, 1.2]
        assert existing.arm("left_leader").joint_offsets == [0.1, 0, 0, 0, 0, 0]
        assert existing.arm("left_leader").rest_pose == [0.2] * 6
        assert [(p.leader, p.follower) for p in existing.pairs] == expected_pairs
        assert existing.validate() == []


def test_rediscovery_can_add_serial_to_interface_only_arm():
    pr = _two_pair_probes()
    existing = suggest_rig(list(pr.values()), [_iface(f"can{i}", None) for i in range(4)])
    existing.arm("left_follower").gripper_limits = [6.4, 1.2]

    again = suggest_rig(list(pr.values()), [_iface(f"can{i}", f"S{i}") for i in range(4)], existing)
    assert set(again.arms) == set(existing.arms)
    assert again.arm("left_follower").can_serial == "S0"
    assert again.arm("left_follower").can_iface is None
    assert again.arm("left_follower").gripper_limits == [6.4, 1.2]
    assert again.validate() == []


@pytest.mark.parametrize("serials", [True, False])
@pytest.mark.parametrize("changed_arm,classification", [
    ("L1", "arm_no_gripper"), ("L1", "follower"), ("F0", "leader"),
])
def test_rediscovery_keeps_known_identity_when_probe_role_differs(serials, changed_arm, classification, caplog):
    pr = _two_pair_probes()
    ifaces = [_iface(f"can{i}", f"S{i}" if serials else None) for i in range(4)]
    existing = suggest_rig(list(pr.values()), ifaces)
    existing.arm("left_leader").joint_offsets = [0.1, 0, 0, 0, 0, 0]
    existing.arm("left_leader").rest_pose = [0.2] * 6
    existing.arm("left_follower").gripper_limits = [6.4, 1.2]
    expected_pairs = [(p.leader, p.follower) for p in existing.pairs]
    probe = pr[changed_arm]
    probe.motors = probe.motors[:6] + ([MotorProbe(7, 10.0)] if classification == "follower" else [])
    probe.encoder_versions = ["dev1:v2.4.0"] if classification == "leader" else []
    assert probe.classification == classification

    again = suggest_rig(list(pr.values()), ifaces, existing)
    assert set(again.arms) == set(existing.arms)
    for name, old in existing.arms.items():
        current = again.arm(name)
        assert current.role == old.role
        assert current.gripper == old.gripper
        assert current.can_serial == old.can_serial
        assert current.can_iface == old.can_iface
        assert current.joint_offsets == old.joint_offsets
        assert current.rest_pose == old.rest_pose
        assert current.gripper_limits == old.gripper_limits
    assert [(p.leader, p.follower) for p in again.pairs] == expected_pairs
    assert "keeping configured" in caplog.text
    assert again.validate() == []
