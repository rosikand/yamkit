"""Passive discovery of what is attached to each CAN interface.

Uses the Damiao register-read frame (sent to 0x7FF, cmd 0x33; answered from the motor's master
id) which every DM motor answers *without being enabled*, so probing never energises a motor. The teaching-handle
encoder is detected with its version request. Result: which channel is a follower (motors
1–7), which is a leader (motors 1–6 + handle encoder), and the gear ratios (motor types).
"""

from __future__ import annotations

import logging
import math
import struct
import time
from dataclasses import dataclass, field

from .can import CanIface, list_can_interfaces
from .config import ArmSpec, ControlSpec, PairSpec, RigConfig

log = logging.getLogger(__name__)

DM_BROADCAST = 0x7FF
DM_CMD_READ = 0x33
REG_GR = 20  # gear ratio (read-only, identifies the motor type)
GEAR_TO_TYPE = {10.0: "DM4310", 40.0: "DM4340", 7.0: "DM3507", 48.0: "DM6248", 9.0: "DM8009"}
YAM_MOTOR_TYPES = {1: "DM4340", 2: "DM4340", 3: "DM4340", 4: "DM4310", 5: "DM4310", 6: "DM4310", 7: "DM4310"}
ARM_IDS = frozenset(range(1, 7))
GRIPPER_ID = 7


@dataclass
class MotorProbe:
    id: int
    gear_ratio: float

    @property
    def motor_type(self) -> str | None:
        return GEAR_TO_TYPE.get(round(self.gear_ratio, 1))


@dataclass
class ChannelProbe:
    iface: str
    motors: list[MotorProbe] = field(default_factory=list)
    encoder_versions: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def motor_ids(self) -> set[int]:
        return {m.id for m in self.motors}

    @property
    def classification(self) -> str:
        ids = self.motor_ids
        if self.error:
            return "error"
        if not ids and not self.encoder_versions:
            return "empty"
        if ARM_IDS <= ids and GRIPPER_ID in ids:
            return "follower"
        if ARM_IDS <= ids and self.encoder_versions:
            return "leader"
        if ARM_IDS <= ids:
            return "arm_no_gripper"
        return "partial"

    @property
    def suggested_gripper(self) -> str:
        return {"follower": "linear_4310", "leader": "yam_teaching_handle"}.get(self.classification, "no_gripper")

    @property
    def type_mismatches(self) -> list[str]:
        out = []
        for m in self.motors:
            exp = YAM_MOTOR_TYPES.get(m.id)
            if exp and m.motor_type and m.motor_type != exp:
                out.append(f"motor {m.id}: found {m.motor_type} (Gr={m.gear_ratio:g}), YAM expects {exp}")
        return out


def _drain(bus) -> None:
    while bus.recv(timeout=0) is not None:
        pass


def read_register_float(bus, motor_id: int, reg: int, retries: int = 3, timeout: float = 0.02) -> float | None:
    """Read a float32 register from a DM motor. Returns None if the motor does not answer."""
    import can

    payload = [motor_id, 0x00, DM_CMD_READ, reg, 0, 0, 0, 0]
    for _ in range(retries):
        _drain(bus)
        try:
            bus.send(can.Message(arbitration_id=DM_BROADCAST, data=payload, is_extended_id=False))
        except can.CanError as e:
            log.debug("send failed on %s: %s", bus.channel_info, e)
            time.sleep(0.005)
            continue
        deadline = time.monotonic() + timeout
        while (remaining := deadline - time.monotonic()) > 0:
            msg = bus.recv(timeout=remaining)
            # The reply comes from the motor's master id (0x10 + id in the YAM's "p16" scheme), not from
            # 0x7FF, so match on the payload echo (motor id + register address) like the vendor tool does.
            if (
                msg is not None
                and len(msg.data) == 8
                and msg.data[0] == motor_id
                and msg.data[2] == DM_CMD_READ
                and msg.data[3] == reg
            ):
                return struct.unpack("<f", bytes(msg.data[4:8]))[0]
    return None


def probe_channel(iface: str, motor_ids: range = range(1, 8), probe_encoder: bool = True) -> ChannelProbe:
    import can

    probe = ChannelProbe(iface=iface)
    try:
        bus = can.Bus(channel=iface, interface="socketcan")
    except Exception as e:  # noqa: BLE001 — surface anything (iface down, no permission...)
        probe.error = f"{type(e).__name__}: {e}"
        return probe
    try:
        for mid in motor_ids:
            gr = read_register_float(bus, mid, REG_GR)
            if gr is not None and not math.isnan(gr):
                probe.motors.append(MotorProbe(mid, gr))
        if probe_encoder:
            try:
                from i2rt.utils.encoder_manager import PassiveJointEncoder

                _drain(bus)
                for v in PassiveJointEncoder(bus).get_version(timeout=0.3):
                    probe.encoder_versions.append(f"dev{v.device}:v{v.major}.{v.minor}.{v.patch}")
            except Exception as e:  # noqa: BLE001
                log.debug("encoder probe failed on %s: %s", iface, e)
    finally:
        bus.shutdown()
    return probe


def probe_all(ifaces: list[CanIface] | None = None) -> list[ChannelProbe]:
    ifaces = ifaces if ifaces is not None else list_can_interfaces()
    out = []
    for i in ifaces:
        if not i.up:
            out.append(ChannelProbe(iface=i.name, error="interface is DOWN"))
            continue
        out.append(probe_channel(i.name))
    return out


def suggest_rig(probes: list[ChannelProbe], ifaces: list[CanIface], existing: RigConfig | None = None, cameras: dict | None = None) -> RigConfig:
    """Draft a rig from probe results.

    An adapter whose serial is already in `existing` keeps that arm's name, side, gripper,
    calibration and rest pose (so re-running discovery after a reboot or a cable change never
    flips a verified left/right). New adapters get provisional names (`left_*`, `right_*`, ... in
    discovery order) — the user must verify which physical arm is which. Arms in `existing`
    whose adapter is not attached right now are kept as they are (the CLI warns about them)."""
    by_name = {i.name: i for i in ifaces}
    old_arms = list(existing.arms.values()) if existing else []
    old_by_serial = {a.can_serial: a for a in old_arms if a.can_serial}
    sides = ["left", "right", "third", "fourth"]
    arms: dict[str, ArmSpec] = {}
    seen_serials: set[str] = set()

    def place(role: str, plist: list[ChannelProbe]) -> None:
        pending = []
        for p in plist:
            iface = by_name[p.iface]
            old = old_by_serial.get(iface.serial or "")
            if iface.serial:
                seen_serials.add(iface.serial)
            if old is not None and old.role == role and old.name not in arms:
                arms[old.name] = ArmSpec(
                    name=old.name, role=role, side=old.side, arm_type=old.arm_type, gripper=old.gripper,
                    can_serial=iface.serial, gripper_limits=old.gripper_limits, rest_pose=old.rest_pose,
                    joint_offsets=old.joint_offsets, notes=f"adapter seen as {p.iface}",
                )
            else:
                pending.append(p)
        used = {a.side for a in arms.values() if a.role == role} | {a.side for a in old_arms if a.role == role and a.name not in arms and a.can_serial not in seen_serials}
        free_sides = [s for s in sides if s not in used]
        for idx, p in enumerate(pending):
            side = free_sides[idx] if idx < len(free_sides) else f"arm{idx}"
            name = f"{side}_{role}"
            iface = by_name[p.iface]
            arms[name] = ArmSpec(
                name=name, role=role, side=side, gripper=p.suggested_gripper,
                can_serial=iface.serial, can_iface=None if iface.serial else p.iface,
                notes=f"discovered on {p.iface} — verify left/right",
            )

    place("leader", [p for p in probes if p.classification == "leader"])
    place("follower", [p for p in probes if p.classification in ("follower", "arm_no_gripper")])
    for a in old_arms:  # adapters not attached right now: keep, the user decides
        if a.name not in arms and (a.can_serial not in seen_serials or not a.can_serial):
            arms[a.name] = a
    order = {s: i for i, s in enumerate(sides)}
    arms = dict(sorted(arms.items(), key=lambda kv: (order.get(kv[1].side or "", 99), 0 if kv[1].role == "leader" else 1, kv[0])))

    pairs: list[PairSpec] = []
    paired: set[str] = set()
    for p in existing.pairs if existing else []:
        if p.leader in arms and p.follower in arms and arms[p.leader].role == "leader" and arms[p.follower].role == "follower":
            pairs.append(PairSpec(p.leader, p.follower))
            paired |= {p.leader, p.follower}
    leaders = [a for a in arms.values() if a.role == "leader" and a.name not in paired]
    followers = [a for a in arms.values() if a.role == "follower" and a.name not in paired]
    for lead in list(leaders):  # same side first
        mate = next((f for f in followers if f.side and f.side == lead.side), None)
        if mate:
            pairs.append(PairSpec(lead.name, mate.name))
            leaders.remove(lead)
            followers.remove(mate)
    for lead, mate in zip(leaders, followers):  # then whatever is left, in order
        pairs.append(PairSpec(lead.name, mate.name))
    pairs.sort(key=lambda p: order.get(arms[p.leader].side or "", 99))

    rig = RigConfig(arms=arms, pairs=pairs, control=existing.control if existing else ControlSpec())
    rig.cameras = cameras if cameras is not None else (existing.cameras if existing else {})
    return rig


def absent_arms(rig: RigConfig, ifaces: list[CanIface]) -> list[ArmSpec]:
    """Arms in the rig whose CAN adapter is not plugged in right now."""
    present = {i.serial for i in ifaces if i.serial} | {i.name for i in ifaces}
    return [a for a in rig.arms.values() if (a.can_serial or a.can_iface) not in present]
