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


def suggest_rig(probes: list[ChannelProbe], ifaces: list[CanIface], existing: RigConfig | None = None) -> RigConfig:
    """Draft a rig from probe results. Names are provisional (`left_*`/`right_*` in discovery order):
    the user should verify which physical arm is which and edit names/sides."""
    by_name = {i.name: i for i in ifaces}
    leaders = [p for p in probes if p.classification == "leader"]
    followers = [p for p in probes if p.classification in ("follower", "arm_no_gripper")]
    sides = ["left", "right", "third", "fourth"]
    arms: dict[str, ArmSpec] = {}
    pairs: list[PairSpec] = []
    for idx in range(max(len(leaders), len(followers))):
        side = sides[idx] if idx < len(sides) else f"arm{idx}"
        lname = fname = None
        if idx < len(leaders):
            p = leaders[idx]
            lname = f"{side}_leader"
            arms[lname] = ArmSpec(
                name=lname, role="leader", side=side, gripper=p.suggested_gripper,
                can_serial=by_name[p.iface].serial, can_iface=None if by_name[p.iface].serial else p.iface,
                notes=f"discovered on {p.iface}",
            )
        if idx < len(followers):
            p = followers[idx]
            fname = f"{side}_follower"
            arms[fname] = ArmSpec(
                name=fname, role="follower", side=side, gripper=p.suggested_gripper,
                can_serial=by_name[p.iface].serial, can_iface=None if by_name[p.iface].serial else p.iface,
                notes=f"discovered on {p.iface}",
            )
        if lname and fname:
            pairs.append(PairSpec(leader=lname, follower=fname))
    rig = RigConfig(arms=arms, pairs=pairs, control=ControlSpec())
    if existing is not None:
        # keep hand-edited fields (rest poses, gripper limits, cameras, control) when serials match
        by_serial = {a.can_serial: a for a in existing.arms.values() if a.can_serial}
        for a in rig.arms.values():
            old = by_serial.get(a.can_serial or "")
            if old:
                a.rest_pose, a.gripper_limits, a.gripper, a.arm_type = old.rest_pose, old.gripper_limits, old.gripper, old.arm_type
        rig.cameras, rig.control = existing.cameras, existing.control
    return rig
