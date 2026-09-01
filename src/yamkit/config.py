"""Rig configuration: which arm is on which CAN adapter, how they pair up, cameras, control knobs.

The rig file (``configs/rig.yaml``) is the single source of truth used by the CLI and by the
LeRobot plugins. CAN adapters are identified by USB serial number so the mapping survives
reboots and re-plugging, without touching udev rules.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .paths import DEFAULT_RIG, resolve

Role = Literal["leader", "follower"]

ARM_TYPES = ("yam", "yam_pro", "yam_ultra", "yam_ultra_2", "big_yam")
GRIPPER_TYPES = (
    "crank_4310",
    "linear_3507",
    "linear_4310",
    "flexible_4310",
    "yam_teaching_handle",
    "no_gripper",
)
MOTOR_GRIPPERS = ("crank_4310", "linear_3507", "linear_4310", "flexible_4310")
N_JOINTS = 6


@dataclass
class ArmSpec:
    name: str
    role: Role
    side: str | None = None  # "left" / "right" (informational)
    arm_type: str = "yam"
    gripper: str = "linear_4310"
    can_serial: str | None = None  # USB serial of the CAN adapter (preferred)
    can_iface: str | None = None  # explicit interface name (overrides serial lookup)
    gripper_limits: list[float] | None = None  # [closed, open] motor rad; set → skips auto-calibration
    rest_pose: list[float] | None = None  # 6 joint angles (rad) for `yamkit rest`
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.role not in ("leader", "follower"):
            raise ValueError(f"{self.name}: role must be 'leader' or 'follower', got {self.role!r}")
        if self.arm_type not in ARM_TYPES:
            raise ValueError(f"{self.name}: arm_type must be one of {ARM_TYPES}, got {self.arm_type!r}")
        if self.gripper not in GRIPPER_TYPES:
            raise ValueError(f"{self.name}: gripper must be one of {GRIPPER_TYPES}, got {self.gripper!r}")
        if self.rest_pose is not None and len(self.rest_pose) != N_JOINTS:
            raise ValueError(f"{self.name}: rest_pose needs {N_JOINTS} values")

    @property
    def has_motor_gripper(self) -> bool:
        return self.gripper in MOTOR_GRIPPERS

    @property
    def has_handle(self) -> bool:
        return self.gripper == "yam_teaching_handle"

    @property
    def n_dofs(self) -> int:
        return N_JOINTS + (1 if self.has_motor_gripper else 0)


@dataclass
class PairSpec:
    leader: str
    follower: str


@dataclass
class ControlSpec:
    teleop_hz: float = 100.0  # yamkit teleop loop rate
    sync_seconds: float = 3.0  # follower catch-up move when engaging
    bilateral_kp: float = 0.0  # 0 = no force feedback on the leader; 0.1–0.2 recommended if used
    engage_button: int = 0  # teaching-handle button index that toggles engage
    max_joint_speed: float = 3.0  # rad/s clamp on follower position targets
    max_gripper_speed: float = 3.0  # (normalized units)/s clamp on the gripper target


@dataclass
class RigConfig:
    arms: dict[str, ArmSpec] = field(default_factory=dict)
    pairs: list[PairSpec] = field(default_factory=list)
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)  # LeRobot camera configs
    control: ControlSpec = field(default_factory=ControlSpec)
    path: Path | None = None

    # ----- accessors --------------------------------------------------------------------------
    def arm(self, name: str) -> ArmSpec:
        try:
            return self.arms[name]
        except KeyError:
            raise KeyError(f"arm {name!r} not in rig (have: {sorted(self.arms)})") from None

    def leaders(self) -> list[ArmSpec]:
        return [a for a in self.arms.values() if a.role == "leader"]

    def followers(self) -> list[ArmSpec]:
        return [a for a in self.arms.values() if a.role == "follower"]

    def pair_for(self, name: str) -> PairSpec | None:
        for p in self.pairs:
            if name in (p.leader, p.follower):
                return p
        return None

    def validate(self) -> list[str]:
        problems: list[str] = []
        for p in self.pairs:
            for n, role in ((p.leader, "leader"), (p.follower, "follower")):
                if n not in self.arms:
                    problems.append(f"pair references unknown arm {n!r}")
                elif self.arms[n].role != role:
                    problems.append(f"{n!r} is used as {role} but has role {self.arms[n].role!r}")
        for a in self.arms.values():
            if not a.can_serial and not a.can_iface:
                problems.append(f"{a.name}: needs can_serial or can_iface")
        return problems

    # ----- (de)serialisation ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        def arm_dict(a: ArmSpec) -> dict[str, Any]:
            d = dataclasses.asdict(a)
            d.pop("name")
            return {k: v for k, v in d.items() if v is not None}

        return {
            "version": 1,
            "arms": {n: arm_dict(a) for n, a in self.arms.items()},
            "pairs": [dataclasses.asdict(p) for p in self.pairs],
            "cameras": self.cameras,
            "control": dataclasses.asdict(self.control),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], path: Path | None = None) -> RigConfig:
        arms = {n: ArmSpec(name=n, **(spec or {})) for n, spec in (d.get("arms") or {}).items()}
        pairs = [PairSpec(**p) for p in (d.get("pairs") or [])]
        control = ControlSpec(**(d.get("control") or {}))
        return cls(arms=arms, pairs=pairs, cameras=d.get("cameras") or {}, control=control, path=path)

    @classmethod
    def load(cls, path: str | Path | None = None) -> RigConfig:
        p = resolve(path) if path else DEFAULT_RIG
        if not p.is_file():
            raise FileNotFoundError(f"rig file not found: {p} (run `yamkit discover --write` to create one)")
        with open(p) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data, path=p)

    def save(self, path: str | Path | None = None) -> Path:
        p = resolve(path) if path else (self.path or DEFAULT_RIG)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write("# yamkit rig configuration — see README.md\n")
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
        self.path = p
        return p
