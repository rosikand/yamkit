"""`YamArm`: a thin, safe wrapper around the i2rt `MotorChainRobot`.

* joint positions in radians (6), gripper normalised 0 (closed) … 1 (open)
* leader arms expose the teaching-handle trigger as `gripper` and its two buttons
* `command()` rate-limits position targets (max joint / gripper speed) so a far-away target
  (teleop engage, policy glitch) turns into a bounded-speed move instead of a jump
* leaders with `joint_offsets` (from `yamkit align`) read and are commanded in their follower's frame
* `go_home()` is the slow, interruptible move to the parked pose used at session start/stop
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from .config import N_JOINTS, ArmSpec

log = logging.getLogger(__name__)

STALE_COMMAND_S = 0.5  # after this long without commands, ramp restarts from the measured pose
COMPLIANT_KP_SCALE = 0.15  # gains for moving a leader home: gentle enough that a hand on the handle wins
COMPLIANT_KD_SCALE = 0.3
HOME_MIN_S = 0.5  # shortest go_home move, so a tiny correction is still a visible glide


@dataclass
class ArmState:
    t: float
    q: np.ndarray  # (6,) rad
    qd: np.ndarray  # (6,) rad/s
    tau: np.ndarray  # (6,) Nm (motor feedback)
    gripper: float | None  # 0..1; follower: motorised gripper opening, leader: trigger
    buttons: tuple[bool, ...] | None  # teaching-handle buttons (leader only)

    def vector(self) -> np.ndarray:
        return np.concatenate([self.q, [self.gripper]]) if self.gripper is not None else self.q.copy()


class YamArm:
    def __init__(
        self,
        spec: ArmSpec,
        channel: str,
        robot: Any,
        *,
        max_joint_speed: float = 3.0,
        max_gripper_speed: float = 3.0,
    ) -> None:
        self.spec = spec
        self.channel = channel
        self._robot = robot
        self.max_joint_speed = float(max_joint_speed)
        self.max_gripper_speed = float(max_gripper_speed)
        self._n_dofs = int(robot.num_dofs())
        info = robot.get_robot_info()
        self.default_kp = np.array(info["kp"], dtype=float)
        self.default_kd = np.array(info["kd"], dtype=float)
        self.gripper_limits = info.get("gripper_limits")
        self._offsets = np.asarray(spec.joint_offsets, dtype=float) if spec.joint_offsets else None
        self._last_cmd: np.ndarray | None = None
        self._last_cmd_t: float | None = None
        self._gains_zeroed = False
        self._closed = False

    # ----- construction ---------------------------------------------------------------------
    @classmethod
    def connect(
        cls,
        spec: ArmSpec,
        channel: str,
        *,
        zero_gravity: bool = True,
        max_joint_speed: float = 3.0,
        max_gripper_speed: float = 3.0,
        encoder_timeout_s: float = 5.0,
    ) -> YamArm:
        """Open the CAN channel, enable the motors in gravity-compensation mode and return the arm.

        Note: on followers with an uncalibrated linear gripper (no `gripper_limits` in the rig) the SDK
        runs a short open/close calibration of the gripper motor during connect.
        """
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import ArmType, GripperType

        log.info("connecting %s (%s, %s) on %s", spec.name, spec.arm_type, spec.gripper, channel)
        robot = get_yam_robot(
            channel=channel,
            arm_type=ArmType.from_string_name(spec.arm_type),
            gripper_type=GripperType.from_string_name(spec.gripper),
            zero_gravity_mode=zero_gravity,
            gripper_limits_override=np.asarray(spec.gripper_limits, dtype=float) if spec.gripper_limits else None,
        )
        arm = cls(spec, channel, robot, max_joint_speed=max_joint_speed, max_gripper_speed=max_gripper_speed)
        if spec.has_handle:
            deadline = time.monotonic() + encoder_timeout_s
            while robot.motor_chain.get_same_bus_device_states() is None:
                if time.monotonic() > deadline:
                    arm.close()
                    raise TimeoutError(f"{spec.name}: teaching-handle encoder never reported on {channel}")
                time.sleep(0.02)
        log.info("%s connected: q=%s", spec.name, np.round(arm.read().q, 3))
        return arm

    # ----- properties -----------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    @property
    def robot(self) -> Any:
        return self._robot

    @property
    def home_pose(self) -> np.ndarray:
        return np.asarray(self.spec.home_pose, dtype=float)

    # ----- reading --------------------------------------------------------------------------
    def read(self) -> ArmState:
        obs = self._robot.get_observations()
        q = np.asarray(obs["joint_pos"], dtype=float)[:N_JOINTS]
        if self._offsets is not None:
            q = q + self._offsets
        qd = np.asarray(obs["joint_vel"], dtype=float)[:N_JOINTS]
        tau = np.asarray(obs["joint_eff"], dtype=float)[:N_JOINTS]
        gripper: float | None = None
        buttons: tuple[bool, ...] | None = None
        if self.spec.has_motor_gripper and "gripper_pos" in obs:
            gripper = float(np.clip(obs["gripper_pos"][0], 0.0, 1.0))
        elif self.spec.has_handle:
            enc = self._robot.motor_chain.get_same_bus_device_states()
            if enc:
                gripper = float(np.clip(1.0 - enc[0].position, 0.0, 1.0))  # trigger released -> open (1)
                buttons = tuple(bool(b) for b in enc[0].io_inputs)
        return ArmState(t=time.time(), q=q, qd=qd, tau=tau, gripper=gripper, buttons=buttons)

    # ----- commanding -----------------------------------------------------------------------
    def _full_target(self, q: np.ndarray, gripper: float | None) -> np.ndarray:
        q = np.asarray(q, dtype=float).reshape(-1)[:N_JOINTS]
        if q.shape[0] != N_JOINTS:
            raise ValueError(f"{self.name}: expected {N_JOINTS} joint values, got {q.shape[0]}")
        if not self.spec.has_motor_gripper:
            return q
        if gripper is None:
            cur = self._last_cmd[-1] if self._last_cmd is not None else self.read().gripper
            gripper = 1.0 if cur is None else cur
        return np.concatenate([q, [float(np.clip(gripper, 0.0, 1.0))]])

    def _to_raw(self, target: np.ndarray) -> np.ndarray:
        """Aligned frame → the motors' own frame (undo `joint_offsets`)."""
        if self._offsets is None:
            return target
        raw = target.copy()
        raw[:N_JOINTS] -= self._offsets
        return raw

    def command(self, q: np.ndarray, gripper: float | None = None, *, limit_speed: bool = True) -> np.ndarray:
        """Command joint targets (rad) and gripper (0..1). Returns the target actually sent."""
        if self._gains_zeroed:
            self.restore_gains()
        target = self._full_target(q, gripper)
        now = time.monotonic()
        if limit_speed:
            fresh = self._last_cmd is None or self._last_cmd_t is None or now - self._last_cmd_t > STALE_COMMAND_S
            if fresh:
                prev = self.read().vector()
                dt = 0.01
            else:
                prev = self._last_cmd
                dt = max(now - self._last_cmd_t, 1e-3)
            step = np.full_like(target, self.max_joint_speed * dt)
            if self.spec.has_motor_gripper:
                step[-1] = self.max_gripper_speed * dt
            target = prev + np.clip(target - prev, -step, step)
        self._robot.command_joint_pos(self._to_raw(target))
        self._last_cmd, self._last_cmd_t = target.copy(), now
        return target

    def move_to(self, q: np.ndarray, gripper: float | None = None, duration: float = 3.0, hz: float = 100.0) -> None:
        """Blocking linear interpolation from the measured pose to the target."""
        start = self.read().vector()
        target = self._full_target(q, gripper)
        if start.shape != target.shape:  # arm without gripper
            target = target[: start.shape[0]]
        steps = max(int(duration * hz), 1)
        for i in range(1, steps + 1):
            a = i / steps
            self.command((1 - a) * start[:N_JOINTS] + a * target[:N_JOINTS],
                         (1 - a) * start[-1] + a * target[-1] if self.spec.has_motor_gripper else None,
                         limit_speed=False)
            time.sleep(1.0 / hz)

    def hold(self) -> None:
        """Hold the current measured pose under PD control."""
        st = self.read()
        self.command(st.q, st.gripper, limit_speed=False)

    def go_home(self, speed: float = 0.5, *, compliant: bool = False, release: bool = False) -> float:
        """Move slowly to the home pose (`rest_pose`, default all joints 0 = folded). Blocking.

        The move takes max|Δq| / `speed` (rad/s), at least 0.5 s; the gripper is left where it is.
        `compliant` uses low gains so a hand holding the arm wins (leaders); `release` leaves the arm
        in gravity-compensation idle afterwards. A KeyboardInterrupt (second Ctrl-C / Stop) releases
        the arm where it is and propagates, so the caller can skip every remaining move.
        Returns how far (rad) the arm had to move."""
        target = self.home_pose
        dist = float(np.max(np.abs(self.read().q - target)))
        duration = max(dist / max(float(speed), 1e-6), HOME_MIN_S)
        log.info("%s: moving home, %.2f rad away over %.1f s%s", self.name, dist, duration, " (compliant)" if compliant else "")
        if compliant:
            self.scale_gains(COMPLIANT_KP_SCALE, COMPLIANT_KD_SCALE)
        try:
            self.move_to(target, None, duration=duration)
        except KeyboardInterrupt:
            log.warning("%s: home move interrupted — releasing here", self.name)
            self.gravity_idle()
            raise
        finally:
            if compliant:
                self.restore_gains()
        if release:
            self.gravity_idle()
        return dist

    # ----- gains / modes --------------------------------------------------------------------
    def set_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(np.asarray(kp, dtype=float), np.asarray(kd, dtype=float))
        self._gains_zeroed = bool(np.all(np.asarray(kp) == 0))

    def scale_gains(self, kp_scale: float, kd_scale: float = 0.0) -> None:
        self.set_gains(self.default_kp * kp_scale, self.default_kd * kd_scale)

    def restore_gains(self) -> None:
        self.set_gains(self.default_kp, self.default_kd)

    def gravity_idle(self) -> None:
        """Compliant, gravity-compensated (the mode the arm starts in)."""
        self._robot.enter_gravity_comp_idle()
        self._last_cmd = self._last_cmd_t = None

    def zero_torque(self) -> None:
        self._robot.zero_torque_mode()
        self._gains_zeroed = True
        self._last_cmd = self._last_cmd_t = None

    def info(self) -> dict[str, Any]:
        return self._robot.get_robot_info()

    # ----- teardown -------------------------------------------------------------------------
    def close(self, settle_s: float = 0.2) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._robot.enter_gravity_comp_idle()
            time.sleep(settle_s)
        except Exception:
            log.exception("%s: could not enter gravity idle before close", self.name)
        # Ordered shutdown (the vendor's close() races its own threads and logs spurious errors):
        # 1. stop the MotorChainRobot server thread, 2. stop the chain's CAN loop, 3. close the socket.
        stop = getattr(self._robot, "_stop_event", None)
        server = getattr(self._robot, "_server_thread", None)
        if stop is not None:
            stop.set()
        if server is not None and server.is_alive():
            server.join(timeout=2.0)
        chain = getattr(self._robot, "motor_chain", None)
        if chain is not None and hasattr(chain, "running"):
            chain.running = False
            time.sleep(0.05)
        self._robot.close()
        log.info("%s closed", self.name)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def resolve_channel(spec: ArmSpec) -> str:
    """Map an ArmSpec to a live socketcan interface name (explicit name > USB serial)."""
    from .can import find_by_name, find_by_serial, list_can_interfaces

    ifaces = list_can_interfaces()
    if spec.can_iface:
        i = find_by_name(spec.can_iface, ifaces)
        if i is None:
            raise RuntimeError(f"{spec.name}: CAN interface {spec.can_iface!r} not found")
    elif spec.can_serial:
        i = find_by_serial(spec.can_serial, ifaces)
        if i is None:
            have = {x.name: x.serial for x in ifaces}
            raise RuntimeError(f"{spec.name}: no CAN adapter with serial {spec.can_serial!r} (present: {have})")
    else:
        raise RuntimeError(f"{spec.name}: rig entry has neither can_serial nor can_iface")
    if not i.up:
        raise RuntimeError(
            f"{spec.name}: {i.name} is DOWN. Bring it up with: sudo ip link set {i.name} up type can bitrate 1000000"
        )
    return i.name
