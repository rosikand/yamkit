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
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from .config import N_JOINTS, ArmSpec
from .validation import finite_scalar, finite_vector, vendor_joint_limits

log = logging.getLogger(__name__)

STALE_COMMAND_S = 0.5  # ramp reset only, NOT a watchdog: SDK threads can keep transmitting
MAX_COMMAND_DT = 0.01  # never accumulate a backlog of movement during an application stall
COMPLIANT_KP_SCALE = 0.15  # gains for moving a leader home: gentle enough that a hand on the handle wins
COMPLIANT_KD_SCALE = 0.3
HOME_MIN_S = 0.5  # shortest go_home move, so a tiny correction is still a visible glide
HOME_SETTLE_S = 1.0  # hold the home target this long before releasing, so a soft (compliant) arm actually gets there


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
        spec.validate()
        self.spec = spec
        self._ownership = None
        self.channel = channel
        self._robot = robot
        self.max_joint_speed = finite_scalar(max_joint_speed, "max_joint_speed", positive=True)
        self.max_gripper_speed = finite_scalar(max_gripper_speed, "max_gripper_speed", positive=True)
        self._n_dofs = int(robot.num_dofs())
        if self._n_dofs != spec.n_dofs:
            raise ValueError(f"{spec.name}: SDK has {self._n_dofs} DOFs, expected {spec.n_dofs}")
        info = robot.get_robot_info()
        self.default_kp = finite_vector(info["kp"], self.n_dofs, "SDK kp")
        self.default_kd = finite_vector(info["kd"], self.n_dofs, "SDK kd")
        if np.any(self.default_kp < 0) or np.any(self.default_kd < 0):
            raise ValueError(f"{spec.name}: SDK gains must be nonnegative")
        self._raw_limits = joint_limits(spec)
        reported = info.get("joint_limits")
        if reported is not None and not np.array_equal(reported, self._raw_limits):
            raise ValueError(f"{spec.name}: SDK joint limits differ from the vendored configuration")
        self.gripper_limits = info.get("gripper_limits")
        self._offsets = np.asarray(spec.joint_offsets, dtype=float) if spec.joint_offsets is not None else None
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

        from .ownership import ArmOwnership

        spec.validate()
        finite_scalar(max_joint_speed, "max_joint_speed", positive=True)
        finite_scalar(max_gripper_speed, "max_gripper_speed", positive=True)
        finite_scalar(encoder_timeout_s, "encoder_timeout_s", positive=True)
        # Validate the home target and alignment before the SDK can activate motors.
        limits = joint_limits(spec)
        offsets = np.zeros(N_JOINTS) if spec.joint_offsets is None else np.asarray(spec.joint_offsets)
        check_joint_bounds(np.asarray(spec.home_pose) - offsets, limits, f"{spec.name}: home pose")
        lease = ArmOwnership.acquire(channel)
        robot = arm = None
        try:
            log.info("connecting %s (%s, %s) on %s", spec.name, spec.arm_type, spec.gripper, channel)
            robot = get_yam_robot(
                channel=channel,
                arm_type=ArmType.from_string_name(spec.arm_type),
                gripper_type=GripperType.from_string_name(spec.gripper),
                zero_gravity_mode=zero_gravity,
                gripper_limits_override=np.asarray(spec.gripper_limits, dtype=float) if spec.gripper_limits is not None else None,
            )
            arm = cls(spec, channel, robot, max_joint_speed=max_joint_speed, max_gripper_speed=max_gripper_speed)
            arm._ownership = lease
            if spec.has_handle:
                deadline = time.monotonic() + encoder_timeout_s
                while robot.motor_chain.get_same_bus_device_states() is None:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"{spec.name}: teaching-handle encoder never reported on {channel}")
                    time.sleep(0.02)
            state = arm.read()
            arm._check_target(arm._measured_vector(state), "measured state; operator recovery required")
            log.info("%s connected: q=%s", spec.name, np.round(state.q, 3))
            return arm
        except BaseException as error:
            try:
                if arm is not None:
                    arm.close()
                elif robot is not None:
                    robot.close()
                    lease.release()
                else:
                    # The SDK factory cleans its partially initialized resources before raising.
                    if not getattr(error, "_yamkit_cleanup_failed", False):
                        lease.release()
            except BaseException:
                log.exception("%s: startup cleanup failed; retaining ownership until process exit", spec.name)
            raise

    # ----- properties -----------------------------------------------------------------------
    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def n_dofs(self) -> int:
        return self._n_dofs

    @property
    def robot(self) -> Any:
        self._ensure_open()
        return self._robot

    @property
    def home_pose(self) -> np.ndarray:
        return finite_vector(self.spec.home_pose, N_JOINTS, f"{self.name}: home pose")

    # ----- reading --------------------------------------------------------------------------
    def _ensure_open(self) -> None:
        if self._ownership is not None:
            self._ownership.check_owner()
        if self._closed:
            raise RuntimeError(f"{self.name}: arm is closed")

    def read(self) -> ArmState:
        self._ensure_open()
        obs = self._robot.get_observations()
        q = finite_vector(obs["joint_pos"], N_JOINTS, f"{self.name}: measured joints")
        if self._offsets is not None:
            q = q + self._offsets
        qd = finite_vector(obs["joint_vel"], N_JOINTS, f"{self.name}: measured velocities")
        tau = finite_vector(obs["joint_eff"], N_JOINTS, f"{self.name}: measured efforts")
        gripper: float | None = None
        buttons: tuple[bool, ...] | None = None
        if self.spec.has_motor_gripper:
            gripper = float(finite_vector(obs.get("gripper_pos"), 1, f"{self.name}: measured gripper")[0])
        elif self.spec.has_handle:
            enc = self._robot.motor_chain.get_same_bus_device_states()
            if enc:
                position = finite_scalar(enc[0].position, f"{self.name}: handle position")
                gripper = float(np.clip(1.0 - position, 0.0, 1.0))
                buttons = tuple(bool(b) for b in enc[0].io_inputs)
        return ArmState(t=time.time(), q=q, qd=qd, tau=tau, gripper=gripper, buttons=buttons)

    # ----- commanding -----------------------------------------------------------------------
    def _to_raw(self, target: np.ndarray) -> np.ndarray:
        raw = target.copy()
        if self._offsets is not None:
            raw[:N_JOINTS] -= self._offsets
        return raw

    def _measured_vector(self, state: ArmState) -> np.ndarray:
        # A teaching handle is an input, not a seventh motor.
        return np.concatenate([state.q, [state.gripper]]) if self.spec.has_motor_gripper else state.q.copy()

    def _check_target(self, target, label: str) -> np.ndarray:
        target = finite_vector(target, self.n_dofs, f"{self.name}: {label}")
        check_joint_bounds(self._to_raw(target)[:N_JOINTS], self._raw_limits, f"{self.name}: {label}")
        if self.spec.has_motor_gripper and not 0 <= target[-1] <= 1:
            raise ValueError(f"{self.name}: {label}: gripper must be in [0, 1]; operator recovery required for measured state")
        return target

    def _full_target(self, q, gripper: float | None, measured: np.ndarray) -> np.ndarray:
        q = finite_vector(q, N_JOINTS, f"{self.name}: joint target")
        if gripper is not None:
            gripper = finite_scalar(gripper, f"{self.name}: gripper target")
            if not 0 <= gripper <= 1:
                raise ValueError(f"{self.name}: gripper target must be in [0, 1]")
        if not self.spec.has_motor_gripper:
            return q
        if gripper is None:
            gripper = self._last_cmd[-1] if self._last_cmd is not None else measured[-1]
        return np.concatenate([q, [gripper]])

    def _prepare_command(self, q, gripper, limit_speed):
        self._ensure_open()
        if not isinstance(limit_speed, bool):
            raise ValueError("limit_speed must be a bool")  # noqa: TRY004 — uniform command rejection API
        finite_scalar(self.max_joint_speed, "max_joint_speed", positive=True)
        finite_scalar(self.max_gripper_speed, "max_gripper_speed", positive=True)
        if self._gains_zeroed:
            for name, gains in (("default kp", self.default_kp), ("default kd", self.default_kd)):
                if np.any(finite_vector(gains, self.n_dofs, name) < 0):
                    raise ValueError(f"{name}: gains must be nonnegative")
        # Always check measurements and previous targets, including unlimited commands.
        measured = self._check_target(self._measured_vector(self.read()), "measured state; operator recovery required")
        if self._last_cmd is not None:
            self._check_target(self._last_cmd, "previous target")
        if self._last_cmd_t is not None:
            finite_scalar(self._last_cmd_t, "previous command time")
            if self._last_cmd_t > time.monotonic():
                raise ValueError(f"{self.name}: previous command time is in the future")
        target = self._check_target(self._full_target(q, gripper, measured), "target")
        return target, measured

    def validate_command(self, q, gripper: float | None = None, *, limit_speed: bool = True) -> np.ndarray:
        """Validate target, measurements and previous target without sending or changing gains.

        Bimanual callers validate both arms before issuing either command. This is preflight,
        not an atomic two-arm transaction: a hardware fault can still occur during execution.
        """
        return self._prepare_command(q, gripper, limit_speed)[0]

    def command(self, q: np.ndarray, gripper: float | None = None, *, limit_speed: bool = True) -> np.ndarray:
        """Reject invalid/out-of-bounds values; return the (optionally speed-clamped) target sent."""
        target, measured = self._prepare_command(q, gripper, limit_speed)
        now = time.monotonic()
        if limit_speed:
            age = None if self._last_cmd_t is None else now - self._last_cmd_t
            if age is not None and age < 0:
                raise ValueError(f"{self.name}: previous command time is in the future")
            fresh = self._last_cmd is None or age is None or age > STALE_COMMAND_S
            prev = measured if fresh else self._last_cmd
            dt = MAX_COMMAND_DT if fresh else min(age, MAX_COMMAND_DT)
            step = np.full_like(target, self.max_joint_speed * dt)
            if self.spec.has_motor_gripper:
                step[-1] = self.max_gripper_speed * dt
            target = self._check_target(prev + np.clip(target - prev, -step, step), "limited target")
        # Replace any obsolete SDK target before restoring PD gains.
        self._robot.command_joint_pos(self._to_raw(target))
        if self._gains_zeroed:
            self.restore_gains()
            self._robot.command_joint_pos(self._to_raw(target))
        self._last_cmd, self._last_cmd_t = target.copy(), now
        return target

    def move_to(self, q: np.ndarray, gripper: float | None = None, duration: float = 3.0, hz: float = 100.0, stop: threading.Event | None = None) -> None:
        """Move to a validated target; extend duration to respect configured target speeds.

        Each interpolation step earns at most one period of elapsed time. A late wakeup
        cannot trigger a catch-up jump. A stop event ends the move before the next command.
        """
        duration = finite_scalar(duration, "duration", positive=True)
        hz = finite_scalar(hz, "hz", positive=True)
        target, start = self._prepare_command(q, gripper, False)
        delta = target - start
        duration = max(duration, float(np.max(np.abs(delta[:N_JOINTS]))) / self.max_joint_speed)
        if self.spec.has_motor_gripper:
            duration = max(duration, abs(float(delta[-1])) / self.max_gripper_speed)
        elapsed = 0.0
        period = 1.0 / hz
        previous_t = time.monotonic()
        while elapsed < duration:
            if stop is not None and stop.is_set():
                return
            delay = min(period, duration - elapsed)
            if stop is not None:
                if stop.wait(delay):
                    return
            else:
                time.sleep(delay)
            now = time.monotonic()
            elapsed = min(duration, elapsed + min(max(now - previous_t, 0.0), delay))
            previous_t = now
            value = start + min(elapsed / duration, 1.0) * delta
            # Bounds/state/gains validation still applies; interpolation enforces target speed.
            self.command(value[:N_JOINTS], value[-1] if self.spec.has_motor_gripper else None, limit_speed=False)

    def hold(self) -> None:
        """Replace the old target with the measured pose before restoring any zeroed gains."""
        state = self.read()
        self.command(state.q, state.gripper if self.spec.has_motor_gripper else None, limit_speed=False)

    def go_home(self, speed: float = 0.5, *, compliant: bool = False, release: bool = False, stop: threading.Event | None = None) -> float:
        """Move home at no more than configured target speed; release on cancellation/error."""
        speed = finite_scalar(speed, "home speed", positive=True)
        state = self.read()
        target, measured = self._prepare_command(self.home_pose, state.gripper if self.spec.has_motor_gripper else None, False)
        dist = float(np.max(np.abs(measured[:N_JOINTS] - target[:N_JOINTS])))
        duration = max(dist / min(speed, self.max_joint_speed), HOME_MIN_S)
        if stop is not None and stop.is_set():
            self.gravity_idle()
            return dist
        log.info("%s: moving home, %.2f rad over at least %.1f s", self.name, dist, duration)
        try:
            if compliant:
                # The SDK gain setter updates defaults only; hold installs the measured
                # target with these low gains, avoiding a full-gain pulse on the leader.
                self.scale_gains(COMPLIANT_KP_SCALE, COMPLIANT_KD_SCALE)
                self.hold()
            self.move_to(target[:N_JOINTS], target[-1] if self.spec.has_motor_gripper else None, duration=duration, stop=stop)
            if release:
                if stop is not None:
                    stop.wait(HOME_SETTLE_S)
                else:
                    time.sleep(HOME_SETTLE_S)
        except BaseException:
            try:
                self.gravity_idle()
                if compliant:
                    self.restore_gains()
            except BaseException:
                log.exception("%s: could not release after failed home move", self.name)
            raise
        if release or (stop is not None and stop.is_set()):
            self.gravity_idle()
        if compliant:
            self.restore_gains()
        return dist

    # ----- gains / modes --------------------------------------------------------------------
    def set_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._ensure_open()
        kp = finite_vector(kp, self.n_dofs, "kp")
        kd = finite_vector(kd, self.n_dofs, "kd")
        if np.any(kp < 0) or np.any(kd < 0):
            raise ValueError("gains must be nonnegative")
        self._robot.update_kp_kd(kp, kd)
        self._gains_zeroed = bool(np.all(kp == 0))

    def scale_gains(self, kp_scale: float, kd_scale: float = 0.0) -> None:
        kp_scale = finite_scalar(kp_scale, "kp_scale", minimum=0)
        kd_scale = finite_scalar(kd_scale, "kd_scale", minimum=0)
        self.set_gains(self.default_kp * kp_scale, self.default_kd * kd_scale)

    def restore_gains(self) -> None:
        self.set_gains(self.default_kp, self.default_kd)

    def gravity_idle(self) -> None:
        """Compliant, gravity-compensated (the mode the arm starts in)."""
        self._ensure_open()
        self._robot.enter_gravity_comp_idle()
        self._last_cmd = self._last_cmd_t = None

    def zero_torque(self) -> None:
        self._ensure_open()
        self._robot.zero_torque_mode()
        self._gains_zeroed = True
        self._last_cmd = self._last_cmd_t = None

    def info(self) -> dict[str, Any]:
        self._ensure_open()
        return self._robot.get_robot_info()

    # ----- teardown -------------------------------------------------------------------------
    def close(self, settle_s: float = 0.2) -> None:
        if self._closed:
            return
        self._ensure_open()
        settle_s = finite_scalar(settle_s, "settle_s", minimum=0)
        errors: list[BaseException] = []
        try:
            self._robot.enter_gravity_comp_idle()
            time.sleep(settle_s)
        except BaseException as exc:  # noqa: BLE001 — finish cleanup, then re-raise
            errors.append(exc)
        # SDK close stops and joins both transmitters before closing the CAN socket.
        # A failed SDK close retains the lease and allows another close attempt.
        try:
            self._robot.close()
        except BaseException as exc:  # noqa: BLE001 — finish cleanup, then re-raise
            errors.append(exc)
        else:
            self._closed = True
            if self._ownership is not None:
                self._ownership.release()
        if errors:
            raise errors[0]
        log.info("%s closed", self.name)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def joint_limits(spec: ArmSpec) -> np.ndarray:
    """Raw motor-coordinate bounds, exactly as configured by the pinned SDK."""
    return vendor_joint_limits(spec.arm_type, spec.gripper).copy()


def check_joint_bounds(raw: np.ndarray, limits: np.ndarray, label: str) -> None:
    outside = (raw < limits[:, 0]) | (raw > limits[:, 1])
    if np.any(outside):
        indices = (np.flatnonzero(outside) + 1).tolist()
        raise ValueError(f"{label}: outside vendor joint bounds at joints {indices} (raw motor coordinates)")


def close_all(arms) -> None:
    """Attempt every close; preserve an active error, otherwise raise the first cleanup error."""
    active_error = sys.exc_info()[0] is not None
    errors: list[BaseException] = []
    for arm in arms:
        try:
            arm.close()
        except BaseException as exc:
            errors.append(exc)
            log.exception("%s: cleanup failed", arm.name)
    if errors and not active_error:
        raise errors[0]


def go_home_all(jobs: list[tuple[YamArm, dict[str, Any]]], *, stop: threading.Event | None = None) -> None:
    """`arm.go_home(**kw)` for every (arm, kw) at the same time — one thread per arm, each arm has its
    own CAN bus. Ctrl-C (raised in the calling thread) stops every move, releases the arms where they
    are, then propagates; an error in any arm's move is re-raised after all moves have ended."""
    # Check every home target before any thread changes gains or sends a command.
    for arm, kw in jobs:
        finite_scalar(kw.get("speed", 0.5), "home speed", positive=True)
        arm.validate_command(arm.home_pose, limit_speed=False)
    stop = stop if stop is not None else threading.Event()
    begin = threading.Event()
    errors: list[BaseException] = []
    finished = [threading.Event() for _ in jobs]

    def run(arm: YamArm, kw: dict[str, Any], done: threading.Event) -> None:
        try:
            begin.wait()
            if stop.is_set():
                return
            arm.go_home(**kw, stop=stop)
        except BaseException as e:  # noqa: BLE001 — surfaced after every worker has stopped
            errors.append(e)
            stop.set()
        finally:
            done.set()

    threads = [threading.Thread(target=run, args=(a, kw, done), daemon=True, name=f"home-{a.name}")
               for (a, kw), done in zip(jobs, finished, strict=True)]
    started = []
    try:
        for t, done in zip(threads, finished, strict=True):
            started.append((t, done))
            t.start()
        begin.set()  # no hardware worker may run until all thread starts have succeeded
        # Event waits avoid CPython's interrupted Thread.join marking a live worker stopped.
        for _, done in started:
            while not done.wait(0.05):
                pass
    except BaseException:
        stop.set()
        begin.set()  # even a late-starting worker now exits without touching an arm
        for t, done in started:
            if t.ident is None:
                continue
            while not done.is_set():
                try:
                    done.wait(0.05)
                except KeyboardInterrupt:
                    stop.set()
        raise
    if errors:
        raise errors[0]


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
