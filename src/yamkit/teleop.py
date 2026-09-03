"""Leader → follower teleoperation with engage/disengage on the teaching-handle button.

Protocol (mirrors the vendor's `minimum_gello.py`, in-process and multi-pair):
  * on start every arm moves slowly to its home pose (`home_speed` > 0): followers under normal
    gains, leaders compliantly so a hand on the handle wins; leaders are then left free
  * press the handle button (or `auto_engage`) → the follower moves to the leader pose over
    `sync_seconds`, then tracks the leader at `hz`; with `bilateral_kp > 0` the leader is pulled
    toward the follower pose with scaled gains (force feedback)
  * press again → the follower holds its pose, the leader goes compliant
  * Ctrl-C / `stop` → every arm returns home the same slow way, then everything is released and
    closed; a second Ctrl-C / stop during that move releases the arms where they are
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .arm import ArmState, YamArm, go_home_all, resolve_channel
from .config import RigConfig

log = logging.getLogger(__name__)


@dataclass
class TeleopPair:
    name: str
    leader: YamArm
    follower: YamArm
    engaged: bool = False
    _button_prev: bool = False
    last_leader: ArmState | None = None
    last_follower: ArmState | None = None

    @property
    def tracking_error(self) -> float:
        if self.last_leader is None or self.last_follower is None:
            return float("nan")
        return float(np.max(np.abs(self.last_leader.q - self.last_follower.q)))


@dataclass
class TeleopStats:
    ticks: int = 0
    overruns: int = 0
    t_start: float = field(default_factory=time.monotonic)

    @property
    def rate_hz(self) -> float:
        el = time.monotonic() - self.t_start
        return self.ticks / el if el > 0 else 0.0


class TeleopSession:
    def __init__(
        self,
        pairs: list[TeleopPair],
        *,
        hz: float = 100.0,
        sync_seconds: float = 3.0,
        bilateral_kp: float = 0.0,
        engage_button: int = 0,
        auto_engage: bool = False,
        home_speed: float = 0.5,
        leader_home_speed: float | None = None,
        on_tick: Callable[[TeleopSession], None] | None = None,
    ) -> None:
        self.pairs = pairs
        self.hz = hz
        self.sync_seconds = sync_seconds
        self.bilateral_kp = bilateral_kp
        self.engage_button = engage_button
        self.auto_engage = auto_engage
        self.home_speed = home_speed
        self.leader_home_speed = home_speed / 2 if leader_home_speed is None else leader_home_speed
        self.on_tick = on_tick
        self.stats = TeleopStats()
        self.stop_event = threading.Event()
        self._home_aborted = False

    # ----- construction helpers ---------------------------------------------------------------
    @classmethod
    def from_rig(cls, rig: RigConfig, pair_names: list[str] | None = None, **kw) -> TeleopSession:
        """Connect every (or the selected) leader/follower pair of the rig."""
        ctrl = rig.control
        wanted = rig.pairs if not pair_names else [p for p in rig.pairs if p.follower in pair_names or p.leader in pair_names]
        if not wanted:
            raise RuntimeError("rig has no leader/follower pairs to teleoperate")
        pairs: list[TeleopPair] = []
        try:
            for p in wanted:
                lspec, fspec = rig.arm(p.leader), rig.arm(p.follower)
                leader = YamArm.connect(lspec, resolve_channel(lspec))
                follower = YamArm.connect(
                    fspec, resolve_channel(fspec),
                    max_joint_speed=ctrl.max_joint_speed, max_gripper_speed=ctrl.max_gripper_speed,
                )
                pairs.append(TeleopPair(name=f"{p.leader}->{p.follower}", leader=leader, follower=follower))
        except Exception:
            for pr in pairs:
                pr.leader.close()
                pr.follower.close()
            raise
        kw.setdefault("hz", ctrl.teleop_hz)
        kw.setdefault("sync_seconds", ctrl.sync_seconds)
        kw.setdefault("bilateral_kp", ctrl.bilateral_kp)
        kw.setdefault("engage_button", ctrl.engage_button)
        kw.setdefault("home_speed", ctrl.home_speed)
        kw.setdefault("leader_home_speed", ctrl.leader_home_speed)
        return cls(pairs, **kw)

    # ----- home -----------------------------------------------------------------------------
    def home_all(self, why: str) -> None:
        """Move every arm to its home pose, all at the same time (no-op if home_speed <= 0)."""
        if self.home_speed <= 0:
            return
        log.info("%s: all arms moving home (followers %.2f rad/s, leaders %.2f rad/s) — let go of the handles (Ctrl-C / Stop again releases immediately)", why, self.home_speed, self.leader_home_speed)
        jobs: list[tuple[YamArm, dict]] = []
        for pair in self.pairs:
            jobs.append((pair.follower, {"speed": self.home_speed}))
            jobs.append((pair.leader, {"speed": self.leader_home_speed, "compliant": True, "release": True}))
        try:
            go_home_all(jobs)
        except KeyboardInterrupt:
            self._home_aborted = True
            raise

    # ----- engage / disengage ---------------------------------------------------------------
    def engage(self, pair: TeleopPair) -> None:
        lead = pair.leader.read()
        log.info("[%s] engaging: follower syncing to leader over %.1fs", pair.name, self.sync_seconds)
        pair.follower.move_to(lead.q, lead.gripper, duration=self.sync_seconds)
        if self.bilateral_kp > 0:
            pair.leader.scale_gains(self.bilateral_kp, 0.0)
        pair.engaged = True
        log.info("[%s] engaged", pair.name)

    def disengage(self, pair: TeleopPair) -> None:
        pair.engaged = False
        pair.follower.hold()
        pair.leader.gravity_idle()
        pair.leader.restore_gains()
        log.info("[%s] disengaged (follower holding, leader free)", pair.name)

    # ----- loop -----------------------------------------------------------------------------
    def step(self) -> None:
        for pair in self.pairs:
            lead = pair.leader.read()
            foll = pair.follower.read()
            pair.last_leader, pair.last_follower = lead, foll
            pressed = bool(lead.buttons[self.engage_button]) if lead.buttons and len(lead.buttons) > self.engage_button else False
            if pressed and not pair._button_prev:  # rising edge toggles
                self.disengage(pair) if pair.engaged else self.engage(pair)
            pair._button_prev = pressed
            if pair.engaged:
                pair.follower.command(lead.q, lead.gripper)
                if self.bilateral_kp > 0:
                    pair.leader.command(foll.q, None, limit_speed=False)

    def run(self, duration: float | None = None) -> TeleopStats:
        period = 1.0 / self.hz
        t_end = None if duration is None else time.monotonic() + duration
        self.stats = TeleopStats()
        try:
            self.home_all("start")
            if self.auto_engage:
                for pair in self.pairs:
                    self.engage(pair)
            next_t = time.monotonic()
            while not self.stop_event.is_set():
                self.step()
                self.stats.ticks += 1
                if self.on_tick:
                    self.on_tick(self)
                if t_end is not None and time.monotonic() >= t_end:
                    break
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    self.stats.overruns += 1
                    next_t = time.monotonic()
        except KeyboardInterrupt:
            log.info("teleop interrupted")
        finally:
            self.shutdown()
        return self.stats

    def shutdown(self) -> None:
        try:
            for pair in self.pairs:
                if pair.engaged:
                    self.disengage(pair)
            if not self._home_aborted:
                self.home_all("stop")
        except KeyboardInterrupt:
            log.warning("home move aborted — releasing the arms where they are")
        finally:
            for pair in self.pairs:
                pair.leader.close()
                pair.follower.close()
        log.info("teleop session closed (%d ticks, %.1f Hz, %d overruns)", self.stats.ticks, self.stats.rate_hz, self.stats.overruns)
