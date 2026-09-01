"""Leader → follower teleoperation with engage/disengage on the teaching-handle button.

Protocol (mirrors the vendor's `minimum_gello.py`, in-process and multi-pair):
  * both arms start compliant in gravity-compensation mode
  * press the handle button (or `auto_engage`) → the follower moves to the leader pose over
    `sync_seconds`, then tracks the leader at `hz`; with `bilateral_kp > 0` the leader is pulled
    toward the follower pose with scaled gains (force feedback)
  * press again → the follower holds its pose, the leader goes compliant
  * Ctrl-C / `stop` → leaders compliant, followers hold, everything closed cleanly
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .arm import ArmState, YamArm, resolve_channel
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
        on_tick: Callable[[TeleopSession], None] | None = None,
    ) -> None:
        self.pairs = pairs
        self.hz = hz
        self.sync_seconds = sync_seconds
        self.bilateral_kp = bilateral_kp
        self.engage_button = engage_button
        self.auto_engage = auto_engage
        self.on_tick = on_tick
        self.stats = TeleopStats()
        self.stop_event = threading.Event()

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
        return cls(pairs, **kw)

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
        if self.auto_engage:
            for pair in self.pairs:
                self.engage(pair)
        t_end = None if duration is None else time.monotonic() + duration
        self.stats = TeleopStats()
        try:
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
        for pair in self.pairs:
            try:
                if pair.engaged:
                    self.disengage(pair)
            finally:
                pair.leader.close()
                pair.follower.close()
        log.info("teleop session closed (%d ticks, %.1f Hz, %d overruns)", self.stats.ticks, self.stats.rate_hz, self.stats.overruns)
