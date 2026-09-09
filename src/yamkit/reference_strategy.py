"""The linked YAM runner's send, rate wait, observation and 1 ms sleep order.

LeRobot still prepares observations/actions and owns the rollout context. The
reference controller supplies literal interpolation points and bounded RPCs.
"""

from __future__ import annotations

import logging
import time

from lerobot.rollout.strategies.base import BaseStrategy
from lerobot.rollout.strategies.core import send_next_action

from yamkit.inference.client import RemoteFault
from yamkit.inference.command_shaping import ACTION_NAMES
from yamkit.validation import finite_scalar

logger = logging.getLogger(__name__)


def _wait_until(deadline, shutdown_event, keep_running, *, poll_s):
    while keep_running():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return True
        shutdown_event.wait(min(poll_s, remaining))
    return False


class ReferenceRate:
    """Upstream Rate.sleep semantics, using an interruptible monotonic clock."""

    def __init__(self, fps, *, shutdown_event, keep_running):
        self.period = 1 / finite_scalar(fps, "reference rate", positive=True)
        self.last = time.monotonic()
        self.shutdown_event = shutdown_event
        self.keep_running = keep_running

    def sleep(self):
        while self.keep_running():
            if self.last + self.period <= time.monotonic():
                # As upstream, an overrun resets the rate origin to now. The
                # command was already sent; there is no pre-send rate floor.
                self.last = time.monotonic()
                return True
            self.shutdown_event.wait(.0001)
        return False


class ReferenceStrategy(BaseStrategy):
    """Serial full chunks with literal upstream post-send rate ordering."""

    def run(self, ctx):
        engine = self._engine
        robot = ctx.hardware.robot_wrapper
        shutdown_event = ctx.runtime.shutdown_event
        cfg = ctx.runtime.cfg
        if cfg.interpolation_multiplier != 1 or cfg.use_torch_compile:
            raise ValueError("Reference strategy requires unchanged points and prewarmed remote inference")

        def keep_running():
            if shutdown_event.is_set():
                return False
            engine._check()
            return time.monotonic() < engine.phase_deadline

        def observe_row():
            if not keep_running():
                return None
            inner = robot.inner
            missing = object()
            previous = getattr(inner, "_reference_observation_role", missing)
            inner._reference_observation_role = "row_anchor"
            try:
                # Upstream dynamic_smoothing reads this extra observation. It
                # does not replace the last post-step policy input or notify
                # the engine; the committed command remains the row anchor.
                return robot.get_observation()
            finally:
                if previous is missing:
                    del inner._reference_observation_role
                else:
                    inner._reference_observation_role = previous

        engine.resume()
        rate = ReferenceRate(cfg.fps, shutdown_event=shutdown_event, keep_running=keep_running)
        previous_observer = engine.observe_row
        engine.observe_row = observe_row
        try:
            if not keep_running():
                return
            obs_raw = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs_raw)
            start = getattr(robot.inner, "_reference_start_action", None)
            if start is None:
                raise RemoteFault("Reference startup reset did not complete before inference")
            # The upstream reset commands both full start vectors, including
            # open grippers, and caches that command before its first policy
            # observation. Keep actual encoders in monitoring, not this cache.
            robot.command_shaper.initialize_position({name: start[name] for name in ACTION_NAMES})
            logger.info("Reference strategy control loop started")
            while keep_running():
                action = send_next_action(obs_processed, obs_raw, ctx, self._interpolator)
                if action is None:
                    # The synchronous controller has no pending-action polling
                    # phase. A completed chunk may wait out the approved tail
                    # when another full request budget would not fit.
                    if not _wait_until(min(engine.phase_deadline, time.monotonic() + rate.period),
                                       shutdown_event, keep_running, poll_s=rate.period):
                        break
                    continue
                if not rate.sleep():
                    break
                engine.rate_steps += 1
                if not keep_running():
                    break
                obs_raw = robot.get_observation()
                obs_processed = self._process_observation_and_notify(ctx.processors, obs_raw)
                if engine.last_step_multipoint and keep_running():
                    engine.multipoint_extra_sleep_calls += 1
                    if not _wait_until(time.monotonic() + .001, shutdown_event,
                                       keep_running, poll_s=.001):
                        break
                self._log_telemetry(obs_processed, action, ctx.runtime)
        finally:
            engine.observe_row = previous_observer
