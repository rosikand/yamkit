"""Serial full-chunk MolmoAct2 execution through LeRobot's base strategy.

The reference's literal 14D interpolation and cached-command policy state are
retained. ReferenceStrategy owns the pinned send/sleep/observe ordering.
Inference waits are bounded and contain no application motor commands.
"""

from __future__ import annotations

import time
from collections import deque
from copy import copy

import numpy as np
import torch
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.rollout.inference.base import InferenceEngine

from yamkit.inference.client import InvalidatedRequest, RemoteFault
from yamkit.inference.command_shaping import ACTION_NAMES


class ReferenceRemoteInferenceEngine(InferenceEngine):
    """One bounded RPC, every original target, then the next policy observation."""

    def __init__(self, *, policy, preprocessor, postprocessor, robot_wrapper, task, fps,
                 shutdown_event, duration, gripper_max_step):
        self._policy, self._preprocessor, self._postprocessor = policy, preprocessor, postprocessor
        self._robot, self._task, self._fps = robot_wrapper, task, fps
        self._shutdown_event = shutdown_event
        self.duration = duration
        self.gripper_max_step = gripper_max_step
        self._failed = False
        self._valid = True
        self._started = False
        self._observation_time = None
        self._plans = []
        self._row = self._point = 0
        self._pending = None
        self._plan_deadline = None
        self.observe_row = None
        self.last_step_multipoint = False
        self.rate_steps = self.multipoint_extra_sleep_calls = 0
        self.action_deadline = None
        self.phase_deadline = None
        self.executed_actions = self.dequeued_actions = 0
        self.predicted_steps = self.completed_steps = self.completed_chunks = 0
        self.admitted_steps = 0
        self.peak_queue_depth = self.last_queue_depth_before_stop = 0
        self.minimum_execution_queue_depth = None
        self.minimum_dispatch_margin_s = None
        self.expired_before_dispatch = 0
        self.expired_plans = 0
        self.stop_detected_at = self.robot_released_at = None
        self.duration_completed = False
        self.home_attempted = self.home_completed = self.home_aborted = False
        self.home_abort_reason = None
        self.home_started_at = self.home_finished_at = self.home_stop_detected_at = None
        self.predictions = deque(maxlen=1000)
        self.dispatch_samples = deque(maxlen=1000)
        self.chunk_samples = deque(maxlen=128)
        policy.on_fault = self._fault

    @property
    def failed(self):
        return self._failed

    def trace_event(self, kind, **values):
        """Read-only instrumentation seam; no I/O in the production controller."""

    def _check(self):
        if not self._valid or self._shutdown_event.is_set():
            raise InvalidatedRequest("Reference execution stopped")
        check = getattr(self._policy.transport, "ensure_session_active", None)
        if check is not None:
            check()
        if not self._valid or self._shutdown_event.is_set():
            raise InvalidatedRequest("Reference execution stopped")

    def reset(self):
        if self._started or not self._valid:
            self._fault()
            raise RemoteFault("A used reference executor cannot resume; start a new supervised session")
        self._policy.reset()
        self._preprocessor.reset()
        self._postprocessor.reset()

    def start(self):
        self._check()
        self._started = True

    def resume(self):
        self._check()
        if self.phase_deadline is not None:
            raise RemoteFault("Reference execution cannot resume a previous phase")
        self.phase_deadline = time.monotonic() + self.duration

    def notify_observation(self, obs):
        # Monitoring still reads cameras and actual encoders each strategy tick.
        # Only a completed chunk permits this observation to reach the policy.
        self._observation_time = time.monotonic()

    def _predict(self, obs_frame):
        from yamkit.inference.reference import plan_reference_row

        guard = self._robot.command_shaper
        now = time.monotonic()
        wait_deadline = min(self.phase_deadline, now + self._policy.config.request_timeout_s,
                            self._observation_time + self._policy.config.max_observation_age_s)
        guard.begin_inference_wait(wait_deadline)
        event = {"chunk_index": len(self.predictions), "prediction_started_monotonic_s": now,
                 "observation_timestamp_monotonic_s": self._observation_time,
                 "executed_actions_at_start": self.executed_actions,
                 "completed_steps_at_start": self.completed_steps,
                 "completed_chunks_at_start": self.completed_chunks,
                 "accepted_steps": 0, "expired_prefix_dropped": 0, "overlap_prefix_dropped": 0,
                 "observation_state_basis": "last committed full action; completed startup-reset command before first chunk"}
        self.predictions.append(event)
        observation = copy(obs_frame)
        observation["observation.state"] = np.array(
            [guard.last_action[name] for name in ACTION_NAMES], dtype=np.float32)
        self._policy._observation_time = self._observation_time
        self._policy._observation_selected_time = now
        self._policy._request_deadline_monotonic_s = wait_deadline
        event["inference_wait_deadline_monotonic_s"] = wait_deadline
        try:
            with torch.inference_mode():
                batch = prepare_observation_for_inference(observation, torch.device("cpu"),
                                                          self._task, self._robot.robot_type)
                actions = self._postprocessor(self._policy.predict_action_chunk(self._preprocessor(batch)))
            returned = time.monotonic()
            event.update(prediction_s=returned - now,
                         observation_age_at_return_s=returned - self._observation_time,
                         actions_executed_during_prediction=self.executed_actions - event["executed_actions_at_start"],
                         **self._policy._last_prediction_timing)
            if tuple(actions.shape) != (1, 30, 14) or not torch.isfinite(actions).all():
                raise RemoteFault("Reference mode requires one complete finite 30 by 14 chunk")
            self.predicted_steps += len(actions[0])
            self._check()
            if returned >= self.phase_deadline:
                event["error"] = "duration_completed_before_admission"
                return False
            guard.end_inference_wait(returned)
            targets = [dict(zip(ACTION_NAMES, row, strict=True)) for row in actions[0].cpu().tolist()]
            # Validate all original targets before any interpolation can hide a
            # malformed later row, including both arms and normalized grippers.
            for target in targets:
                self._robot.inner.validate_action_target(target)
            start = dict(guard.last_action)
            plans = []
            deltas = []
            for target in targets:
                deltas.append(max(abs(target[name] - start[name]) for name in ACTION_NAMES))
                plan = plan_reference_row(start, target, joint_limits=self._robot.inner.joint_command_limits(),
                                          gripper_max_step=self.gripper_max_step, period_s=1 / self._fps)
                plans.append(plan)
                start = target
            self._check()
            admitted = time.monotonic()
            if admitted >= wait_deadline:
                raise RemoteFault("Reference plan admission exceeded its original request deadline")
            points = sum(len(plan.commands) for plan in plans)
            # No elapsed-prefix expiry or artificial execution-time dilation.
            # The approved phase/session boundary remains authoritative.
            self._plan_deadline = self.phase_deadline
            self._plans, self._row, self._point = plans, 0, 0
            self.admitted_steps += len(targets)
            self.peak_queue_depth = max(self.peak_queue_depth, len(targets))
            event.update(accepted_steps=len(targets), error=None,
                         plan_dispatches=points, plan_deadline_monotonic_s=self._plan_deadline,
                         plan_admitted_monotonic_s=admitted,
                         reference_row_max_deltas=deltas,
                         reference_row_counts=[len(plan.commands) for plan in plans],
                         planned_duration_s=points / self._fps)
            self.chunk_samples.append(dict(event))
            self.trace_event("reference_chunk_admitted", **event)
            return True
        except BaseException as exc:
            event["error"] = "invalidated" if isinstance(exc, InvalidatedRequest) else type(exc).__name__
            event.update(prediction_s=time.monotonic() - now,
                         actions_executed_during_prediction=self.executed_actions - event["executed_actions_at_start"])
            raise

    def get_action(self, obs_frame):
        try:
            self._check()
            if self._pending is not None:
                raise RemoteFault("Reference action was not committed before requesting its successor")
            if time.monotonic() >= self.phase_deadline:
                return None
            if (not self._plans and self._robot.command_shaper.generation
                    and self.phase_deadline - time.monotonic() < min(
                        self._policy.config.request_timeout_s, self._policy.config.max_observation_age_s)):
                # At a stationary completed chunk, do not start an RPC whose
                # entire existing freshness budget cannot fit in this phase.
                # Continue monitoring until normal duration/home; never turn a
                # genuine request timeout into a successful completion.
                return None
            if not self._plans and not self._predict(obs_frame):
                return None
            self._check()
            now = time.monotonic()
            if now >= self.phase_deadline:
                return None
            if now >= self._plan_deadline:
                self.expired_plans += 1
                raise RemoteFault("Reference chunk plan deadline expired; no retiming or retry")
            plan = self._plans[self._row]
            if self._point == 0 and self.observe_row is not None:
                self.observe_row()  # Upstream dynamic_smoothing acquires one row-anchor observation.
                self._check()
                if time.monotonic() >= self.phase_deadline:
                    return None
            self._pending = {"chunk_index": len(self.predictions) - 1, "row_index": self._row,
                             "point_index": self._point, "points_in_row": len(plan.commands),
                             "progress": float(plan.progress[self._point]),
                             "endpoint": self._point == len(plan.commands) - 1,
                             "target": dict(plan.commands[-1])}
            self.action_deadline = self._plan_deadline
            self.dequeued_actions += 1
            return torch.tensor([plan.commands[self._point][name] for name in ACTION_NAMES], dtype=torch.float64)
        except InvalidatedRequest:
            if self._shutdown_event.is_set():
                self.invalidate()
                return None
            self._fault()
            raise
        except BaseException:
            self._fault()
            raise

    def record_execution(self):
        self.executed_actions += 1  # Includes a successful send whose feedback later faults.

    def record_commit(self):
        if self._pending is None:
            raise RemoteFault("Reference command commit has no corresponding interpolation point")
        self._check()
        event = {**self._pending, "dispatch_index": self.executed_actions - 1,
                 "monotonic_s": time.monotonic(), "deadline_monotonic_s": self.action_deadline}
        self.dispatch_samples.append(event)
        self.trace_event("reference_dispatch", **event)
        self.last_step_multipoint = len(self._plans[self._row].commands) > 1
        self._point += 1
        if self._point == len(self._plans[self._row].commands):
            self.completed_steps += 1
            self._row, self._point = self._row + 1, 0
            if self._row == len(self._plans):
                self.completed_chunks += 1
                self._plans = []
        depth = len(self._plans) - self._row if self._plans else 0
        if self.minimum_execution_queue_depth is None or depth < self.minimum_execution_queue_depth:
            self.minimum_execution_queue_depth = depth
        self._pending = None

    def record_dispatch(self, margin_s):
        if margin_s is not None:
            if self.minimum_dispatch_margin_s is None or margin_s < self.minimum_dispatch_margin_s:
                self.minimum_dispatch_margin_s = margin_s
            self.expired_before_dispatch += int(margin_s <= 0)

    def _fault(self):
        self._failed = True
        self._shutdown_event.set()
        self.invalidate()

    def invalidate(self):
        if self.stop_detected_at is None:
            self.stop_detected_at = time.monotonic()
        self._valid = False
        self.last_queue_depth_before_stop = self.predicted_steps - self.completed_steps
        self._robot.invalidate_shaping()
        self._policy.close()

    def pause(self):
        self._shutdown_event.set()
        self.invalidate()

    def stop(self):
        self.invalidate()

    def metrics(self):
        from yamkit.inference.qualification import REFERENCE_CONTRACT

        return {"controller_mode": "reference", "reference_contract": dict(REFERENCE_CONTRACT),
                "phase_deadline_monotonic_s": self.phase_deadline,
                "rate_steps": self.rate_steps,
                "multipoint_extra_sleep_calls": self.multipoint_extra_sleep_calls,
                "predicted_steps": self.predicted_steps,
                "admitted_steps": self.admitted_steps,
                "completed_steps": self.completed_steps, "completed_chunks": self.completed_chunks,
                "interpolation_dispatches": self.executed_actions,
                "uncompleted_steps_at_stop": self.predicted_steps - self.completed_steps,
                "expired_prefix_dropped": 0, "overlap_prefix_dropped": 0,
                "prefix_drop": 0, "expired_plans": self.expired_plans,
                "partial_chunk_at_stop": self.predicted_steps != self.completed_steps,
                "coherence_violations": self._robot.command_shaper.postclamp_modified_count,
                "next_observation_after_full_chunk": True,
                "policy_state_basis": "completed startup reset, then last committed 14D command; actual state remains in monitoring and hardware guards",
                "deadline_basis": "fixed RPC freshness/timeout admission; approved phase and session during execution",
                "chunks": list(self.chunk_samples), "dispatch_samples": list(self.dispatch_samples),
                "dispatch_samples_dropped": max(0, self.executed_actions - len(self.dispatch_samples))}
