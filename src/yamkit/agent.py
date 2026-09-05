"""Bounded, synchronous tool decisions above the LeRobot robot interface.

No hardware lifecycle or servo lives here. Live construction is deliberately blocked in
agent_robot until the plugin can guarantee state/image acquisition freshness.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .agent_robot import GRIPPER_KEY, JOINT_KEYS


class AgentError(RuntimeError):
    pass


class DeadlineExceeded(AgentError):
    pass


@dataclass(frozen=True)
class AgentConfig:
    model: str
    task: str
    max_steps: int = 50
    settle_s: float = 0.5
    max_joint_delta: float = 0.10
    motion_timeout_s: float = 5.0
    api_timeout_s: float = 30.0
    episode_timeout_s: float = 300.0

    def validate(self) -> None:
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", self.model):
            raise ValueError("model must be an explicit model ID (1–128 characters)")
        if not isinstance(self.task, str) or not self.task.strip() or len(self.task) > 4096:
            raise ValueError("task must contain 1–4096 characters")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 1000:
            raise ValueError("max_steps must be an integer from 1 to 1000")
        for name, low, high in (
            ("settle_s", 0, 60), ("max_joint_delta", 0.000001, 0.10),
            ("motion_timeout_s", 0.01, 60), ("api_timeout_s", 0.01, 300),
            ("episode_timeout_s", 0.01, 3600),
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and between {low} and {high}")


def validate_tool(name: str, arguments: str) -> dict:
    """Independently validate JSON, including duplicates and Python's permissive NaN parser."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate argument field")
            result[key] = value
        return result

    if not isinstance(arguments, str) or len(arguments) > 4096:
        raise ValueError("arguments must be a bounded JSON object")
    try:
        args = json.loads(arguments, object_pairs_hook=pairs)
    except (ValueError, RecursionError) as exc:
        raise ValueError("invalid argument JSON") from exc
    fields = {
        "observe": set(), "move_joints": {"delta"}, "open_gripper": set(),
        "close_gripper": set(), "finish": {"success", "reason"},
    }
    if name not in fields or type(args) is not dict or set(args) != fields[name]:
        raise ValueError("unknown tool or malformed/unknown argument fields")
    if name == "move_joints":
        delta = args["delta"]
        if type(delta) is not list or len(delta) != 6:
            raise ValueError("delta must contain exactly six finite numbers")
        try:
            valid = all(type(v) in (int, float) and math.isfinite(v) for v in delta)
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("delta must contain exactly six finite numbers; booleans are invalid")
    if name == "finish":
        if type(args["success"]) is not bool or not isinstance(args["reason"], str):
            raise ValueError("finish requires boolean success and string reason")
        if not 1 <= len(args["reason"].strip()) <= 512:
            raise ValueError("finish reason must contain 1–512 characters")
    return args


class JsonlLog:
    """Exclusive-create, 2 MiB journal with room reserved for the final termination event."""
    MAX_BYTES = 2 * 1024 * 1024
    MAX_LINE = 64 * 1024

    def __init__(self, path: Path, clock=time.monotonic):
        from .paths import ROOT

        path = Path(path).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("agent logs must stay inside the repository")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("x", encoding="utf-8")
        self.clock, self.start, self.size = clock, clock(), 0
        self.secrets = [os.environ[n] for n in ("YAMKIT_OPENAI_API_KEY", "OPENAI_API_KEY") if os.environ.get(n)]

    def write(self, event: str, *, final=False, **fields):
        line = json.dumps({"event": event, "elapsed_s": self.clock() - self.start, **fields},
                          allow_nan=False, ensure_ascii=True)
        for secret in self.secrets:
            line = line.replace(secret, "[REDACTED]")
            # JSON-escaped credentials are equally sensitive.
            line = line.replace(json.dumps(secret)[1:-1], "[REDACTED]")
        encoded = (line + "\n").encode("utf-8")
        ceiling = self.MAX_BYTES if final else self.MAX_BYTES - self.MAX_LINE
        if len(encoded) > self.MAX_LINE or self.size + len(encoded) > ceiling:
            raise AgentError("log budget exceeded")
        self.file.write(encoded.decode("utf-8"))
        self.file.flush()
        self.size += len(encoded)

    def close(self):
        self.file.close()


def _redact(value):
    if isinstance(value, str):
        for name in ("YAMKIT_OPENAI_API_KEY", "OPENAI_API_KEY"):
            secret = os.environ.get(name)
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    return value


def _close(adapter, provider):
    try:
        adapter.close()
    finally:
        close = getattr(provider, "close", None)
        if close is not None:
            close()


def observation_record(obs) -> dict:
    return {"state": obs.state, "camera_names": sorted(obs.images), "captured_at": obs.captured_at,
            "sequence": obs.sequence, "source": obs.source}


def _check(deadline, clock, cancelled):
    if cancelled():
        raise KeyboardInterrupt
    if clock() >= deadline:
        raise DeadlineExceeded("deadline exceeded")


def _pause(seconds, deadline, clock, sleep, cancelled):
    end = clock() + seconds
    while clock() < end:
        _check(deadline, clock, cancelled)
        sleep(min(0.05, end - clock(), max(0, deadline - clock())))
    _check(deadline, clock, cancelled)


def fixed_target_operation(adapter, name, args, config, episode_deadline, log,
                           *, clock=time.monotonic, sleep=time.sleep, cancelled=lambda: False):
    """Read once to anchor the target, submit that same target at at most 10 Hz, measure.

    send_action's return is the limited command, never evidence of arrival. The operation
    terminates on the first feedback fault; a timeout cannot extend or restart the target.
    """
    deadline = min(episode_deadline, clock() + config.motion_timeout_s)
    _check(deadline, clock, cancelled)
    start = adapter.observe()
    _check(deadline, clock, cancelled)
    target = dict(start.state)
    requested = dict(start.state)
    bounded_delta = None
    if name == "move_joints":
        bounded_delta = [max(-config.max_joint_delta, min(config.max_joint_delta, v)) for v in args["delta"]]
        for key, delta, bounded in zip(JOINT_KEYS, args["delta"], bounded_delta, strict=True):
            requested[key] += delta
            target[key] += bounded
    elif name in ("open_gripper", "close_gripper"):
        target[GRIPPER_KEY] = requested[GRIPPER_KEY] = 1.0 if name == "open_gripper" else 0.0
    else:
        raise ValueError("not a motion tool")
    log.write("target", tool=name, arguments=args, requested=requested, bounded=target,
              bounded_delta=bounded_delta, **observation_record(start))
    while True:
        _check(deadline, clock, cancelled)
        sent = adapter.send(dict(target))
        sent_at = clock()
        # Bound submission cadence and require feedback acquired after this command.
        _pause(0.1, deadline, clock, sleep, cancelled)
        measured = adapter.observe(after=sent_at)
        _check(deadline, clock, cancelled)
        log.write("readback", sent=sent, bounded=target, **observation_record(measured))
        if any(abs(sent[k] - measured.state[k]) > 0.35 for k in target):
            raise AgentError("excessive tracking error")
        if (all(abs(target[k] - measured.state[k]) <= 0.01 for k in JOINT_KEYS)
                and abs(target[GRIPPER_KEY] - measured.state[GRIPPER_KEY]) <= 0.03):
            break
    settled_after = clock() + config.settle_s
    _pause(config.settle_s, episode_deadline, clock, sleep, cancelled)
    # A new acquisition after settling, not the cached readback above.
    fresh = adapter.observe(after=settled_after)
    _check(episode_deadline, clock, cancelled)
    if any(abs(sent[k] - fresh.state[k]) > 0.35 for k in target):
        raise AgentError("excessive tracking error after settling")
    result = {"ok": True, "requested": requested, "bounded": target, "sent": sent,
              "measured": measured.state, "post_settle": observation_record(fresh)}
    log.write("action_complete", **result)
    return result, fresh


def run_episode(adapter, provider, config: AgentConfig, log_path: Path,
                *, clock=time.monotonic, sleep=time.sleep, cancelled=lambda: False) -> dict:
    try:
        config.validate()
        log = JsonlLog(log_path, clock)
    except BaseException:
        _close(adapter, provider)
        raise
    deadline = clock() + config.episode_timeout_s
    result = {"status": "max_steps", "success": None, "success_basis": "unverified",
              "reason": "decision budget exhausted", "steps": 0, "usage": {}}
    seen = set()
    try:
        log.write("start", task=config.task, model=config.model, max_steps=config.max_steps,
                  settle_s=config.settle_s, max_joint_delta=config.max_joint_delta,
                  motion_timeout_s=config.motion_timeout_s, api_timeout_s=config.api_timeout_s,
                  episode_timeout_s=config.episode_timeout_s)
        _check(deadline, clock, cancelled)
        obs = adapter.observe()
        for step in range(config.max_steps):
            _check(deadline, clock, cancelled)
            log.write("observation", **observation_record(obs))
            result["steps"] = step + 1  # every attempt counts, including empty/malformed responses
            api_deadline = min(deadline, clock() + config.api_timeout_s)
            decision = provider.decide(obs, timeout_s=api_deadline - clock())
            _check(api_deadline, clock, cancelled)
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = decision.usage.get(key, 0)
                if type(value) is int and value >= 0:
                    result["usage"][key] = result["usage"].get(key, 0) + value
            calls = decision.calls
            if len(calls) > 16:
                raise AgentError("excessive tool calls")
            ids = [c.call_id for c in calls]
            if any(not isinstance(i, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", i) for i in ids):
                raise AgentError("invalid call ID")
            rejection = None
            if len(calls) != 1:
                rejection = "exactly one tool call required; none executed"
            elif ids[0] in seen:
                rejection = "duplicate call ID; no action replayed"
            seen.update(ids)  # even rejected decisions consume their IDs
            args = None
            if rejection is None:
                try:
                    args = validate_tool(calls[0].name, calls[0].arguments)
                except ValueError:
                    rejection = "invalid tool arguments; none executed"
            log.write("decision", response_id=str(decision.response_id)[:128], call_ids=ids,
                      tool=calls[0].name[:128] if len(calls) == 1 else None,
                      arguments=args, error=rejection, usage=result["usage"])
            if rejection:
                provider.record_result(decision, {"ok": False, "error": rejection})
                obs = adapter.observe()
                continue
            call = calls[0]
            if call.name == "finish":
                result.update(status="finished", success=args["success"], success_basis="model_declared",
                              reason=args["reason"])
                provider.record_result(decision, {"ok": True, "success_basis": "model_declared", **args})
                break
            if call.name == "observe":
                obs = adapter.observe()
                output = {"ok": True, **observation_record(obs)}
            else:
                output, obs = fixed_target_operation(adapter, call.name, args, config, deadline, log,
                                                    clock=clock, sleep=sleep, cancelled=cancelled)
            _check(deadline, clock, cancelled)
            provider.record_result(decision, output)
    except KeyboardInterrupt:
        result.update(status="cancelled", reason="cancelled; no further actions submitted")
    except DeadlineExceeded:
        result.update(status="deadline", reason="deadline exceeded; no further actions submitted")
    except Exception as exc:  # noqa: BLE001 — every provider/feedback fault must release ownership.
        # Never log provider/transport error bodies, which can contain credentials or image data.
        from .agent_openai import ProviderError

        reason = str(exc) if isinstance(exc, (AgentError, ProviderError)) else type(exc).__name__
        result.update(status="error", reason=f"{reason}; no further actions submitted")
        try:
            log.write("error", error_type=type(exc).__name__)
        except (AgentError, OSError):
            pass  # Preserve the reserved termination record when the journal fills.
    finally:
        try:
            _close(adapter, provider)
        except BaseException as exc:  # noqa: BLE001 — record even a second cancellation during cleanup.
            result.update(status="error", success=None, success_basis="unverified",
                          reason=f"cleanup failed ({type(exc).__name__})")
        try:
            log.write("termination", final=True, **result)
        finally:
            log.close()
    return _redact(result)
