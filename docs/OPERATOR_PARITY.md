# Native teleop and recording operator behavior

Use `yamkit teleop` with recording off and `yamkit record` with recording on. Both use
one small per-pair operator state (`PairGate`) for teaching-handle button edges,
engagement, synchronization, and disengaged hold. `yamkit teleoperate` supplies the
same processor to LeRobot's existing teleoperation entry point.

- A rising edge on `control.engage_button` toggles engagement. Holding the button
  does not repeatedly toggle it. Each bimanual side has its own button state.
- A disengaged follower holds a captured measured pose, including before first
  engagement. Moving the leader while disengaged does not move the follower.
- Engagement captures the leader target and synchronizes from the follower pose.
  Synchronization advances per tick, earns at most one loop period after a stall,
  and respects `control.sync_seconds` and configured speed bounds. Disengagement or
  Stop can interrupt synchronization; neither outer loop waits inside a blocking sync move.
- Both paths use `YamArm.read()`'s aligned joint frame and normalized gripper direction
  (0 closed, 1 open). `YamArm.command()` remains the command/limit boundary.
- A release obtains a fresh measured hold. Only that measured hold can replace an
  obsolete target without a speed-limited return toward it; a caller-supplied target
  cannot disable the clamps by claiming to be a hold.
- Normal completion/Stop retains configured home behavior. Errors and partial startup
  release through hardened no-home cleanup. Native `--no-home` remains available for
  explicitly skipping its startup/normal-stop home moves.

Native teleop and LeRobot keep their own loops, rates, processes and dataset handling.
The existing follower connection and recorder camera owner are reused. Recording reset
continues through LeRobot's existing loop and preserves the operator state and live previews.

## Recorded action labels

Pinned LeRobot 0.6.1 records the output of `teleop_action_processor`, after calling
`robot.send_action`, but ignores `send_action`'s return value. Gating only in the robot
plugin would therefore record moving leader targets while a disengaged follower holds.

The YAM wrapper injects an `OperatorStep` into that public recording hook. Raw leader
positions carry button metadata outside their public feature keys. The processor emits
a gated action object; the follower acknowledges the actual bounded values into that
same object before LeRobot builds its dataset frame. Captured holds also update the
shared operator state from the acknowledged measured pose. Existing feature names,
order, units and shapes are unchanged.

Use the `yamkit record` / `yamkit teleoperate` wrappers for this behavior. Raw
`lerobot-record` / `lerobot-teleoperate` with YAM leader input has no operator processor
and is rejected with wrapper guidance. Advanced direct Python integration must supply
`yamkit.lerobot_teleop.make_teleop_processor`; it must retain the pinned identity action
processor and acknowledgment path. No LeRobot recording code was copied or patched.

Native bilateral feedback remains supported. Recording and LeRobot teleoperation
bilateral feedback are explicitly unsupported: set `control.bilateral_kp: 0` before
using those wrappers, or use native teleop. This change does not create a feedback subsystem.

## Hardware-free verification

```bash
.venv/bin/pytest -q tests/test_operator_parity.py tests/test_teleop.py tests/test_plugins.py
.venv/bin/pytest -q tests/test_preview_plugins.py tests/test_preview_sessions.py
```

Deterministic tests exercise equivalent trajectories/buttons/grippers through native
teleop and the actual installed LeRobot recording loop, including recorded action
labels, independent sides, partial synchronization, holds and cancellation/fault cleanup.
Physical operator acceptance is a separate activity in [acceptance-test.md](acceptance-test.md).
