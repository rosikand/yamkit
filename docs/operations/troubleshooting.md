# Troubleshooting

Start with read-only diagnostics:

```bash
yamkit doctor
yamkit can
yamkit discover
```

Add `--verbose` before a command name to see yamkit, I2RT, CAN, and HTTP logs:

```bash
yamkit --verbose read left_leader
```

| Symptom | Check | Action |
| --- | --- | --- |
| CAN interface is `DOWN` | `yamkit can` | Run `scripts/can_up.sh`. |
| Bus errors or an unresponsive arm | Check error counters in `yamkit can`; confirm no competing controller | Run `scripts/can_up.sh --reset`, then power-cycle the arm if needed. |
| No adapter with the configured serial | Compare `yamkit can` with `configs/rig.yaml` | Reconnect the expected adapter or run `yamkit discover --write` for replacement hardware, then verify all physical names. |
| Wrong physical arm responds | Read one arm at a time | Use `yamkit swap` between arms with the same role. |
| Released trigger reads near `0` | `yamkit read <leader>` | Fully release it, then run `yamkit zero-handle <leader>`. |
| Follower moves too quickly | Inspect `control.max_joint_speed` and `max_gripper_speed` | Lower the rig values; do not bypass the clamp. |
| Teleop loop overruns | Observe the reported Hz and overrun count | Reduce `control.teleop_hz` or isolate expensive work from the control process. |
| Camera is absent | `lerobot-find-cameras opencv`; inspect `/dev/video*` | Correct its stable device path or RealSense serial in the rig. |
| Policy is slow on CPU | Run `yamkit policy-check` | Use `yamkit rollout --rtc` or LeRobot async inference from a GPU host. |
| TorchCodec/libavutil warning | Check whether recording continues via PyAV | The documented setup expects LeRobot to fall back to PyAV. |
| Plugin type is unknown | `yamkit doctor` | Run `uv sync --extra dev` after edits to plugin metadata or dependencies. |

## Inspect generated LeRobot commands

Wrapper failures are easier to separate from hardware failures with `--dry-run`:

```bash
yamkit teleoperate --dry-run
yamkit record --name smoke --task "smoke test" --dry-run
yamkit rollout --policy <checkpoint> --task "smoke test" --dry-run
yamkit train --dataset smoke --dry-run
```

Copy the printed command when you need to compare its options with the installed LeRobot CLI.

## Recover a CAN bus carefully

1. Stop every process that can drive the bus.
2. Inspect state and error counters with `yamkit can`.
3. Reset interfaces with `scripts/can_up.sh --reset`.
4. If errors continue, remove arm power, inspect cabling and termination, then power it back on.
5. Resume with passive discovery and a single-arm `read` before commanding motion.
