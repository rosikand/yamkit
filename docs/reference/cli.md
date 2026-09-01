# CLI reference

Run `yamkit --help` or `yamkit <command> --help` for the installed option details. Global `--verbose` must appear before the command name.

## Inspection and configuration

| Command | Purpose | Motor behavior |
| --- | --- | --- |
| `version` | Print the package version and resolved repository root. | None |
| `env` | Print repository-local cache environment exports. | None |
| `doctor` | Check environment, dependencies, plugins, CAN, cameras, and rig. | Does not connect to arms |
| `can` | List CAN interfaces, USB serials, bitrate, state, traffic, and errors. | Passive |
| `discover` | Probe buses and propose or write a rig. | Passive; does not enable motors |
| `swap A B` | Exchange physical mapping and calibration for same-role rig entries. | None |

```bash
yamkit can --udev
yamkit discover --rig configs/rig.yaml --write
yamkit doctor --rig configs/rig.yaml
```

## Arm operations

| Command | Purpose |
| --- | --- |
| `read [ARMS...]` | Stream joint, gripper, and button state at a selected rate. |
| `calibrate-gripper ARM` | Force I2RT gripper calibration and save its limits. |
| `zero-handle ARM` | Re-zero a released teaching-handle trigger in device EEPROM. |
| `set-rest ARM` | Save the current six-joint pose. |
| `rest [ARMS...]` | Move arms to saved poses over a requested duration. |
| `teleop` | Run yamkit's direct leader-to-follower loop. |

These commands connect to hardware; several command motion. Review [Safety](../operations/safety.md) first.

```bash
yamkit read left_leader --hz 10 --duration 5
yamkit teleop --pair left_follower --duration 30
yamkit rest left_follower --duration 6
```

## LeRobot wrappers

| Command | Delegates to | Required yamkit options |
| --- | --- | --- |
| `teleoperate` | `lerobot-teleoperate` | none |
| `record` | `lerobot-record` | `--name`, `--task` |
| `rollout` | `lerobot-rollout` | `--policy`, `--task` |
| `train` | `lerobot-train` | `--dataset` |
| `policy-check` | in-process LeRobot policy loading | `--policy` |

The four delegated wrappers accept additional LeRobot flags. Each supports `--dry-run`; `policy-check` instead executes against synthetic data without connecting to arms.

```bash
yamkit record --name demo --task "move the block" --arms left --dry-run
yamkit rollout --policy outputs/train/demo/checkpoints/last/pretrained_model \
  --task "move the block" --arms left --rtc --dry-run
```

## Rig selection

The default rig path is `configs/rig.yaml`. Override it consistently when testing another setup:

```bash
yamkit --verbose teleop --rig configs/bench-rig.yaml --pair left_follower
```
