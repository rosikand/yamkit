# LeRobot workflows

yamkit installs four LeRobot plugins and wraps the standard record, train, teleoperate, and rollout scripts with rig-derived defaults.

| Role | Single arm | Bimanual |
| --- | --- | --- |
| Robot | `yam_follower` | `bi_yam_follower` |
| Teleoperator | `yam_leader` | `bi_yam_leader` |

## Record a dataset

Ensure cameras are configured and the complete teleoperation path has been tested first.

```bash
yamkit record \
  --name pick_cube \
  --task "pick up the red cube and place it in the bowl" \
  --episodes 20 \
  --episode-s 30 \
  --reset-s 10 \
  --fps 30
```

The dataset is written to `data/datasets/pick_cube` with repository ID `yamkit/pick_cube`. Add `--push` to upload after recording or `--resume` to continue an existing recording.

Before moving hardware, inspect the exact LeRobot invocation:

```bash
yamkit record --name pick_cube --task "pick up the red cube" --dry-run
```

Dataset features are:

- `observation.state` and `action`: six joint positions in radians plus normalized gripper position where present;
- bimanual feature names prefixed with `left_` and `right_`;
- camera frames under `observation.images.<camera>` in the LeRobot dataset.

Inspect an episode with LeRobot:

```bash
lerobot-dataset-viz \
  --repo-id yamkit/pick_cube \
  --root data/datasets/pick_cube \
  --episode-index 0
```

## Train on a GPU host

The repository's default Torch source is CPU-only. Move the dataset or push it to the Hub, prepare this repository on a GPU machine, and run a supported LeRobot policy:

```bash
yamkit train \
  --dataset pick_cube \
  --policy-type smolvla \
  --pretrained lerobot/smolvla_base \
  --steps 20000
```

Other examples already supported by the wrapper:

```bash
yamkit train --dataset pick_cube --policy-type pi05 --pretrained lerobot/pi05_base --batch-size 4
yamkit train --dataset pick_cube --policy-type act --pretrained "" --steps 50000
```

Outputs go to `outputs/train/<policy>_<dataset>` unless `--job-name` overrides the final directory name.

## Validate a checkpoint offline

`policy-check` loads a policy, builds a synthetic observation from the selected rig, and times action selection without connecting to any arm:

```bash
yamkit policy-check \
  --policy outputs/train/smolvla_pick_cube/checkpoints/last/pretrained_model \
  --task "pick up the red cube"
```

Use `--keep-policy-features` when you intentionally want the checkpoint's own input feature schema instead of adapting it to this rig.

## Run a policy

Review [Safety](safety.md), test the hardware path manually, and dry-run the command before deployment:

```bash
yamkit rollout \
  --policy outputs/train/smolvla_pick_cube/checkpoints/last/pretrained_model \
  --task "pick up the red cube" \
  --duration 60 \
  --rtc \
  --dry-run
```

Remove `--dry-run` only after reviewing the generated command. `--rtc` selects real-time chunking inference, intended to hide policy latency. Follower commands still pass through `YamArm.command()` and its rig-derived speed clamps.

## Pass options through

The `record`, `teleoperate`, `rollout`, and `train` wrappers accept unknown options and append them to the underlying LeRobot command:

```bash
yamkit record \
  --name pick_cube \
  --task "pick up the red cube" \
  --dataset.streaming_encoding=true \
  --display_data=true
```

Prefer `--key=value` for nested LeRobot options so Typer does not consume their values.
