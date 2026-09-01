# yamkit — I2RT YAM arms: CAN setup → teleop → LeRobot datasets → VLA inference

> Practical guides, architecture, API examples, and the full operator reference are in
> [`docs/`](docs/index.md). Preview the Material for MkDocs site with
> `uv run --extra docs mkdocs serve`.

Self-contained toolkit for the four YAM arms on this machine. **Everything lives in this directory**:
the Python interpreter (`.uv-python/`), the virtualenv (`.venv/`), the uv cache (`.uv-cache/`), the
vendored vendor SDK (`third_party/i2rt`, pinned in `third_party/i2rt.VERSION`), datasets/models
(`data/`) and training outputs (`outputs/`). Nothing is installed system-wide.

```
src/yamkit/                 core package + `yamkit` CLI
  can.py                    SocketCAN enumeration, USB-serial mapping, bring-up commands
  discovery.py              passive bus probe → leader / follower classification
  config.py                 rig file schema (configs/rig.yaml)
  arm.py                    YamArm: safe wrapper over i2rt MotorChainRobot (speed-clamped targets)
  teleop.py                 leader→follower session (button engage, sync move, optional bilateral)
  cli.py                    all commands
plugins/lerobot_robot_yamkit/         LeRobot Robot plugin: yam_follower, bi_yam_follower
plugins/lerobot_teleoperator_yamkit/  LeRobot Teleoperator plugin: yam_leader, bi_yam_leader
configs/rig.yaml            which adapter (by USB serial) is which arm, pairs, cameras, control knobs
scripts/env.sh              `source` it to activate the env;  scripts/can_up.sh brings CAN up (sudo)
system/                     optional templates for boot-time CAN bring-up (NOT installed)
third_party/i2rt            vendored I2RT SDK (+ one documented local patch)
tests/                      hardware-free unit tests (fake robot)
```

## Setup on a fresh machine

```bash
git clone <this repo> yamkit && cd yamkit
./setup.sh             # installs uv (if missing), Python 3.12 and all deps — all inside this directory
source scripts/env.sh  # activate; also exports HF_HOME/HF_LEROBOT_HOME/... into ./data
scripts/can_up.sh      # bring the CAN adapters up at 1 Mbit/s (needs sudo, once per boot)
yamkit doctor          # sanity check: venv, torch, lerobot, plugins, CAN, cameras, rig
```

Prerequisites on the box: Linux with SocketCAN (any Ubuntu), `build-essential` + `curl`, internet.
Optional: `can-utils` (`candump`), `sudo` for bringing CAN up.
`configs/rig.yaml` identifies the arms by CAN-adapter USB serial, so the same arms + adapters work
unchanged on a new computer; with different adapters run `yamkit discover --write` (then check
left/right with `yamkit read`/`yamkit teleop` and fix with `yamkit swap` if needed).
The lockfile (`uv.lock`) pins every package, so installs are reproducible.

You can also skip activation and use `uv run yamkit ...` or `.venv/bin/yamkit ...`.
Any Python started from `.venv` automatically redirects HuggingFace/LeRobot/torch caches into
`./data` (via `yamkit_env.pth` → `yamkit._env`), so plain `lerobot-*` commands are self-contained too.

## 1. CAN and arm discovery

```bash
yamkit can                 # adapters, state, bitrate, USB serial. Prints sudo commands if any are down
scripts/can_up.sh          # bring all adapters up at 1 Mbit/s (needs sudo); --reset to recover a wedged one
yamkit discover --write    # passive probe (no motor is enabled) → writes configs/rig.yaml
```

Discovery classifies a bus as **leader** (motors 1–6 + teaching-handle encoder) or **follower**
(motors 1–7). Left/right names are assigned in discovery order — **verify physically** and edit
`configs/rig.yaml` (`side`, names, pairs) if needed. Arms are matched to adapters by USB serial, so
the mapping survives reboots and re-plugging without udev rules.

For boot-time bring-up, see `system/80-yam-can.network` (systemd-networkd) or
`system/yamkit-can.service`; installing either is a system change and is left to you.

## 2. Reading, calibration, rest poses

```bash
yamkit read left_leader left_follower      # gravity-comp mode, streams q / gripper / buttons
yamkit calibrate-gripper left_follower     # SDK gripper auto-calibration → limits stored in the rig
yamkit set-rest left_follower              # store current pose as rest pose
yamkit rest                                # move every arm with a rest pose there, slowly
yamkit zero-handle right_leader            # re-zero a trigger encoder (only if trigger reads wrong)
```

Connecting an arm enables its motors in the vendor's gravity-compensation mode (it stays free to
move). On exit the arm is left compliant; the motors fall back to firmware damping after 400 ms.

## 3. Teleop

```bash
yamkit teleop                       # all pairs; press the handle's top button to engage / release
yamkit teleop --pair left_follower --auto-engage --duration 20
yamkit teleop --bilateral-kp 0.15   # force feedback on the leader (0.1–0.2 recommended)
```

On engage the follower moves to the leader pose over `control.sync_seconds`, then tracks at
`control.teleop_hz`. Follower targets are always clamped to `control.max_joint_speed` (rad/s) and
`control.max_gripper_speed`, so a jump in the target becomes a bounded-speed move.

## 4. Cameras

Add cameras to `configs/rig.yaml`; they are used by the LeRobot plugins automatically:

```yaml
cameras:
  top:   {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}
  wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}
  # depth: {type: intelrealsense, serial_number_or_name: "1234", width: 640, height: 480, fps: 30}  # needs `uv sync --extra realsense`
```

`lerobot-find-cameras opencv` lists what is attached. There were **no cameras** on this machine
when the repo was set up; recording for VLAs and VLA inference need at least one.

## 5. Record datasets (LeRobot)

```bash
yamkit record --name pick_cube --task "pick up the red cube and place it in the bowl" \
              --episodes 20 --episode-s 30 --reset-s 10 --fps 30
# → data/datasets/pick_cube  (LeRobot v3 dataset; --push to upload to the Hub)
yamkit teleoperate                    # same plugins, LeRobot's teleop loop (no recording)
lerobot-dataset-viz --repo-id yamkit/pick_cube --root data/datasets/pick_cube --episode-index 0
```

`yamkit record` is a thin wrapper around `lerobot-record`; any extra `--flag=value` is passed through
(e.g. `--dataset.streaming_encoding=true --display_data=true`). Keys in the dataset:
`observation.state` / `action` = `left_joint_1.pos … left_gripper.pos, right_…` (radians, gripper 0–1),
`observation.images.<camera>`.

Equivalent raw command (single arm shown):

```bash
lerobot-record --robot.type=yam_follower --robot.rig=configs/rig.yaml --robot.arm=left_follower \
               --teleop.type=yam_leader --teleop.rig=configs/rig.yaml --teleop.arm=left_leader \
               --dataset.repo_id=yamkit/pick_cube --dataset.root=data/datasets/pick_cube \
               --dataset.single_task="pick up the red cube" --dataset.num_episodes=20
```

## 6. Fine-tune a VLA

This box has no NVIDIA GPU, so training happens elsewhere: copy `data/datasets/<name>` (or `--push`
it to the Hub) to a GPU machine with the same repo, then

```bash
yamkit train --dataset pick_cube --policy-type smolvla --pretrained lerobot/smolvla_base --steps 20000
yamkit train --dataset pick_cube --policy-type pi05 --pretrained lerobot/pi05_base --batch-size 4   # heavier
yamkit train --dataset pick_cube --policy-type act --pretrained "" --steps 50000                   # small, fast
# checkpoints → outputs/train/<job>/checkpoints/last/pretrained_model
```

Bring the `pretrained_model` directory back under `outputs/` (or push it to the Hub).

## 7. Run a policy on the arms

First check that a checkpoint loads for this rig and how fast it runs here (no arm is energised):

```bash
yamkit policy-check --policy lerobot/smolvla_base --task "pick up the red cube"
# smolvla on this CPU: ~35 s load, ~0.8 s per 50-step action chunk, 14-d state/action, 1 camera
```

Then deploy:

```bash
yamkit rollout --policy outputs/train/smolvla_pick_cube/checkpoints/last/pretrained_model \
               --task "pick up the red cube" --duration 60 --rtc
yamkit rollout --policy lerobot/smolvla_base --task "..." --dry-run    # print the lerobot-rollout command only
```

`--rtc` enables LeRobot's real-time-chunking inference, which hides most of the latency of a slow VLA
on CPU. The same speed clamps as in teleop bound every commanded step. For a remote GPU, run
`lerobot-rollout` here with the policy served by LeRobot's async inference (`lerobot[async]`) over
Tailscale, or copy the checkpoint over.

## Safety notes

* Nothing is enabled during `yamkit can` / `yamkit discover`. Everything else energises motors.
* Keep the workspace clear when engaging teleop: the follower moves to the leader pose first.
* Another program on this machine (`ctrl_pi` Docker container) can drive the same buses; make sure
  it is idle (`candump can0` shows nothing) before starting yamkit.
* Motor timeout (400 ms) is left at the factory default — do not disable it.

## Command reference

| command | what it does |
|---|---|
| `yamkit can` | List CAN adapters (state, bitrate, USB serial) and how to bring them up. |
| `yamkit discover` | Passively probe each CAN interface (no motor is enabled) and classify leader/follower arms. |
| `yamkit read` | Connect (gravity-compensation mode, arm stays free to move) and stream joint state. |
| `yamkit teleop` | Leader→follower teleoperation (press the teaching-handle button to engage/disengage). |
| `yamkit calibrate-gripper` | Run the SDK gripper limit auto-calibration once and store the limits in the rig (skipped afterwards). |
| `yamkit swap` | Swap the physical arms behind two rig names (e.g. after finding "left_leader" is really the right one). |
| `yamkit zero-handle` | Re-zero a leader's teaching-handle trigger encoder at its current (released) position. |
| `yamkit set-rest` | Store the arm's current pose as its rest pose (used by `yamkit rest`). |
| `yamkit rest` | Move arm(s) slowly to their stored rest pose, then release. |
| `yamkit teleoperate` | Teleop through LeRobot's `lerobot-teleoperate` (same plugins used for recording). |
| `yamkit record` | Record teleop episodes into a LeRobot dataset (`lerobot-record`). |
| `yamkit rollout` | Run a policy/VLA on the follower arm(s) (`lerobot-rollout`). |
| `yamkit train` | Fine-tune a policy with `lerobot-train` (needs a GPU box; see README for the remote workflow). |
| `yamkit policy-check` | Load a policy/VLA for this rig and run it on a synthetic frame (no arm is energised). |
| `yamkit doctor` | Check the environment: venv, torch, CAN, plugins, cameras, rig file, data dirs. |
| `yamkit env` | Print the environment variables that keep everything inside this repo (for `eval`). |

Every command accepts `--help`; `record`/`teleoperate`/`rollout`/`train` pass unknown `--flags` straight to the underlying `lerobot-*` script and `--dry-run` prints the exact command instead of running it.

## How it fits together

```
teaching handle ─┐                      ┌─ YamArm.command() (speed-clamped) ─► follower motors
leader motors ───┴─► YamArm.read() ──► TeleopSession (yamkit teleop)
                                  └──► YamLeader.get_action() ─► lerobot-record / -teleoperate ─► YamFollower.send_action()
cameras (rig.yaml) ─► YamFollower.get_observation() ─► LeRobotDataset (data/datasets/<name>)
checkpoint ─► lerobot-rollout ─► policy.select_action() ─► YamFollower.send_action()
```

* `yamkit.arm.YamArm` is the only place that talks to the vendor SDK (`i2rt.MotorChainRobot`).
* The LeRobot plugins (`plugins/`) adapt `YamArm` to LeRobot's `Robot` / `Teleoperator` interfaces;
  everything LeRobot offers (datasets, viz, training, rollout strategies, async inference) works
  through them.
* The rig file is the single source of truth for hardware identity and control limits.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CAN interface … is DOWN` | `scripts/can_up.sh` |
| `no CAN adapter with serial …` | adapter unplugged / different adapters → `yamkit can`, `yamkit discover --write` |
| bus errors, arm unresponsive | `scripts/can_up.sh --reset`, power-cycle the arm |
| wrong arm responds to a name | `yamkit swap <a> <b>` |
| trigger reads ~0 while released | `yamkit zero-handle <leader>` |
| follower moves too fast | lower `control.max_joint_speed` in `configs/rig.yaml` |
| torchcodec / libavutil errors in logs | harmless: LeRobot falls back to PyAV for video |
| policy too slow on CPU | `yamkit rollout --rtc`, or serve the policy from a GPU box |

## Development

```bash
make test      # hardware-free tests (fake robot)
make lint
uv sync --extra dev        # after editing pyproject.toml or plugins/*
```
