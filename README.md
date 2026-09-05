# yamkit — I2RT YAM arms: CAN setup → teleop → LeRobot datasets → VLA inference

Self-contained toolkit for the four YAM arms on this machine. **Everything lives in this directory**:
the Python interpreter (`.uv-python/`), the virtualenv (`.venv/`), the uv cache (`.uv-cache/`), the
vendored vendor SDK (`third_party/i2rt`, pinned in `third_party/i2rt.VERSION`), datasets/models
(`data/`) and training outputs (`outputs/`). Nothing is installed system-wide. Cooperative arm
ownership uses shared runtime lock files in `/tmp/yamkit-arm-locks` across checkouts.

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
configs/rig.yaml            YOUR rig: which adapter (by USB serial) is which arm, pairs, cameras, control knobs
                            (machine-specific, not in git; configs/rig.example.yaml shows the format)
scripts/env.sh              `source` it to activate the env
scripts/install_system.sh   one-time: CAN adapters come up at boot/hot-plug (sudo; offered by setup.sh)
scripts/can_up.sh           bring CAN up by hand (sudo) if you skipped the above
system/                     the systemd-networkd unit that install_system.sh installs (+ alternatives)
third_party/i2rt            vendored I2RT SDK (+ one documented local patch)
tests/                      hardware-free unit tests (fake robot)
```

## Setup on a fresh machine

Plug in the CAN adapters and cameras, power the arms, then:

```bash
git clone <this repo> yamkit && cd yamkit
./setup.sh             # 1. installs uv, Python 3.12 and all deps — all inside this directory
                       # 2. asks once for sudo: CAN adapters come up at boot / hot-plug from now on
                       # 3. writes configs/rig.yaml from the attached arms + cameras (no motor is enabled)
source scripts/env.sh  # activate (per terminal); also keeps HF/LeRobot/torch caches in ./data
yamkit ui              # http://127.0.0.1:8400 — camera feeds, arm status; never energises a motor
```

Prerequisites on the box: Linux with SocketCAN (any Ubuntu), `build-essential` + `curl`, internet.
Optional: `can-utils` (`candump`). `./setup.sh --no-system` skips the sudo step (then run
`scripts/can_up.sh` after every boot, or `scripts/install_system.sh` later).

The one thing setup cannot know is which physical arm is *left*: two followers look the same on
the bus. Check once with `yamkit read left_follower` (the arm stays free to move — wiggle it) and
`yamkit swap left_follower right_follower` if it was the other one; same for the wrist cameras
(`yamkit swap left_wrist right_wrist`). The rig file remembers it from then on.

**Changed cables?** Run `yamkit discover --write` again. Arms keep their names, calibration and
left/right (matched by adapter serial) and cameras keep theirs (matched by serial, then USB port,
then model). `yamkit doctor` tells you when the rig no longer matches what is plugged in.
Moving a CAN adapter to another USB port needs nothing at all. Re-running `./setup.sh` is safe
(it never overwrites an existing rig file); `make sync` only refreshes Python packages.

`configs/rig.yaml` is machine-specific and not in git — see `configs/rig.example.yaml` for the
format; it is written with comments so it can be edited by hand. The lockfile (`uv.lock`) pins
every package, so installs are reproducible.

You can also skip activation and use `uv run yamkit ...` or `.venv/bin/yamkit ...`.
Any Python started from `.venv` automatically redirects HuggingFace/LeRobot/torch caches into
`./data` (via `yamkit_env.pth` → `yamkit._env`), so plain `lerobot-*` commands are self-contained too.

## 1. CAN and discovery

```bash
yamkit can                 # adapters, state, bitrate, USB serial
yamkit cameras             # attached cameras (model, serial, USB port) and which rig name uses each
yamkit discover --write    # passive probe (no motor is enabled) + camera detection → configs/rig.yaml
scripts/install_system.sh  # one-time (sudo): adapters come up at boot and on hot-plug
scripts/can_up.sh          # by hand instead (sudo, every boot); --reset recovers a wedged adapter
```

Discovery classifies a bus as **leader** (motors 1–6 + teaching-handle encoder) or **follower**
(motors 1–7). New arms get provisional `left_*` / `right_*` names in discovery order — **verify
physically** (`yamkit read`, then `yamkit swap`). Arms are matched to adapters by USB serial, so
the mapping survives reboots, re-plugging and the kernel renumbering `can0…can3`.
Cameras: a RealSense D405 is taken to be a wrist camera (`left_wrist`, `right_wrist` in USB-port
order), any other camera becomes `top` (then `cam2`, …); only the colour stream of a RealSense is
used. Devices are stored as `/dev/v4l/by-path/…` links, which follow the USB port.

The boot-time bring-up is `system/80-yam-can.network` for systemd-networkd, which only touches
interfaces named `can*` (NetworkManager keeps wifi/ethernet). `scripts/install_system.sh --uninstall`
removes it. `system/yamkit-can.service` is an alternative for machines without networkd.

## 2. Reading, calibration, rest poses

```bash
yamkit read left_leader left_follower      # gravity-comp mode, streams q / gripper / buttons
yamkit calibrate-gripper left_follower     # SDK gripper auto-calibration → limits stored in the rig
yamkit align left_follower                 # once per pair: fold both arms to their stops → leader offsets stored
yamkit rest                                # park: every arm moves slowly to its home pose and is released
yamkit set-rest left_follower              # optional: store the current pose as that arm's home pose
yamkit zero-handle right_leader            # re-zero a trigger encoder (only if trigger reads wrong)
```

Connecting an arm enables its motors in the vendor's gravity-compensation mode (it stays free to
move). On exit the arm is left compliant; the motors fall back to firmware damping after 400 ms.

**Home** is the folded pose the vendor zeroed every joint at (all joints 0), unless `yamkit set-rest`
stored another one. Teleop, recording and rollout move every arm home slowly at Start and back home at
Stop, all arms at the same time (`control.home_speed` for followers and `control.leader_home_speed`
for leaders, both 0.25 rad/s by default; 0 turns it off). Leaders move with low gains so a hand on
the handle simply wins. A second Stop / Ctrl-C during the return releases the arms immediately.

**Align** fixes a follower that points slightly off its leader: the two arms' motor zeros never agree
exactly. `yamkit align` reads both arms folded against their stops and stores the per-joint difference
on the leader (`joint_offsets`); teleop, recording and rollout then all work in the follower's frame.

## 3. Teleop

```bash
yamkit teleop                       # all pairs; press the handle's top button to engage / release
yamkit teleop --pair left_follower --auto-engage --duration 20
yamkit teleop --bilateral-kp 0.15   # force feedback on the leader (0.1–0.2 recommended)
```

On start every arm moves to home; on engage the follower moves to the leader pose over
`control.sync_seconds`, then tracks at `control.teleop_hz`; on Ctrl-C every arm returns home before
being released (`--no-home` skips both moves). Follower targets are always clamped to
`control.max_joint_speed` (rad/s) and `control.max_gripper_speed`, so a jump in the target becomes a
bounded-speed move.

## 4. Cameras

`yamkit discover --write` fills in the `cameras:` section; `yamkit cameras` shows what is attached
and `yamkit doctor` flags a rig camera that is no longer there. Entries are plain LeRobot camera
configs plus informational `serial` / `model` / `notes` written by discovery:

```yaml
cameras:
  top:        {type: opencv, index_or_path: /dev/v4l/by-path/pci-…-usb-0:1.1:1.3-video-index0, width: 640, height: 480, fps: 30}
  left_wrist: {type: opencv, index_or_path: /dev/v4l/by-path/pci-…-usb-0:1.2:1.0-video-index4, width: 640, height: 480, fps: 30}
  # depth: {type: intelrealsense, serial_number_or_name: "1234", width: 640, height: 480, fps: 30}  # needs `uv sync --extra realsense`
```

Camera names become dataset keys (`observation.images.<name>`), so settle them before recording.
Wrist cameras crossed? `yamkit swap left_wrist right_wrist`. RealSense cameras are used as plain
colour webcams (no depth) unless `pyrealsense2` is installed. Recording for VLAs and VLA inference
need at least one camera. Note: a RealSense on a USB 2 port (`yamkit cameras` shows "USB 480 Mb/s")
can drop frames at 640x480@30 when it shares the hub with another camera.

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

## 5b. Hugging Face Hub (optional)

Sign in once, then recordings can go to the Hub instead of (or as well as) this computer, and
models trained anywhere can be pulled straight from the Hub for rollout. The token is stored in
`data/hf/token` (git-ignored), never in the rig file.

```bash
yamkit hub login                      # paste a "write" token from huggingface.co/settings/tokens
yamkit hub status
yamkit record --name pick_cube --task "…" --to hub      # local | hub | both (default: hub.datasets in the rig)
yamkit push-dataset pick_cube          # upload an existing local dataset  (--remove-local to free the disk)
yamkit pull-dataset andre/pick_cube    # download one into data/datasets/
yamkit train --dataset andre/pick_cube --push          # on any GPU box: pull the dataset, push the checkpoint
yamkit push-model outputs/train/<job>/checkpoints/last/pretrained_model
yamkit rollout --policy andre/act_pick_cube --task "…"  # a Hub id works wherever a checkpoint path does
```

The rig's `hub:` section holds the account name, whether uploads are private (default yes) and
where recordings go by default; the Settings page edits it, the Record page overrides it per
recording, and the Datasets / Models pages list local and Hub entries side by side.

## 6. Fine-tune a VLA

This box has no NVIDIA GPU, so VLA fine-tuning happens elsewhere: copy `data/datasets/<name>` (or
`--push` it to the Hub) to a GPU machine with the same repo, then

```bash
yamkit train --dataset pick_cube --policy-type smolvla --pretrained lerobot/smolvla_base --steps 20000
yamkit train --dataset pick_cube --policy-type pi05 --pretrained lerobot/pi05_base --batch-size 4   # heavier
yamkit train --dataset pick_cube --policy-type act --pretrained "" --steps 50000                   # small, fast
# checkpoints → outputs/train/<job>/checkpoints/last/pretrained_model
```

ACT (52M parameters) does train on this CPU: about 2.5 s per step at batch 2 with three 640x480
cameras, so a few thousand steps is an overnight job; `yamkit train` keeps the data loader in-process
on CPU boxes automatically.

Bring the `pretrained_model` directory back under `outputs/` (or push it to the Hub).

## 7. Run a policy on the arms

First check a checkpoint without activating hardware. Reviewed base-model checks use
checkpoint-native fixtures; compatible custom checkpoint checks use the rig's feature spec:

```bash
yamkit policy-check --policy lerobot/smolvla_base --task "pick up the red cube"
# smolvla base: three fresh 50-step chunks, native 6-d state/action; no physical YAM mapping implied
```

Then deploy:

```bash
yamkit rollout --policy outputs/train/smolvla_pick_cube/checkpoints/last/pretrained_model \
               --task "pick up the red cube" --duration 60 --rtc
yamkit rollout --policy outputs/train/my_policy/checkpoints/last/pretrained_model --task "..." --dry-run
```

`--rtc` enables LeRobot's real-time-chunking inference for compatible local policies.
Measure end-to-end latency before relying on chunk buffering. The same speed clamps
as in teleop bound every commanded step.

For optional Modal GPU inference and browser deployment, see [docs/MODAL.md](docs/MODAL.md).
Local remains the default. The reviewed MolmoAct2-YAM profile allows local synchronous
inference or Modal unguided async. The SmolVLA and pi05 base profiles support native
checks and are blocked from physical
rollout because they lack a reviewed YAM mapping. Guided remote RTC is unsupported.

For a small multimodal LLM controller, see [the agent guide](docs/AGENT.md). `yamkit agent` offers
an offline fixture mode and paid OpenAI calls with fixtures; live execution is disabled pending
the documented plugin cleanup and observation freshness fixes. VLA `rollout` is unchanged.

## 8. Web UI

```bash
yamkit ui        # → http://127.0.0.1:8400
```

Local dashboard: live camera/state/CAN view, teleop + recording control, dataset browser with an
episode viewer, policy-run history, checkpoint list. It is a thin wrapper — every hardware action
spawns the corresponding `yamkit` command as a child process, and opening pages never energises a
motor. See `docs/UI.md` and `docs/ui-screenshots/`.

## Safety notes

Arm commands now validate exact dimensions, finite values, measured state and vendor-configured
joint bounds before commands or gain changes. Bimanual actions prevalidate both sides. Homing
and synchronization respect configured target speeds even when their requested duration is too
short. See [hardware guarantees, ownership and supervised acceptance](docs/HARDWARE_HARDENING.md)
for the precise limits and optional `disconnect(home=False)` / `shutdown(home=False)` cleanup.

The stale-command ramp reset is not a watchdog: SDK threads can keep transmitting during an
application stall. Target bounds are not collision avoidance, measured-velocity guarantees, or
a safety-rated emergency stop. Cooperative locks do not protect against unrelated drivers.

* Nothing is enabled during `yamkit can` / `yamkit discover`. Everything else energises motors.
* Keep the workspace clear when engaging teleop: the follower moves to the leader pose first.
* Another program on this machine (`ctrl_pi` Docker container) can drive the same buses; make sure
  it is idle (`candump can0` shows nothing) before starting yamkit.
* Motor timeout (400 ms) is left at the factory default — do not disable it.

## Command reference

| command | what it does |
|---|---|
| `yamkit can` | List CAN adapters (state, bitrate, USB serial) and how to bring them up. |
| `yamkit cameras` | List attached cameras (model, serial, USB port) and which rig name uses each. Never streams. |
| `yamkit discover` | Passively probe each CAN interface (no motor is enabled), classify leader/follower arms, detect cameras; `--write` saves the rig (keeps names/calibration). |
| `yamkit read` | Connect (gravity-compensation mode, arm stays free to move) and stream joint state. |
| `yamkit teleop` | Leader→follower teleoperation (press the teaching-handle button to engage/disengage). |
| `yamkit calibrate-gripper` | Run the SDK gripper limit auto-calibration once and store the limits in the rig (skipped afterwards). |
| `yamkit swap` | Swap the physical devices behind two rig names — arms or cameras (e.g. "left_leader" is really the right one). |
| `yamkit zero-handle` | Re-zero a leader's teaching-handle trigger encoder at its current (released) position. |
| `yamkit align` | Once per pair: fold leader and follower to their stops, store the per-joint offset on the leader so both point the same way. |
| `yamkit set-rest` | Store the arm's current pose as its home pose (default home: all joints 0). |
| `yamkit rest` | Park: move arm(s) slowly to their home pose, then release them there. |
| `yamkit teleoperate` | Teleop through LeRobot's `lerobot-teleoperate` (same plugins used for recording). |
| `yamkit record` | Record teleop episodes into a LeRobot dataset (`lerobot-record`). |
| `yamkit rollout` | Run a policy/VLA on the follower arm(s) (`lerobot-rollout`). |
| `yamkit agent` | Bounded multimodal LLM controller with labeled fixtures; `--dry-run --offline` makes no API calls. Live execution is blocked; see [docs/AGENT.md](docs/AGENT.md). |
| `yamkit train` | Fine-tune a policy with `lerobot-train` (needs a GPU box; see README for the remote workflow). |
| `yamkit policy-check` | Load a policy/VLA for this rig and run it on a synthetic frame (no arm is energised). |
| `yamkit ui` | Serve the local web UI (viewer + launcher for the commands above; pages never energise a motor). |
| `yamkit doctor` | Check the environment: venv, torch, CAN (and boot-time bring-up), plugins, cameras, rig file vs attached hardware. |
| `yamkit hub login/status/logout` | Hugging Face sign-in (token kept in `data/hf`, never in the rig). |
| `yamkit push-dataset` / `pull-dataset` / `push-model` | Move datasets and checkpoints between this computer and the Hub. |
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
* The rig file is the single source of truth for hardware identity and control limits. It is
  machine-specific (git-ignored), written with comments for hand editing, and regenerated by
  `yamkit discover --write` without losing names, calibration or settings.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CAN interface … is DOWN` | `scripts/install_system.sh` once (adapters then come up by themselves), or `scripts/can_up.sh` now |
| `no CAN adapter with serial …` | adapter unplugged / different adapters → `yamkit can`, `yamkit discover --write` |
| camera black in the UI / `could not open …` | camera moved or unplugged → `yamkit cameras`, then `yamkit discover --write` |
| wrist cameras crossed | `yamkit swap left_wrist right_wrist` |
| bus errors, arm unresponsive | `scripts/can_up.sh --reset`, power-cycle the arm |
| wrong arm responds to a name | `yamkit swap <a> <b>` |
| trigger reads ~0 while released | `yamkit zero-handle <leader>` |
| follower moves too fast | lower `control.max_joint_speed` in `configs/rig.yaml` |
| follower points slightly off its leader | `yamkit align <arm>` (both arms folded to their stops) |
| arms should not move by themselves at Start/Stop | `control.home_speed: 0` in `configs/rig.yaml`, or `yamkit teleop --no-home` |
| torchcodec / libavutil errors in logs | harmless: LeRobot falls back to PyAV for video |
| policy too slow on CPU | `yamkit rollout --rtc`, or serve the policy from a GPU box |
| training stops silently at step 0 (leaked semaphores) | forked data-loader workers; `yamkit train` adds `--num_workers=0` on CPU boxes, pass it yourself to plain `lerobot-train` |

## Development

```bash
make test      # hardware-free tests (fake robot)
make lint
uv sync --extra dev        # after editing pyproject.toml or plugins/*
```
