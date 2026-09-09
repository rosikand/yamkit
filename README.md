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
third_party/i2rt            vendored I2RT SDK (local patches listed in i2rt.VERSION)
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
stored another one. Teleop and recording normally move connected arms home slowly at Start and
back home at Stop, all arms at the same time (`control.home_speed` for followers and
`control.leader_home_speed` for leaders, both 0.25 rad/s by default; 0 turns it off). Local rollout
retains its normal return behavior. Leaders move with low gains so a hand on the handle simply
wins. A second Stop / Ctrl-C during the return releases the arms immediately. Failed startup or
operator-session faults release without an additional home move. Physical Modal rollout requires
a passing qualification on the robot host and explicit supervised confirmation.

**Align** fixes a follower that points slightly off its leader: the two arms' motor zeros never agree
exactly. `yamkit align` reads both arms folded against their stops and stores the per-joint difference
on the leader (`joint_offsets`); teleop and recording then map leader input into the follower's
frame. Policy targets must already use the matching follower frame.

## 3. Teleop

In the dashboard, click **Start Teleop** and wait for **Teleop ready**.
Start connects the arms, runs configured startup homing and automatically synchronizes
both followers to their leaders. **Stop** returns the arms home and releases them;
the page returns to idle after cleanup. There is no separate Park Arms control.
Handle buttons remain available to pause/resume an individual follower, but no button
press is required to start following. **Start Recording** uses the same automatic engagement.
Its episode timer, frame count and videos begin only after both followers finish
synchronizing and are ready. Preparation does not consume the requested episode duration.
The command-line defaults below still wait for the handle button; add `--auto-engage`
to `yamkit teleop`, `yamkit record` or `yamkit teleoperate` to select automatic engagement.

```bash
yamkit teleop                       # all pairs; press the handle's top button to engage / release
yamkit teleop --pair left_follower --auto-engage --duration 20
yamkit teleop --bilateral-kp 0.15   # force feedback on the leader (0.1–0.2 recommended)
```

On start every arm moves to home; on engage the follower moves to the leader pose over at least
`control.sync_seconds`, then tracks at `control.teleop_hz`. Synchronization advances each tick and
can be disengaged; disengaged followers hold a freshly measured pose. On normal Ctrl-C every arm
returns home before being released (`--no-home` skips both moves). Follower targets are clamped to
`control.max_joint_speed` (rad/s) and `control.max_gripper_speed`, so a jump in the target becomes a
bounded-speed move.

Native `yamkit teleop` supports bilateral feedback. Recording and LeRobot teleoperation reject
nonzero `control.bilateral_kp`; use `0` for those paths. See [operator parity](docs/OPERATOR_PARITY.md).
The native `--duration` interval starts after startup homing and engagement preparation.

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
yamkit record --name pick_cube --task "put the red cube into the black container" \
              --episodes 20 --episode-s 30 --reset-s 10 --fps 30
# → data/datasets/pick_cube  (LeRobot v3 dataset; --push to upload to the Hub)
yamkit teleoperate                    # same plugins, LeRobot's teleop loop (no recording)
lerobot-dataset-viz --repo-id yamkit/pick_cube --root data/datasets/pick_cube --episode-index 0
```

`yamkit record` is a thin wrapper around `lerobot-record`; extra `--flag=value` options are passed through
(e.g. `--dataset.streaming_encoding=true --display_data=true`). Keys in the dataset:
`observation.state` / `action` = `left_joint_1.pos … left_gripper.pos, right_…` (radians, gripper 0–1),
`observation.images.<camera>`.

Use the yamkit wrappers for YAM leader input. They install the same engage, synchronization and
hold processing used by native teleop, and recording stores the action actually sent after safety
clamps. Raw `lerobot-record` / `lerobot-teleoperate` reject unprocessed YAM leader actions with
guidance to use these wrappers. A single-arm recording is:

```bash
yamkit record --arms left_follower --name pick_cube \
              --task "put the red cube into the black container" --episodes 20
```

During recording or reset, the first Stop / Ctrl-C ends the acquisition loop and saves the
episode before finalization, configured homing and any requested upload. A further interrupt
keeps the normal cancellation behavior. Failed recordings are never uploaded or deleted;
inspect the retained local data before retrying an upload. Use `--name`, `--repo-id` and `--to`
for storage and Hub settings; nested flags that override those settings are rejected.

The dashboard reports startup, synchronization, readiness, saving and finishing from
the running process. Its first **Stop** finishes the current episode save, then returns
the arms home and releases them. A further Stop interrupts that work and may discard
an unfinished episode. During saving, followers hold their last command. Command-line
Ctrl-C during saving still interrupts the save. Stop during incomplete startup cancels
startup and releases connected arms without initiating an additional home move.

Dashboard recordings and CLI recordings with `--auto-engage` prepare each episode
before starting its clock or saving frames. The separate **session elapsed** display
includes startup time. If a follower was paused before the next episode, resume it
with its handle button; preparation waits without silently engaging it again. Pausing
during an episode keeps recording and does not restart its clock. Manual CLI recording
without `--auto-engage` retains its ability to record held poses before engagement.

After updating this branch, restart an idle `yamkit ui` process and refresh the page.
An already running dashboard keeps its old Python backend until restarted.

Recording defaults to LeRobot's `--dataset.encoder_threads=1` to leave CPU headroom for
arm control while multiple camera videos encode. Saving can take longer than automatic
parallelism. You can override it explicitly, for example `--dataset.encoder_threads=2`.
For SVT-AV1 this selects its parallelism level, not a strict operating-system thread limit.

## 5b. Hugging Face Hub (optional)

Sign in once, then recordings can go to the Hub instead of (or as well as) this computer, and
models trained anywhere can be pulled straight from the Hub for rollout. The token is stored in
`data/hf/token` (git-ignored), never in the rig file.

```bash
yamkit hub login                      # paste a "write" token from huggingface.co/settings/tokens
yamkit hub status
yamkit record --name pick_cube --task "…" --to hub      # local | hub | both (default: hub.datasets in the rig)
yamkit push-dataset pick_cube          # upload an existing local dataset  (--remove-local to free the disk)
yamkit push-dataset pick_cube --repo-id andre/pick_cube_v2  # optional explicit Hub destination
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
yamkit policy-check --policy lerobot/smolvla_base --task "put the red cube into the black container"
# smolvla base: three fresh 50-step chunks, native 6-d state/action; no physical YAM mapping implied
```

Then deploy:

```bash
yamkit rollout --policy outputs/train/smolvla_pick_cube/checkpoints/last/pretrained_model \
               --task "put the red cube into the black container" --duration 60 --rtc
yamkit rollout --policy outputs/train/my_policy/checkpoints/last/pretrained_model --task "..." --dry-run
```

`--rtc` enables LeRobot's real-time-chunking inference for compatible local policies.
Measure end-to-end latency before relying on chunk buffering. The same speed clamps
as in teleop bound every commanded step.

For optional Modal GPU inference and browser deployment, see [docs/MODAL.md](docs/MODAL.md).
For an existing Lambda GPU with a direct, private Lenovo SSH tunnel, use
[docs/LAMBDA.md](docs/LAMBDA.md). Its separate inference setup keeps CUDA dependencies inside
the GPU checkout and retains host qualification before any supervised robot rollout.
Local remains the default. MolmoAct2-YAM has a reviewed source mapping and a local synchronous
path. Supervised five-second run `20260908-220559-rollout-81cf2f4d` completed 131 bimanual policy dispatches at
29.9 Hz, with zero queue underruns, all three 30 fps recordings intact, and return home
completed in 2.1 seconds before release. Remote joint command shaping removed the earlier
sharp command reversals; the operator said it “seemed ok.” These command-space results
do not guarantee measured robot dynamics. **The orange-lid task still failed: no grasp
was requested or observed.** Videos, original RGB frames, traces and full metrics are in
**Runs → `20260908-220559-rollout-81cf2f4d`** and its
[private archive](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/e3d6f1a641cdab6aabdd125fd57568b6e28b6b33/runs/20260908-220559-rollout-81cf2f4d).
The subsequent ten-second attempt stopped early after a slow first response left only
eight valid actions. Startup now requires half a chunk of fresh actions before its
first policy dispatch; short initial queues are discarded without retiming targets.
The fix passed a hardware-free failure reproduction; physical validation is pending.
Model plans still vary; see the
[physical trial evidence and limits](docs/VLA_HTTP_ROLLOUT.md).

**Physical Modal rollout requires a current passing qualification on the actual robot host**,
separate mapping acceptance and explicit supervised confirmation. The trial's retained session
was retired; each new session must qualify again. The Inference page can now attach the exact
qualified retained session and launch a managed trial with optional debug video and joint traces.
The latest supervised UI trial saved all 150 frames per camera at nominal 30 fps, preserving
observation timestamps, with no capture drops or trace/export errors. Normal completion returns
the followers home slowly; Stop, faults and expiry retain prompt release. Warm physical inference
requests measured 338 ms p95. The trial's GPU shutdown was verified. Expired, stale or
mismatched sessions remain blocked; readiness or confirmation alone cannot enable motion. See the
[managed trial and debugging workflow](docs/VLA_DEBUGGING.md). Checks and probes remain available. SmolVLA
and pi05 base profiles support native checks and are blocked from physical rollout because they
lack a reviewed YAM mapping. Guided remote RTC and local Molmo guidance are unsupported. See
[remote performance and its measurement limits](docs/REMOTE_PERFORMANCE.md).

To inspect a rollout on another computer, optionally upload its finalized debug bundle
to a private HF dataset. The bundle retains all three videos, original RGB frames,
timing/trajectory JSONs, metrics, logs and sanitized provenance under `runs/<run-id>/`.
Upload runs after hardware release and artifact finalization; local originals remain.
See [rollout archives and download commands](docs/ROLLOUT_ARCHIVES.md).

Modal defaults to raw RGB, cached `.remote` calls and `us-west` placement, with earlier
requests through the same LeRobot async worker. JPEG qualities 85, 90 and 95 all exceeded
the gripper-difference limit in paired H100 fixture tests; raw RGB preserves the image
values. JPEG remains selectable for diagnostics. See [the H100 investigation](docs/MOLMO_H100.md).
An opt-in persistent HTTP path with `cuda_graph10` preserves raw RGB and ten-step
Molmo inference, warms the actual task before hardware connection, and requires fresh
qualification of its exact source, task, graph and container. See the
[HTTP preparation and qualification commands](docs/MODAL.md#transport-and-qualification).
The [HTTP rollout investigation](docs/VLA_HTTP_ROLLOUT.md) records Lenovo measurements,
numerical fidelity checks and the current qualification outcome.
Collect evidence on the robot host without opening its arms or cameras:

```bash
yamkit modal-prepare --policy molmoact2 --region us-west --routing-region us-west
yamkit modal-qualify --policy molmoact2 --requests 50
yamkit modal-shutdown
```

These are billable cloud operations. Qualification uses generated images and fake arms
with the real service; its record stays under `data/qualifications/`, applies only to
the measured host/settings and expires after 24 hours. It requires healthy queue execution,
Stop rejection of late actions, and warm p95 within 80% of the remaining usable action
horizon. Mapping acceptance and supervised confirmation remain separate. Cloud results
cannot qualify the Lenovo, and failed or expired records keep physical rollout blocked. Image,
transport and scheduling options are described in [the Modal guide](docs/MODAL.md);
see [the earlier latency investigation](docs/MODAL_LATENCY.md) for its measured settings and limits.

For a small multimodal LLM controller, see [the agent guide](docs/AGENT.md). `yamkit agent` offers
an offline fixture mode and paid OpenAI calls with fixtures; live execution is disabled pending
verified sensor acquisition freshness (hardened no-home cleanup is available).

## 8. Web UI

```bash
yamkit ui        # → http://127.0.0.1:8400
```

Local dashboard: live camera/state/CAN view, teleop + recording control, dataset browser with an
episode viewer, policy-run history, checkpoint list. It is a thin wrapper — every hardware action
spawns the corresponding `yamkit` command as a child process, and opening pages never energises a
motor. The Inference page starts with camera previews off; **Show camera previews** explicitly
opens them through the existing ownership mechanism. See [UI behavior](docs/UI.md) and
the [staged acceptance checklist](docs/acceptance-test.md) for command effects and Stop behavior.

## Safety notes

Arm commands now validate exact dimensions, finite values, measured state and vendor-configured
joint bounds before commands or gain changes. Bimanual actions prevalidate both sides. Homing
and synchronization respect configured target speeds even when their requested duration is too
short. See [hardware guarantees, ownership and supervised acceptance](docs/HARDWARE_HARDENING.md)
for the precise limits and optional `disconnect(home=False)` / `shutdown(home=False)` cleanup.

The stale-command ramp reset is not a watchdog: SDK threads can keep transmitting during an
application stall. Target bounds are not collision avoidance, measured-velocity guarantees, or
a safety-rated emergency stop. Cooperative locks do not protect against unrelated drivers.

* `yamkit can`, discovery and hardware-free policy checks do not energise motors. Arm read,
  calibration, teleoperation, recording and permitted rollout commands can energise them; an
  active-read probe requires its separate approval. See the acceptance checklist for each command.
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
| `yamkit teleoperate` | LeRobot teleoperation with shared YAM operator processing; bilateral feedback unsupported. |
| `yamkit record` | Record sent teleop actions into a LeRobot dataset with shared YAM operator processing. |
| `yamkit rollout` | Run a compatible policy/VLA on the followers; Modal additionally requires current host qualification, mapping acceptance and supervised confirmation. |
| `yamkit agent` | Bounded multimodal LLM controller with labeled fixtures; `--dry-run --offline` makes no API calls. Live execution is blocked; see [docs/AGENT.md](docs/AGENT.md). |
| `yamkit train` | Fine-tune a policy with `lerobot-train` (needs a GPU box; see README for the remote workflow). |
| `yamkit policy-check` | Load a policy/VLA for this rig and run it on a synthetic frame (no arm is energised). |
| `yamkit modal-prepare` / `modal-shutdown` | Explicitly prepare or shut down the owned Modal service; billable preparation, no hardware activation. |
| `yamkit modal-qualify` | Measure the real Modal service with generated images and fake hardware; write a host-bound qualification record. Failed records cannot authorize motion. |
| `yamkit ui` | Serve the local web UI (viewer + launcher for the commands above; pages never energise a motor). |
| `yamkit doctor` | Check the environment: venv, torch, CAN (and boot-time bring-up), plugins, cameras, rig file vs attached hardware. |
| `yamkit hub login/status/logout` | Hugging Face sign-in (token kept in `data/hf`, never in the rig). |
| `yamkit push-dataset` / `pull-dataset` / `push-model` | Move datasets and checkpoints between this computer and the Hub. |
| `yamkit env` | Print the environment variables that keep everything inside this repo (for `eval`). |

Every command accepts `--help`. Local `record`/`teleoperate`/`rollout`/`train` pass extra
`--flags` to LeRobot, subject to wrapper validation; `--dry-run` prints the command without running it.
Modal rollout rejects extra LeRobot flags and requires the same qualification and confirmation
checks even with `--dry-run`. Browser Modal Start is available through the managed retained-session
workflow only after its exact session passes the current qualification and confirmation checks;
expired or mismatched sessions remain blocked. A supervised UI debug trial completed with saved
videos and joint traces; manipulation success remains unvalidated. See [VLA debugging](docs/VLA_DEBUGGING.md).

## How it fits together

```
teaching handle ─┐                      ┌─ YamArm.command() (speed-clamped) ─► follower motors
leader motors ───┴─► YamArm.read() ──► TeleopSession (yamkit teleop)
                                  └──► YamLeader.get_action() ─► shared operator processor
                                       ─► LeRobot record / teleoperate ─► YamFollower.send_action()
cameras (rig.yaml) ─► YamFollower.get_observation() ─► LeRobotDataset (data/datasets/<name>)
checkpoint ─► lerobot-rollout ─► policy.select_action() ─► YamFollower.send_action()
```

* `yamkit.arm.YamArm` is the only place that talks to the vendor SDK (`i2rt.MotorChainRobot`).
* The LeRobot plugins (`plugins/`) adapt `YamArm` to LeRobot's `Robot` / `Teleoperator` interfaces.
  LeRobot owns datasets, visualization, training and supported rollout paths. YAM leader input
  requires the yamkit operator wrappers; the model and deployment restrictions above still apply.
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
| arms should not home at Start/Stop | Set both `control.home_speed: 0` and `control.leader_home_speed: 0`, or use `yamkit teleop --no-home` |
| torchcodec / libavutil errors in logs | harmless: LeRobot falls back to PyAV for video |
| policy too slow on CPU | Use a suitable local GPU or supported local RTC; physical Modal rollout requires a passing queue qualification on the robot host |
| training stops silently at step 0 (leaked semaphores) | forked data-loader workers; `yamkit train` adds `--num_workers=0` on CPU boxes, pass it yourself to plain `lerobot-train` |

## Development

```bash
make test      # hardware-free tests (fake robot)
make lint
uv sync --extra dev        # after editing pyproject.toml or plugins/*
```
