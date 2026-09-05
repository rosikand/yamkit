# Integrated yamkit acceptance test

This checklist separates software verification, paid inference, camera capture, and
supervised motor operation. No physical robot or camera acceptance was performed by
the integration agents. An inference result is not a manipulation result.

**Current motion restrictions:** live LLM execution is blocked because state/image
acquisition freshness cannot be verified. Physical Modal rollout is blocked by the
remote performance gate. The MolmoAct2 mapping is reviewed from source; no individual
rig's calibration, motor frames, camera geometry, or manipulation performance is
validated. Passing a check or probe does not remove either block.

The exact integrated source revisions, milestone commits and verification results are in
[integration-yamkit-v1.md](integration-yamkit-v1.md). The performance gate and measured
synthetic/real-service distinctions are in [REMOTE_PERFORMANCE.md](REMOTE_PERFORMANCE.md).

## Effects and preparation

Every command block below lists six effects, in this order:
**motors / calibration / home-or-motion / camera capture / paid cloud-or-API / physical policy actions**.
“Possible” means the command can have that effect with the rig's current settings.
Gravity compensation energises motors and is not guaranteed motion-free. Operator
teleop commands are physical motion, but are not policy-generated actions.

Work from the repository root. Keep the interpreter, environment, caches, recordings,
models, snapshots and logs inside this repository. Use an already bootstrapped
environment; do not run interactive hardware discovery/setup on the cloud VM.

Effects for preparation commands: **no / no / no / no / no / no**. They select the repo-local
environment and create an ignored report directory.

```bash
source scripts/env.sh
mkdir -p .context/acceptance .context/tmp
export TMPDIR="$PWD/.context/tmp"
```

Optional dependency refresh, if required: **no / no / no / no / no / no**. It downloads
packages but does not call an inference service. Keep the CPU PyTorch source configured
by the project. Modal's CUDA requirements belong only in its separate service image.

```bash
uv sync --extra dev --extra agent --extra modal
```

Credentials remain server-side in the operator's process environment. Preserve
`YAMKIT_OPENAI_API_KEY`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`, and
`HF_NAMESPACE`; do not paste values into browser fields, rig YAML, commands, logs,
or this checklist. Hugging Face login state stays under `data/hf/`. No database
configuration is needed. Never publish the ignored acceptance artifacts without
reviewing their scene/task content.

## A. Offline tests

Run on the cloud workspace or rig computer without connected hardware. All commands
below: **no / no / no / no / no / no**. Tests use fake robots/cameras and mock services;
the browser and preview benchmark use synthetic images and localhost sockets.

```bash
HF_HUB_OFFLINE=1 make test
make lint
git diff --check
.venv/bin/python scripts/browser_smoke.py --output .context/acceptance/browser.json
.venv/bin/python scripts/benchmark_preview.py --duration 10 --warmup 1 --output .context/acceptance/preview-benchmark.json
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m scripts.benchmark_remote --output .context/acceptance/remote-performance.json
```

For focused diagnosis, the following also have all six effects **no**:

```bash
.venv/bin/pytest -q tests/test_preview.py tests/test_preview_plugins.py tests/test_preview_sessions.py
.venv/bin/pytest -q tests/test_agent.py tests/test_agent_robot.py tests/test_agent_cli.py tests/test_agent_openai.py tests/test_agent_review.py
.venv/bin/pytest -q tests/test_remote_policy.py tests/test_local_rollout.py tests/test_probes.py tests/test_probe_runner.py
```

Pass criteria: full suite and lint pass; no whitespace errors; browser report has
no JavaScript exceptions or forbidden real-service/hardware calls. Recording,
reset, upload, and direct-preview transitions work with a single capture owner.
Benchmark reports show bounded latest-frame handoff and cleanup with no remaining
viewer threads. Compare timing to the committed integration benchmark, retaining
host/CPU details; do not treat benchmark timing as robot tracking performance.

The complete suite includes deterministic native-teleop/recording parity tests in
`tests/test_operator_parity.py` ([implementation notes](OPERATOR_PARITY.md)):
identical starting state, leader motion, gripper input and button transitions must
produce equivalent gated follower commands. Dataset `action` must label the gated
command, including follower hold while the leader moves disengaged.

## B. CPU local policy inference

Effects: **no / no / no / no / no / no**. This loads real local weights and can download
checkpoint assets to the repo cache. It does not contact Modal/OpenAI or acquire a rig.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 yamkit policy-check --policy smolvla --backend local --device cpu --steps 3
```

Require three distinct fresh `predict_action_chunk` calls, each finite with shape
`50 x 6`, saved preprocessing/postprocessing, pinned revision and an explicitly
synthetic source. Do not count cached action pops as fresh inference. The integrated
baseline artifact is `docs/integration-results/baseline-smolvla.json`: all three calls
passed; observed RTTs were about 3.784, 3.677 and 4.497 seconds on the cloud VM.
These results do not establish local continuous-control speed or a 14-dimensional
YAM mapping. Local fake-robot rollout coverage is in A.

## C. Cloud inference using fixtures/fake robot

This is an optional, separately approved paid retest. Standalone C already obtained
real L40S fresh-chunk results for SmolVLA, MolmoAct2-YAM and pi05. The integration does
not need to spend money repeating those calls unless an integration regression requires
it. Inference page load and changing model/backend must not deploy or warm a service.

Preparation and check effects: **no / no / no / no / yes / no**. Each command may incur
build, GPU, idle-container or storage charges. Preparation explicitly creates/warms
one owned service; checks require an explicitly prepared service.

```bash
yamkit modal-prepare --policy molmoact2 --gpu L40S --development
yamkit policy-check --policy molmoact2 --backend modal --steps 3
```

Require immutable model/saved processor identity, the expected `30 x 14` finite chunk,
three fresh requests and clear synthetic source labels. Record sample count, observed
request timings, server timings and payload sizes. A tiny forward-pass check is not
a p95/p99 performance qualification and has no hardware action queue.

Cleanup effects: **no / no / no / no / cloud control request; existing costs may persist until stopped / no**.
Shutdown stops only the service in this workspace's ownership receipt and preserves
the weight cache. Inspect the shutdown result; stopping local execution alone does
not shut down cloud compute.

```bash
yamkit modal-shutdown
```

To retest another reviewed model, shut down the prior owned pool first, then use the
same two commands with `--policy smolvla` or `--policy pi05`. Their effects are identical.
Do not deploy concurrent test pools or delete unrelated cloud resources.

## D. Camera-only preview

Perform only on the rig host after separately authorizing camera access. Camera
enumeration effects: **no / no / no / no (enumeration only) / no / no**.

```bash
yamkit cameras --rig configs/rig.yaml
```

UI effects: **no / no / no / yes only when a page subscribes to preview / no paid inference / no**.
On Inference, camera streams are opt-in through **Show camera previews**: initial page
load must open no camera streams, energise no arms, load no heavy weights and start no
GPU. Hub metadata may be fetched according to existing login/settings.

```bash
yamkit ui --rig configs/rig.yaml
```

For the no-capture page-load check, first open `http://127.0.0.1:8400/#/inference`
with no other preview viewers. Opening the root URL instead selects Live and starts
camera previews. Then inspect Live/Record previews or opt in with **Show camera
previews** on Inference, and verify the physical top/left-wrist/right-wrist views.
Do not press Read, Rest, Teleop Start,
Record Start, Live active read, or Start rollout in this stage. Confirm browser feeds
use the existing `/api/cameras/<name>/stream` route. Closing a preview subscription
releases its camera when no other viewer owns it; stop the UI process with Ctrl-C to
end the camera-only session. Closing a browser is not a general robot Stop.

## E. Basic physical lifecycle

These are supervised rig-host tests. Keep all other CAN controllers idle and verify
the operator can reach Stop and the rig's physical emergency-stop/power procedure.
Do not disable speed clamps, bypass an ownership lock, or change the 400 ms firmware
timeout. Save the current rig before discovery/calibration changes.

Each command below: **no / no / no / no / no / no**. `discover --write` updates rig
identity/configuration by serial; it enumerates cameras without streaming them.

```bash
cp configs/rig.yaml .context/acceptance/rig-before.yaml
yamkit can
yamkit doctor --rig configs/rig.yaml
yamkit discover --rig configs/rig.yaml --write
```

First state read: **yes / possible if gripper limits are absent / possible calibration or
gravity-compensation motion; no deliberate home trajectory / no / no / no**.

```bash
yamkit read left_leader left_follower --rig configs/rig.yaml --duration 5
```

Verify finite states, physical left/right identity, handle buttons and gripper direction.
Repeat only with the intended other pair after the first pair passes. Ctrl-C stops the
read and releases its arms without a home trajectory. A stale ownership error must
stop the test; do not manually remove a live process's ownership state.

Intentional gripper calibration, only if needed: **yes / yes / yes (open/close calibration;
no commanded home) / no / no / no**.

```bash
yamkit calibrate-gripper left_follower --rig configs/rig.yaml
```

Intentional parking, after independently confirming the path is clear:
**yes / possible if unsaved calibration / yes (home) / no / no / no**.

```bash
yamkit rest --rig configs/rig.yaml
```

This stage passes only with supervised lifecycle evidence. Passing software cleanup
tests cannot substitute for checking this rig's mechanical zero, alignment, stops or
gripper calibration. Ctrl-C during `yamkit rest` cancels its current home move and
releases the arms where they are. The second-Stop rule applies when teleop/recording
has already begun its return-home after the first Stop.

## F. Teleop

First bounded native session: **yes / possible / yes on engagement; `--no-home` skips
start/stop home moves / no / no / no physical policy actions**.

```bash
yamkit teleop --rig configs/rig.yaml --pair left_follower --duration 15 --no-home
```

Start disengaged. Move the leader a small distance, press the top button once to
engage, observe bounded synchronization, then move slowly. Press again to disengage;
the follower must hold while the leader moves freely, including its initial disengaged
pose. Synchronization advances per control tick, so disengaging during synchronization
must stop tracking and enter hold promptly. Check gripper direction and that a held
button does not repeatedly toggle. Repeat with both pairs only after
single-pair acceptance. Do not use auto-engage for initial acceptance.

The normal native session omits `--no-home`: **yes / possible / yes, including normal
start/stop home moves / no / no / no physical policy actions**.

```bash
yamkit teleop --rig configs/rig.yaml --pair left_follower --duration 15
```

Use this to accept intended normal Stop/homing separately. Fault cleanup must not
start a new home move. Native bilateral feedback remains available under
`control.bilateral_kp` or `--bilateral-kp`; recording bilateral feedback is unsupported.
Do not infer bilateral parity from position-command parity.

## G. Recording + live preview + reset/upload

Recording effects: **yes / possible / yes, including engagement and normal startup/stop
homing / yes / no paid inference (`--to local`) / no physical policy actions**.
Use a new dataset name on repeat runs. The actual CLI flag is singular `--episode-s`.
Use `yamkit record` / `yamkit teleoperate` wrappers for YAM leader input: direct raw
`lerobot-record` / `lerobot-teleoperate` invocations with YAM leader actions must reject
with guidance instead of bypassing the operator gate.

```bash
yamkit record --rig configs/rig.yaml --name acceptance_parity_01 --task "supervised slow teleop acceptance" --arms left_follower --episodes 2 --episode-s 10 --reset-s 5 --fps 30 --to local
```

To accept live recorder-owned previews, stop the CLI session before starting a session
from the **Record** page in the already running UI. **The Record page has no arm selector
and starts all configured pairs**, so first complete F's acceptance for both pairs and
supervise both sides; the single-pair CLI example above has a narrower hardware scope.
The CLI alone does not create a UI-owned preview session.
Use the same safe trajectory/button/gripper sequence as F. While disengaged, move the
leader without moving the follower; inspect the recorded dataset's action/state plot:
`action` must remain the follower hold command rather than the free-moving leader input.
Schema/key order must stay compatible with prior YAM datasets.

Across recording and the reset interval between these two episodes, verify frames keep
arriving from the recorder's single camera owner; no second capture opens and preview
failure does not stop recording.
After Stop, inspect the saved episode in Datasets and verify the final dataset is readable.
This stage is position/gripper/operator parity only; recording bilateral feedback is unsupported.

For the upload transition, check both **this computer (data/datasets)** and **also upload
to Hugging Face Hub** on the Record page (again supervising all configured pairs), or run:
**yes / possible / yes / yes / no paid inference; Hub upload/network/storage possible / no physical policy actions**.

```bash
yamkit record --rig configs/rig.yaml --name acceptance_upload_01 --task "supervised recording preview upload acceptance" --arms left_follower --episodes 2 --episode-s 10 --reset-s 5 --fps 30 --to both
```

Observe recorder-owned preview → recorder/camera cleanup → direct preview during Hub
upload → finished session. Use `both` so the local dataset remains available. `--to hub`
instead deletes the local copy after successful upload. An upload failure/cancellation
must preserve the local dataset.

Upload-only retry effects: **no / no / no / no / no paid inference; Hub upload/network/storage possible / no**.

```bash
yamkit push-dataset acceptance_upload_01 --rig configs/rig.yaml
```

## H. Saved-observation VLA probe

Generate a clearly labeled synthetic snapshot, without touching a rig. Effects:
**no / no / no / no / no / no**. This writes only `data/probes/acceptance-synthetic.npz`.

```bash
.venv/bin/python - <<'PY'
import time
import numpy as np
from yamkit.inference.profiles import get_profile
from yamkit.probes import ProbeObservation, save_observation

profile = get_profile("molmoact2")
state = np.zeros(len(profile.state_names), dtype=np.float32)
state[6] = state[13] = 0.5
images = {name: np.full((480, 640, 3), 127, dtype=np.uint8) for name in profile.image_keys}
save_observation("data/probes/acceptance-synthetic.npz", ProbeObservation(
    state, profile.state_names, images,
    source="synthetic acceptance fixture; no robot/camera capture", captured_at=time.time(),
))
PY
```

After explicit cloud preparation in C, probe effects: **no / no / no / no / yes / no**.

```bash
yamkit policy-probe --policy molmoact2 --backend modal --saved data/probes/acceptance-synthetic.npz --task "inspect synthetic acceptance fixture"
```

The UI equivalent is **Probe saved observation** with that path. Require source/age,
ordered state and action names, signed deltas, unclipped extrema, saved processing
identity and diagnostic flags. An old snapshot can legitimately be stale: preserve
its original timestamp instead of making it appear fresh. No prediction is sent to
an arm, stored as an executable chunk, or reused by a later rollout.

If a justified image transform/transport change is made, repeat this same saved input
with the changed path and compare camera transforms and outputs. With approved paid
budget, the explicit crop comparison is **no / no / no / no / yes / no**:

```bash
yamkit policy-probe --policy molmoact2 --backend modal --saved data/probes/acceptance-synthetic.npz --center-crop --task "inspect synthetic acceptance fixture"
```

Cropping changes the view and cannot validate this rig's training-camera geometry.
Use `modal-shutdown` from C when paid work finishes.

## I. Gravity-compensation ACTIVE-READ probe

This is a separate supervised motor-activation decision on the rig host. Readiness
and schema/calibration preflight must finish before motor activation. Saved gripper
limits are mandatory; probes reject missing/invalid limits rather than auto-calibrating.

Effects: **yes / no (missing calibration rejected) / possible gravity-compensation
motion, no home or policy trajectory / yes / yes with Modal / no**.

```bash
yamkit policy-probe --rig configs/rig.yaml --policy molmoact2 --backend modal --live --approve-active-read --arms left_follower --arms right_follower --task "inspect supervised rig observation"
```

UI equivalent: select MolmoAct2/Modal/both followers, choose **Probe live active read**,
and accept its **GRAVITY-COMPENSATION ACTIVE READ** confirmation only after inspecting
the physical rig. This confirmation is distinct from rollout motion confirmation.
Cameras use the same exclusive ownership handshake as recording. Arms/cameras close
before inference output is inspected; no predicted position is executed.

Require unclipped targets/deltas, robot units, source/age and all anomaly flags. Check
camera orientation and physical mapping with the operator. A probe result does not
demonstrate exposure-level sensor freshness and does not enable the LLM or Modal gate.
Stop with Ctrl-C/UI **Stop local execution**; verify local session exit, then separately
shut down cloud compute when no further paid checks are intended.

## J. Short Modal rollout — BLOCKED

Physical Modal acceptance is blocked. A nominal Molmo chunk has 30 actions at 30 Hz:
one second of horizon. Standalone C measured about 1.48-second warm RPC and 2.38–3.17-second
rig-resolution saved probes. Those measurements did not establish continuous queue supply.
[Final integration performance results](REMOTE_PERFORMANCE.md) report observed/injected
timings separately: two 32-second healthy fake scenarios each completed 69 requests
with zero underruns; a 700 ms spike caused a safe underrun, and all three historical
Molmo delay scenarios sent zero actions.
Fake-delay overlap tests do not establish real Modal network p95/p99.

On the final guarded branch, the following is a **negative gate check**, not a request
to move a robot. Expected effects: **no / no / no / no / no / no** because static
performance validation must reject it before readiness, GPU calls or plugin connection.
Run only after the final offline gate regression is green.

```bash
yamkit rollout --rig configs/rig.example.yaml --policy molmoact2 --backend modal --task "blocked acceptance gate check" --duration 3
```

Require an explicit physical-Modal-BLOCKED/performance explanation. In the UI, selecting
Modal must expose the same restriction; **Start rollout** must not bypass it through
confirmation, custom input, stale readiness or a direct POST. The control remains part
of the combined Inference page, with unsupported-combination messaging.

Do not bypass this gate or lower FPS to obtain a passing test. Future approval requires
the final LeRobot execution path to demonstrate warm p50/p95/p99 over enough consecutive
requests, observation age, transforms/serialization/payload, dispatch/network/server
queue/preprocess/model/postprocess/decode timing, queue depth at request start/return,
remaining valid horizon, expired prefixes dropped and no systematic underruns. Predictions
must overlap execution and start early enough to keep the valid queue supplied. Do not
reuse stale chunks or call unguided async “RTC”.

There is no physical Modal run to approve in this release. The future short-run effects,
if separately qualified and explicitly authorized, would be **yes / possible / yes
(startup home and policy motion) / yes / yes / yes**. Local Stop/fault must invalidate
late remote results and release without a new home trajectory; cloud shutdown remains
separate. No automatic CPU fallback is permitted.

## K. LLM tool test

Offline fixture effects: **no / no / no / no / no / no**. Simulated commands only change
in-memory state. Leave out an explicit log path to obtain a unique ignored log on repeats.

```bash
yamkit agent --rig configs/rig.example.yaml --arm left_follower --model fixture --task "observe the fixture, move slightly, and finish" --dry-run --offline --max-steps 5 --settle-s 0
```

Inspect observation/target/readback/action_complete/termination events. Delta commands
are bounded to at most 0.10 rad per joint; gripper tools preserve joints. Each completed
action must supply advancing synthetic state/image acquisitions before the next decision.
Malformed/stale readback or post-settle feedback must prevent another action/decision.

Optional paid fixture effects: **no / no / no / no / yes (OpenAI) / no**. This is a
separately approved retest only if an integration regression needs it; standalone B's
real fixture smoke remains valid. Named fixture images/state and the task leave the machine.

```bash
yamkit agent --rig configs/rig.example.yaml --arm left_follower --model gpt-5.4 --task "observe the labeled fixture and finish" --dry-run --max-steps 3 --api-timeout-s 30 --episode-timeout-s 90
```

`--dry-run` alone does not eliminate API charges; `--offline` does. Cancellation ends
fixture execution and closes the provider; a delayed response must not submit actions.
Model-declared success is never independent manipulation success.

## L. Short LLM episode — live execution BLOCKED

Effects: **no / no / no / no / no / no**. This is a negative guard test; `--execute`
must fail before importing/constructing physical hardware or making an OpenAI request.

```bash
yamkit agent --rig configs/rig.example.yaml --arm left_follower --model gpt-5.4 --task "blocked live execution check" --execute --max-steps 1
```

Require the explicit sensor-acquisition freshness blocker and “No arm or camera was
opened.” The hardened plugin now supports `disconnect(home=False)`, including camera
failure/cancellation cleanup, but its observation boundary still lacks verifiable state
and per-image acquisition evidence. Re-timestamping cached reads is not sufficient.
There is no live LLM episode to approve until that contract and cancellation behavior
are implemented and qualified. There is no LLM-agent UI in this integration.

## Model/backend support at handoff

“Tested” means the specified inference/software path, not physical manipulation. The
source field `mapping_verified` identifies a reviewed source convention; it must not
be interpreted as this rig having passed physical validation.

| Model | Local inference tested? | Modal inference tested? | Physical YAM mapping validated on a rig? | Async | Guided RTC | Physical rollout status/reason |
|---|---|---|---|---|---|---|
| SmolVLA base | Yes: real CPU, three fresh 50×6 chunks | Yes: standalone L40S, three fresh chunks | No; native six-dimensional base profile has no reviewed YAM mapping | Hardware-free transport/worker tests only; no approved physical async path for this base profile | Unsupported for this physical profile; remote RTC unsupported | Blocked locally and remotely: missing physical YAM mapping; Modal additionally performance-gated |
| MolmoAct2-BimanualYAM | Local synchronous code path exists; no real local Molmo forward pass established by this integration | Yes: standalone L40S, three fresh 30×14 chunks and saved probes | No; source-defined 14-dimensional convention only, individual calibration/camera/frame acceptance absent | Unguided async implemented/tested with fake robot; physical Modal use blocked by performance gate | Local Molmo guidance unqualified/rejected; remote guided RTC unsupported | Modal BLOCKED for queue/performance qualification; local sync is a software path requiring separate rig/compute acceptance, not approved by these results |
| pi05 base | No real local pi05 forward pass established by this integration | Yes: standalone L40S, three fresh 50×32 chunks | No; native 32-dimensional base profile lacks YAM mapping/statistics | Hardware-free transport/worker tests only; no approved physical async path for this base profile | Unsupported for this physical profile; remote RTC unsupported | Blocked locally and remotely: missing physical mapping/statistics; Modal additionally performance-gated |

Compatible custom local checkpoints retain the existing LeRobot rollout path. Their
own saved feature/statistics/mapping and RTC support must be checked independently;
none inherits qualification from a base-model fixture or the Molmo source convention.

## Stop, rollback, and acceptance evidence

- **Native teleop/record:** use the session Stop control or Ctrl-C. Normal configured
  Stop can home the arms; the initial `--no-home` teleop test omits that motion. A second
  Stop/Ctrl-C during return-home releases immediately. Observe session exit and arm
  release before starting another controller. Fault cleanup must not initiate homing.
- **Remote execution:** stop the local managed process first; in-flight requests and
  queued actions must be invalidated. Do not treat stopping a GPU, closing the browser,
  or merely losing network connectivity as a robot Stop.
- **Probes/LLM:** stop the local command/session. Probes never dispatch predictions;
  LLM physical construction remains blocked. Use the rig's physical stop procedure
  for unexpected motion during any approved active-read/teleop/record test.
- **Cloud:** after the local session exits, use `yamkit modal-shutdown` (effects listed
  in C), inspect shutdown status, and preserve the weight cache and unrelated services.
- **Recording upload:** cancellation/failure preserves local data; retry upload-only
  rather than re-running a motor session. Keep local datasets with `--to both` during acceptance.
- **Rollback:** first stop local execution and verify ownership release; stop owned
  cloud compute. Preserve datasets, logs, rig calibration and source-SHA records.
  Restore only the deliberately backed-up rig file if its values caused the regression,
  or check out an independently reviewed known-good integration revision with a clean
  tree. Do not modify main, erase calibration, or delete active ownership locks.

Optional rig restoration after stopping every hardware session:
**no / no / no / no / no / no**. It changes the saved configuration used by future sessions.

```bash
cp .context/acceptance/rig-before.yaml configs/rig.yaml
```

For each accepted stage record branch/final SHA, operator/date/host, exact command/UI
selection, source/checkpoint revisions, rig identity/calibration/camera notes, expected
and observed effects, Stop outcome, log/report paths and pass/fail reason. Keep software
results separate from physical evidence. Current blockers for supervised policy-motion
acceptance are Modal performance, unavailable LLM acquisition freshness, and the absence
of any physically validated model/rig mapping. Camera-only, basic lifecycle and manual
operator acceptance are separate supervised activities, not implied by the software merge.
