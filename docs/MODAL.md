# Optional Modal inference and browser checks

**Physical Modal rollout requires a passing qualification on the actual robot host,
accepted mapping and explicit supervised confirmation.** No passing qualification
has been demonstrated by the current measurements. `modal-qualify` collects evidence
without hardware; failed, expired or foreign-host records cannot authorize motion.
CLI validation and the integrated runner enforce these conditions before activation;
direct raw LeRobot rollout cannot bypass that runner. Cloud workspaces cannot activate
physical Modal rollout, and browser Modal Start remains disabled.
See [the H100 and image-fidelity investigation](MOLMO_H100.md),
[the earlier latency investigation](MODAL_LATENCY.md),
[the performance gate](REMOTE_PERFORMANCE.md) and [staged acceptance](acceptance-test.md).

Local inference remains the default. Install the optional client with
`source scripts/env.sh` then `uv sync --extra dev --extra modal`. Add
`--extra molmoact2` when loading Molmo locally. The normal lockfile
still installs CPU PyTorch. Modal builds its separately pinned CUDA image; it never
replaces the local interpreter or PyTorch installation.

The runtime and dataset use LeRobot 0.6.1. The remote policy is a registered lightweight
`yamkit_remote` proxy with no local VLA weights. Execution uses the existing LeRobot
context, background worker, base strategy and YAM plugin. See
[the integration audit](REMOTE_LEROBOT_AUDIT.md) and
[model revisions/mapping evidence](INFERENCE_MODELS.md).

## Support matrix

| Checkpoint/path | Fresh CPU/GPU check | Physical rollout |
|---|---|---|
| `smolvla` / `lerobot/smolvla_base` | Native 6-dimensional fixture | Blocked: no published YAM physical mapping |
| `molmoact2` / `lerobot/MolmoAct2-BimanualYAM-LeRobot` | Native 14-dimensional fixture | Modal requires current robot-host qualification, mapping acceptance and supervised confirmation; no passing qualification demonstrated. Local sync path available; physical validation not performed |
| `pi05` / `lerobot/pi05_base` | Native 32-dimensional fixture | Blocked: no published YAM physical mapping/statistics |
| Compatible custom local checkpoint | Existing local policy-check | Existing local LeRobot CPU/GPU path and supported local RTC |
| Unreviewed custom Modal checkpoint | Rejected | Blocked; profile/mapping review would not remove the performance gate |

MolmoAct2's saved processor uses absolute joint pose control. Its local synchronous
path requires both standard YAM followers, calibrated LINEAR_4310 grippers, RGB
top/left_wrist/right_wrist cameras and 30 Hz. No supervised physical mapping or
rollout validation was performed. The pinned model advertises RTC support for continuous inference,
but yamkit rejects local Molmo RTC before activation because its physical profile and
prefix processing have not been qualified with guidance. A narrow local preflight parses the effective LeRobot
configuration, validates schema/options, then calls the unchanged upstream rollout.
For the reviewed Molmo checkpoint, immutable local metadata pins its nested model and
saved processor to the same revision. Custom compatible checkpoint weights remain
unchanged; Molmo metadata bundles live under `data/local_policy_bundles/`. Local devices
such as `cuda:0` and duration `0` retain upstream meaning. Upstream temporary checkpoint
configuration files default to `data/tmp/`.

This release does not claim guided remote RTC support for any profile. Prefix
conversion, normalized/relative coordinates and re-anchoring need further qualification
before guiding a remote denoiser. There is no automatic CPU takeover.

Check results are not a physical compatibility certificate. Base checkpoint fixtures
retain their original features and saved processors; they do not use policy-check's
legacy synthetic rig statistics. Compatible custom local checks still use that legacy
synthetic diagnostic path and now request fresh chunks rather than cached action pops.
No such synthetic statistics are used for rollout.

## Credentials on the robot computer

Credentials configured in Conductor do **not** arrive on the Lenovo after a Git pull.
Start yamkit in a shell containing `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. When the
model or its nested tokenizer/backbone requires Hub access, also set `HF_TOKEN`.
For example, read values without printing or embedding them in shell commands:

```bash
source scripts/env.sh
read -rs -p 'Modal token ID: ' MODAL_TOKEN_ID; echo
read -rs -p 'Modal token secret: ' MODAL_TOKEN_SECRET; echo
read -rs -p 'HF token (if needed): ' HF_TOKEN; echo
export MODAL_TOKEN_ID MODAL_TOKEN_SECRET HF_TOKEN
yamkit ui
```

Use environment/session credential management suitable for the operator's machine.
Do not put tokens in rig YAML, the frontend, browser storage, source files or Git.
Existing `yamkit hub login` continues to keep its token under `data/hf/`; remote
preparation forwards only `HF_TOKEN`, through a Modal Secret. Modal SDK credentials
remain on the client. `HF_NAMESPACE` is preserved and is not required for inference.
Neither `YAMKIT_OPENAI_API_KEY` nor `DATABASE_URL` is forwarded to inference children or
Modal. No OpenAI generation is involved. The UI reports only SET/MISSING.

## CLI workflow

These first commands do not activate arms or open cameras:

```bash
yamkit policy-check --policy smolvla --device cpu --steps 3
yamkit modal-prepare --policy molmoact2                         # billable cloud preparation
yamkit policy-check --policy molmoact2 --backend modal --steps 3 # billable inference
yamkit modal-qualify --policy molmoact2 --requests 50            # billable, fake hardware only
```

`modal-prepare` explicitly deploys a dedicated app with a randomized `yamkit-vla-…`
name, loads the immutable checkpoint, and validates readiness. It records only resource
IDs and public metadata in `outputs/modal/owned-service.json`. Repeated preparation
of the same ready profile warms that pool; switching models requires shutting it down
first. A check or qualification never implicitly deploys an app. `--modal-app` can
select an explicitly prepared dedicated service. Failed preparations attempt to shut
down their own app.

Compute placement and request routing both default to `us-west`. They are explicit
configuration choices, not a promise of available GPU capacity or observed routing.
Use `--region` and `--routing-region` when preparing a different pool; readiness
reports the requested compute region and the observed container region separately.
Changing either invalidates a previous qualification. For example, the defaults are:

```bash
yamkit modal-prepare --policy molmoact2 --region us-west --routing-region us-west
```

Preparation reserves 65,536 MiB of host memory by default. `--memory-mib` accepts
49,152–65,536 MiB for bounded diagnostic comparisons; it does not change model dtype,
weights or GPU count.

The service has one fixed profile and at most one GPU (L40S by default, or explicit
`--gpu H100` for a prepared H100 service), zero minimum/buffer containers,
serialized model state, finite startup/request timeouts, and no permanent heartbeat.
Ordinary requests retain warmth; production idle scale-down defaults to 300 seconds
(configurable to 300–600 through the app factory). Idle warm time is billable.
`--development` caps startup/request/idle timeouts at 240/90/15 seconds. The development
validation ledger and overall deadline are separate from these production commands.

### Transport and qualification

Modal images default to raw RGB (`--image-encoding rgb8`). Three images, ordered state
and task travel in one request. Encoding happens once per camera; recording resolution,
the optional center crop and the model's saved preprocessing remain unchanged.

JPEG qualities 85, 90 and 95 all exceeded the maximum gripper-difference limit of 0.02
in paired H100 tests with fixed images and model noise; repeated raw requests matched
exactly. The fixture selector also requires maximum joint difference no greater than
0.01 rad. Raw RGB is therefore the production fallback. `--image-encoding jpeg
--jpeg-quality 85` (or quality 90/95) remains available for explicit comparisons;
these settings have not established acceptable policy fidelity. See
[the H100 investigation](MOLMO_H100.md). Changing the codec requires new qualification
evidence and cannot reuse an earlier record.

The client reuses its service handle and calls Modal `.remote` from its request
worker. LeRobot's existing background inference overlaps local action execution.
`--call-mode spawn` retains the earlier transport for diagnostics on compatible
`us-east` routing. With `.remote`, Stop/reset invalidates the local response and
queue promptly; the SDK offers no cancellation handle for that remote call. Its
worker remains busy until the call returns, and the late result cannot execute.
Cloud shutdown remains a separate operation.

The next prediction starts as soon as a fresh observation is available after the
previous request, using a full-chunk queue threshold by default: 30 steps for Molmo
at its unchanged 30 Hz. `--prediction-queue-threshold` makes that trigger explicit
(0–30). There is still at most one prediction in flight. Earlier requests do not
extend action deadlines or preserve expired prefixes. This is unguided async,
not guided RTC.

Run qualification on the **Lenovo robot host**, using its configured image dimensions
and intended transport settings:

```bash
yamkit modal-qualify --policy molmoact2 --requests 50 --rig configs/rig.yaml \
  --image-encoding rgb8 --call-mode remote \
  --prediction-queue-threshold 30
```

The command reads the rig file but never enumerates or opens arms or cameras. It
first measures one initial request and at least 50 warm requests using generated
images. It then exercises the actual LeRobot worker, queue and YAM command path with
fake arms and generated camera frames against the same real service. It deliberately
stops during an in-flight request and checks that no later fake SDK commands execute.
The images contain generated texture so JPEG does not get an unrealistically small
payload from all-black frames. Measurements describe this host's networking and
software, not real camera exposure timestamps or physical robot performance.

The result is written to git-ignored `data/qualifications/modal-molmoact2.json`.
A failed measurement attempt replaces an older success with failed evidence; readiness failures
retain unknown placement as `null` and do not launch another inference stage.
Qualification requires all of the following:

- At least the requested number of completed warm requests in both paths, with
  raw timing samples and one matching container.
- Warm RPC p95 no greater than 80% of the effective usable action horizon. That
  horizon includes observation age and the measured horizon remaining after queue
  merge and expired-prefix removal; the nominal one-second chunk alone is insufficient.
- A supplied execution queue, successful fake commands, zero underruns or expired
  queued/dispatched actions, and zero commands after Stop. Normal expired-prefix
  removal remains enabled.
- Matching host identity, model/dependency revisions, image dimensions, JPEG settings,
  crop, transport, scheduling, cadence and compute/routing configuration. The record
  expires after 24 hours; changed settings require another qualification.

Cloud measurements are labeled `READY_FOR_LENOVO_QUALIFICATION` only when their
measurements pass. They never qualify the Lenovo, and copying their reports or record
cannot transfer that qualification. Physical mapping checks remain separate. The
qualified rollout interface requires both `--accept-mapping` and
`--confirm-supervised`; these flags cannot override failed, expired, mismatched or
foreign-host evidence. Browser Modal Start remains disabled.

A saved probe reads a bounded `.npz` snapshot without accessing arms or cameras:

```bash
yamkit policy-probe --policy molmoact2 --backend modal --saved data/probes/observation.npz
```

Snapshots use `yamkit.probes.save_observation(path, ProbeObservation(...))`: ordered
14-element state names/values, named HWC RGB uint8 images, source description and
capture time. Loading rejects pickle, malformed shapes, invalid order and oversized
payloads. Preserve the original source/age; a saved probe can diagnose an old observation
but its output is never executable or cached for a subsequent rollout.

Live probing requires explicit active-read approval:

```bash
yamkit policy-probe --policy molmoact2 --backend modal --live --approve-active-read
```

This is **GRAVITY-COMPENSATION ACTIVE READ**, not motors off or guaranteed motion-free.
Before any arm opens, every selected arm's gripper limits, hardware type, order and
camera schema must pass preflight. Readiness also finishes before activation. The
probe connects with `zero_gravity=True`, reads once, then closes; it never calls
policy position commands, `move_to` or `go_home`. Cameras have exclusive ownership.

Reports show state, first targets, signed deltas, joint radians, gripper values, and
full-chunk extrema before the model's saved clamp and local motor clipping. They flag
stale observations, nonfinite output, implausible deltas and gripper ranges. Those are
diagnostic flags, not physical joint limits. A successful probe never authorizes
motion. `--center-crop` is optional for Modal checks/probes, uses the same logged
policy-boundary transform, and leaves recording settings unchanged. It cannot restore
training camera geometry.

The physical command requires a passing record from the same host and configuration.
It rejects a missing or invalid qualification before activating hardware. Preparing
a pool or confirming motion alone is insufficient:

```bash
yamkit rollout --policy molmoact2 --backend modal --task 'pick up the red cube' --duration 30 \
  --accept-mapping --confirm-supervised
```

The gated runner is exercised with fake hardware and RPC in regression tests.
Qualification can additionally use real RPC with fake hardware. Its readiness,
cold first-forward warmup and static mapping checks precede follower connection/startup homing.
The integrated queue drops expired and overlapping prefixes and checks the original
action deadline again immediately before dispatch. Existing speed clamps and the
400 ms firmware timeout remain intact. Remote Stop/fault invalidates in-flight replies
and queues, stops local execution, and releases the arms without fault-triggered homing,
replaying actions or switching to CPU. Stop during readiness prevents activation; Stop
during the existing startup home interrupts it. Local custom rollout retains its
existing normal homing behavior.

Stop an active check/probe process with Ctrl-C (or UI Stop), then shut down the cloud separately:

```bash
yamkit modal-shutdown
```

Shutdown uses only this workspace's ownership receipt and verifies its containers have
retired. It does not stop unrelated apps. Production weights remain cached in the
named `yamkit-policy-weights` volume; shutting down compute does not delete that cache.
Use a dedicated `--cache-volume` for tests, and remove only that owned test volume after
all its apps stop. Never delete shared production storage as part of robot Stop.

## Browser workflow

Open the existing **Inference** page. Choose preset/custom local checkpoint, backend,
task, follower selection, duration and relevant device/GPU/RTC/crop options. Local is
the initial backend. The page and offline profile catalog do not load weights or start
GPUs. Camera previews initially stay off: **Show camera previews** explicitly requests
streams through the existing ownership mechanism. Opening Inference alone does not
capture camera images or activate motors.

- **Check** launches a hardware-free fresh-chunk check.
- **Prepare Modal** explicitly deploys/warms; it never activates a robot.
- **Probe saved observation** reads a snapshot without arms or cameras; the selected
  backend loads local weights or uses the prepared billable service.
- **Probe live active read** requires its own motor-activation confirmation.
- **Start rollout** is disabled for Modal, with the performance blocker displayed;
  a direct HTTP request is rejected too. Compatible local rollout requires a separate
  motion confirmation.
- **Stop local execution** signals the managed local process group. Stopping only a
  cloud request is insufficient, and closing the browser is not Stop.
- **Shut down owned cloud service** is a separate action, refused while another managed
  session is active.

Long operations run as existing managed CLI child processes. HTTP handlers return job
status without waiting for GPU builds or inference. One session owns the rig/cameras;
duplicate/concurrent launches are refused. Results include an operation ID and exact
configuration key. Changing configuration or saved snapshot invalidates displayed
completion/readiness; a stale job cannot mark another selection ready. Invalid request
responses omit input values so accidentally submitted credentials are not reflected.
Endpoints remain local by default (`yamkit ui` binds 127.0.0.1).

Inference reuses `camsHTML`/`syncCams`, `/api/cameras/.../stream`, and the merged camera
ownership handshake. A failed camera release retains ownership; preview failure does
not open a competing capture. See [UI preview behavior](UI.md). Recording and LeRobot
teleoperation use the yamkit wrappers' [shared operator processing](OPERATOR_PARITY.md).

## Acceptance and operational measurements

Physical Modal acceptance requires a passing current robot-host qualification,
followed by supervised physical
left/right, gripper, zero/alignment and camera checks. No such hardware acceptance was
performed, and the current measurements have not produced a passing qualification.
An active-read probe's **unclipped** targets and deltas remain diagnostic;
never execute a saved probe's old chunk. The [acceptance checklist](acceptance-test.md)
separates checks from commands that activate hardware.

Measure from the actual robot host: cold load, warm round-trip p50/p95/p99 with sample
count, encoding/payload/server times, observation age, queue depth/peak and underruns.
The native check reports no hardware queue metrics. Remote rollout emits bounded
request/queue metrics on success and faults. Camera timestamps currently indicate local
snapshot receipt, not sensor exposure time. A one-second Molmo chunk does not guarantee
latency coverage. Container changes stop the session and require preparation again.

The earlier raw transport sends 2,764,800 image bytes for three 640×480 RGB frames.
JPEG payload size depends on scene content and is measured per request. The benchmark
can observe the pinned SDK's actual serialization and blob-transfer costs without
serializing the input a second time. Requested routing and observed compute placement
are distinct; network-only latency and actual internal routing are not independently
observable. See [the latency investigation](MODAL_LATENCY.md) for measured results and limits.
See [historical C validation](MODAL_VALIDATION.md) for its real CPU/GPU results,
resource IDs and costs. [Integrated performance results](REMOTE_PERFORMANCE.md)
distinguish real-service measurements from injected fake RPC delays. Neither a
cloud benchmark nor a synthetic queue run qualifies physical deployment on another host.
