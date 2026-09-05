# Optional Modal inference and browser deployment

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
| `molmoact2` / `lerobot/MolmoAct2-BimanualYAM-LeRobot` | Native 14-dimensional fixture | Local sync or Modal unguided async; both YAM followers, calibrated LINEAR_4310 grippers, RGB top/left_wrist/right_wrist, 30 Hz |
| `pi05` / `lerobot/pi05_base` | Native 32-dimensional fixture | Blocked: no published YAM physical mapping/statistics |
| Compatible custom local checkpoint | Existing local policy-check | Existing local LeRobot CPU/GPU path and supported local RTC |
| Unreviewed custom Modal checkpoint | Rejected | Requires a reviewed immutable profile and mapping |

MolmoAct2's saved processor uses absolute joint pose control. Its local synchronous
path is allowed. The pinned model advertises RTC support for continuous inference,
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
yamkit modal-prepare --policy molmoact2
yamkit policy-check --policy molmoact2 --backend modal --steps 3
```

`modal-prepare` explicitly deploys a dedicated app with a randomized `yamkit-vla-…`
name, loads the immutable checkpoint, and validates readiness. It records only resource
IDs and public metadata in `outputs/modal/owned-service.json`. Repeated preparation
of the same ready profile warms that pool; switching models requires shutting it down
first. A check never implicitly deploys an app. `--modal-app` can select an explicitly
prepared dedicated service. Failed preparations attempt to shut down their own app.

The service has one fixed profile and one L40S maximum, zero minimum/buffer containers,
serialized model state, finite startup/request timeouts, and no permanent heartbeat.
Ordinary requests retain warmth; production idle scale-down defaults to 300 seconds
(configurable to 300–600 through the app factory). Idle warm time is billable.
`--development` caps startup/request/idle timeouts at 240/90/15 seconds. The development
validation ledger and overall deadline are separate from these production commands.

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
motion. `--center-crop` is optional for Modal checks/probes/rollout, uses the same logged
policy-boundary transform, and leaves recording settings unchanged. It cannot restore
training camera geometry.

After operator acceptance and a separate motion decision:

```bash
yamkit rollout --policy molmoact2 --backend modal --task 'pick up the red cube' --duration 30
```

Rollout uses new observations and fresh predictions. Readiness and all static mapping
checks occur before follower connection/startup homing. Existing speed clamps and the
400 ms firmware timeout remain intact. Remote Stop/fault invalidates in-flight replies
and queues, stops local execution, and releases the arms without fault-triggered homing,
replaying actions or switching to CPU. Stop during readiness prevents activation; Stop
during the existing startup home interrupts it. Local custom rollout retains its
existing normal homing behavior.

Stop the robot process with Ctrl-C (or UI Stop), then shut down the cloud separately:

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
GPUs; the existing preview tiles retain their original camera ownership mechanism.

- **Check** launches a hardware-free fresh-chunk check.
- **Prepare Modal** explicitly deploys/warms; it never activates a robot.
- **Probe saved observation** opens only the snapshot file.
- **Probe live active read** requires its own motor-activation confirmation.
- **Start rollout** requires a separate motion confirmation and fresh observations.
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

Integration with the separate preview/hardening branches touches only the inference
page in `ui/app.js`, inference routes in `ui/server.py`, and narrow session lifecycle
handling in `ui/sessions.py`. Resolve overlapping edits there deliberately. This feature
reuses `camsHTML`/`syncCams`, `/api/cameras/.../stream`, and CameraHub suspension; it adds
no competing preview reader or capture/preview IPC mechanism.

## Acceptance and operational measurements

Before a physical run, confirm physical left/right devices, gripper open/closed
calibration, matching motor zero/alignment frame, camera view placement, safe workspace
and accessible Stop. Inspect an active-read probe's **unclipped** targets and deltas,
then make a separate supervised decision to run a short rollout. Do not execute a
saved probe's old chunk. No real robot acceptance was performed during development.

Measure from the actual robot host: cold load, warm round-trip p50/p95/p99 with sample
count, encoding/payload/server times, observation age, queue depth/peak and underruns.
The native check reports no hardware queue metrics. Remote rollout emits bounded
request/queue metrics on success and faults. Camera timestamps currently indicate local
snapshot receipt, not sensor exposure time. A one-second Molmo chunk does not guarantee
latency coverage. Container changes stop the session and require preparation again.

All spawned SDK payloads use US storage, independently of the selected compute region;
a typical three-camera 640×480 RGB observation is 2.76 MB. See the source-linked model
document for routing details. WebSocket transport is a future measured upgrade.
See [development validation results](MODAL_VALIDATION.md) for real/mocked outcomes,
resource IDs, costs, retained storage and unresolved access/dependency limits.
