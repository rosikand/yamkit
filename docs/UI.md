# yamkit web UI (`yamkit ui`)

A small local web UI on top of the existing toolkit. Start it with:

```bash
yamkit ui                # http://127.0.0.1:8400
yamkit ui --port 9000 --rig configs/rig.yaml
```

Screenshots: [`docs/ui-screenshots/`](ui-screenshots/).
Current command effects and blocked stages: [staged acceptance](acceptance-test.md).

## Design rule: yamkit stays the source of truth

The UI adds **no second robot driver and no new control loop**:

* **Read-only pages** (Live status, Datasets, Deployments, Models) read only the rig file,
  sysfs (CAN), and the filesystem (`data/datasets/`, `outputs/`). Serving the UI or opening any
  page never connects to — and never energises — an arm.
* **Hardware actions** (Start state stream / Start Teleop / Start Recording / rollout) spawn the
  unmodified CLI as a child process — `yamkit read`, `yamkit teleop`, `yamkit record`,
  `yamkit rollout`, `yamkit policy-check` — and parse its stdout for display. Teleop is launched
  with `--print-state` (opt-in CLI flag, off by default in a terminal) so per-arm joint state
  lines flow to the Live page during teleop. A session started in a separate terminal is a
  different process — the UI cannot see its output. Stop sends the
  process group a SIGINT (identical to Ctrl-C in a terminal), escalating to SIGTERM/SIGKILL only
  if the child hangs. One session at a time.
* **Camera previews** keep the same MJPEG browser URL (`/api/cameras/<name>/stream`). Live and Record request their tiles; Inference starts with previews off and opens streams only after **Show camera previews**. While no session owns the cameras, the UI opens requested direct previews lazily. While a follower plugin owns them, the UI proxies images already acquired by that plugin's observation loop. Previews target 10 fps independently of the recording rate, and continue through resets while observations are acquired. There is only one capture owner at a time; the UI confirms that its direct captures have stopped before the plugin opens its cameras. A preview failure during ownership displays an unavailable tile and never reopens a competing capture.
* **UI-started sessions get no display**: `DISPLAY` / `WAYLAND_DISPLAY` are stripped from the child environment so LeRobot's recorder does not install its system-wide keyboard hook (Esc / arrows / n / r / q pressed in any window would otherwise end or skip an episode). Sessions are driven by the UI buttons only.

## Pages

Navigation is a fixed left sidebar; the theme switcher (Light / Dark / System) lives at its foot
(preference in `localStorage`; `?theme=light|dark` overrides for a single load).

| page | contents |
|---|---|
| **Live** | camera tiles (top / left wrist / right wrist), follower joint+gripper state, CAN/camera/rig status, current mode + loop rate, read-only controls (`yamkit read` stream), Park arms (`yamkit rest`: every arm moves slowly home and is released) |
| **Record** | camera tiles, teleop pair status (engaged, tracking error, Hz), dataset name / task / episodes / durations form (recording rate fixed at 30 fps unless changed under "Advanced", capped by the slowest camera), Start Teleop / Start Recording / Park / Stop, episode + elapsed progress, live log. Normal Start/Stop use configured homing; clicking Stop again during the return releases immediately. Startup and operator-session faults release without another home move |
| **Datasets** | LeRobot v3 datasets under `data/datasets/` and, when signed in, the account's Hub datasets in the same table with a local / cloud / both tag, Upload and Download buttons and Hub links; per-episode detail page with synchronized videos and state/action small-multiple charts |
| **Inference** | local/Modal selection, model/task/options, checks, explicit cloud preparation/shutdown and saved/live probes; camera previews opt in. Compatible local rollout requires motion confirmation. All physical Modal rollout is disabled by the performance gate. Run history includes latency, status, termination reason, logs and replay videos (`outputs/ui/deployments/`) |
| **Models** | checkpoint directories under `outputs/` and the account's Hub models, tagged local / cloud / both; per-checkpoint detail page with file sizes and `config.json` / `train_config.json` contents; Upload buttons; Hub models get their own detail page and can be typed straight into the rollout form |
| **Settings** | Hugging Face sign-in (the token goes to `data/hf/token`, never through the rig) and the rig's hub settings (account, private, default destination); view/edit `configs/rig.yaml`: structured fields for the control knobs, read-only arm/camera tables (with the discovery notes: model, serial, USB port), and a raw-YAML editor. Every save is validated server-side first (parse → `RigConfig` → `validate()`), raw YAML is written verbatim (comments kept), saving is refused while a hardware session runs, and the camera feeds are reloaded from the saved file (no restart). The rig file holds hardware identifiers only — no credentials pass through the UI. |

## Code layout

```
src/yamkit/ui/
  server.py     FastAPI app: /api/* endpoints + serves the frontend
  sessions.py   SessionManager (one yamkit child process, log ring buffer, output parsers)
  camstream.py  direct MJPEG camera hub (lazy open, confirmed release during session ownership)
  catalog.py    read-only scanners: datasets / models / deployment records
ui/             standalone frontend (vanilla HTML/JS/CSS, no build step)
tests/test_ui.py  hardware-free tests (parsers, sessions with stub children, API via TestClient)
```

The frontend is a single-page app (`ui/app.js`) with hash routing; it polls `/api/session` (1 s)
and `/api/overview` (5 s). Chart colors are the validated categorical pair per theme
(light: state `#2a78d6` / action `#eb6834`; dark: `#3987e5` / `#d95926`). The Inter variable
font is vendored at `ui/InterVariable.woff2`, so the UI needs no network access.

Screenshots (light + dark for every page): `docs/ui-screenshots/<page>-<theme>.png`.

The Inference page's initial load reads no cameras, loads no weights and starts no
cloud service. **Show camera previews** can open physical cameras but never arms.
Prepare Modal and remote checks/probes are explicit billable operations; live probes
require their own active-read motor approval. Stop local execution and shutting down
the owned cloud service are separate actions. Readiness or confirmation cannot bypass
the [physical Modal performance gate](REMOTE_PERFORMANCE.md). Molmo's source mapping
is reviewed, but physical validation was not performed; base SmolVLA/pi05 physical
mapping and guided remote RTC remain unsupported. See [the Modal workflow](MODAL.md).

Record launches `yamkit record`, which installs the same operator processing used by
native teleop. Unprocessed raw LeRobot YAM leader actions are rejected. Native bilateral
feedback remains supported; recording and LeRobot teleoperation reject nonzero
`control.bilateral_kp`. See [operator parity](OPERATOR_PARITY.md) for button edges,
interruptible synchronization, measured holds and recorded action labels.

## Camera previews during sessions

The follower's existing observation path hands the acquired image to the optional preview
publisher after assigning it to the observation. It performs no extra camera read. Demand and
rate checks run before the publisher makes a private image copy; copying selected frames also
protects against cameras or callers reusing an owning NumPy buffer. Observations, dataset images,
camera settings, and control behavior are unchanged. The observation thread does no encoding,
color conversion, socket/file I/O, or per-frame logging. Preview errors are contained.

One background worker converts RGB/BGR as needed and encodes JPEG once per selected frame.
Each camera has one replaceable pending image and one latest JPEG; there is at most one image
being encoded by the worker. Viewers share the JPEG and skip superseded frames. Connections,
send buffers, and socket waits are bounded, so a slow browser does not create an image queue.
Closing the publisher is bounded too. The benchmark and its scope are documented in
[`preview-benchmark.md`](preview-benchmark.md).

The UI supplies a random per-session token and session ID through `YAMKIT_PREVIEW_TOKEN` and
`YAMKIT_PREVIEW_SESSION`. The child binds its small MJPEG server to `127.0.0.1` on an OS-assigned
port and registers it with one versioned `@yamkit-preview/1` JSON stdout line containing the
session, camera owner, port, and camera names. The token is sent only in the authentication
header, never in the registration line, browser URL, or logs. SessionManager parses this protocol explicitly.
The parent accepts only its active session's validated loopback endpoint and known cameras;
it does not accept arbitrary URLs or follow redirects.

Camera ownership is a separate claim/release handshake from preview registration. Before
acquisition, the plugin sends a versioned `@yamkit-cameras/1` ownership claim on stdout and waits
up to 15 seconds for the parent's JSON stdin acknowledgement that direct captures have released
their devices. Each claim/acknowledgement identifies both the session and the camera owner.
Release is reported only after camera disconnection is confirmed, including termination of its
existing capture thread. A failed release retains the lease and prevents reconnect until cleanup
succeeds. This follows actual camera ownership, so a
command that does not connect cameras leaves direct previews available. Registration does not
grant ownership, and loss of preview transport does not release it. Confirmed process-group
death also clears ownership and registration after a crash. Rapid session changes reject
previous registrations and stop previous streams.

Tiles show **waiting** before their first published frame and **stale** when acquisition stops
producing fresh frames (currently after one second), or **unavailable** after a preview failure.
If the camera owner never registers a preview, waiting becomes unavailable after ten seconds.
Source sequence and age track the image handed to the publisher; they are not verified sensor
exposure timestamps. Reconnecting or replaying an old image does not make it fresh.
A stale last image can remain visible with its state label.
Failed or stale streams retry at two-second intervals. Healthy streams renew every thirty
seconds to recover silent MJPEG disconnects that a browser may not report as errors.

In pinned LeRobot 0.6.1, the reset interval runs the same `record_loop` with `dataset=None`.
That loop still calls `robot.get_observation()`, so previews keep updating during a normal
reset. `dataset.save_episode()` and `dataset.finalize()` can pause the observation loop;
the UI reports stale images during such pauses. It does not start a second camera reader to
hide them. LeRobot disconnects the robot before its optional Hub upload; the explicit camera
release restores direct previews even when the upload process is still running. A progress
message about saving or uploading alone is insufficient to release cameras.

## Supervised camera acceptance (manual only)

**Do not run this checklist automatically.** Teleoperation and recording can energise and move
the arms. A person at the rig must supervise the existing hardware controls and keep the work
area clear. No physical acceptance is included in the automated tests or synthetic benchmark.

1. **Idle:** Open Live/Record and confirm top, left_wrist, and right_wrist show the expected
   physical views. Open a second browser window and confirm both windows update.
2. **Teleop:** Start the existing Teleop action with supervision. Confirm previews match actual
   ownership: direct views continue if that command does not acquire cameras; a camera-owning
   follower session uses its acquired images. Confirm the existing engage/Stop behavior.
3. **Record:** Stop teleop, start a short recording with at least two episodes, and confirm all
   three tiles resume from the recording images without camera-busy errors. Confirm the saved
   dataset's resolution, fps, image colors, and camera keys remain correct. Close/reopen a
   viewer and use multiple windows; recording should continue.
4. **Reset:** Between episodes, move an object in each view and confirm previews update during
   the reset. During any long save/finalize pause, confirm age grows and the tile becomes stale;
   fresh frames should restore the live state when acquisition resumes.
5. **Upload:** If the operator has separately chosen a Hub upload, confirm direct previews return
   after camera disconnect while upload progress is still visible. Do not infer release from
   the start of saving or uploading. This step needs an explicitly authorised upload and may
   be recorded as unverified when only local storage is used.
6. **Idle:** Stop/end the session and confirm direct views reconnect. With the rig safely idle,
   repeat a short start/stop and browser reconnect to check for stale ownership or registrations.

Record observed behavior and any missing checks; synthetic results do not establish physical
camera compatibility, control-loop timing, real recording throughput, or upload performance.

## Historical preview-workstream validation

Before integration, on 2026-09-05, the preview workstream's `make test` passed
**151 tests in 34.74 seconds** with an isolated, offline
Hugging Face environment. One existing Starlette/httpx deprecation warning remained.
`make lint`, benchmark-script Ruff, and `git diff --check` passed.

A Chrome 152 smoke run through the actual frontend, SessionManager, and HTTP proxy passed
**14 checks with zero JavaScript exceptions**. Direct captures and the child camera sources
were synthetic; attempts to create a physical capture were trapped. It checked idle views,
dynamic ownership for a camera-owning teleop command, waiting/live states, stable connections,
browser reload, publisher failure without direct fallback, recording/reset frames, increasing
stale age during a pause, and direct previews during simulated upload only after lease release.
The three healthy MJPEG images retained their DOM nodes and URLs for 4.3 seconds even though
Chrome reported `complete=true`; that property is not treated as an ended stream.

The [synthetic benchmark](preview-benchmark.md) records handoff/loop percentiles, CPU, memory,
and drops for three 640×480 RGB sources at 30 Hz. These checks did not exercise physical arms,
USB cameras, real recording throughput, or a real Hub upload.
These historical counts are not the final integrated suite results. Follow the
[acceptance checklist](acceptance-test.md) for the current combined workflow.
