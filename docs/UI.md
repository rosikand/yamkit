# yamkit web UI (`yamkit ui`)

A small local web UI on top of the existing toolkit. Start it with:

```bash
yamkit ui                # http://127.0.0.1:8400
yamkit ui --port 9000 --rig configs/rig.yaml
```

Screenshots: [`docs/ui-screenshots/`](ui-screenshots/).

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
* **Camera previews**: the tiles poll `/api/cameras/<name>/frame` about ten times a second for the newest JPEG (read with OpenCV from the cameras in `configs/rig.yaml`); no long-lived connection, so a dropped stream or the browser's per-host connection limit can never freeze a tile. The MJPEG `/stream` endpoint is still there for other clients. While a record / teleoperate / rollout session owns the cameras, the follower plugin publishes each camera's newest frame as a JPEG into `outputs/ui/frames/` (`$YAMKIT_FRAMES_DIR`, a few Hz) and the same stream endpoints serve those, so the tiles keep showing what the robot sees.
* **UI-started sessions get no display**: `DISPLAY` / `WAYLAND_DISPLAY` are stripped from the child environment so LeRobot's recorder does not install its system-wide keyboard hook (Esc / arrows / n / r / q pressed in any window would otherwise end or skip an episode). Sessions are driven by the UI buttons only.
  They are released automatically while a record/teleoperate/rollout session runs, because the
  LeRobot child process needs exclusive access to the V4L2 devices.

## Pages

Navigation is a fixed left sidebar; the theme switcher (Light / Dark / System) lives at its foot
(preference in `localStorage`; `?theme=light|dark` overrides for a single load).

| page | contents |
|---|---|
| **Live** | camera tiles (top / left wrist / right wrist), follower joint+gripper state, CAN/camera/rig status, current mode + loop rate, read-only controls (`yamkit read` stream), Park arms (`yamkit rest`: every arm moves slowly home and is released) |
| **Record** | camera tiles, teleop pair status (engaged, tracking error, Hz), dataset name / task / episodes / durations form (recording rate fixed at 30 fps unless changed under "Advanced", capped by the slowest camera), Start Teleop / Start Recording / Park / Stop, episode + elapsed progress, live log. Start moves every arm home first; Stop sends Ctrl-C, the arms return home and are released; clicking Stop again during that move releases them immediately |
| **Datasets** | LeRobot v3 datasets under `data/datasets/` and, when signed in, the account's Hub datasets in the same table with a local / cloud / both tag, Upload and Download buttons and Hub links; per-episode detail page with synchronized videos and state/action small-multiple charts |
| **Inference** | policy runs launched from the UI (rollout + policy-check) with model, task, latency, status, termination reason; per-run detail page with log and replay videos (`outputs/ui/deployments/`) |
| **Models** | checkpoint directories under `outputs/` and the account's Hub models, tagged local / cloud / both; per-checkpoint detail page with file sizes and `config.json` / `train_config.json` contents; Upload buttons; Hub models get their own detail page and can be typed straight into the rollout form |
| **Settings** | Hugging Face sign-in (the token goes to `data/hf/token`, never through the rig) and the rig's hub settings (account, private, default destination); view/edit `configs/rig.yaml`: structured fields for the control knobs, read-only arm/camera tables (with the discovery notes: model, serial, USB port), and a raw-YAML editor. Every save is validated server-side first (parse → `RigConfig` → `validate()`), raw YAML is written verbatim (comments kept), saving is refused while a hardware session runs, and the camera feeds are reloaded from the saved file (no restart). The rig file holds hardware identifiers only — no credentials pass through the UI. |

## Code layout

```
src/yamkit/ui/
  server.py     FastAPI app: /api/* endpoints + serves the frontend
  sessions.py   SessionManager (one yamkit child process, log ring buffer, output parsers)
  camstream.py  MJPEG camera hub (lazy open, idle release, suspend while recording)
  catalog.py    read-only scanners: datasets / models / deployment records
ui/             standalone frontend (vanilla HTML/JS/CSS, no build step)
tests/test_ui.py  hardware-free tests (parsers, sessions with stub children, API via TestClient)
```

The frontend is a single-page app (`ui/app.js`) with hash routing; it polls `/api/session` (1 s)
and `/api/overview` (5 s). Chart colors are the validated categorical pair per theme
(light: state `#2a78d6` / action `#eb6834`; dark: `#3987e5` / `#d95926`). The Inter variable
font is vendored at `ui/InterVariable.woff2`, so the UI needs no network access.

Screenshots (light + dark for every page): `docs/ui-screenshots/<page>-<theme>.png`.
