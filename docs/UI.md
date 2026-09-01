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
  `yamkit rollout`, `yamkit policy-check` — and parse its stdout for display. Stop sends the
  process group a SIGINT (identical to Ctrl-C in a terminal), escalating to SIGTERM/SIGKILL only
  if the child hangs. One session at a time.
* **Camera previews** are MJPEG streams read with OpenCV from the cameras in `configs/rig.yaml`.
  They are released automatically while a record/teleoperate/rollout session runs, because the
  LeRobot child process needs exclusive access to the V4L2 devices.

## Pages

| page | contents |
|---|---|
| **Live** | camera tiles (top / left wrist / right wrist), follower joint+gripper state, CAN/camera/rig status, current mode + loop rate, read-only controls (`yamkit read` stream) |
| **Record** | camera tiles, teleop pair status (engaged, tracking error, Hz), dataset name / task / episodes / durations form, Start Teleop / Start Recording / Stop, episode + elapsed progress, live log |
| **Datasets** | LeRobot v3 datasets under `data/datasets/` (episodes, frames, fps, tasks, cameras); per-episode viewer with synchronized videos and state/action small-multiple charts |
| **Deployments** | policy runs launched from the UI (rollout + policy-check), with model, task, latency, status, termination reason, log, and replay videos when present (`outputs/ui/deployments/`) |
| **Models** | checkpoint directories under `outputs/` with policy type, train steps, dataset, size |

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
and `/api/overview` (5 s). Chart colors are the validated dark-mode categorical pair
(state `#3987e5`, action `#d95926`).
