# Debugging a real VLA rollout

Use the Lenovo browser at **http://127.0.0.1:8400/#/inference**. The dashboard
runs in `/home/andre/rohan-new`; Conductor connects directly over Tailscale.
No Mac is required. Starting the dashboard or showing camera previews does not
enable motors.

## Live view and existing logs

Click **Show camera previews** on Inference. For the first physical trial, open
**Runs → `20260909-002337-rollout-first-live`**. That imported run contains the
preserved console log and a metric summary. It has no video or joint trajectory
recording, and its raw tool output was truncated. Its zero exit code means the
bounded control test completed; the orange-lid manipulation task failed.

The dashboard itself writes to `outputs/ui/dashboard.log`. UI-managed inference
runs appear under **Operation** while active and **Runs** after completion.
Their saved artifacts are under `outputs/ui/deployments/<run-id>/`.
Standalone SSH commands are not automatically attached to the dashboard: its
Stop button, session log and camera handoff apply only to its managed child.

## Prepare a managed debug trial

1. Have Conductor prepare and qualify a retained MolmoAct2 HTTP graph session.
   Its private endpoint credential and exact qualification must exist on the
   Lenovo. Cloud-account credentials stay in Conductor.
2. Click **Use owned MolmoAct2 session** in Inference. Select both followers and
   enter the exact qualified task: `pick up the orange lid and place it into the black circular container`.
   Use a five-second first debug trial and enable **Save debug video and joint traces**.
   The initial debug helper supports five or ten seconds and this exact task.
   Accept the physically checked mapping before checking the session.
3. Click **Check retained session (no hardware)** after setting all options.
   It reads local qualification and configuration;
   it neither opens hardware nor starts a GPU. Expired, changed-task, changed-source
   or mismatched sessions stay blocked. Changing any option requires another check.
4. Start only when supervising the cleared workspace and after the explicit
   motion confirmation. For an assistant-run trial, approve its exact command
   first. No yellow-button press is needed.

Camera acquisition precedes motor connection and startup homing. The duration
starts at the policy control phase, and includes the wait for its first chunk.
On Stop, a fault, expiry, or the duration limit, remote rollout invalidates queued
actions and releases the followers without a return-home move. Do not mistake
that expected release for a completed manipulation task. Keep the arm power
cutoff accessible during supervised tests.

## What the debug artifacts show

Debug capture observes the existing control path and preserves the model,
30 Hz cadence, action freshness checks, joint/gripper clamps, and firmware timeout.
It does not open a second camera or robot reader. Camera frames are copied into
bounded memory at up to 5 fps. Video encoding, JSON export and plotting happen
after hardware release, so saving may continue after motion has stopped.

- **Video:** top and both wrist cameras. These are sampled views, not 30 fps
  policy recordings. `frame_timestamps.json` records actual receipt times and gaps;
  camera exposure timestamps are unavailable.
- **Joint plots:** measured positions, requested policy targets and targets after
  the yamkit clamp, with chunk merges marked. A sent target is not a measurement
  of where the arm moved. SDK gripper force limiting can modify its target further.
- **Raw evidence:** `trace.json` contains observations, per-side sends/errors,
  chunks and merge events; `metrics.json` preserves the complete runtime result.
  `summary.json` includes frame/event limits, dropped samples and export errors.

Compare a jerk in the video to these traces. A changing requested target points
toward policy/chunk behavior; a persistent gap between sent and measured positions
points toward tracking, load, or hardware behavior. Neither observation alone
identifies a root cause. Missing samples, a failed partial send, and overflow
must remain visible when interpreting the plots.

The standalone capture helper defaults to a no-hardware plan:

```bash
.venv/bin/python scripts/trace_rollout.py --plan --duration 5
```

Its `--run` mode energizes and moves the arms and requires separate explicit
approval. Artifacts from standalone use remain under `.context/rollout-traces/`;
managed UI runs also copy their reviewed artifacts into the run's history.
