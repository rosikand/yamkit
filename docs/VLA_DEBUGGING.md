# Debugging a real VLA rollout

Use the Lenovo browser at **http://127.0.0.1:8400/#/inference**. The dashboard
runs in `/home/andre/rohan-new`; Conductor connects directly over Tailscale.
No Mac is required. Starting the dashboard or showing camera previews does not
enable motors.

## Live view and existing logs

Click **Show camera previews** on Inference. Open the latest debug trial under
**Runs → `20260908-184743-rollout-578b88cc`** for all three videos, joint plots,
the HTML report, full metrics and console log. The operator confirmed that the
camera feed and return home now work; the orange-lid manipulation task failed,
and the arm motion remained abrupt and jittery.

That five-second trial acquired 150 observations and completed 132 bimanual
dispatches at about 29.9 Hz after the first action, with zero queue underruns.
The followers returned home concurrently in 3.306 seconds, then released about
475 ms later. Debug video contains 150 frames per camera at nominal 30 fps,
with original observation timestamps and no dropped samples, overflow or
trace/export errors. Its 14 warm physical requests had a 299 ms p95; the
separate qualification before motion measured 50 integrated warm requests
at 318 ms p95. The GPU was stopped
with zero remaining containers verified. A fresh session is needed for another trial.

The trial's console log omitted INFO phase messages because a LeRobot import
configured logging before the CLI. The helper now initializes logging before those
imports; fresh hardware-free subprocess tests verify all four phase messages.
That follow-up logging fix has not been used in a new physical run. The trial's
home metrics and the operator's confirmation establish the completed home move.

The preceding **`20260908-181138-rollout-deefc67a`** run retains its original
25-frame, 5 fps videos and release-only completion. Its 399 ms warm-request p95
was higher than the latest run's; that comparison does not establish the cause
of either run's jitter.

The earlier **`20260909-002337-rollout-first-live`** run retains its partial console
log and metric summary. It has no video or joint trajectory recording, and its
raw tool output was truncated. A zero exit code establishes control-loop completion,
not successful manipulation or the cause of any jitter.

The dashboard itself writes to `outputs/ui/dashboard.log`. UI-managed inference
runs appear under **Operation** while active and **Runs** after completion.
Their saved artifacts are under `outputs/ui/deployments/<run-id>/`.
Standalone SSH commands are not automatically attached to the dashboard: its
Stop button, session log and camera handoff apply only to its managed child.

## Prepare a managed debug trial

1. Have Conductor prepare and qualify a retained MolmoAct2 HTTP graph session.
   Its private endpoint credential and exact qualification must exist on the
   Lenovo. Cloud-account credentials stay in Conductor.
2. Open Inference and review its attached service, exact qualified task and both
   followers. A currently qualified task such as `put the red cube into the green bowl`
   is preserved verbatim through inference and recording; a changed task requires a
   matching qualification. Enable **Save recording locally**, or **Also upload to Hugging Face**
   to retain a local recording and upload a private copy. Captured policy durations are
   5, 10, 20, 30, 45, 60 and 90 seconds. Accept the physically checked mapping.
3. Check the displayed readiness after setting all options (or use **Recheck readiness**).
   It reads local qualification and configuration;
   it neither opens hardware nor starts a GPU. Expired, changed-task, changed-source
   or mismatched sessions stay blocked. The session must cover the selected duration
   plus 30 seconds for startup and 30 seconds for return home. Changing any option
   requires another check. Available memory must also cover the complete selected
   recording plus 512 MiB of headroom; 90 seconds requires 8,010,125,312 bytes. The
   check reads Linux host/cgroup limits and disk space without allocating frames.
4. Start only when supervising the cleared workspace and after the explicit
   motion confirmation. For an assistant-run trial, approve its exact command
   first. No yellow-button press is needed.

Camera acquisition precedes motor connection and startup homing. The duration
starts at the policy control phase, and includes the wait for its first chunk.
After normal completion, remote rollout invalidates queued policy actions, then
returns both followers home concurrently at their configured `control.home_speed`
(0.25 rad/s on this rig) before releasing them. Return home is limited to 30 seconds;
Stop interrupts it. A fault, expiry, failed startup or operator Stop releases the
followers without starting another home move. The dashboard distinguishes policy
execution, return home, release and saving. Keep the arm power cutoff accessible
during supervised tests. Completing this sequence does not establish task success.

## What the debug artifacts show

Debug capture observes the selected control path and preserves its model,
30 Hz cadence, action freshness/bounds checks and firmware timeout. Reference mode
keeps its literal interpolation without policy speed/acceleration shaping; capture
does not change controller semantics.
It does not open a second camera or robot reader. Every existing camera observation
is copied into bounded memory at the policy's nominal 30 Hz cadence. The helper
checks available memory before hardware startup. Video encoding, JSON export and
plotting happen after return home and hardware release, so saving may continue
after motion has stopped. The saved trial above exercised 30 fps capture and
normal-completion homing on the rig, and the operator confirmed both improvements.

- **Video:** top and both wrist cameras during the policy phase, at nominal 30 fps.
  Playback timestamps preserve observation timing and gaps rather than speeding
  through missing time. `video_timeline.json` maps video frames to observations;
  `frame_timestamps.json` retains receipt times. Camera exposure timestamps are
  unavailable. The model consumes the latest observation for each predicted action
  chunk; 30 Hz actions do not imply 30 cloud requests per second. Original RGB PNGs
  remain in the trace directory. Saved video does not include the return-home phase;
  the dashboard's live camera previews remain available during that phase.
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
.venv/bin/python scripts/trace_rollout.py --plan --duration 5 \
  --task 'put the red cube into the green bowl' --controller-mode reference
```

Its `--run` mode energizes and moves the arms and requires separate explicit
approval. Artifacts from standalone use remain under `.context/rollout-traces/`;
managed UI runs also copy their reviewed artifacts into the run's history. Click the
rollout to view recorded camera playback after saving finishes. No recording is
available for an earlier run that did not enable capture.
