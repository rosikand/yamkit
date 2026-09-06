# Integration development and Lenovo validation — 2026-09-06

Development branch: `codex/yamkit-integration-validation`, based on the requested
`integration/yamkit-v1` revision `b2cf14df37489bccfecc635ecf42a2b170306c3d`.
Development and automated testing run in the Conductor cloud checkout. The hardware
host is `yam-lenovo`, accessed as `andre`, with checkout `/home/andre/rohan-new`.

## Fixes

- Rediscovery preserves Hub preferences, verified arm identities, calibration and pairs,
  including adapters without a serial and ambiguous teaching-handle replies.
- Structured Settings updates validate control values before writing the rig. Invalid
  speeds, rates, buttons and nonfinite values leave the rig and camera configuration intact.
- Native teleop duration and rate statistics begin after startup homing/engagement preparation.
- Native teleop freezes the final rate before return-home/close time, reports successful button
  transitions immediately, and logs successful session closure once while retaining cleanup retries.
- Recording Stop during acquisition/reset uses LeRobot's existing Stop events to save the
  current episode. Subsequent interrupts and interruptions during startup/saving/homing
  retain their normal behavior. Stop before any captured frame cancels without an empty save.
- Failed, empty or invalid recordings cannot trigger upload or deletion. Local storage and
  explicit Hub targets stay consistent; upload retries support `push-dataset --repo-id`.
- UI history retains Stop intent after the process exits. Dataset navigation releases videos,
  playback timers and chart observers, and ignores responses for departed pages.
- Direct camera previews release disconnected viewers even when acquisition stalls. A synthetic
  stalled-camera HTTP regression closes 45 successive viewers while preserving another viewer
  and confirming that the camera and session APIs remain responsive.
- The SDK's default fail-fast CAN loop captures one complete pending command under its lock,
  then releases the lock while waiting for CAN replies. Opt-in automatic recovery retains the
  original full-lock behavior. This targets observation-read delays measured on the two-pair rig;
  the approved follow-up run reached 100 Hz with zero application-loop overruns.

No target-speed clamp, joint limit, firmware timeout, model mapping or qualification gate was
relaxed. The CAN command-lock patch is recorded in `third_party/i2rt.VERSION`.

## Software checks

The complete hardware-free suite after the CAN command-lock patch passed **1,176 tests**
in 157.63 seconds, with four existing
Starlette/fork deprecation warnings. `make lint`, Ruff on all three diagnostic scripts,
the offline lockfile check and `git diff --check` passed.

The standalone teleop profiler passed 14 dedicated hardware-free tests; those
tests plus the existing teleop suite passed **66 tests** in 41.14 seconds before the CAN patch.
The CAN patch then passed **35 vendor tests**, including five new concurrency, fault and
recovery cases. `make lint`, explicit Ruff checking of the profiler and compilation of the
modified SDK module passed. No hardware tests were run by the automated suite.

Actual Chrome passed **31 checks**, with zero JavaScript exceptions or attempted real
hardware/service calls. The browser harness additionally plays a three-camera episode from
its concatenated-video offset and a chart-only episode, then verifies resource release after
navigation.

A real two-step ACT training smoke used the committed `pick_red_cube_2demo_dummy` dataset,
CPU, batch size 2 and a small transformer configuration. Training and checkpoint saving
completed. The saved checkpoint produced fresh finite actions with 14 state/action dimensions,
three image inputs and four-step chunks in a hardware-free CPU check (271–360 ms per call).
This checkpoint is a software diagnostic and has no physical policy qualification.

The offline agent fixture completed without API calls. Detailed local logs and browser reports
are under `.context/validation/`; generated checkpoints remain under `outputs/`.

## Passive and camera-only Lenovo evidence

The initial measurements used the exact requested base revision before deployment:

| Check | Observed result |
|---|---|
| Environment | Python 3.12.14, LeRobot 0.6.1, CPU Torch 2.11.0; both YAM plugin families registered |
| Rig | Four configured arms, two leader/follower pairs; saved calibration retained |
| CAN | All four adapters up at 1 Mbps, zero reported errors |
| Passive traffic sample | No received/transmitted frames or errors during the three-second window |
| Register discovery | Both followers and both leaders, including both teaching handles, identified |
| Cameras | Top D435 and two wrist D405 cameras present on USB 3 |
| Concurrent LeRobot capture | Each camera delivered 90 distinct frames and acquisition timestamps at approximately 30 FPS, 640×480 RGB |
| Capture teardown | No errors; no surviving acquisition threads |

Camera reports and samples remain on the Lenovo under `.context/validation-20260906/`.
No rig configuration was rewritten. The separate `ctrl_pi` service was running; the passive
traffic sample establishes only that the buses were idle during that observation window.

## Deployed dashboard and local policies

The development branch was pushed and deployed to the Lenovo checkout. The dashboard runs as
an `andre` process at `http://100.98.214.86:8400`, bound to the Lenovo's Tailscale address. The
saved rig file's SHA-256 before and after deployment was unchanged:
`5eb387fedbfc60a59c27859016409fe2d27e5614b376857246457492f778eae8`.

Actual Chrome in Conductor verified all three physical camera previews at 640×480, existing
three-camera dataset episode playback, Settings navigation, and release of every camera when
viewers close. The catalog listed 14 datasets and eight model entries. There were no JavaScript
exceptions, non-GET requests, or operator sessions. Independent Tailscale connections were used
for HTTP: a shared SSH forwarding connection delayed requests under concurrent MJPEG traffic,
while the Lenovo-local API remained responsive in 1.5–2.8 ms with fresh camera frames.

Offline, camera-free and robot-free policy checks on the Lenovo used four CPU threads:

| Policy | Observed result | Qualification limit |
|---|---|---|
| Existing ACT two-demo dummy checkpoint | Finite 14-dimensional rig state/action contract, three image keys; fresh calls 438–488 ms | Dummy data establishes software execution, not task performance |
| Reviewed SmolVLA base | Three fresh finite 50×6 chunks; 2.835–2.938 s per call | Native fixture has no verified YAM mapping; inference exceeds its 1.667 s chunk horizon |

These checks do not authorize or qualify physical policy execution. Detailed policy logs are
`.context/validation-20260906/policy-check-act.txt` and `policy-check-smolvla.txt` on the Lenovo.

## Approved left-pair state read

After the operator approved this exact command, it ran once on deployed source `29787b1`:

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && yamkit read left_leader left_follower --rig configs/rig.yaml --duration 5 --hz 5'
```

The dashboard had no active session, and a three-second passive sample showed all four CAN
buses idle before connection. The command exited successfully after printing 25 finite state
samples per arm. The leader connected on `can2` and follower on `can3`; saved follower gripper
limits were reused with automatic calibration explicitly disabled. Leader joint values were
constant at the printed precision; the follower's largest printed change was 0.018 rad (about
1.03 degrees) at joint 4 between its first and second sample. Gripper values were 0.96–0.97
for the leader and 0.95 for the follower. Both leader buttons remained unpressed (`00`).

Both arms logged successful closure. No read process remained, all four CAN buses were idle
again during a three-second postflight sample, and reported RX/TX errors remained zero.
The saved rig file's SHA-256 was unchanged. The local log is
`.context/validation/left-pair-read.txt`; passive postflight evidence is on the Lenovo at
`.context/validation-20260906/left-pair-read-postflight.json`.

## Approved right-pair state read

After separate operator approval, this exact command ran once on deployed revision `b9a8c55`
(executable source unchanged from `29787b1`):

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && yamkit read right_leader right_follower --rig configs/rig.yaml --duration 5 --hz 5'
```

The dashboard and all four CAN buses were idle beforehand. The command exited successfully,
printing 25 finite samples per arm. The leader connected on `can1`, follower on `can0`, and
saved follower gripper calibration was reused without recalibration. The largest printed
follower joint change was 0.013 rad (about 0.74 degrees) at joint 4 between the first and second
sample; subsequent follower joint values were unchanged at printed precision. Leader joint
variation was at most 0.001 rad. Leader gripper values were 0.98–0.99, follower gripper remained
0.99, and both handle buttons remained unpressed (`00`).

Both arms logged successful closure. No read process remained, all CAN traffic was idle during
the three-second postflight sample, RX/TX errors stayed zero, and the rig checksum was unchanged.
The local command log is `.context/validation/right-pair-read.txt`; preflight and postflight
reports are on the Lenovo under `.context/validation-20260906/right-pair-read-*.json`.

## Approved left-pair teleop: engagement not yet verified

After operator approval, the following command ran once on deployed revision `f0c1753`
(executable source `29787b1`):

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && yamkit teleop --rig configs/rig.yaml --pair left_follower --duration 30 --no-home --bilateral-kp 0 --print-state'
```

The command exited successfully with 3,001 control ticks and zero overruns. All 60 printed
status samples were idle, with buttons `00`, leader gripper 0.96–0.97 and follower gripper 0.95.
The leader joints were unchanged except 0.001 rad variation at joint 5. The follower's largest
printed change was 0.016 rad at joint 4 between its first two samples, then it held its pose.
Both arms closed; postflight found no remaining arm process, no CAN traffic during three
seconds, zero RX/TX errors and the original rig checksum. Logs are
`.context/validation/left-teleop.txt` locally and `left-teleop-{preflight,postflight}.json` in
the Lenovo validation directory.

The operator reported pressing the top yellow button without activation or movement. This
run therefore establishes initial hold and bounded cleanup, **not successful teleop tracking**.
The old status logger samples only about twice per second, which can miss short button states.
Follow-up code logs successful button transitions directly. The original final 98.7 Hz summary
included close time despite approximately 100 Hz during acquisition; final rate timing now
excludes cleanup. These are diagnostic fixes, not an explanation for the missing input.

A subsequent motor-free diagnostic polled both teaching handles on their resolved interfaces
(`left_leader`: `can2`, `right_leader`: `can1`) for 75 seconds using only CAN request `0x50E`
with payload `ff02`. It did not construct a robot or write encoder configuration. Each handle
returned 3,067 raw reports without a missed request; all digital-input bytes were zero and
trigger counts varied only from 14–25 on the left and 4087–4098 on the right. The raw log is
`.context/validation-20260906/handle-inputs-01.jsonl` on the Lenovo, with summary
`.context/validation/handle-inputs-01.txt` locally. A subsequent coordinated held-button sample
established which configured handle receives the operator's yellow-button input.

## Motor-free yellow-button confirmation

After the operator reported holding the same top yellow button, sensor-only requests read both
handle encoder reports and raw GPIO on deployed source `781cfa3`. All 12 left-leader samples
on `can2` reported input byte `1`, confirmed independently by raw GPIO `1`. All 12 right-leader
samples on `can1` remained `0`. After the operator was told to release the button, all six
follow-up samples per handle reported `0` in both the encoder report and raw GPIO.

The SDK decodes input byte `1` as button 0 pressed and button 1 released, matching the saved
`control.engage_button: 0`. This confirms that the tested button's press and release reach the
configured left leader. No arm-name or button-index change is warranted. It does not establish
why the earlier powered run stayed idle or validate engagement under powered SDK polling.

These requests used only teaching-encoder report reads (`0x50E`, payload `0002`) and raw
ADC/GPIO reads (`0x50E`, payload `000600`); no robot was constructed and no motor or encoder
configuration was changed. The dashboard remained idle and the saved rig checksum unchanged.
Local logs are `.context/validation/handle-held-snapshot.txt` and
`handle-released-snapshot.txt`; matching JSON evidence is in the Lenovo validation directory.

## Approved left-pair teleop retry: engagement and tracking observed

After fresh approval, the same 30-second, left-pair, no-home, zero-bilateral-feedback command
ran once on deployed revision `c2a24b6` (executable source `781cfa3`). It completed 3,001 ticks
at 100.0 Hz with zero overruns. Of the 60 printed status samples, 45 were idle and 15 engaged.

The direct event log recorded button-0 engagement at Lenovo log time 23:35:59, with three-second
synchronization. Two status samples showed `btn=10`; later samples returned to `00` while the
pair remained engaged, confirming that releasing the button does not disengage tracking.
The initial maximum joint discrepancy fell from 0.081 to 0.019 rad. Subsequent manual leader
movement reached joint 4 positions of 0.870 rad on the leader and 0.820 rad on the follower,
demonstrating tracking beyond the captured synchronization pose. The largest sampled discrepancy
during movement was 0.253 rad. These twice-per-second samples do not measure continuous peak
error or latency. The operator confirmed that movement worked.

At 23:36:07 the normal duration limit ended the run, about eight seconds after engagement.
Shutdown disengaged into measured hold and closed both arms without homing. There was no
button-driven disengagement event or logged fault. Postflight found no remaining arm process,
idle CAN traffic for three seconds, zero RX/TX errors and the unchanged rig checksum. The local
log is `.context/validation/left-teleop-02.txt`; preflight/postflight JSON is in the Lenovo
validation directory. A separate receive-only encoder listener started too late to capture
traffic and provides no additional evidence about the active loop.

Gripper values remained near open (leader 0.96–0.97, follower 0.95–0.97). Deliberate trigger
travel and button-controlled disengagement/re-engagement still need a longer supervised window.

## Approved 60-second left-pair teleop: gripper and button disengagement

After separate approval, the following command ran once on deployed revision `14f831c`
(executable source `781cfa3`):

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && yamkit teleop --rig configs/rig.yaml --pair left_follower --duration 60 --no-home --bilateral-kp 0 --print-state'
```

The command completed 6,001 ticks at 100.0 Hz with zero overruns. The 119 paired status samples
comprised 29 initial idle, 83 engaged and seven final idle samples. Button 0 engaged at Lenovo
log time 23:42:09 and explicitly disengaged at 23:42:51, before the normal timed shutdown at
23:42:55. Both transitions were recorded directly, including the disengagement press that fell
between the twice-per-second button-state samples.

Synchronization reduced the maximum sampled joint discrepancy from 0.226 to 0.016 rad within
seven engaged samples, later settling near 0.012 rad. Manual motion exercised multiple joints;
the engaged median discrepancy was 0.058 rad and maximum 0.371 rad during wrist movement.
These are sampled position differences, not continuous peak-error or latency measurements.

The gripper partially closed and reopened with the trigger. Printed leader gripper values
ranged 0.64–0.97 and follower values 0.54–0.97; corresponding samples included 0.66/0.66,
0.78/0.77, 0.64/0.65, then 0.97/0.96 after reopening. These samples establish partial gripper
operation, not full travel. One transient 0.96/0.54 pair also limits timing inference from
the sparse printed snapshots. The operator reported that operation seemed correct.

After the second button press, the follower's joints varied at most 0.001 rad across the seven
idle samples. The leader also moved at most 0.001 rad, and the operator confirmed forgetting
to move it deliberately after disengagement. Stationary hold is verified; independence while
the disengaged leader moves remains an open check. No re-engagement was attempted.

Both arms closed successfully. Postflight found no arm process, no CAN traffic during three
seconds, zero RX/TX errors, unchanged counters on the unused right pair and the unchanged rig
checksum. The local log is `.context/validation/left-teleop-03.txt`; matching preflight/postflight
JSON is in the Lenovo validation directory.

## Approved 90-second two-pair teleop: left disengaged hold and timing limitation

After fresh approval, this command ran once on deployed revision `92816cf`
(executable source `781cfa3`):

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && yamkit teleop --rig configs/rig.yaml --pair left_follower --pair right_follower --duration 90 --no-home --bilateral-kp 0 --print-state'
```

All four arms connected. The left pair engaged through button 0 at Lenovo log time 23:51:00
and disengaged through the button at 23:51:18, ahead of normal timed cleanup at 23:51:36–37.
There were 178 status samples: 108 initially idle, 34 left-engaged and 36 finally idle.
After disengagement, the left leader's joints 4, 5 and 6 spanned 0.204, 0.312 and 0.409 rad,
respectively, while the left follower's printed joints remained unchanged at 0.001 rad
precision. This verifies that moving the disengaged left leader does not move its follower.
The operator confirmed the behavior was correct. During engagement, sampled joint discrepancy
had a median of 0.0415 rad, a maximum of 0.227 rad and settled near 0.009–0.011 rad.

The right pair remained idle throughout. This run does not establish right-pair tracking or
simultaneous tracking of both pairs.

The loop completed 8,638 ticks at **96.0 Hz with 2,082 overruns**, compared with 100.0 Hz and
zero overruns in the preceding single-pair tests. The reduced rate was already present while
both pairs were idle. These overruns count late application-loop deadlines; they do not by
themselves establish missed motor firmware deadlines or quantify maximum latency. The two-pair
timing requirement remains unresolved despite successful left-pair operation.

Source inspection found that SDK observation reads can wait on a state lock held across motor
updates, which can in turn wait for synchronous CAN request/reply work. This is a candidate
source of delay, not a measured cause. No lock, state-validation, speed-clamp or firmware-timeout
change was made. Follow-up profiling must measure the delays before changing control behavior.

`scripts/profile_teleop.py` forwards arguments after `--` to the existing native teleop CLI.
It measures control-thread SDK observation/handle reads, each control step and the status
callback, and captures the SDK's existing periodic timing reports without verbose console
logging. Sample buffers are bounded; counts, means and maxima cover the whole run, while
percentiles cover the most recent 8,192 calls per measurement. The harness adds measurement
overhead, so its results must be interpreted accordingly. It writes a new, repository-local
JSON report only after the normal CLI cleanup confirms every tracked arm closed. Help and
argument rejection do not connect hardware. Diagnostic failures retain the normal movement
error and cleanup behavior. A powered invocation still requires separate approval.

The command exited successfully and all four arms logged closure. Postflight found no teleop
process, an idle dashboard, no CAN traffic during three seconds, zero RX/TX errors and the
unchanged rig checksum. The local log is `.context/validation/bimanual-teleop-01.txt`; matching
preflight/postflight JSON is in the Lenovo validation directory.

## Approved profiled two-pair run: right control and simultaneous tracking

After approval, the following command ran once on deployed revision `494ddc1` (production
control source remained `781cfa3`):

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && python scripts/profile_teleop.py --output .context/validation-20260906/bimanual-profile-01.json -- --rig configs/rig.yaml --pair left_follower --pair right_follower --duration 90 --no-home --bilateral-kp 0 --print-state'
```

The right pair engaged at Lenovo log time 00:10:15, disengaged through button 0 at 00:10:28,
and re-engaged at 00:10:55. The left pair engaged at 00:10:57; both stayed engaged until normal
timed shutdown at 00:11:06. The 179 status snapshots comprised 79 both-idle, 26 right-only,
52 both-idle, four right-only and 18 both-engaged samples.

During the 52-sample disengaged interval, the right leader's joints 3 and 4 spanned 0.193 and
0.067 rad while follower joints varied at most 0.001 rad. The left follower remained stationary
during right-only operation. Right gripper values reached 0.02/0.03 leader/follower in two
samples and reopened near 0.99, demonstrating nearly full normalized travel, without proving
mechanical endpoints or loaded grasp performance.

Both pairs tracked beyond synchronization. One simultaneous sample showed left joint 2 at
0.740/0.728 rad leader/follower and right joint 2 at 0.703/0.701 rad. Maximum sampled discrepancies,
including synchronization, were 0.401 rad left and 0.287 rad right. The operator confirmed the
run seemed to work correctly. These approximately 2 Hz snapshots do not measure continuous
peak error or response latency.

The run completed 8,684 ticks at **96.5 Hz with 1,825 overruns**. The profiler reported no
diagnostic or CLI error and confirmed every arm closed before writing its JSON.

| Profiled operation | Mean | Recent p95 | Recent p99 | Whole-run maximum |
|---|---:|---:|---:|---:|
| Control step | 5.566 ms | 11.936 ms | 13.213 ms | 17.413 ms |
| Status callback | 0.043 ms | 0.002 ms | 1.989 ms | 3.679 ms |
| Left-leader observation read | 0.577 ms | 2.057 ms | 2.493 ms | 4.548 ms |
| Left-follower observation read | 0.697 ms | 2.424 ms | 2.780 ms | 6.399 ms |
| Right-leader observation read | 0.517 ms | 1.916 ms | 2.214 ms | 4.657 ms |
| Right-follower observation read | 0.513 ms | 2.201 ms | 2.706 ms | 5.894 ms |

Percentiles use the most recent 8,192 calls per measurement; means and maxima cover all calls.
Summing observation and handle-read totals gives 4.940 ms per control step, about 88.8% of
the measured step mean. These nested measurements are not additional time on top of the step.
The handle getters alone contribute about 0.014 ms per step. Status printing therefore does
not explain most of the measured delay.

SDK reports show leader CAN loops near 195–200 Hz, follower CAN loops near 250–254 Hz, and
gravity loops above 249 Hz. The largest CAN period in the reported 30-second windows was
9.739 ms. These reports do not establish a whole-run worst-case motor deadline or validate
the firmware timeout. The application still misses its requested 100 Hz timing.

Postflight again found no arm process, an idle dashboard, zero CAN traffic during three seconds,
zero RX/TX errors and the unchanged rig checksum. The command log, timing JSON and postflight
JSON are saved locally under `.context/validation/bimanual-profile-01*`; original JSON files
remain under `.context/validation-20260906/` on the Lenovo.

## Timing correction prepared after profiling

The CAN worker previously held `command_lock` across each synchronous scan. A gravity-update
thread posting its next command could therefore wait for the scan while holding the robot's
state lock, which also blocked application observation reads. The patch limits the command lock
to capturing a complete command list in the default fail-fast path. Producers already replace
the entire list with new entries, so a scan uses one consistent list and the next scan picks up
the latest pending list. Opt-in automatic recovery retains its previous lock scope throughout
the scan and retries, including exclusion of new targets while motors are being re-enabled.

Feedback returned by `set_commands` remains the latest completed CAN scan. It may now return
the preceding completed scan while another is in flight; it is not an acknowledgment of the
just-posted target. Motor error checks, state publication, the robot state lock, command
validation and all speed/timeout limits are unchanged. This removes a measured source of lock
coupling. The following approved profile measures the resulting hardware behavior.

## Approved timing verification: both pairs at 100 Hz

After fresh approval, this command ran once on deployed revision `0920017`:

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && python scripts/profile_teleop.py --output .context/validation-20260906/bimanual-profile-02.json -- --rig configs/rig.yaml --pair left_follower --pair right_follower --duration 90 --no-home --bilateral-kp 0 --print-state'
```

The session completed **9,001 ticks at 100.0 Hz with zero overruns**. Right button engagement
occurred at Lenovo log time 00:34:45 and left at 00:34:48. Both explicitly disengaged through
their buttons at 00:35:15, then re-engaged right at 00:35:49 and left at 00:35:51. Both remained
engaged until normal timed release at 00:36:00; all four arms closed by 00:36:01.

The 179 snapshots comprised 31 both-idle, six right-only, 52 both-engaged, 68 both-idle,
four right-only and 18 both-engaged samples. During the disengaged interval, right leader
joint 1 spanned 0.336 rad while follower joint 1 remained unchanged; other follower joints
varied by at most 0.003 rad. During the earlier right-only phase, left leader joint 5 spanned
0.226 rad while its disengaged follower varied at most 0.001 rad. Engaged median sampled
discrepancies were 0.0315 rad left and 0.039 rad right; maxima including synchronization were
0.223 and 0.467 rad. The operator confirmed correct operation after being asked about tracking,
disengaged hold, re-engagement and grippers.

Left gripper samples remained 0.96–0.97 on both leader and follower, so full left-gripper travel
is still not demonstrated by the log. Right values covered 0.80–1.00 leader and 0.81–0.99
follower in this run; its earlier profile had demonstrated nearly full normalized travel.

| Application measurement | Before command-lock patch | After patch |
|---|---:|---:|
| Native loop | 96.47 Hz | 100.01 Hz |
| Overruns | 1,825 | 0 |
| Mean control step | 5.566 ms | 0.693 ms |
| Recent p95 control step | 11.936 ms | 1.338 ms |
| Recent p99 control step | 13.213 ms | 1.883 ms |
| Maximum control step | 17.413 ms | 4.414 ms |
| Aggregate SDK reads per tick, mean | 4.940 ms | 0.155 ms |

Mean step time fell 87.6% and SDK-read time per tick fell 96.9%, despite more engaged follower
reads in this run. Status callback mean remained near 0.042 ms; its maximum was 7.140 ms.
Percentiles still cover only the most recent 8,192 calls per measurement, and instrumentation
adds overhead. These measurements establish application scheduling for this supervised window,
not sensor freshness or end-to-end physical latency.

There is a background-work tradeoff to monitor under recording load: gravity loops increased
from approximately 250–404 Hz to 734–791 Hz, while CAN scan rates fell roughly 6–8% (leaders
approximately 182–189 Hz, followers 229–238 Hz in the full reporting windows). Logged CAN
intervals over 7 ms increased from 258 to 911; the largest reported interval was 18.296 ms,
with later 30-second windows peaking at 12.679 ms. No motor faults were reported. These
statistics do not establish a whole-run worst-case firmware deadline. The evidence supports
advancing to a bounded camera/recording workload test without another speculative SDK change.

The profiler reported no diagnostic or CLI error. Postflight found no arm process, an idle
dashboard, zero CAN traffic during three seconds, zero RX/TX errors and the unchanged rig
checksum. Logs, timing JSON and postflight JSON are saved as `.context/validation/bimanual-profile-02*`
locally; original JSON evidence remains in the Lenovo validation directory.

## Remaining physical acceptance

Both pairs now have successful connection, state acquisition and orderly cleanup evidence.
The left pair additionally has confirmed button engagement/disengagement, synchronization,
manual tracking, partial gripper operation, timed release and follower hold while its disengaged
leader moves. The right pair now has the same button/tracking/hold evidence plus re-engagement
and nearly full normalized gripper travel. Simultaneous tracking of both pairs is verified.
Both pairs have now re-engaged successfully, and the 100 Hz application target passed the
90-second test. Remaining checks include full left-gripper travel, recording under camera and
encoding load, homing and physical policy execution.
Printed samples do not measure sensor freshness or the firmware timeout.

Each additional powered test requires approval of its exact command and effects first. The
next proposed test records both pairs with all three configured cameras, saves locally and
uses a separate rig copy with both home speeds set to zero. The original rig remains unchanged.
The wrapper dry-run, upstream `RecordConfig` parsing and shared operator-processor validation
passed on the Lenovo without constructing runtime devices. The new dataset/PID paths were
absent and 805 GiB of disk space was available. The proposed sequence includes one SIGINT
during the second episode to check partial-episode saving. That first Stop saves and encodes
before disconnecting, so arms may hold their last command during finalization. No recording
has yet been run. Preparation evidence is `record-stop-01-preparation.json` in the Lenovo
validation directory; the temporary rig checksum is
`7482a97ff7517ef4126715d36ed18addcdb69c709633e6a6522cf51e6c40b35a`.
During future teleop, keep the top button released through startup: a held button at the first
tick counts as engagement. Existing live-LLM freshness and remote policy qualification
restrictions remain in effect; see [the acceptance checklist](acceptance-test.md).
