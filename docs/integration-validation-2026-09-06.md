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
- Recording and LeRobot teleoperation report the same button transitions after successful
  sent-action acknowledgment, without repeated messages for held buttons or duplicate acknowledgments.
- Recording logs entry into upstream episode saving. The dashboard distinguishes acquisition,
  saving and finishing, labels saving's Stop as an interrupt, and clears the hold description
  before configured homing. Signal dispatch and interruption semantics are unchanged.
- Recording Stop during acquisition/reset uses LeRobot's existing Stop events to save the
  current episode. Subsequent interrupts and interruptions during startup/saving/homing
  retain their normal behavior. Stop before any captured frame cancels without an empty save.
- Failed, empty or invalid recordings cannot trigger upload or deletion. Local storage and
  explicit Hub targets stay consistent; upload retries support `push-dataset --repo-id`.
- UI history retains Stop intent after the process exits. Dataset navigation releases videos,
  playback timers and chart observers, and ignores responses for departed pages.
- Dataset playback handles canceled Play promises during seeking, pausing and navigation;
  current playback failures stop the chart timer and display the error. Obsolete failures
  cannot interrupt a newer Play request.
- Settings ignores late configuration/Hub responses after navigation or Reload, including
  failed Hub requests, instead of updating removed or replacement elements.
- Direct camera previews release disconnected viewers even when acquisition stalls. A synthetic
  stalled-camera HTTP regression closes 45 successive viewers while preserving another viewer
  and confirming that the camera and session APIs remain responsive.
- The SDK's default fail-fast CAN loop captures one complete pending command under its lock,
  then releases the lock while waiting for CAN replies. Opt-in automatic recovery retains the
  original full-lock behavior. This targets observation-read delays measured on the two-pair rig;
  the approved follow-up run reached 100 Hz with zero application-loop overruns.
- Recording defaults to LeRobot's `encoder_threads=1`, preserving explicit overrides. The
  measured AV1 parallelism setting trades longer warm-cache saves for lower CPU use.
  Subsequent powered recording showed less average control-thread slowdown while encoding;
  long CAN intervals still occurred, so this does not establish worst-case latency.

No target-speed clamp, joint limit, firmware timeout, model mapping or qualification gate was
relaxed. The CAN command-lock patch is recorded in `third_party/i2rt.VERSION`.

## Software checks

The complete hardware-free suite after the saving-phase follow-up passed
**1,197 tests** in 159.03 seconds, with four existing
Starlette/fork deprecation warnings. `make lint`, Ruff on all three diagnostic scripts,
the offline lockfile check and `git diff --check` passed.

The standalone teleop profiler passed 14 dedicated hardware-free tests; those
tests plus the existing teleop suite passed **66 tests** in 41.14 seconds before the CAN patch.
The CAN patch then passed **35 vendor tests**, including five new concurrency, fault and
recovery cases. `make lint`, explicit Ruff checking of the profiler and compilation of the
modified SDK module passed. No hardware tests were run by the automated suite.

The final finishing-phase refinement then passed **49 focused tests** in 17.30 seconds.
Actual Chrome passed **42 checks**, with zero JavaScript exceptions or attempted real
hardware/service calls. The browser harness additionally plays a three-camera episode from
its concatenated-video offset and a chart-only episode, then verifies resource release after
navigation. Five Settings regressions cover delayed configuration/Hub success and failure
after navigation, plus obsolete Hub status after Reload.
Three playback regressions cover canceled Play requests, visible playback failures and
obsolete failures after restarting playback.
Three recording-phase checks cover the actual saving banner, the finishing transition
before configured homing, and restoration of normal Stop wording during acquisition.

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

## Approved recording and Stop: both episodes and all camera videos saved

The following separately approved launch and one planned SIGINT ran once at deployed
revision `b892ebe` (control source `0920017`):

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && printf "%s\n" "$$" > .context/validation-20260906/record-stop-01.pid && exec yamkit record --rig .context/validation-20260906/record-no-home.yaml --arms left_follower --arms right_follower --name integration_record_stop_01 --task "supervised bimanual button gripper and Stop acceptance" --episodes 2 --episode-s 45 --reset-s 10 --fps 30 --to local'
tailscale ssh andre@yam-lenovo 'kill -INT "$(cat /home/andre/rohan-new/.context/validation-20260906/record-stop-01.pid)"'
```

The copied rig disabled both home speeds and preserved all other settings. Recorder PID
516989 was checked against its exact module and dataset path before signaling. Episode 0
started at Lenovo log time 01:00:21; the ten-second reset began at 01:01:06. After encoding,
episode 1 began at 01:01:43. One SIGINT during that episode triggered partial saving. Encoding
finished at 01:02:28, all three cameras disconnected, and all four arms closed by 01:02:29.
The process exited successfully. No exact SIGINT-receipt time appears in the child log, so
the test does not provide a precise Stop-to-release latency.

Local-only dataset `integration_record_stop_01` contains **2,154 frames across two episodes**:
1,343 frames in episode 0 and 811 in the stopped second episode, representing 44.77 and 27.03
seconds at the dataset's 30 FPS. These are frame-count durations, not measured acquisition
latencies. Metadata and Parquet frame counts match; per-episode indices/timestamps are
consistent and every state/action vector is finite with 14 dimensions and normalized grippers.

Each of the three AV1 videos decoded completely to 2,154 RGB frames at 640×480, with increasing
presentation timestamps. Sampling every 30th frame produced 72 distinct frame hashes per
camera. Offline LeRobot/PyAV access succeeded for the first frame, both sides of the episode
boundary and the final frame, returning all three image tensors at 3×480×640.

The action/state traces show movement on both sides, and the operator confirmed correct
tracking, gripper response and disengaged hold. Saved data does not contain raw leader/button
states, so it cannot independently establish button times or leader motion during hold.
Gripper travel in this dataset was limited: left actions 0.939–0.965 with states 0.947–0.965,
right actions 0.888–1.000 with states 0.890–0.995. Full left-gripper travel remains open.

Actual Chrome played and sought both episodes with all three videos and 14 charts, including
episode 1's 44.7667-second offset within the concatenated videos. Navigation released players,
timers and observers. The test also exposed a late Settings Hub response causing an uncaught
DOM error, now fixed with render-specific element ownership. A deployed retest then exposed
uncaught canceled Play promises during seeking/pausing, also fixed with playback-generation
checks and explicit rejection handling. The initial report and screenshots
are under `.context/validation/record-stop-playback-j9fvxkzb/`. Saved views mainly show floor
and clothing; camera placement for task datasets still needs operator confirmation.

After deployment at `e208ad6`, the same actual Chrome test passed both episodes, all
three videos, 14 charts, play/seek/resume and navigation cleanup with zero JavaScript
exceptions. All 59 browser requests were GETs; no live camera or operator session was
started. The final report/screenshots are under
`.context/validation/record-stop-playback-186s9jgl/`.

Nonfatal startup warnings concerned optional TorchCodec shared libraries (working PyAV
fallback), unavailable headless keyboard controls (the approved PID-specific SIGINT worked),
and SVT mapping preset 12 to 10. No system packages were installed. Postflight confirmed the
recorder was gone, the dashboard was idle, CAN traffic stayed zero for three seconds, error
counters were zero and the original rig checksum was unchanged. Command, postflight and full
dataset-validation reports are under `.context/validation/record-stop-01*` locally and the
Lenovo validation directory.

## Hardware-free encoder parallelism measurements

During acquisition, SDK CAN reports averaged approximately 160–166 Hz for leaders and
199–205 Hz for followers. Windows overlapping three concurrent AV1 encoders fell to
116–119 Hz and 137–138 Hz, with reported CAN intervals reaching 65.6–70.4 ms. Gravity
reports fell from roughly 642–711 Hz to 319–459 Hz. The windows overlap stages, so they
do not isolate exact per-stage latency or establish a motor fault.

After all arms/cameras closed, a benchmark decoded the same 300 consecutive frames per
camera (starting at frame 600) into repository-local PNGs, then used the pinned LeRobot
encoder with all three camera processes concurrently. Six trials compared automatic, 1 and
2 in forward/reverse order. Codec, CRF, GOP, pixel format and source frames stayed constant;
every output decoded fully. Imports, source extraction and output validation were excluded
from encode wall/CPU timing.

| `encoder_threads` | Wall time, two trials | Aggregate CPU seconds | Peak native threads, three processes |
|---|---:|---:|---:|
| automatic | 6.218 / 2.692 s | 29.815 / 21.649 | 318 |
| 1 | 4.175 / 4.134 s | 16.679 / 16.464 | 144 |
| 2 | 2.666 / 2.627 s | 20.831 / 20.703 | 237 |

Automatic's first trial was slower than its repeat; comparisons must retain that warm-up
variation. Setting 1 used approximately four busy CPU cores during encoding, versus eight
for warm automatic/2, but took about 1.5 seconds longer for this ten-second video sample.
The wrapper now defaults to 1 to leave control headroom and preserves user overrides.
SVT interprets this as its parallelism level, not a strict OS-thread cap. This benchmark
had no concurrent robot/camera acquisition workload. Later UI02 powered measurements
showed encoding-overlap CAN rates of 150–157 Hz on leaders and approximately 193 Hz on
followers, versus 116–119 Hz and 137–138 Hz in the earlier automatic-encoder run; the
largest reported CAN interval still reached 65.4 ms. These overlapping windows indicate
less average contention, not a controlled comparison or a recording-loop latency bound.
Script, report and local validation evidence are under `.context/validation/`;
remote inputs/outputs remain in `.context/validation-20260906/encoder-threads-benchmark/`.

## Approved dashboard recording: previews, saving and cleanup

The approved Start and one planned Stop used `/api/session/record` and
`/api/session/stop` on the temporary dashboard at port 8401, running the no-home rig
at revision `4611dda`. Recorder PID 529227 started episode 0 at Lenovo log time
01:38:58, reset at 01:39:58 and episode 1 at 01:40:47. The Stop response reported
32.8 seconds into episode 1; all cameras disconnected and all four arms closed by
01:41:44, with exit 0 and retained `stop_requested=true`. No homing occurred.

The operator subsequently reported being away and performing none of the requested
button, trigger or leader movements. This run therefore validates recording and
camera behavior with connected arms holding, not engagement or manual tracking.
No button-transition messages were observed, consistent with that report.

Dataset `integration_record_ui_02` contains 2,771 frames: 1,792 in the first episode
and 979 in the stopped second episode, nominally 59.73 and 32.63 seconds at 30 FPS.
Offline validation passed matching metadata, contiguous indices, timestamps, finite
14-dimensional state/actions, normalized grippers, complete decoding of all three
2,771-frame videos and LeRobot/PyAV sample reads across both episodes. The left
gripper stayed at approximately 0.955; full travel remains untested.

Actual Chrome displayed fresh 640×480 recorder-owned images from all three cameras
during both episodes and the active reset interval, with advancing source/preview
sequences and zero direct camera clients while the recorder owned the devices.
Previews became stale while acquisition paused for saving, then recovered. After
Stop, direct previews resumed; navigation and browser closure released all camera
clients. There were zero JavaScript exceptions and all observer requests were GETs.
The observer recorded the actual `encoder_threads=1` configuration before older
session log entries rolled out of the bounded log. The original dashboard on port
8400 had no camera clients at preflight and the later sampled check.

Reports and screenshots are under `.context/validation/record-ui-observer-65x81kl5/`;
Start/Stop responses and dataset validation are saved as `record-ui-02-*` locally.
The dataset and matching validation report remain in the Lenovo checkout. A fresh
three-second sample after closure found zero CAN traffic/errors; both rig checksums
were unchanged. The separately approved `integration_record_ui_03` retry is documented below.

## Dashboard retry: tracking passed, late Stop interrupted the second save

After fresh approval, `integration_record_ui_03` ran on revision `4f53e49` with
the same no-home dashboard settings. Both followers acknowledged engagement during
episode 0: right at 01:49:31 and left at 01:49:33, each with at least three seconds
of synchronization. Left disengaged at 01:49:53 and right at 01:50:03. The operator
confirmed tracking and disengaged hold, but reported skipping the triggers. There
were no second-episode engagement events, and full gripper travel remains unverified.

The first episode saved successfully with 1,792 frames at 30 FPS. Both sides show
substantial joint movement in the finite 14-dimensional state/action traces. All
three videos decoded completely to 1,792 RGB frames at 640×480; metadata/indices
and offline LeRobot/PyAV access passed. The left gripper stayed near 0.95–0.96 and
the right near 0.99–1.00. Saved data does not independently show disengaged leader
motion because it stores follower states and commanded actions.

The second episode began at 01:50:54. The Stop request reached the dashboard at
61.3 seconds into that phase, after the nominal episode timer had expired. It
interrupted `save_episode → compute_episode_stats → PIL decoding`, producing exit
−2 (`SIGINT`). This was a late test command, not a successful partial-episode save.
Only the first episode is present in finalized dataset metadata; remaining files
are retained for inspection. UI02 remains the successful dashboard partial-save test.

All cameras and all four arms closed by 01:51:57. Recorder-owned previews were
fresh during acquisition; direct previews resumed afterward and all viewers/camera
captures closed. Chrome reported no JavaScript errors. Postflight confirmed the
recorder was gone, both dashboards idle, zero CAN traffic/errors during three
seconds and both rig checksums unchanged. Artifacts are `record-ui-03-*` and
`record-ui-observer-ixzk2nki/` under the local validation directory; the readable
first-episode validator report is `record-ui-03-preserved-episode-validation.json`.

This late Stop motivated an observation-only UI improvement: explicit saving logs replace
elapsed-time estimates, and upstream's `Stop recording` transitions to a finishing phase
before finalization/disconnect/homing. During saving the Stop button explains that it
interrupts saving; the banner says that episode may be lost. Finishing no longer claims
the followers are holding. No signal handler or motor-control behavior changed.

## Approved automatic homing: all four arms returned and released

After approval of the exact command and its conditional PID-specific abort, this command
ran once at deployed revision `9126157`:

```bash
tailscale ssh andre@yam-lenovo 'cd /home/andre/rohan-new && source scripts/env.sh && printf "%s\n" "$$" > .context/validation-20260906/home-01.pid && exec yamkit rest --rig configs/rig.yaml'
```

All four arms connected using the original rig and saved follower gripper calibration.
Homing began concurrently at Lenovo log time 02:10:06, with zero joint targets and
0.25 rad/s configured target speeds. The logged largest displacements/minimum durations
were left leader 2.03 rad/8.1 s, left follower 1.97 rad/7.9 s, right leader 0.64 rad/2.6 s,
and right follower 0.61 rad/2.4 s. Leaders used compliant gains; follower gripper targets
retained their measured opening. All four arms closed by 02:10:16 and the command exited 0.
The conditional abort was not used. There were no logged warnings or errors.

The operator confirmed that the return and release looked correct. No final measured
joint positions were logged, so this is operator-confirmed physical home arrival rather
than a measured convergence bound. Postflight verified PID 538351 was gone, the dashboard
and cameras idle, zero CAN frames/errors during three seconds and the unchanged original
rig checksum. Local artifacts are `.context/validation/home-01*`; matching preparation
and postflight JSON remain under `.context/validation-20260906/` on the Lenovo.

## Remaining physical acceptance

Both pairs now have successful connection, state acquisition and orderly cleanup evidence.
The left pair additionally has confirmed button engagement/disengagement, synchronization,
manual tracking, partial gripper operation, timed release and follower hold while its disengaged
leader moves. The right pair now has the same button/tracking/hold evidence plus re-engagement
and nearly full normalized gripper travel. Simultaneous tracking of both pairs is verified.
Both pairs have now re-engaged successfully, and the 100 Hz application target passed the
90-second test. Recording, partial-episode Stop, saved videos and cleanup passed. Remaining
checks include full left-gripper travel and physical policy execution. The operator
chose to leave full trigger travel unverified. All-four-arm homing and release passed.
Engaged recording with reduced encoder parallelism, recorder-owned dashboard previews
and powered recording/encoding while followers hold have passed. UI03's second
episode was interrupted during saving and must not be counted as a successful save.
Printed samples do not measure sensor freshness or the firmware timeout.

Each additional powered test requires approval of its exact command and effects first.
No further motor command is approved or running. Physical policy execution remains subject
to the existing mapping and qualification requirements; software inference checks do not
qualify a manipulation policy. Task camera framing and real Hub transfers also have not
been end-to-end qualified in this validation.
The preceding dashboard recordings used a separate rig copy with both home speeds
set to zero. The original rig remains unchanged.
The first approved recording used the copied rig described above. First Stop saves and encodes
before disconnecting, so arms may hold their last command during finalization. Preparation
evidence is `record-stop-01-preparation.json` in the Lenovo validation directory; the temporary rig checksum is
`7482a97ff7517ef4126715d36ed18addcdb69c709633e6a6522cf51e6c40b35a`.
During future teleop, keep the top button released through startup: a held button at the first
tick counts as engagement. Existing live-LLM freshness and remote policy qualification
restrictions remain in effect; see [the acceptance checklist](acceptance-test.md).
