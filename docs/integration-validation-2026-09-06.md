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

No target-speed clamp, joint limit, firmware timeout, model mapping or qualification gate was
relaxed. No vendored SDK code changed.

## Software checks

The complete hardware-free suite passed **1,157 tests** in 151.31 seconds, with four existing
Starlette/fork deprecation warnings. `make lint`, Ruff on all three diagnostic scripts,
the offline lockfile check and `git diff --check` passed.

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

## Approved left-pair teleop: input issue unresolved

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
`.context/validation/handle-inputs-01.txt` locally. A coordinated held-button sample is still
needed to establish which handle/button is being pressed and whether its raw input changes.

## Remaining physical acceptance

Both pairs now have successful connection, state acquisition and orderly cleanup evidence.
Operator confirmation of physical identity/behavior, button transitions, gripper travel,
teleop tracking, recording, homing and physical policy execution remain unverified. The approved
left teleop session exercised idle hold only. Printed
samples do not measure sensor freshness or the firmware timeout.

Each additional powered test requires approval of its exact command and effects first. Resolve
the handle input first using a motor-free sensor read before requesting another powered run.
During future teleop, keep the top button released through startup: a held button at the first
tick counts as engagement. Existing live-LLM freshness and remote policy qualification
restrictions remain in effect; see [the acceptance checklist](acceptance-test.md).
