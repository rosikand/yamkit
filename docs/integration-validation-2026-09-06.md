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

The complete hardware-free suite passed **1,154 tests** in 149.53 seconds, with four existing
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

## Physical acceptance still requires approval

No command that enables an arm motor has been run for this development task. Camera and
software results do not establish physical teleop, recording, homing or policy acceptance.
Each powered test requires the operator's approval of its exact command and effects first.

The next stage is a short gravity-compensation state read, followed by separately approved
operator and recording tests. Existing live-LLM freshness and remote-policy qualification
restrictions remain in effect; see [the acceptance checklist](acceptance-test.md).
