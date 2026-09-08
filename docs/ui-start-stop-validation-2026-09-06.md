# Dashboard Start/Stop correction

The operator reported that ordinary teleop moved, its buttons/status were confusing,
and recording never moved the followers. This follow-up starts from integration
validation revision `f3fc0a3` on `codex/yamkit-integration-validation`.

## Reproduced causes

- Recording promised automatic tracking but initialized every follower disengaged.
  Its API/CLI had no automatic engagement option. The Lenovo's last user recording,
  `movejacketsmokev1`, ran for ten seconds, saved 299 frames and exited normally,
  with no engagement event in its log. Completing a file was insufficient evidence
  that this workflow worked as presented to a new operator.
- The page inferred readiness from engagement/rate samples during synchronization,
  and retained old engagement/progress during homing and after process exit.
- Replacing camera tiles did not cancel detached MJPEG requests. Real Chrome
  reproduced two direct streams per camera after a handoff, exhausting its six
  HTTP connections to the dashboard and queuing Stop and status requests. Clearing
  the image sources immediately unblocked the queued Stop.
- First Stop during episode saving signaled the encoder process group and could
  discard that episode. The old warning described the problem but did not provide
  the requested simple Stop-and-return workflow.

## Resulting behavior

Dashboard Start Teleop and Start Recording both request automatic engagement.
After connection and configured startup homing, followers synchronize to the leaders.
Readiness is reported only after every pair completes synchronization and accepts its
command. No handle-button press is needed to start. Handle buttons can still pause
individual followers; paused pairs stay paused across resets and subsequent episodes.
CLI manual defaults remain available; `--auto-engage` opts into automatic operation.

Park Arms and the dashboard auto-engage checkbox are removed. Start remains visible
but disabled while starting/running/stopping. Stop updates the state immediately;
configured homing and cleanup finish before the page restores idle controls. Old live
progress clears; output and recording form values remain available.

First dashboard Stop during acquisition or saving tells the registered recorder to
finish its current episode and return home. It sends SIGUSR1 only to the verified
recorder PID, with session, process-group and start-time checks. Encoders receive no
signal. A further Stop uses SIGINT to interrupt, including if both signals become
pending together. Incomplete startup keeps its previous immediate cancellation and
release behavior. Existing LeRobot loops, dataset formats, speed clamps and the
400 ms motor firmware timeout remain unchanged.

Camera sources are explicitly removed before tile replacement, navigation or hiding
previews. Completed status requests remain usable while later requests are pending,
and an older response cannot overwrite a newer applied state.

## Validation

`make test`: **1,237 passed**, four existing deprecation warnings. `make lint` and
`git diff --check` passed. **55 Chrome fixture checks passed**, with zero JavaScript
exceptions or real hardware/service calls. The reset-duration input now also keeps
an explicitly entered zero instead of silently substituting ten seconds.

- Actual pinned LeRobot record/reset/record loops with fake single/bimanual rigs:
  automatic tracking, gripper commands, held/released startup buttons, persistent
  pauses, speed bounds and labels matching commands actually sent.
- Actual pinned recording lifecycle with fake rigs: Stop during acquisition, reset
  and saving; first-frame cancellation; no extra episode after Stop; home/release
  of all four arms; immediate second Stop; startup and failure cleanup.
- Real child/encoder subprocess test: targeted Stop reaches the recorder child and
  leaves its encoder grandchild running until save completion.
- Chrome fixture tests: Start/boot/sync/ready/Stop/home/idle, failed and pending
  requests, camera handoffs, one direct stream per camera, stream cleanup on
  navigation, playback and settings. No physical hardware or external model/service
  calls are made by these tests.

The hardware session below qualifies the new automatic engagement and dashboard
lifecycle. Deliberate operator movement and full gripper travel still require
confirmation; earlier manual-button tests do not qualify those changed defaults.

After deployment the idle dashboard must be restarted, then the browser refreshed:
`git pull` alone leaves an already running Python backend on its old imported code.

## Lenovo automatic recording test — 2026-09-08 UTC

Conductor connected directly to `andre@yam-lenovo` over Tailscale. No Mac was used.
The user requested deployment of the fixed branch; the clean Lenovo checkout was
switched from `andre-dev` to `codex/yamkit-integration-validation` at `8a3ddf8`,
preserving `andre-dev` at `31793da`. The idle dashboard was restarted so its imported
backend and the recording child used the same source. Rig configuration and saved
calibration were unchanged.

The previously approved dashboard Start ran once for `ui_autostart_01`: one episode,
180-second limit, 30 FPS, zero reset, local storage. Both pairs logged automatic
engagement and then readiness after startup homing and synchronization. The first
dashboard Stop was sent during acquisition after approximately 103 seconds. Saving
took about 55 seconds, during which followers held their last commands. Normal
homing and closing followed for all four arms; the process exited 0. No second Stop
was needed. No additional powered test was run.

- Dataset validation passed: 3,085 frames (102.83 seconds), finite 14-dimensional
  state/action vectors, consistent indices and timestamps, and three complete
  640×480 videos with matching frame counts. First/middle/last LeRobot samples read
  successfully using PyAV.
- Real Chrome observed recorder previews, Saving, Returning home, Closing and Idle.
  Both Start buttons were restored and Stop was hidden. Ten checks passed with
  244 GET requests, no POST requests from the observer and no JavaScript errors.
- After cleanup the recorder was gone, camera ownership was released, all three
  camera clients/captures were closed, and a three-second passive CAN sample showed
  no traffic or new errors. The rig file retained its pre-test checksum.

This was a lifecycle and recording-integrity result, not full physical acceptance.
Joint spans were small (largest measured spans: left 0.121 rad, right 0.084 rad),
and measured grippers stayed near open (left 0.941–0.996, right 0.980–0.995).
These data do not establish deliberate leader movement or squeeze/release coverage.
Operator confirmation of tracking, grippers and physical home arrival was still
pending when this report was written. Saving latency also remains an improvement
opportunity; this test did not change the existing encoder CPU budget.

Evidence is retained under Lenovo `.context/validation-20260908/`, with a cloud
archive at `.context/validation/lenovo-autostart-01-artifacts.tar`. Browser evidence
is in `.context/validation/autostart-observer-pmrx9et6/` in the Conductor checkout.
