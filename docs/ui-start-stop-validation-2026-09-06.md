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

Physical acceptance of these changed defaults requires a fresh supervised test.
Earlier manual-button hardware tests do not qualify this new dashboard flow.

After deployment the idle dashboard must be restarted, then the browser refreshed:
`git pull` alone leaves an already running Python backend on its old imported code.
