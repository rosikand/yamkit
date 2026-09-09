# HTTP rollout validation — 2026-09-09

## Ten-second attempt and startup admission fix

Run **`20260908-222613-rollout-6a7f7a12`** failed after eight policy actions,
about one second into the requested ten-second phase. Its first live RPC took
671 ms (206 ms of reported server work), leaving only 269 ms of valid action
horizon. The second prediction was still in flight when this queue emptied.
The underrun guard released the followers in 497 ms without attempting home.
All 31 observation triplets and the failed-run evidence were retained in the
[private archive](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/0e9c428cd6d35a4a8af0b675f53a372a7d47354b/runs/20260908-222613-rollout-6a7f7a12).
The task was not accomplished. These timings do not isolate the source of delay
outside the reported server work.

Remote startup now requires at least half the usable chunk horizon (0.5 seconds
for MolmoAct2) before the first policy dispatch, measured against both remaining
action count and original deadlines. A shorter first queue is discarded while
the worker obtains a fresh observation within the existing startup timeout.
No action timestamps are extended, and post-start underruns still release.
`metrics.json.startup_queue` records admission and discarded queues. This is
startup headroom, not a guarantee against future latency spikes.

A hardware-free reproduction of the slow first response failed after eight
actions without the reserve. With it, the same injected delays completed ten
seconds and 268 fake actions with no underrun, discarding the initial eight
targets. Physical validation of this change remains pending. Evidence:
`.context/validation/smooth-rollout/ten-second-failed/`.

## Physical command-shaping trial

Run **`20260908-220559-rollout-81cf2f4d`**, on commit `a480f42`, completed the
supervised five-second trial with **131 actions at 29.9 Hz**, zero underruns or
expired dispatches, and all **150 frames per camera** retained without export
errors. Warm inference p95 was 338 ms. Normal home completed in **2.112 seconds**,
followed by confirmed resource release. The operator said it “seemed ok.”

Sharp command reversals disappeared on both arms. The recorded preparation-time
command slopes respected 0.6 rad/s and 2 rad/s², and every shaped target matched
the actual postclamp target. These are command-space measurements, not guarantees
of measured robot dynamics. **The task failed:** visual review shows the lid
remaining on the table; no grasp was requested or observed. Raw predicted chunk
joins still reached 1.413 rad, so the policy problem remains.

The [complete private archive](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/e3d6f1a641cdab6aabdd125fd57568b6e28b6b33/runs/20260908-220559-rollout-81cf2f4d)
contains videos, original RGB frames and requested/shaped/sent command records.
Local analysis is in `.context/validation/smooth-rollout/first-physical-analysis.md`
and `.json`, with visual evidence in `first-physical/visual-outcome.md` and `.json`.

## Investigation preceding the shaping trial

The automatic-upload trial `20260908-195737-rollout-1c73c0d4` at commit
`1e4e580` completed five seconds, normal home and release, and uploaded its
[full private archive](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/63cedc21cb330d5d0426ea8e90e08508946d44e5/runs/20260908-195737-rollout-1c73c0d4).
Its 150 observations and three videos have no capture drops. Neither this
successful recording nor its zero exit code establishes manipulation success.
All predicted grippers remained above 0.98 open; no grasp closure was requested.

A hardware-free probe then repeated each of four exact saved observation triplets
and measured states five times, using the same pinned bfloat16/ten-step graph,
raw RGB, task and saved processors. These were labeled `saved_probe`, unseeded,
and never dispatched to a robot. Identical-input predictions differed by as much
as **1.094 radians**; both sampling variation and changed observations contribute
to the inconsistent plans. Observation 63 mostly predicted near-home targets,
while observations 36, 45 and 54 mostly predicted a substantial reach. Five draws
per input cannot isolate the causal contribution of images versus joint state.

Offline replay exactly reconstructed both earlier runs' 132 requested and sent
commands. Replacing the queued tail reduced observation age but did not reliably
reduce abrupt reversals across both recordings. A joint command shaper limited
to **0.6 rad/s and 2 rad/s²** removed immediate reversals over 0.1 rad/s in both
saved command sequences (74 and 67 events). It increased lag behind the model's
requested targets; neither replay nor those limits establish physical smoothness,
Cartesian clearance or task success.

The remote-rollout change applies joint shaping before the existing
hardware speed clamp. Original model targets are validated before shaping;
all action deadlines, session expiry, Stop and fault release remain enforced.
Gripper targets, normal home behavior, teleop and recording are unchanged.
`metrics.json.command_shaping` retains limits and timestamped original, shaped
and sent commands so filtering cannot be mistaken for a model prediction.
The supervised trial above tested this change after a fresh service qualification.

Visual review found no supported reason to swap cameras or invert joint/gripper
values. The objects changed sides between the two archived runs. The overhead
view currently clips the grippers at its horizontal edges; the reference sample
shows both clearly. This is a possible scene improvement, not a proven cause.
Camera exposure timestamps remain unavailable, and one selected wrist frame is
visibly blurred. The initial smoothing trial kept the scene and camera settings
fixed to isolate the execution change.

Local diagnostic evidence is under `.context/validation/smooth-rollout/`:
`consistency-results.json`, `consistency-findings.md`, `execution-replay.md`, and
`visual-audit.md`. No motor was activated for these diagnostics.

## Earlier trial: 30 fps capture and return home

**Run `20260908-184743-rollout-578b88cc` validated capture and normal return home;
the orange-lid task still failed.** The operator confirmed improved camera playback
and the expected home pose, while reporting abrupt, jittery policy motion.
This run used commit `97b78061d6b9ccc1c04f22f9524026c1104f3c0d`.

The 5.018-second policy phase acquired 150 observations and completed 132 bimanual
dispatches at 29.910 Hz after the first action. Dispatch interval p95 was 36.529 ms,
maximum 39.786 ms. There were zero queue underruns, expired dispatches or send
errors. The first dispatch arrived 605 ms after the policy phase started.
Fifteen RPCs completed: the first took 521 ms; the 14 warm requests measured
286 ms median and 299 ms p95. The separate pre-motion qualification measured
50 integrated warm requests at 318 ms p95. One in-flight request was invalidated
at normal shutdown; its closed-transport log message did not indicate a control fault.

Normal return home began 2 ms after policy Stop and completed in 3.306 seconds.
Release followed 475 ms later, for 3.783 seconds from policy Stop to release.
That total includes the commanded home trajectory and is not emergency Stop latency.
The operator confirmed that both arms reached the expected home pose.

All three videos decode to 150 frames over 5.017350 seconds, retaining original
observation presentation timestamps at nominal 30 fps. There were no missing
observations, capture drops, overflow, trace errors or export errors. Frame-copy
p95 was 1.332 ms, maximum 1.871 ms. The UI serves the videos with working partial
content requests. Original RGB PNGs remain under the trace directory. Video covers
the policy phase; live previews cover the home move. Full local evidence is under
`.context/validation/fps-home-validation/actual-run/` (git-ignored).

The helper's INFO phase messages were suppressed by import-time logging setup.
The follow-up fix initializes normal CLI logging before the LeRobot hook imports;
fresh hardware-free subprocess tests verify running, home, release and saving
phase markers while preserving inherited handlers and quiet vendor logging.
This logging fix was not part of the physical trial above.

Offline trace review found requested joint changes over 0.1 rad at 12 of 13
prediction boundaries, including a 1.503-rad left J3 change around 2.811 seconds.
All 132 requests match saved prediction indices and nominal deadlines; adjacent
predictions also disagree at closely matched future times. Sent joint steps remain
limited to 0.030 rad, but repeated direction reversals persist. This points toward
prediction continuity as an investigation target; it does not establish why the
predictions differ or justify changing safety clamps. At that point no smoothing change had
been deployed or physically tested.

The pinned Ai2 YAM example also uses different execution semantics: its
[client](https://github.com/allenai/molmoact2/blob/66b87e64efd99dfd103241418113955cf64dfa9c/examples/yam/molmoact_client.py#L81)
selects 25 actions, and its
[runner](https://github.com/allenai/molmoact2/blob/66b87e64efd99dfd103241418113955cf64dfa9c/examples/yam/launch_yaml_eval_molmoact.py#L187)
queries synchronously, executes the fresh prefix, and interpolates from measured
joints. Each interpolation command consumes an environment tick, extending target
timing. Our remote adapter continuously replans and appends the unexpired,
nonoverlapping tail. Matching 30 Hz and checkpoint chunk length does not reproduce
the reference's behavior. This difference merits investigation; copying its
interpolation would change the current action-freshness assumptions. RTC disabled
matches the released checkpoint default and is not itself an identified defect.
The next diagnostic is repeated inference on saved observations around the
contradictory chunks, with no robot connection, to separate variability for identical
inputs from changes caused by successive observations.

After completion, the UI was idle and both follower CAN transmit counters were
unchanged during a two-second read-only check. The retained GPU was stopped with
zero containers verified. No additional physical run was performed. The improved
latency and zero underruns do not establish smooth motion or successful manipulation.

## Previous trial: initial managed debug capture

**The managed UI/debug rollout completed; the orange-lid manipulation task still failed.**
The preceding Lenovo run is **`20260908-181138-rollout-deefc67a`** in the dashboard's
Runs list. Its five-second policy phase acquired 150 observations and completed
132 bimanual action dispatches, about 29.925 Hz from the first to last completed
dispatch. There were zero queue underruns or expired dispatches. Both followers
released normally about 473 ms after local Stop detection. The operator reported
that the UI and camera view seemed fine, but the lid was not placed in the container.

Debug capture saved 25 frames per camera at 5 fps without dropped samples,
overflow, trace errors or export errors. This sampling rate applies to saved
debug video; the configured policy cadence remained 30 Hz. All three videos,
joint plots, the HTML report and full metrics were imported into Runs. The
artifact summary is `TRACE_SAVED`, with resources released. Full local evidence
is in `.context/validation/debug-rollout-preparation/actual-run/` (git-ignored).

The physical trial completed 13 warm requests with a 399 ms round-trip p95;
its first request took 560 ms. These short-run measurements are distinct from
the fresh, uninstrumented qualification before motion: 50 integrated warm
requests with a 312 ms p95. One final in-flight request was invalidated during
normal shutdown; the corresponding closed-transport log message did not mark
the control loop failed. The retained GPU was stopped and zero containers were
verified. Another trial requires a freshly qualified session and explicit approval.

This validates one supervised managed control and evidence-capture trial. It
does not establish manipulation success, identify the cause of jitter or rule
out future latency stalls. The historical measurements below retain their
original scope.

The subsequent software update keeps the stock 30 Hz action cadence, records every
observation with its video presentation timestamp, and adds a bounded slow return
home after successful duration completion. Operator Stop, faults and expiry retain
release without an additional home move. The earlier physical run did not validate
these changes; the latest trial above now does. The model's pinned
[dataset metadata](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/blob/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c/meta/info.json)
specifies 30 fps. In the recorded trial, the policy requested approximately
1.3-radian joint changes at a chunk transition while the configured clamps limited
sent joint changes to 0.03 rad. That discrepancy is an investigation target, not
proof of a mapping, network or hardware fault. Some blur was already present in
original wrist-camera PNGs before video encoding.

The update passed 1,951 hardware-free tests and nine subtests, plus Ruff. Real
video encode/decode tests preserve 150 frames over five seconds, 300 over ten,
and deliberately irregular frame timestamps. A Lenovo benchmark using saved
images copied all 150 observations over five seconds with no drops and a
0.71 ms p95 callback time (1.29 ms maximum). It opened no cameras or motors and
made no inference requests; this measures capture-copy overhead, not complete
physical-loop performance. Return-home tests exercise Stop, expiry, timeout,
partial failure and late inference replies with fake arms.

## First physical CLI trial

**The first real VLA control rollout completed; the manipulation task failed.**
On 2026-09-09 at 00:23 UTC, the Lenovo ran the pinned MolmoAct2 policy with live
camera images and both follower arms. The five-second policy phase completed
132 bimanual action dispatches with zero queue underruns or expired dispatches.
Both followers released normally at the duration limit. The operator reported
that the left arm lifted itself, jittered briefly, and never approached the
orange lid. It did not pick up the lid or place it in the black container.

This establishes live observation → cloud inference → physical command flow.
It does not establish useful task behavior. The cause of the jitter is unknown:
we did not record measured/requested joint trajectories or video for this run.
The tool output was truncated, so the preserved leading metrics do not support
aggregate physical-run latency claims. Full current-session qualification,
partial console output, metric summary and operator feedback are in
`.context/validation/returned-approved-rollout/` (git-ignored). The partial log
was imported into the Lenovo dashboard as `20260909-002337-rollout-first-live`.

The retained H100 session passed a fresh, uninstrumented qualification before
motion: 50 integrated warm requests, 269 ms p95 and zero simulated underruns.
The physical run used app `ap-bOj3JjiSuNMS0mGxiWwiMQ`, instance
`a7b906fa-252a-406c-b8b2-a6b11834e644`, and commit `791804f`. The app subsequently
expired, was stopped with zero containers, and its Lenovo credential was retired.
A new supervised trial needs a freshly qualified retained session and approval.

An earlier physical startup homed both followers but failed camera acquisition
before policy execution. Commit `39696eb` corrected the ordering: acquire all
cameras before enabling either follower. That fix passed 1,804 software tests,
9 subtests and Ruff, plus 177 targeted tests on the Lenovo; the completed physical
trial confirmed the corrected camera-before-arm startup order.

The water jugs stay on the table for stability. The user accepts their possible
effect on policy performance. Conductor prepares the GPU, and the Lenovo owns
camera acquisition and robot control. The Mac is not involved.

See [the debugging workflow](VLA_DEBUGGING.md) for live previews, managed session
logs, saved video, and measured-versus-commanded joint traces. The historical
measurements below retain their original scopes and do not establish task success.

## Hardware-free check while the operator was away

A bounded client-trace diagnostic on a fresh H100 in `us-west4` completed 50 warm
native-fixture requests and 50 warm integrated requests. Integrated p95 was
292 ms, with 420 simulated actions, a minimum executing queue of 11 actions,
zero underruns, zero commands after Stop, and all fake arms released. Direct
p95 was 496 ms, with a 944 ms maximum. The slow direct requests waited for HTTP
response headers while measured server runtime stayed near 200 ms; their client
TCP counters showed no retransmissions, and the two slowest had no client GC.
The observations do not isolate the remaining delay to a specific network or
server stage.

Instrumentation changes the client, so this is diagnostic evidence only and
cannot qualify physical rollout. App `ap-zcDxVkLKKs2scJtJZ9RWh4` was stopped with
zero containers. The source snapshots, full report and compact summary are in
`.context/validation/coffee-trace-20260908/` (git-ignored). No arm or camera was
opened during this check. A separate uninstrumented session must qualify before
the next approved physical command.

## Execution and software checks

This path keeps the pinned MolmoAct2 checkpoint and saved processors, bfloat16
parameters, ten denoising steps, three raw 640 × 480 RGB images and 30 × 14 action
chunks at 30 Hz. Startup warms the actual task and image shape before connecting
arms. Real requests may replay only that graph; task changes, cache replacement,
recapture and eager fallback are rejected. Queue deadlines, speed clamps and the
400 ms motor firmware timeout are unchanged.

The HTTP credential is separate from the public service receipt and rig file.
The matching public receipt and private endpoint credential are transferred to
the Lenovo; Modal account credentials remain in Conductor.
See [preparation commands](MODAL.md#transport-and-qualification).

Validation: **1,796 software tests and 9 subtests passed**, plus Ruff, after tunnel
integration. The earlier HTTP integration also passed 289 targeted inference tests
on the Lenovo. A flaky UI test now waits for its harmless
child's ready marker before measuring Stop; production UI behavior did not change.

## H100 numerical fidelity

One NVIDIA H100 80 GB loaded the pinned model once. All eager references completed
before a second `ModelRuntime` wrapper enabled production graphs on the same policy
and saved processors. The graph mode used no diagnostic step/graph override.

| Comparison | Result |
|---|---|
| Ten seeded pairs with varied generated states and independent RGB images | Bit-identical raw and saved-processed chunks; maximum difference 0 |
| Generated input through the robot request schema | Bit-identical replay; no graph capture |
| Session reset and fresh-session replay | Same cache and warm proof; bit-identical output |
| Changed task and retired session | Rejected before prediction |
| Historical real-rig observation, preserved as `saved_probe` | Bit-identical raw, processed and unclipped robot-unit chunks; maximum difference 0 |

All comparisons include both gripper dimensions. The saved observation contains
the recorded bin/table scene without a red cube; it demonstrates data flow and
numerical equivalence, not task success. Diagnostic observers copy raw outputs,
so these comparisons are not inference latency measurements.

Fidelity app: `ap-JA2qnvTorgGzg9bXhqRcYO`; shutdown verified zero containers.
Fidelity build identity: `75181186f1ac45acb89f9b50e9ba50de647f979e4df3529604a38a9e677b1861`.
Full report: `.context/validation/production-graph-fidelity.json`.
The historical observation SHA-256 is
`3d6d2b3839308006501720f6c246892b4d96233dbb4486b8c9cdcbb65e0b8b2b`.

## Production HTTP qualification attempt

The bounded production session (`06cc5cd`, build
`d6dfbf64d6985746bf82981983afdf4e90bf3de5f9710a3167fde24240cf6d41`)
kept one H100 container in `us-west4`. The Lenovo sent generated raw images,
then exercised the real LeRobot worker with fake arms and cameras.

| Direct requests, 50 warm plus a separate first request | Measured time |
|---|---:|
| Warm round trip p50 / p95 / p99 | 372 / 400 / 539 ms |
| Maximum warm round trip | 578 ms |
| Model inference p50 / p95 | 166 / 170 ms |
| First prediction, including graph capture | 9.729 s |

**Qualification failed.** The integrated loop completed five model requests and
67 fake action dispatches before an underrun. One 525 ms request had ordinary
193 ms server work; another 568 ms request included 394 ms server work. These
are separate sources of delay. After the latter response, the next request had
393 ms of action validity remaining and was invalidated when that queue drained.
No stale action was replayed. All fake arms released, with zero commands after
Stop. Four completed integrated warm samples are insufficient for qualification.

App `ap-bvIV2We2ATD7h2gNFtji7F` was stopped with zero remaining containers and its
Lenovo endpoint credential removed. Full evidence is in
`.context/validation/production-http-qualification-attempt2.json`.
The latency distributions above establish neither physical readiness nor task success.

An earlier persistent min-zero-container deployment returned readiness, then
its container exited with success status and was replaced before the first
prediction completed. The control plane supplied no cause. A separate fresh
runtime successfully captured its graph on its first worker-thread prediction
in 9.429 s, without eager priming. Neither observation establishes the reason
for the earlier retirement. Both test apps were stopped with zero containers.

## Regional HTTP baseline

The Lenovo tested two authenticated CPU endpoints sequentially, with one CPU core
and 512 MiB per container, `min_containers=max_containers=1`, and a 180-second bound
per region including cleanup. Each payload had a separate first request and 50
warm requests over one persistent connection. Uploads were exactly 1,024 bytes or
2,764,856 seeded random bytes, without compression; responses were exactly 2,048 bytes.

| Requested compute/routing → observed compute | 1 KiB warm p95 | Raw upload warm p50 / p95 / maximum |
|---|---:|---:|
| `us-east` → `us-east5` | 138 ms | 159 / 175 / 196 ms |
| `us-west` → `westus3` | 136 ms | 163 / 175 / 184 ms |

Every payload hash matched, invalid credentials were rejected, and both apps were
stopped with zero containers. Local evidence: [east report](../.context/validation/regional-http-us-east-report.json)
and [west report](../.context/validation/regional-http-us-west-report.json), both git-ignored.

Both raw-upload p95 values were approximately 175 ms. This comparison establishes
no clear regional improvement. It includes HTTP ingress and server receive/hash
work, excludes policy framing, codecs and inference, and is not qualification.
These CPU placements differ from the tested H100's `us-west4`; requested routing
does not independently establish the actual network path. See
[Modal's region documentation](https://modal.com/docs/guide/region-selection).

## Identified model pause

A separate instrumented H100 run observed full Python garbage collections
inside the actual model call: **226 ms** at request 52 and **165 ms** at request
94. Those requests took 388 ms and 313 ms of model execution respectively;
warm median model execution was 156 ms. This identifies a concrete source of
server pauses. CPU throttling counters were unavailable in this container.

The observer changes allocations and response sizes, so its round trips are
not production qualification evidence. Its fake integrated loop also drained
and released all fake arms without commands after Stop. App
`ap-JxQ1Ukr7TH5BzdgUtqgtwb` was stopped with zero containers. Commit `80be17a` collects and freezes the loaded heap once at the end of the
dedicated HTTP graph container's startup. Automatic collection and its thresholds
remain unchanged for later request and graph-warm allocations. Guards prevent
this process-wide operation from running in a local policy/UI process; startup
fails if its preparation cannot complete. A container retains one model for its
lifetime, and retires the frozen heap when it exits. No freezing occurs on task
changes or session resets. The updated service passed 107 targeted software tests.
Its source build is
`872d2dfbee532896a76ced1f86f19d4772c3cb21611a6b674f1877611418d1e1`;
the following production test used no observer.

Compact public measurements and diagnostic helper hashes are in
[vla-http-rollout-2026-09-08.json](vla-http-rollout-2026-09-08.json).

## Production test after startup GC preparation

App `ap-zwczXiaShxkw8TGFTTeNvT` confirmed 481,163 frozen startup objects with
automatic GC enabled. All 50 completed warm model calls took **156–170 ms**.
Direct warm round trips were **357 / 684 / 789 ms p50/p95/p99**, with an 812 ms
maximum. Four slow requests had ordinary model timings but 489–619 ms outside
the measured runtime. Server idle intervals place the added delay before those
requests entered the runtime; they do not distinguish client upload, network
or Modal's request bridge.

The integrated loop completed 13 requests (12 warm, 381 ms median and 388 ms
p95). The next request exceeded its remaining 592 ms of action validity and was
cancelled without a response. **Qualification still failed.** All fake arms
released, with no expired action dispatched and zero commands after Stop.
The app was stopped with zero containers and its Lenovo credential removed.
Evidence: `.context/validation/production-http-qualification-attempt3.json`.

## Direct TLS tunnel comparison

One CPU container in `westus3` exposed the same authenticated application through
both a Modal ASGI Web Function and a direct TLS tunnel. The Lenovo alternated
requests between the two routes. Each uploaded exactly 2,764,856 raw bytes and
received a 14,336-byte binary-codec reply with 420 synthetic action values. Each
route had a separate first request and 50 warm requests per phase.

| Client phase and route | Warm p50 / p95 / maximum |
|---|---:|
| Normal GC, ASGI Web Function | 168 / 325 / 561 ms |
| Normal GC, direct tunnel | 52 / 73 / 84 ms |
| Diagnostic frozen heap, ASGI Web Function | 217 / 229 / 462 ms |
| Diagnostic frozen heap, direct tunnel | 62 / 74 / 90 ms |

The tunnel was faster in all 100 paired warm comparisons. Warm local uploads
completed within 4.7 ms and reply bodies were read within 0.43 ms. The slow ASGI
requests waited for response headers while the server reported long body-receive
intervals. No warm request had a major page fault or a client GC pause above
1.04 ms. This supports investigating the ASGI ingress/body-delivery path; it does
not attribute every earlier delay to the same cause.

Freezing was restricted to the dedicated diagnostic client and restored on exit;
these results do not justify changing client GC in production. The comparison
loaded model libraries but no model, opened no hardware and produced no
qualification record. All payload hashes matched and invalid credentials were
rejected. App `ap-pdPdTEFj1iFizS4vivntgJ` was stopped with zero containers.
Source snapshots and full measurements are under
`.context/validation/tunnel-attempt3/` (git-ignored).

Modal's [tunnel documentation](https://modal.com/docs/guide/tunnels) describes
direct TCP forwarding with TLS termination. Such traffic needs a retained
container and a bounded owner; its lifetime is not controlled by ordinary
Function request accounting. The inference integration keeps one container,
an explicit session expiry and independent shutdown verification. Full H100
qualification remains required before physical rollout.
The tested tunnel integration source build is
`7ebf3e8700e954a2dad79b006d2735f490b247a39bc2bd75a08dfe486582ae3d`.

## Passing H100 tunnel qualification

Commit `011b0f0` was pushed and deployed to the Lenovo with the same inference
build above. The Lenovo passed 109 targeted tests and its rig configuration hash
was unchanged. Conductor then retained one H100 container in `us-west4`, using
the production ten-step graph runtime without diagnostic execution overrides.

| Measurement | Result |
|---|---:|
| Direct requests | 51 completed, including 50 warm |
| Direct warm p50 / p95 / maximum | 272 / 284 / 456 ms |
| Model inference warm p50 / p95 | 170 / 176 ms |
| First direct prediction, including graph capture | 9.206 s |
| Integrated requests | 51 completed, including 50 warm |
| Integrated warm p50 / p95 / maximum | 268 / 281 / 284 ms |
| Simulated action dispatches | 419 |
| Minimum executing queue depth | 11 actions |
| Underruns / expired actions dispatched / commands after Stop | 0 / 0 / 0 |

The measured effective usable action horizon was 681 ms. Its required 20%
margin permits a warm RPC p95 up to 545 ms; the 284 ms direct p95 passed.
The integrated test requested Stop during an additional in-flight request;
that request was invalidated locally, all simulated arms released, and no later
action was dispatched. The simulated Stop-to-release interval was 400 ms.
All completed requests matched the same source, task, model, graph, container,
tunnel endpoint hash and session expiry. The task remained `pick up the red cube`.

This was a 15.4-second simulated control-loop test, not a physical manipulation
result or a guarantee about future network tails. At that checkpoint, real cameras,
live arm state and task success still needed supervised testing. The qualification-only owner
stopped app `ap-2pCN40LN9qgvVgKCgjzmVk`, verified zero containers and retired the
Lenovo endpoint credential. Full evidence is in
`.context/validation/production-http-qualification-attempt4.json`.
At that checkpoint, the conservative cloud compute estimate was **$5.37 of the
original $15 budget**, with all experiment containers stopped. Cached storage is retained, and the
estimate is not a billing invoice.

## Physical rollout boundary

Physical execution additionally requires current passing qualification from the
Lenovo, accepted mapping, and explicit supervised approval. Qualification binds
the exact task, model, source, wire format, endpoint, container and warmed graph.
Stopping the test service retires that container; a replacement must qualify again.
The qualification-only helper shuts down its app and retires the Lenovo endpoint
credential when it finishes, even if its measurements pass. A supervised session
must retain the same `app.run()` context and container (`min_containers=1`) through
qualification, inspection and rollout, within its explicit time and spending caps.
The managed UI/debug workflow acquires live observations during each separately
approved short rollout and preserves video, requested/sent targets and measured state.
Rollout includes configured follower startup homing before policy control and a
bounded slow return home after normal completion. Stop, faults and expiry release
without starting another home move. The motion approval must cover startup homing,
predicted joint/gripper movements and the final return home.
Required cameras are acquired before either follower is enabled. If a camera is
busy or fails to open, startup releases any acquired cameras without connecting
an arm. UI-managed sessions hand off camera ownership automatically; for a
standalone CLI rollout, close dashboard camera previews before starting.

A supervised orange-lid placement attempt on 2026-09-08 exposed the previous
startup order: a dashboard restarted during cloud preparation, and its previews
reclaimed the cameras. The followers homed before camera acquisition failed;
both were released and no policy actions executed. The camera-first startup
order prevents that failure from enabling the followers. Validation passed
1,804 software tests and 9 subtests, plus Ruff, including camera failure before
arm activation and camera release after an arm connection failure. Subsequent physical
CLI and managed UI trials completed their control loops; the manipulation task remains unsuccessful.

The existing `policy-probe` CLI uses SDK/eager execution; inspecting a saved capture
on this HTTP/graph service requires a matching HTTP `saved_probe` request. Saved
predictions remain diagnostic and are never replayed as physical commands.
