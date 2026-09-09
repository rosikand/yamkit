# HTTP rollout preparation — 2026-09-08

**The first live policy rollout is still pending.** Bounded TLS tunnel sessions
have passed Lenovo qualification using the real H100 model and LeRobot control
loop with simulated arms and cameras. Intermittent latency stalls subsequently
recurred and drained the simulated action queue. Historical passes do not qualify
a new container: the next supervised session needs a current passing qualification
for its retained container and the user's fresh GO after returning.

The most recent approved physical attempt homed both followers, then failed to
acquire the top camera before any policy actions ran. Both followers released.
Commit `39696eb` moves camera acquisition before arm activation; it is pushed and
deployed on the Lenovo. Validation passed 1,804 software tests and 9 subtests,
Ruff, and 177 targeted tests on the Lenovo. A separate camera-only check read all
three camera streams without connecting an arm; this does not establish physical
policy execution.

The rollout task is to place the orange lid into the black circular container.
The water jugs stay on the table for stability; the user accepts their possible
effect on policy performance and prioritizes the first bounded live rollout.

The HTTP transport and production `cuda_graph10` runtime are available on
`codex/yamkit-integration-validation`, implemented in `cfdcce9`; `06cc5cd` adds
one-container retention for bounded sessions. Conductor deploys the cloud GPU;
the Lenovo runs the LeRobot observation, queue and action loop.
The Mac is not involved. The historical measurements below distinguish synthetic
qualification from saved observations and the later physical startup attempt.

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

This is a 15.4-second simulated control-loop test, not a physical manipulation
result or a guarantee about future network tails. Real cameras, live arm state
and task success still need supervised testing. The qualification-only owner
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
The next physical step is an explicitly approved current observation capture,
followed by inspecting predictions before a separately approved short rollout.
Rollout includes configured follower startup homing before policy control; its
cleanup releases the followers without a return-home move. The motion approval
must cover startup homing as well as predicted joint and gripper movements.
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
arm activation and camera release after an arm connection failure. A subsequent physical
policy rollout is still required to validate control and task behavior.

The existing `policy-probe` CLI uses SDK/eager execution; inspecting a saved capture
on this HTTP/graph service requires a matching HTTP `saved_probe` request. Saved
predictions remain diagnostic and are never replayed as physical commands.
