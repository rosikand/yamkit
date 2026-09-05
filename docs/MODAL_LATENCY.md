# Modal latency investigation, 2026-09-05

The original raw transport cannot supply Molmo's one-second action horizon. A
100-warm-request baseline separates about 0.364 s of server work from 1.358 s of
median client/transport overhead. The first cached-handle experiment saves about
68 ms of dispatch work outside input upload, but does not improve the latency tail.
The optimized path reaches 0.645 / 0.695 / 0.707 s warm p50/p95/p99, but still
fails integrated qualification. Physical rollout now requires a strict host-bound
qualification; this run's failed record authorizes no motion.

All measurements here originate in the Conductor cloud workspace, with generated
RGB fixtures and repository fake hardware. No real arm or camera was opened. They
cannot qualify the Lenovo or its network. Compact evidence is in
[modal-latency.json](modal-latency.json); full request traces are under
`.context/latency/` in the measurement workspace.

## Real baseline and isolated handle reuse

Both phases use the same pinned Molmo model, saved processors, L40S container and
three 640×480 RGB frames. Each phase has 101 requests; the first is excluded from
its 100 warm samples. These phases use `spawn() → get()`; JPEG and `.remote()` are
subsequent changes. Compute was reliably observed as `eu-frankfurt-1`, with the
original `us-east` routing request. The routing region is configuration metadata,
not independently measured packet routing.

| Quantity | Original raw, uncached handle | Raw, cached handle |
|---|---:|---:|
| Warm RTT p50 / p95 / p99, seconds | 1.719 / 2.070 / 2.745 | 1.663 / 2.363 / 2.885 |
| Server total p50 / p95 / p99, seconds | .364 / .395 / .404 | .357 / .391 / .401 |
| Model forward p50 / p95 / p99, seconds | .339 / .369 / .382 | .334 / .367 / .379 |
| Saved preprocessing median, seconds | .0214 | .0208 |
| Saved postprocessing median, seconds | .00030 | .00034 |
| Input blob upload p50 / p95 / p99, seconds | .511 / .722 / .755 | .567 / .788 / 1.007 |
| Dispatch p50 / p95 / p99, seconds | .643 / .844 / .897 | .640 / .855 / 1.056 |
| Response wait p50 / p95 / p99, seconds | 1.030 / 1.379 / 2.121 | 1.038 / 1.580 / 2.229 |
| RTT minus server p50 / p95 / p99, seconds | 1.358 / 1.708 / 2.389 | 1.301 / 2.000 / 2.498 |
| Encoded image bytes | 2,764,800 | 2,764,800 |
| SDK serialized input bytes | 2,765,751 | 2,765,751 |
| SDK serialized response bytes | 6,051 | 6,051 |

SDK diagnostic hooks observe one serialization per input/output, one input blob
upload, and no response blob download. They wrap existing SDK calls without extra
serialization. Dispatch includes upload; stages in this table must not be added
as independent components. Residual time includes SDK, routing, scheduling and
network; network-only time is unknown. Modal-side request deserialization and
response serialization are outside the model runtime timer and remain unobserved.

The same container handled predictions 1–101 and 102–202. Its model loaded once
(145.27 s), with 153.87 s container initialization and 158.91 s client preparation.
The first inference took 2.999 s RTT / 1.638 s server / 1.605 s model. First-forward
initialization is separated from warm inference. An earlier 240-second development
startup failed while populating the dedicated test cache; it was stopped before
the cached retry. No warm distribution mixes these starts.

## Optimized real-service result

The final deployed service used JPEG q85, a cached handle and `.remote()`, with
broad `us` compute requested and `us-west` routing requested. Compute was observed
as **`us-ashburn-1`**. Three preceding `us-west` compute attempts received no
container; this result is not a west-compute measurement. The unchanged model and
saved processors loaded once in 137.89 s; container initialization took 147.21 s.
The final app was stopped with zero remaining containers.

| Quantity | Original raw baseline, 100 warm | Optimized diagnostic, 50 warm |
|---|---:|---:|
| Warm RTT p50 / p95 / p99, seconds | 1.719 / 2.070 / 2.745 | **.645 / .695 / .707** |
| Server total p50 / p95 / p99, seconds | .364 / .395 / .404 | .391 / .438 / .442 |
| Model forward p50 / p95 / p99, seconds | .339 / .369 / .382 | .358 / .405 / .408 |
| RTT minus server p50 / p95 / p99, seconds | 1.358 / 1.708 / 2.389 | .254 / .257 / .289 |
| Saved preprocessing median, seconds | .0214 | .0194 |
| Saved postprocessing median, seconds | .00030 | .00027 |
| Median encoded image bytes | 2,764,800 | **698,281** |
| Median three-camera JPEG encode / server decode, ms | Not applicable | 7.43 / 9.90 |

Median RTT falls 62.5%; p95 falls 66.4%. This comparison combines transport, call
path and geography changes; it cannot assign the total improvement to any single
change. The input fixture geometry, crop, checkpoint and model cadence are the
same. The new direct distribution has no private SDK hooks; the baseline and
paired diagnostic have scoped hooks. The remaining median server work now exceeds
the median client/transport residual, so transport is no longer the dominant
component in this direct distribution. Network-only latency remains unknown.

The paired diagnostic observes raw input serialization of 2,765,771 bytes and an
input blob upload, versus 699,337 serialized bytes and **no input blob upload** for
JPEG. Those requests include a diagnostic seed, so their envelope differs slightly
from the baseline. Four deterministic noise seeds share the exact same generated
camera fixtures, state, task and saved processing. Across them the greatest joint
prediction difference is **.01750 radians**, and the greatest gripper difference
is **.08789** on its normalized 0–1 scale. Mean absolute differences across all
420 action components range .00184–.00533; that aggregate mixes units. These are
sensitivity measurements on noisy fixtures, not evidence of physical fidelity.

The final integrated real-service attempt used the actual LeRobot worker and
fake YAM dispatch. Its first RPC took .9214 s (.4184 s server, .5030 s residual).
Observation age was .0449 s at dispatch and .9668 s at return. The existing
conservative timestep rounding and deadline checks dropped all 30 actions: zero
accepted, zero sent, one expired chunk, zero warm integrated samples, zero effective
queued horizon. The .033 s continuous-time remainder did not contain an executable
step. The underrun count is zero because execution never began. All fake robots
released; Stop-under-load cannot be claimed from this failed first chunk.

The hardware-free regression suite was running on the same cloud client during
these optimized measurements; client contention is a measurement limitation.
Even the better direct warm p95 of .695 s leaves at most .305 s of the nominal
horizon before observation/encoding costs, allowing at most .244 s with a 20%
margin. It therefore cannot meet the qualification inequality. The real collector
saved a failed, host-bound record to `data/qualifications/modal-molmoact2.json`.
No Lenovo qualification or supervised physical validation occurred.

## Small transport and scheduling changes

JPEG q85 with 4:2:0 subsampling is the Modal default; `rgb8` remains selectable.
Each camera is encoded once at its original configured dimensions. Header
validation avoids a second pixel decode. Server decoding precedes the unchanged
optional center crop and the model's saved preprocessor. No recording resolution,
model resize, model weights, dtype, control FPS, motor speed clamp or firmware
timeout changes. Local inference explicitly retains raw RGB.

On 100 warm local measurements with three deterministic, noisy 640×480 frames:

| Quantity per three-camera request | rgb8 | JPEG q85 |
|---|---:|---:|
| Median image bytes | 2,764,800 | 698,283 |
| Median / p95 encode time, ms | .317 / .662 | 6.898 / 7.140 |
| Median / p95 decode time, ms | .012 / .015 | 10.092 / 10.558 |

This is a 74.74% payload reduction (3.96× smaller). Decode timings above are local
CPU codec measurements, not L40S server measurements. Random RGB is deliberately
hard to compress and has a mean absolute decoded pixel difference of 46.75/255;
that pixel statistic is not an action fidelity test. Paired real-model fixtures
use identical state, task and seeded Molmo noise to isolate JPEG sensitivity.

The CHW-float to HWC-uint8 validation path now uses NumPy instead of repeated small
Torch reductions. Pixels are exactly identical. With four CPU threads, median
per-camera conversion falls from 2.04 to 1.02 ms (p95 2.87 to 1.06 ms).

The Modal service handle resolves during readiness and is reused. A synchronous
`.remote()` call runs in a daemon transport thread beneath the existing LeRobot
worker. The SDK's lack of a public direct-call cancellation handle does not defer
local Stop: generation invalidation, shutdown checks and action deadlines reject
late results; the transport remains busy until its underlying call finishes.
The old `spawn` path remains diagnostic-only and requires `us-east` routing.

The prediction threshold is configurable, defaulting to 30 steps: request the
next chunk immediately after the previous merge with exactly one request in
flight. A new observation timestamp is required. The original 45-step capacity
cap remains unchanged; stale and overlapping prefixes are still dropped, and
queued actions retain observation-relative deadlines. This is unguided async,
not RTC. Starting earlier cannot recover an entirely expired one-second chunk.

## Queue and qualification

The raw real-service integrated test reused LeRobot's worker, observation
processing, base strategy and YAM dispatch path with fake arms/cameras. Its one
RPC took 1.670 s; observation age was .0249 s at dispatch and 1.6955 s at return.
All 30 actions had expired: zero accepted, zero sent, effective horizon zero,
one expired chunk, and all fake robots released. The zero underrun count reflects
that execution never began, and is not a healthy-queue result.

The qualification collector runs a separate direct distribution and the actual
integrated fake-robot loop against the deployed service. Records under
`data/qualifications/` are git-ignored. Validation binds the hostname and machine
fingerprint, measurement origin, model/dependency revision, Modal app and placement,
encoding/quality/subsampling, configured dimensions, crop and scheduling. Records
expire after 24 hours. Warm p95 must fit 80% of the smaller of the nominal horizon
minus measured observation age and the actual merged horizon. At least 50 warm
direct and integrated requests, healthy queue supply, zero deadline failures,
normal release and zero commands after Stop during an RPC are required. Mapping
acceptance and explicit supervised confirmation remain separate requirements.

A Conductor result can only establish readiness for a fresh Lenovo qualification.
It cannot authorize motion here or become valid by copying its record to the Lenovo.

The final immediate-prefetch synthetic replay ran after the competing regression
process had finished. Injecting the measured .644683 s median yielded nine fake
bimanual sends and then one underrun; injecting the .695181 s p95 yielded eight
sends and one underrun. Both released all fake arms. These are software replays of
measured delays, not additional Modal samples. They confirm that even the better
direct-call distribution cannot sustain this queue at 30 Hz.

A separate healthy final scheduling fixture completed 52 predictions (51 warm),
64 fake bimanual sends and zero underruns. Its deliberately requested Stop occurred
during an RPC, with zero fake SDK commands afterward and all arms released. This
proves the Stop/overlap mechanism under an injected fast RPC; it does not qualify
real-service throughput. Unit regressions additionally cover uncancellable direct
calls, reset invalidation, late responses and unchanged action deadlines.

## Placement, resources and measurement limits

Production requests `us-west` compute and `us-west` routing, with one L40S maximum,
zero minimum containers, a persistent production weight cache and bounded idle
scale-down. Compute/routing and host-memory reservation are configurable. Observed
compute comes from `MODAL_REGION`; actual routing is reported as unknown.

Three west-region attempts did not obtain a container, including a 660-second
readiness wait with a 48 GiB host-memory reservation. Pending queue time is not
reported as GPU inference or west-region latency. The completed broader-US diagnostic retained
west routing and observed Ashburn; it does not establish west-compute availability.

Tests used only their own `yamkit-latency-20260905-d3570a79` volume, deleted after
all owned apps stopped. No shared production volume was removed. Every owned test app is shut down after the test;
there is never more than one requested L40S at a time. The diagnostic governor
caps the final allocation at 280 seconds, within the remaining original bounded
compute budget. No billing limits change and no WebSocket transport is added.

Modal's [region documentation](https://modal.com/docs/guide/region-selection)
explains direct-call restrictions for non-default routing and US-east blob
storage for large payloads. Its [data residency documentation](https://modal.com/docs/guide/data-residency)
states that region pinning does not fall back when capacity is unavailable. SDK
1.5.5's synchronous direct path uses a 2 MiB blob threshold; the raw input exceeds
it, while these JPEG fixtures fit below it. End-to-end measurement is still
required to determine how much latency that avoids.

The first two test app lifetimes conservatively total about 844 seconds; the final
allocation monitor accounts for another 205.79 seconds (including pending start
and cleanup). This stays below the original 1,200-second compute bound. All six
owned app IDs are recorded in the compact evidence. The three west-only attempts
had no observed containers. Shared pre-existing apps and volumes were left intact.
The final host RSS peak was 45.76 GB (42.62 GiB), with CUDA allocation peak 21.81 GB;
48 GiB was sufficient for this test, while production retains its 64 GiB reservation.

The checkpoint exposes 30 actions both in its LeRobot chunk configuration and its
nested backbone maximum horizon. No trained longer horizon is available without
changing policy semantics. The next measurement needed is actual west-compute
placement from the Lenovo network, subject to the same qualification gate. There
is no evidence here to justify adding a custom persistent transport.

**STILL TOO SLOW:** .358 s median model work plus .254 s median client/transport
residual yields .645 s median RTT, which cannot replenish the remaining valid
portion of a one-second chunk with the required margin. The actual integrated
attempt rejected the entire first chunk.

## Validation and changed files

The final qualification-enabled regression run passed **1,028 tests** (four
existing dependency/fork deprecation warnings). Ruff and `git diff --check` passed.
Real Modal testing included 100 warm baseline requests, 100 warm cached-handle
requests, eight paired encoding requests, 50 warm optimized requests and the two
integrated first-chunk attempts. All execution tests used fake hardware.

Changed files in this patch:

- `README.md`
- `docs/INFERENCE_MODELS.md`
- `docs/MODAL.md`
- `docs/MODAL_LATENCY.md`
- `docs/MODAL_VALIDATION.md`
- `docs/REMOTE_LEROBOT_AUDIT.md`
- `docs/REMOTE_PERFORMANCE.md`
- `docs/acceptance-test.md`
- `docs/integration-yamkit-v1.md`
- `docs/modal-latency.json`
- `scripts/benchmark_remote.py`
- `src/yamkit/cli.py`
- `src/yamkit/deployment.py`
- `src/yamkit/inference/client.py`
- `src/yamkit/inference/modal_service.py`
- `src/yamkit/inference/performance.py`
- `src/yamkit/inference/protocol.py`
- `src/yamkit/inference/qualification.py`
- `src/yamkit/inference/service.py`
- `src/yamkit/inference_check.py`
- `src/yamkit/modal_ops.py`
- `src/yamkit/modal_qualification.py`
- `src/yamkit/probe_runner.py`
- `src/yamkit/remote_policy/configuration_yamkit_remote.py`
- `src/yamkit/remote_policy/modeling_yamkit_remote.py`
- `src/yamkit/remote_rollout.py`
- `src/yamkit/ui/server.py`
- `tests/test_benchmark_remote.py`
- `tests/test_deployment.py`
- `tests/test_inference_service.py`
- `tests/test_modal_ops.py`
- `tests/test_modal_qualification.py`
- `tests/test_modal_transport.py`
- `tests/test_probe_runner.py`
- `tests/test_qualification.py`
- `tests/test_remote_performance.py`
- `tests/test_remote_policy.py`
