# Molmo H100, denoising and JPEG fidelity follow-up

This investigation keeps the existing LeRobot async worker, one request in flight,
30 Hz YAM cadence, stale-prefix removal and Stop/deadline guards. It adds no
pipelining or WebSockets. All inputs are generated fixtures; no real arm or camera
is opened. Cloud evidence cannot qualify the Lenovo network.

The comparison starts at `34925a0e1ab16373b2a9ef7529092db039214bc5`. The prior L40S
JPEG q85 distribution was 0.645 / 0.695 / 0.707 s p50/p95/p99 over 50 warm
requests, with 0.391 s median server work and 0.358 s model forward. That run
observed Ashburn compute and requested west routing; its integrated queue failed.
The [earlier report](MODAL_LATENCY.md) remains the record of those measurements.

## Method and budget

The new service requests one exact `H100!`, preventing Modal's automatic H200
upgrade, and requests `us-west` compute/routing. It uses the same pinned model,
processors, image geometry (three 640×480 RGB frames), crop (`none`), 10-step
production denoising, four CPU cores and 64 GiB host reservation. Placement and
loaded runtime properties are recorded separately from requested settings.

The independent governor limits the app to 1,200 observed allocated seconds and
1,500 seconds overall, then shuts it down even if the benchmark fails. It uses the
persistent `yamkit-policy-weights` volume; that production cache is preserved.
There is at most one GPU/container at a time. Regression tests are paused during
warm latency collection to avoid client CPU contention.

[Modal pricing](https://modal.com/pricing) is $0.001097/s for H100, $0.0000131 per
physical CPU core-second and $0.00000222 per GiB-second. The
[us-west region multiplier](https://modal.com/docs/guide/region-selection) is 1.75×.
At the full reservation, the estimate is $0.00226009/s: $2.712 at the allocated
cap, with additional room for cleanup and other charges within the user's $4
limit. Reported costs are resource-time estimates, not an invoice.

## Precision and inference settings

The runtime explicitly configures CUDA Molmo as bfloat16. New telemetry counts the
actual parameter and buffer elements by dtype and observes representative attention
and action-expert inputs/outputs once, removing hooks immediately afterward. This
distinguishes configured dtype from loaded weights and sampled operations; it does
not claim that every normalization or flow accumulator executes in bf16.

The pinned backbone sets `flow_matching_num_steps=10`; the policy's
`num_inference_steps=None` inherits those 10 steps. The action expert has 36 layers.
The model returns 30 actions at the unchanged cadence. Training
`num_flow_timesteps=8` is a separate setting and is not the inference step count.

Upstream has an action CUDA graph manager, including fixed-shape input updates and
a graph cache. The current production loader explicitly disables it. A diagnostic
request can enable it temporarily under the existing runtime lock; telemetry records
whether the graph was actually called and whether its cache populated. Its prior
enabled state and methods are restored even on failure.

Only Molmo `native_fixture` requests may override denoising to 5 steps or toggle
graphs. Requests for robot execution and probes reject those options. Qualification
reassesses saved evidence and rejects inference experiments; an attractive result
from reduced steps cannot authorize motion. Production denoising stays at 10.

## JPEG fidelity method

All three cameras use the same JPEG quality: q85, q90 or q95, with the existing
4:2:0 subsampling. Four fixed noise seeds compare the same RGB fixtures, zero state,
task and saved processing. Each seed also repeats raw RGB to check deterministic
prediction before interpreting JPEG differences. The selector requires raw repeat
agreement within 1e-6, maximum joint difference ≤0.01 rad and maximum gripper
difference <0.02. The joint bound is a conservative screening choice; fixture
agreement does not establish physical task fidelity.

Select the smallest measured payload satisfying both limits. Raw RGB is the fallback
if none of the tested JPEG qualities passes. Per-camera and total payload, encode,
local diagnostic decode and server decode timings are retained. Encoding and image
resolution are independent: no extra resize or crop is introduced.

Default-versus-5-step denoising and default-versus-graph comparisons use raw images
and identical seeded noise. Eleven paired seeds provide ten warm model samples per
variant; their small RTT tails are diagnostic only. The main H100 profiles use a
separate first request and 50 warm requests each.

## Independent model horizons and readiness

| Model and cadence basis | Actions | Horizon | Ideal RTT <H/2 | Conservative p95 target ≤0.4H |
|---|---:|---:|---:|---:|
| MolmoAct2-YAM, pinned dataset 30 Hz | 30 | 1.000 s | .500 s | .400 s |
| SmolVLA, published async example/current wrapper 30 Hz | 50 | 1.667 s | .833 s | .667 s |
| pi05, published robot setting 50 Hz | 50 | 1.000 s | .500 s | .400 s |
| pi05, wrapper's diagnostic 30 Hz assumption | 50 | 1.667 s | .833 s | .667 s |

Molmo's [pinned dataset metadata](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/blob/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c/meta/info.json)
establishes 30 Hz. Its policy chunk, execution chunk, normalization tag and backbone
maximum all cap the horizon at 30 actions; no trained extension is available.
SmolVLA's [paper §3.3](https://arxiv.org/html/2506.01844v1) describes the 30 fps
async example. pi05's [paper §IV-E and appendix](https://arxiv.org/html/2504.16054v1)
describes 50 Hz control and 50 predicted actions. Neither base checkpoint config
specifies one universal FPS, so pi05 must not be credited with a 1.667-second
physical horizon solely because yamkit's diagnostic profile uses 30 Hz.

Applying the *previous Molmo* p95 .695 s as a scenario misses all conservative
targets above, although it is below the ideal half-horizon limit for 50 actions at
30 Hz. This does not measure SmolVLA or pi05. Their historical standalone results
had only two warm samples each, with p95 1.200 s and 2.783 s respectively; different
payloads and placement make them unsuitable as optimized latency qualification.

**SmolVLA and pi05 are not latency-qualified by this task.** They need their own
optimized service distributions at their actual robot cadence. Both also lack
validated physical YAM action mappings. A future latency pass would not satisfy
that separate mapping requirement. No cadence or mapping profile was changed here.

## Measured H100 results

The single deployed app `yamkit-vla-598e1349522b4188`
(`ap-zJqJcUDk0ZBMITAatDmosR`) observed compute **`us-west4`** and requested west
routing. Readiness confirmed 5,442,196,208 bf16 model parameters, including
577,564,448 bf16 expert parameters. The 64 floating-point buffer elements were
float32. Both sampled attention projection and action embedding received and
returned bf16 tensors. Policy and processed action outputs were float32, as the
pinned policy explicitly casts its returned actions. Graphs were supported and
disabled for ordinary requests.

The exact `H100!` request prevents automatic H200 substitution. CUDA device-name
telemetry was added locally after this deployment and was not observed in this
run; the GPU identification here relies on Modal's exact-SKU contract.

| H100 path, 50 warm requests each | Model p50 / p95 | Server p50 | RTT p50 / p95 / p99 | Residual p50 / p95 |
|---|---:|---:|---:|---:|
| JPEG q85, current 10-step inference | .515 / .546 s | .557 s | **.795 / .831 / .972 s** | .245 / .249 s |
| Selected raw RGB, current 10-step inference | .533 / .560 s | .564 s | **1.578 / 1.931 / 2.128 s** | 1.012 / 1.354 s |
| Raw RGB, 10 steps + experimental CUDA graphs | .155 / .162 s | .183 s | **1.237 / 1.515 / 1.713 s** | 1.056 / 1.328 s |

Each distribution excludes its first request. Residual is measured per request
as RTT minus server runtime; it includes serialization, upload, routing, service
scheduling and network. It is not a network-only measurement. Raw images exceed
the SDK's 2 MiB inline threshold; JPEG fixtures stay below it. This run did not
instrument individual SDK upload stages, so the residual cannot be attributed
entirely to blob upload.

The H100's ordinary forward did not beat the earlier L40S forward. These are
separate hosts/placements; GPU type alone does not explain the difference. The
large graph improvement is consistent with substantial launch/execution overhead
inside eager inference, but this test did not separately profile CPU launch and
GPU kernel execution time.

A 22.91-second local regression process overlapped approximately 21:48:23–21:48:46
UTC with part of the raw graph RTT distribution. Its model timings are server-local,
but client RTT tails may include contention. The q85 profile, paired comparisons
and raw default profile finished before that process began. Even the graph run's
minimum RTT was 1.093 s, far above the .400 s target; this limitation does not
establish readiness.

## Paired forward fidelity

Eleven paired seeds, with the first separated, gave:

| Inference settings, raw RGB | Warm model p50 / p95 (n=10) | Maximum joint delta | Maximum gripper delta |
|---|---:|---:|---:|
| Default 10 steps, graphs off | .526 / .565 s | 0 | 0 |
| Experimental 5 steps, graphs off | .309 / .324 s | **.09626 rad** | **.20703** |
| Default 10 steps, experimental graphs on | .179 / .184 s | **0** | **0** |

Every graph request actually used the graph manager and populated its cache.
Subsequent ordinary requests confirmed graphs disabled and 10 steps restored;
all 51 raw default-profile requests used the unchanged settings. All 51 graph
profile requests used graphs with 10 steps. Graphs retain their experimental
request interface in this patch; production settings are unchanged. The 5-step
experiment has no demonstrated task fidelity and is not promoted.

## JPEG decision

The four-seed results below apply the same quality to all three cameras. Times
are medians per three-camera request; encode timing is on the cloud client,
decode timing on the H100 service. The separately recorded local decode diagnostic
is excluded from RPC timing. Raw repeats matched exactly for every seed.

| Encoding | Image bytes | Encode / server decode | Max joint delta | Max gripper delta | Fixture screen |
|---|---:|---:|---:|---:|---|
| Raw RGB | 2,764,800 | .921 / .058 ms | 0 | 0 | Pass |
| JPEG q85 | 698,346 | 12.264 / 12.108 ms | .021877 rad | .062500 | Fail |
| JPEG q90 | 828,044 | 10.772 / 13.124 ms | .008908 rad | .050781 | Fail |
| JPEG q95 | 1,082,906 | 13.044 / 14.147 ms | .013126 rad | .064453 | Fail |

Higher JPEG quality did not monotonically reduce action differences. All tested
JPEG qualities missed the requested gripper goal. **Raw RGB is selected as the
smallest passing payload among these tested encodings**, and becomes the production
default across rollout, checks, probes and qualification. JPEG remains explicitly
selectable for diagnostics; no claim is made that every JPEG configuration has
been exhausted or that noisy-fixture agreement proves physical task fidelity.

## Queue, cleanup and decision

After GPU shutdown, the actual LeRobot worker/strategy ran synthetic replays of
the measured p95 latencies, using matching encodings and generated camera frames:

- q85 at .830781 s: 27 expired prefix steps, three fake bimanual sends, then one underrun.
- Raw default at 1.931038 s: all 30 steps expired, zero sends.
- Raw graph timing at 1.514893 s: all 30 steps expired, zero sends.

All fake arms released. No warm queue sequence was sustained. The graph replay
injects the measured delay and does not execute a graph or model. Effective queued
horizon was zero for both raw scenarios. Existing freshness/deadline/Stop behavior
was preserved; lowering control FPS or weakening stale-prefix removal was not tried.

The app stopped immediately after the bounded measurements, with zero containers.
The governor observed 623.26 seconds including pending startup and cleanup
(**10.39 minutes**); its total wall time was 657.40 seconds. Charging that entire
wall interval at the full H100 + four-core + 64 GiB reservation and 1.75× multiplier
gives a conservative **$1.486 compute estimate**, below the $4 and 30 GPU-minute
limits. This is not a billing invoice. The persistent production cache remains;
pre-existing apps and volumes are unchanged.

**MOLMO STILL TOO SLOW.** With the tested fidelity-preserving encoding, raw transport
has about a one-second median residual even when graphs reduce the model to .155 s.
No tested JPEG quality satisfies the fidelity screen, and its ordinary H100 forward
also exceeds the .400 s target on its own. Neither model speed nor queue behavior
justifies Lenovo qualification from these measurements. SmolVLA/pi05 latency
readiness remains unmeasured separately from their unvalidated YAM mappings.

Full measurement details and source hashes are in [molmo-h100.json](molmo-h100.json).
Raw traces remain under `.context/h100/` in this workspace.

## Validation and changed files

The final suite passed **1,071 tests** with four existing dependency/fork warnings.
Ruff and `git diff --check` passed. Tests cover experiment rejection/restoration,
actual dtype telemetry, raw defaults and exact pixels, deterministic quality
selection, single-GPU restrictions, qualification guards and Stop behavior.

Changed files in this follow-up:

- `README.md`
- `docs/MODAL.md`
- `docs/MODAL_LATENCY.md`
- `docs/MOLMO_H100.md`
- `docs/molmo-h100.json`
- `scripts/benchmark_remote.py`
- `src/yamkit/cli.py`
- `src/yamkit/deployment.py`
- `src/yamkit/inference/modal_service.py`
- `src/yamkit/inference/protocol.py`
- `src/yamkit/inference/qualification.py`
- `src/yamkit/inference/service.py`
- `src/yamkit/inference_check.py`
- `src/yamkit/modal_ops.py`
- `src/yamkit/modal_qualification.py`
- `src/yamkit/probe_runner.py`
- `src/yamkit/remote_policy/configuration_yamkit_remote.py`
- `tests/test_benchmark_remote.py`
- `tests/test_inference_service.py`
- `tests/test_modal_ops.py`
- `tests/test_modal_qualification.py`
- `tests/test_probe_runner.py`
- `tests/test_qualification.py`
- `tests/test_remote_performance.py`
