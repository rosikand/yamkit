# Development validation — 2026-09-05

Starting revision: `dbd01f1e70be66bb0e639789d0d37d7ecb5bd166`.
All development stayed on the feature branch. No real arms/cameras, SSH, CAN/system
changes, OpenAI generation calls, robot deployment, main push, merge or force-push.

## Implementation milestones and files

Completed milestones were committed and pushed incrementally:

- `f86d00c973972b9c472c5a1599c54403efd9ce5f`: optional Modal dependency and isolated test credentials.
- `330e0164716b847edfe0644aa7a4f953bd1be2d5`: profiles, CUDA service, LeRobot adapter and probes.
- `fd54faa5f5145800117dc4402cb8f5e6f31619cf`: existing Inference page and managed deployment jobs.
- `787d7bf8cb3ed200f0695a992972627f2f138fad`: process ownership and complete readiness checks.
- `9006b4544b7164489cf71b61af339f6a0f976ae2`: effective local configuration validation and preserved local options.

The final documentation commit is identified in the task's completion message.

| Area | Main files |
|---|---|
| Model profiles, mapping, protocol and service | `src/yamkit/inference/`, `configs/modal-requirements.{in,txt}` |
| LeRobot integration | `src/yamkit/remote_policy/`, `remote_rollout.py`, `local_rollout.py`, narrow follower-plugin and startup-stop hooks |
| Deployment and probes | `src/yamkit/{deployment,modal_ops,inference_check,probes,probe_runner,cli}.py` |
| Existing browser/session integration | `ui/app.js`, `src/yamkit/ui/{server,sessions}.py` |
| Verification and operator instructions | `tests/test_{remote_policy,local_rollout,inference_service,modal_ops,deployment,probes,probe_runner,inference_ui}.py`, this report and linked audit/runtime docs |

## Real inference

All successful cases used pinned model and saved pre/postprocessors with three **fresh
`predict_action_chunk` calls**. No cached `select_action` pops, synthetic normalization
statistics or physical-vector padding/truncation counted as success.

| Test | Result | Chunk shape, each of 3 calls | Server model load | Compute region |
|---|---|---|---|---|
| SmolVLA CPU | Passed, finite | 50×6 | 29.38 s local readiness | This x86_64 cloud workspace; 4 OpenMP threads |
| SmolVLA L40S | Passed, finite | 50×6 | 23.91 s | us-east-2 |
| MolmoAct2-YAM L40S, first attempt | Readiness timeout during downloads/loading; no successful inference claimed | — | Startup capped at 240 s; client readiness 300 s | App stopped before retry |
| MolmoAct2-YAM L40S, cached retry | Passed, finite | 30×14 | 149.35 s | eu-frankfurt-1 |
| pi05 L40S | Passed, finite | 50×32 | 135.37 s | ap-osaka-1 |

The pi05 preparation and check ran through the real FastAPI Inference endpoints,
SessionManager, and real `yamkit modal-prepare` / `policy-check` CLI children. The test
session manager appended development limits and an owned test-cache volume; it did
not replace the service or prediction. Other GPU tests used the same shared prepare
and check functions. No hardware rig file was required for native fixtures.

Revisions:

- SmolVLA: `c83c3163b8ca9b7e67c509fffd9121e66cb96205`.
- MolmoAct2-YAM: `fdade02d1f1c1dd819114b0478f735072fb6b212`.
- pi05: `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`.
- Molmo dataset metadata: `e9f21ae15074330839f2ac25ed4b49d76dfa1f9c`.

Nested dependency revisions, gripper direction evidence and exact camera/state mapping
are recorded in [INFERENCE_MODELS.md](INFERENCE_MODELS.md).

## Latency and payloads

Only two warm observations per model were collected. Quantiles below are descriptive
interpolations over those two samples, not reliable estimates of production tails.
First request is after preparation and therefore **not** a cold-container measurement.

| Case | First prediction RTT | Warm RTT p50 / p95 / p99 (n=2) | Warm server inference | RGB payload |
|---|---|---|---|---|
| SmolVLA CPU | 3.676 s | 3.550 / 3.614 / 3.620 s | Local total prediction 3.48–3.62 s | 589,824 bytes |
| SmolVLA L40S | 1.819 s | 1.082 / 1.200 / 1.211 s | 0.305–0.312 s | 589,824 bytes |
| MolmoAct2 L40S | 2.959 s | 1.482 / 1.503 / 1.505 s | 0.405–0.412 s | 2,073,600 bytes |
| pi05 L40S | 4.314 s | 2.712 / 2.783 / 2.789 s | 0.184–0.188 s | 451,584 bytes |

Client fixture generation/encoding ranged approximately 0.0007–0.011 seconds. JSON
artifacts retain separate encoding, server preprocessing, inference, postprocessing,
round-trip and locally measured observation age for every successful request.

Each successful pool has just **one** cold preparation observation (p50/p95/p99 would
all equal that one observation): SmolVLA **434.64 s**, pi05 **153.83 s**, and the cached
Molmo retry **166.88 s**. Readiness includes deployment/scheduling and, for the first
app, a 387.63-second CUDA image build. The table above separately reports server model loading.
These measurements came from
this cloud VM, not the Lenovo or its network connection.

Two additional Molmo **saved synthetic** probes used 14 ordered state values and
three 640×480 RGB frames, with explicit synthetic source and capture age. They never
opened hardware. Both returned unclipped robot-unit diagnostics:

| Transform | Input payload | Round trip | Observed transform | Diagnostic issue |
|---|---|---|---|---|
| Crop off | 2,764,800 bytes | 3.166 s | 480×640 retained before saved processing | observation_stale |
| Center crop | 2,764,800 bytes | 2.378 s | 480×640 → 360×640, top offset 60 | observation_stale |

The crop occurs on the server, so it does not reduce the transmitted RGB payload.
These probes passed numerical/schema/pre-clamp reporting, **not** live-motion freshness.
The Molmo chunk lasts one second, while even native warm RPC took about 1.5 seconds.
Continuous remote robot execution is therefore **not validated** by these measurements;
the default two-second freshness limit also rejects the observed rig-resolution probe
latencies. Do not infer that buffering hides this latency. Measure from the actual
robot host and review transport/region/latency before supervised motion.

Actual hardware queue depth/underruns were not measured. Queue capacity, expired actions,
underruns, Stop, pause/reset generations and no post-stop actions are covered by tests
using the actual upstream worker/strategy and fake hardware. The runtime emits these
metrics on success and failure; no fake metrics are substituted for a physical run.

## Automated and browser coverage

Normal tests are hardware-free and isolate inherited Hub/Modal credentials. Tests use
actual pinned LeRobot policy factories, processor factory and rollout context; the
remote service is mocked for that suite. A tiny real local ACT forward pass runs through
the unmodified upstream local strategy with the fake YAM robot.

Coverage includes explicit bidirectional joint-name mapping, saved processing and clamp
boundaries, forbidden RTC continuation, complete readiness identity, isolation/reset,
deadlines, queue bounds, underruns, startup Stop and no-home release. UI tests exercise
confirmation distinctions, concurrent clicks, child failure, Stop, stale operation and
snapshot results, credential exclusion, lingering previews, recording preservation and
local arm selection. Page handlers execute in QuickJS. Chrome also rendered the existing
Inference page against a hardware-free local preview server.

The final validation commands passed: **306 tests**, `make lint`, and `git diff --check`.
An earlier full run had one intermittent failure in the existing
`test_go_home_all_runs_arms_together_and_ctrl_c_releases_all` assertion; its isolated
rerun and the subsequent full suite passed. The underlying homing implementation was
not changed beyond accepting the optional remote startup-stop event. This intermittent
Ctrl-C release check remains relevant to the separate hardware-hardening work.
Known environment warnings:
Starlette's httpx adapter deprecation, multiprocessing fork warnings in ownership-lock
tests, and the repository's pre-existing uv.toml/tool.uv overlap. No system package was
installed to suppress these warnings.

## Budget, resources and cleanup

Persistent local ledger: `.context/modal-development-ledger.json` (never reset).
The total authorized limits were $8 and 3,600 allocated GPU seconds, with one L40S pool
at a time. Each operation reserved $2 and 900 GPU seconds before starting. No test had
minimum containers, buffer containers, persistent heartbeats or configured retries.
The single Molmo retry was explicit after its previous app stopped.

[Modal pricing](https://modal.com/pricing) was checked before paid work: L40S
$0.000542/s, CPU $0.0000131/core/s, RAM $0.00000222/GiB/s, volume $0.09/GiB/month.
The estimate conservatively charges **all elapsed wall time, including image build**, as
one GPU plus 4 CPU cores and 64 GiB RAM, applies the largest 1.75× regional multiplier,
adds 30 seconds of cleanup per operation and $0.20 per operation for build/storage
contingency. It does not subtract free credits or included storage. Egress billing
starts October 2026, according to [Modal's announcement](https://modal.com/docs/guide/network-egress-billing).

| App | Outcome | Conservative allocated-time upper bound | Estimated usage upper bound |
|---|---|---|---|
| `ap-mjSEqBAbNGBn3MbRhi3xKS` | SmolVLA passed | 470.02 s | $0.806 |
| `ap-7JWXvJyGbsG2RMOn3dnFfS` | Molmo readiness timeout | 340.69 s | $0.639 |
| `ap-d0OAbmu3APevKvwtNxZo6K` | pi05 UI/CLI passed | 199.05 s | $0.457 |
| `ap-zcaXzBUEVXy1hBVcbRySYy` | Molmo retry + probes passed | 215.81 s | $0.478 |
| **Total** | | **1,225.57 s (20.43 min)** | **$2.380** |

Remaining allowance: **$5.620** and **2,374.43 GPU seconds**, using these conservative
upper bounds. These are estimates, not a retrieved billing invoice or exact GPU
allocation measurements. Container `ta-01M1QMW6JN0PF6Y18M2NF7KCAR` was recorded for the
last app; the ledger retains app IDs and bounded wall times for every operation.

Final checks confirmed all four owned apps in `stopped` state and each app's container
list empty. The owned volume `yamkit-vla-dev-dbd01f1e-20260905` was deleted and its absence
verified. **No named test-weight storage or test GPUs remain.** Modal-managed build
image cache (`im-gN9LwYNusxpmX92ae19AzI`) may remain; its retained size was not measured.
Unrelated applications/volumes were not stopped or modified. Local downloaded model
fixtures remain under git-ignored `data/hf/`; logs and result JSON remain in `.context/`.

## Outstanding acceptance limits

No physical robot mapping/calibration/camera acceptance or actual Lenovo-network
latency test was performed. Base SmolVLA/pi05 physical mapping, unreviewed custom Modal
profiles, guided remote RTC, DAgger/recording remote strategies, sensor-exposure
freshness and WebSocket transport remain unsupported or unvalidated as documented.
Local Molmo synchronous inference is subject to sufficient local RAM/GPU/dependencies;
its physical execution was not tested. Source-defined conventions and finite outputs
alone do not approve motion.
