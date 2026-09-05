# Remote inference performance gate

Physical Modal rollout is **BLOCKED**. The integrated real-service path has not
demonstrated a continuously supplied, fresh action queue. Checkpoint mapping,
successful GPU inference, readiness, and operator confirmation cannot override
this gate. It applies to shared CLI/UI validation, the remote runner, and direct
construction of the LeRobot remote policy. There is no environment/CLI bypass.
Local rollout remains the default. Policy checks and saved-observation probes
remain available; preparing a service remains an explicit paid operation.

## Findings on the integrated execution path

The benchmark uses LeRobot 0.6.1's actual background worker, observation processing,
base strategy, action dispatcher, YAM plugin and `YamArm.command`. Only motor SDK
objects, cameras and remote RPC are fake. It retains 30 Hz, 30-step chunks, the
15-step prediction threshold, three 640×480 RGB images, speed clamps and no-home
fault cleanup. Every reported execution is a successful `Robot.send_action`,
independently checked against fake SDK command timestamps.

Measured on 2026-09-05 in this cloud workspace:

| Injected fake RPC delay | Successful requests / warm n | Warm RPC p50 / p95 / p99 | Actual queue outcome |
|---|---:|---|---|
| 50 ms, 32 seconds | 69 / 68 | 50.13 / 50.34 / 50.51 ms | 955 bimanual sends, zero underruns |
| Repeated 50–250 ms jitter, 32 seconds | 69 / 68 | 105.14 / 250.13 / 250.44 ms | 955 bimanual sends, zero underruns |
| Three 50 ms requests then a 700 ms spike | 3 / 2 | Diagnostic only | One underrun after 55 sends; released without replay |
| Historical Molmo warm latency represented by 1.48 s | 1 / 0 | No warm sequence possible | Entire 30-step chunk expired; zero sends |
| Historical rig-resolution latency represented by 2.378 / 3.166 s | 0 / 0 each | No warm sequence possible | Observation expired before queue merge; zero sends |

These are **synthetic delay reproductions**, not new Modal, network, model-compute,
camera or hardware measurements. They exercise the final software execution path.
68 warm samples expose the injected jitter but cannot estimate real network tails.
The full traces are generated under `.context/integration/`; the compact committed
result is [remote-performance.json](remote-performance.json).

The healthy scenarios begin subsequent predictions with at most 15 queued actions
(0.5 seconds). Up to eight successful actions execute during a prediction in the
jitter scenario. At 250 ms the queue remains supplied; the 700 ms spike drains it.
The existing half-chunk threshold therefore needs no scheduling change for the
measured healthy fixtures. Starting earlier cannot rescue a result whose entire
one-second observation-relative action horizon has already passed.

## Historical real-service evidence and its limits

[MODAL_VALIDATION.md](MODAL_VALIDATION.md) records the source workstream's real
SmolVLA CPU and SmolVLA/MolmoAct2/pi05 L40S fresh-chunk checks. Their original JSON
artifacts are not present in this integration workspace. The documented Molmo
warm RPC p50 is about 1.482 seconds, with only **two warm observations**; model
forward took about 0.405–0.412 seconds. The roughly one-second difference includes
unseparated dispatch, transport, scheduling and other processing. It is evidence
that model forward alone does not explain total RPC; it does not identify a
specific transport or queue bottleneck. The rig-resolution saved probes took
2.378–3.166 seconds and were stale.

Those real results did not run a physical queue, were not measured from the robot
host, and cannot demonstrate continuous control. Existing pools were stopped and
this integration did not redeploy them or spend additional cloud/API money.
Reliable real warm p50/p95/p99 and real queue overlap remain unmeasured on the
integrated path. A future explicitly authorized diagnostic should collect enough
consecutive requests with these timings and actual robot-host networking before
any proposal to qualify physical deployment.

## Correct action-time handling

The source implementation appended a whole returned chunk and only enforced a
two-second observation-age limit. That shifted already expired action prefixes
into the future. The integrated queue now:

1. Drops timesteps elapsed since the observation, using the greater of local
   observation age and the upstream worker's measured inference delay.
2. Drops the additional prefix overlapping commands still queued ahead of the
   new tail. Existing queued commands retain their original intended times.
3. Assigns every accepted action its observation-relative timestep deadline,
   bounded by the existing maximum observation age.
4. Rejects a wholly expired chunk, expired queued action, or underrun; it never
   replays an old target or takes over with a local policy.
5. Rechecks the last dequeued action's deadline immediately before canonical
   `Robot.send_action`, so a processing stall after queue pop cannot send it late.

The original append operation and background worker are reused. This remains
**unguided async**: no prefix is sent to a denoiser, and it is not guided RTC.
Stop invalidates the session/queue, rejects late replies, and releases hardware
without homing before waiting for the worker to finish.

## Timing coverage and honest unknowns

| Requested quantity | Integrated metric / meaning |
|---|---|
| Observation timestamp and age | Local receipt timestamp; age at prediction start, dispatch, response and merge. Camera exposure timestamp is explicitly `null`. |
| Local camera/image transformation | Upstream observation-processing time plus CHW tensor → HWC uint8 conversion time. Hardware capture is outside this synthetic benchmark. |
| JPEG/image serialization | Raw `rgb8` byte serialization time. JPEG is unused (`jpeg_encoding_s: 0`); image encoding stays unchanged. |
| Request size | Sum of raw RGB bytes, 2,764,800 for three 640×480 frames. Complete SDK-framed wire size is `null`. |
| Client dispatch/network | SDK handle lookup, `spawn` dispatch, and `get` response wait timed separately. Dispatch/wait include network and internal serialization; network-only time is `null`. |
| Server queue/preprocessing | Runtime lock wait, reset/state setup, image decode/crop/tensor conversion, saved preprocessor timed separately. Queueing before Modal method entry is `null`. |
| Model forward/postprocessing | Synchronized GPU model-forward time, saved postprocessor time and response tensor-to-list conversion. Fake benchmark server times are zero, not real model measurements. |
| Response decode | SDK-internal decode is `null`; client schema validation and list-to-tensor conversion are measured separately. |
| Total RPC | Locally measured round trip, first request, warm count and p50/p95/p99; failed attempts retain available timings. |
| Queue at prediction start/return | Step counts, start horizon and successfully sent actions during prediction. |
| Usable horizon / discarded prefix | Remaining observation-relative horizon, elapsed prefix dropped, queued-overlap prefix dropped, accepted steps and queue horizon after merge. |
| Underruns / Stop | Underrun count, expired-chunk count, queue depth before stop, total successful sends and invalidation/failure metrics. |

No server/client clocks are subtracted. Raw observations, image bytes, tasks and
credentials are not added to telemetry. Failed transport attempts cannot inherit
the previous request's timings. Samples are bounded in memory.

No image transformation, checkpoint processor, compression, region, transport,
model settings, control FPS or firmware timeout changed. Saved-observation probe
and saved-processor regression tests were rerun; a paid saved-probe rerun was not
needed. Region tuning, payload optimization or persistent transport require
measurements that distinguish their contribution before changing implementation.

## Reproduce without hardware or spending

```bash
source scripts/env.sh
mkdir -p .context/tmp .context/integration
TMPDIR="$PWD/.context/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/pytest tests/test_remote_performance.py tests/test_remote_policy.py \
  tests/test_inference_service.py tests/test_deployment.py tests/test_probe_runner.py \
  tests/test_probes.py --basetemp=.pytest_cache/remote-review
TMPDIR="$PWD/.context/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  .venv/bin/python -m scripts.benchmark_remote \
  --output .context/integration/remote-performance.json \
  --summary-output docs/remote-performance.json
```

The benchmark imports the repository's fake robot and patches the performance
gate only inside the diagnostic process after replacing every hardware and RPC
boundary. It does not offer a physical deployment override. Running it for a
shorter duration is useful for regression debugging, but does not reproduce the
68-warm-sample summaries above.
