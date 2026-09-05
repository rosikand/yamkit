# Synthetic camera preview benchmark

Run from the repository after installing its normal development environment:

```bash
mkdir -p .context/tmp .context/test-hf
HF_HOME="$PWD/.context/test-hf" HF_TOKEN_PATH="$PWD/.context/test-hf/token" \
HF_HUB_OFFLINE=1 TMPDIR="$PWD/.context/tmp" \
  .venv/bin/python scripts/benchmark_preview.py --duration 10 --warmup 1 \
  --output .context/preview-benchmark.json
```

This opens only a synthetic publisher and authenticated HTTP clients on `127.0.0.1`.
It never creates a robot, physical camera, dataset, or Hub connection. The four scenarios
run sequentially in fresh subprocesses to keep each process's RSS high-water mark separate:

| Scenario | HTTP viewers | Behavior |
|---|---:|---|
| off | 0 | Publisher running, every source frame offered, no viewer demand |
| on | 3 | One reader per camera drains each JPEG |
| slow | 3 | One reader per camera reads 1 KiB at most every 100 ms, requesting a 4 KiB receive buffer |
| multiple | 9 | Three readers per camera drain the same JPEGs |

All scenarios use three 640×480 `uint8` RGB sources at a requested 30 Hz and a 10 fps preview
target. Every tick creates distinct allocated images from fixed random textured backgrounds
(seed `20260905`) with a moving colored bar. The source work is identical across scenarios;
there is no large preallocated frame ring. This pattern yields relatively large JPEGs, so its
encoding/network cost need not match images from a physical rig.

**Handoff** measures each call to `offer`, including calls that do not select a frame. Its
median therefore mostly reflects rate/demand rejection; p95/p99 include selected-frame copies.
**Loop work** includes synthetic frame creation and all three handoffs, excluding pacing sleep.
**Loop interval** includes pacing and scheduler jitter. CPU is process CPU time divided by wall
time, as a percentage of one core; it includes the publisher, synthetic source, and client
threads. It does not include a FastAPI proxy, browser JPEG decoding, recording, or robot work.

Current RSS is sampled from Linux `/proc/self/statm` about every 100 ms; the report also gives
initial/final RSS and the process high-water mark from `getrusage`. These include imports,
source images, and clients; they exclude kernel socket buffers and other processes. Sampling
can miss short peaks, and allocator behavior can affect the values. Pending/busy handoff drops
are reported separately from deliberate rate skips and from socket-level viewer disconnects.
Slow clients count bytes, without parsing complete JPEG frames. After a stalled socket times
out, the publisher can stop encoding because no viewers remain; this is expected and makes
that scenario's CPU average lower. The JSON includes final viewer demand per camera.

Treat these as short synthetic measurements, not zero-overhead claims, a sustained load test,
or a guarantee of real camera or recording fps. Physical acceptance remains the separate, supervised checklist in
[`UI.md`](UI.md#supervised-camera-acceptance-manual-only).

## Measured results

Run on 2026-09-05 in this workspace's Linux x86_64 VM (8 reported CPUs, Python 3.12.14,
NumPy 2.2.6, OpenCV 4.13.0), using the command above. All scenarios maintained a measured
30.000 Hz synthetic source loop over approximately 10 seconds after the 1-second warmup.

Each timing cell is **p50 / p95 / p99**.

| Scenario | Handoff (µs per call) | Loop work (ms per three cameras) | Loop interval (ms) |
|---|---|---|---|
| off | 1.22 / 3.22 / 4.49 | 0.58 / 0.73 / 1.25 | 33.333 / 33.354 / 33.367 |
| on | 5.41 / 128.60 / 156.90 | 0.87 / 1.60 / 1.84 | 33.333 / 33.349 / 33.379 |
| slow | 1.76 / 95.13 / 132.60 | 0.52 / 1.39 / 1.76 | 33.334 / 33.347 / 33.363 |
| multiple | 5.72 / 143.21 / 169.68 | 0.85 / 1.76 / 2.05 | 33.334 / 33.355 / 33.382 |

| Scenario | CPU (% one core) | RSS start → sampled peak (MiB) | JPEGs encoded, all cameras | Rate skips | Pending/busy drops | Preview errors |
|---|---:|---|---:|---:|---:|---:|
| off | 2.07 | 60.72 → 60.79 | 0 | 0 | 0 | 0 |
| on | 13.00 | 66.32 → 69.14 | 303 | 600 | 0 | 0 |
| slow | 6.34 | 66.80 → 69.53 | 126 | 246 | 0 | 0 |
| multiple | 16.77 | 67.13 → 69.95 | 303 | 600 | 0 | 0 |

RSS starts at the beginning of the measured interval, after warmup. On and multiple encoded
101 frames per camera, shared by one and three viewers respectively; each normal viewer
received 101 frames. All selected images were copied; the off scenario copied none. Slow
viewers reached the send timeout and final server demand was zero for every camera, which
explains the smaller JPEG count and lower average CPU. Pending replacement was unnecessary
in this run; pending-replacement and busy-slot tests cover that overload behavior.

Publisher/client cleanup took at most 0.043 seconds in this run, with zero surviving benchmark
viewer threads. These measurements do not exercise a hung native encoder, long-lived memory
growth, actual USB camera buffer behavior, the UI proxy/browser path, or Hub uploads. The full
JSON report is generated at `.context/preview-benchmark.json` (gitignored); rerun the command
to reproduce the methodology on another machine.
