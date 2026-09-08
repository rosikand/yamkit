# Lenovo → cloud VLA: first HTTP inference milestone

The hardware-free inference path met the initial **400 ms warm p95 RPC target**:
**363.6 ms p95** from the Lenovo to one Modal H100, using three full raw RGB
images and the pinned 10-step MolmoAct2 model. This is a diagnostic milestone,
**not physical rollout qualification**. No arms or cameras were opened and no
predicted action was executed. Conductor and `andre@yam-lenovo` were the only
development/test machines; no Mac was involved.

Runtime source was `66960b5808d6540d73293b8e6d719edcefe83fbc`. These experiments
did not change production transport, graph settings, control limits, or gates.

## Measured results

| Measurement | Warm samples | p50 | p95 | p99 | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU HTTP echo, 1 KiB request | 51 | 129.6 ms | 134.2 ms | 135.7 ms | 136.1 ms |
| CPU HTTP echo, raw real images/state, 2,764,856 B | 51 | 156.4 ms | 174.9 ms | 330.5 ms | 481.8 ms |
| CPU HTTP echo, lossless zlib-1, 685,929 B | 51 | 135.8 ms | 142.9 ms | 149.2 ms | 154.3 ms |
| **H100 HTTP inference, raw generated RGB** | **50** | **352.8 ms** | **363.6 ms** | **417.0 ms** | **465.4 ms** |
| Model execution within those H100 requests | 50 | 159.1 ms | 161.5 ms | 162.0 ms | 162.2 ms |
| Server processing within those H100 requests | 50 | 190.0 ms | 195.2 ms | 196.5 ms | 196.6 ms |

Echo measurements include upload plus an approximately 1.8 KiB response. They
exclude model execution and compression/decompression. Payload hashes matched
for every request; invalid authentication was rejected. The CPU container ran
in `westus3`. Its first, cold request took 3.91 seconds and is excluded above.

The GPU experiment used an exact H100 80 GB in `us-west4`, four CPU cores and
64 GiB host memory, one container, a persistent HTTPS connection and an
authenticated bounded binary envelope. Model parameters were bf16. Payloads
contained three 640×480 RGB images (2,764,800 image bytes; approximately
2,765,997 bytes including the request envelope), 14 ordered state values and a
fixed task. There was no JPEG, crop, pixel compression, reduced denoising, or
change to the 30-action/30-Hz profile.

The existing `scripts.benchmark_remote.profile_modal` harness generated fresh
seeded random images per request and validated the protocol and response
identity. All **51 requests completed**, with the first graph-capture request
excluded and **50 warm samples**, no failures, finite 30×14 outputs, and one
stable instance shared by SDK readiness, HTTP readiness, the saved probe and
the benchmark. All benchmark responses reported actual graph use and exactly
10 inference steps. The first graph request took 4.57 seconds. Including local
image encoding/validation, warm total-request p95 was 364.0 ms. One warm request
exceeded 400 ms; a short direct benchmark does not establish queue resilience.

## Real saved observation

A separate eager-mode diagnostic processed the successful `ui_autostart_02`
recording's episode 0, frame 2290, with the original three camera images and
14 measured arm/gripper values. The input is saved on Lenovo at
`data/probes/hello-world-real.npz`, alongside a provenance JSON file. Its capture
timestamp is explicitly an uncertain historical log-receipt estimate, not a
fresh camera timestamp.

This request remained `saved_probe`, used the original rig camera names and
saved model processors, and returned finite 30×14 ordinary and unclipped
diagnostic actions. The first target's largest joint difference from the saved
state was 0.0332 rad. Neither target stream was executed. The first eager
request took 6.70 seconds, including first-forward initialization; it is not a
warm latency measurement.

The recorded scene contains a table and empty black bin, with no red cube.
The requested cube-to-bin instruction therefore tests data flow only; it does
not demonstrate task success. Real saved observations were never relabeled as
native fixtures to enable graph diagnostics.

## Server startup findings

The initial web-only startup attempt failed before readiness. A subsequent
mixed SDK/HTTP class loaded the model, then replaced its container during the
first prediction. The terminated container's control-plane record reports
success/exit 0 without an exception or termination reason; this does not
establish an OOM or heartbeat timeout.

The successful experiment retained the same resources and moved synchronous
runtime calls out of the ASGI event loop using `asyncio.to_thread`. It warmed
the class through SDK readiness before making Lenovo HTTP requests. SDK and
HTTP methods shared one class pool, verified by instance identity. The binary
endpoint rejected live/robot request modes, checked a short-lived bearer token
before body processing, enforced 4 MiB message limits, and decoded inert
JSON/bytes/tuple values without pickle. Six codec self-tests passed.

All three GPU attempts and the CPU echo app were stopped, with zero containers
confirmed. The conservative combined compute estimate was about $1.53,
including failed attempts and counting full app lifetimes at the configured
resource rate, including shutdown and timestamp-rounding margins; this is not
an invoice. Images and the existing model cache were
retained. Rate basis: [Modal pricing](https://modal.com/pricing).

## Remaining before a supervised rollout

1. Add the authenticated HTTP path and reviewed 10-step graph execution mode
   to the production transport and service. Preserve single-flight requests,
   total deadlines, local Stop invalidation and late-result rejection.
2. Warm the **actual task and image shape before hardware connection**. The
   upstream graph cache depends on context shape, including task token length;
   generic readiness or a previous request is insufficient. Unexpected graph
   recapture must not happen after the arms connect.
3. Bind the exact transport, execution mode, model/runtime identity, image
   settings and placement in fresh Lenovo qualification. Existing SDK/eager
   qualification cannot qualify HTTP/graphs. Test the actual LeRobot worker,
   queue and dispatcher with fake hardware and at least 50 completed warm
   predictions, including Stop during a request.
4. Stage and inspect a concrete object/task, check the actual observation and
   action mapping, then request approval for the exact bounded motor command.
   The 400 ms firmware timeout, speed clamps and current qualification gates
   remain unchanged.

## Session artifacts

Cloud artifacts are in `.context/validation/`; Lenovo copies are in
`.context/validation-20260908/` unless otherwise stated. These contain no
account tokens or reusable endpoint credentials. Diagnostic scripts are
temporary session tools, not a supported rollout interface.

| Artifact | SHA-256 |
| --- | --- |
| `hello-world-real.npz` | `3d6d2b3839308006501720f6c246892b4d96233dbb4486b8c9cdcbb65e0b8b2b` |
| `http-network-report.json` | `8830d63f7dc801ba9f2e4aacd4b34f8a666a56f167a9b3d0e7b7fa0143dd0557` |
| `http-gpu-benchmark-report.json` (cloud, HTTP labels added) | `f583227279ad9fd3c4865e603121290cd39533754d6e139135dc85109800019a` |
| `http-gpu-real-probe.json` | `b41a9707ba45a7da3d84a5ebe37a3561bd8c86bb0ec0f24f7de0bdd179bd3c35` |
| `http_gpu_benchmark.py` (successful attempt) | `41d7a41988569738f8afdc9632b6ea46488b136a8a9d409bd8263f0740b9e313` |
| `http_diag_wire.py` | `bb81e524dfc1408d28eb8c97cff5d5f870c04dc9051eee3b7279acf9581f521d` |

Ownership/shutdown receipts are `http-network-run.json` and
`http-gpu-run-attempt{1,2,3}.json`. Successful GPU app:
`ap-HNms7CsITcEXN5tKZ5gOoz`, `yamkit-http-h100-a4e68093c649`.
