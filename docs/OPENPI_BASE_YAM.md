# Official frozen OpenPI π0.5 base on YAM

Use this label: **official frozen pi05_base + documented experimental YAM adapter**.
This is not an official pretrained YAM deployment or a task-success guarantee.

## Terminal use

From the configured Lenovo checkout, with `scripts/env.sh` sourced:

```bash
yamkit rollout \
  --backend lambda \
  --policy pi05_base \
  --task "put the red cube into the black container" \
  --duration 60
```

There is no separate prepare step. The command connects to the existing Lambda service,
or starts its bounded model process if absent; it reuses an authenticated matching service
when warm. It warms the actual task and validates the current host/rig/model/interface
with saved observations and explicit fake arms. Matching, unexpired evidence is reused.
Only then does it show the command's effects and ask for on-site confirmation.

Before confirming: verify the left/right arm and camera mapping, empty grippers, secure
mounts/table, a clear movement area, and accessible local Stop and physical power cutoff.
Remain with the arms. `Ctrl-C` is terminal Stop. Confirmation permits one run, not retries.
Healthy completion homes while preserving the final gripper opening, then releases;
Stop or a fault releases without homing. A failed software or capture-admission check
does not open hardware, shorten the requested run, or weaken limits.

For software-only testing through the same real service:

```bash
yamkit rollout --backend lambda --policy pi05_base \
  --task "put the red cube into the black container" --duration 60 \
  --fake-hardware --capture-trace --upload-repo-id namespace/private-rollout-dataset
```

Omit `--upload-repo-id` for local-only recording, and omit `--capture-trace` if no RGB
recording is wanted. Upload implies capture. Fake mode uses retained images and guarded
fake arms/cameras; it rejects physical approval flags and never asks for a physical GO.
Its output explicitly has `hardware_tested=false`, no motion approval and no task-success
claim. Each model request replays an **intact saved measured-state/RGB observation**.
Independent perfect-tracking fake SDK state tests initial/cross-chunk command transitions;
it is not substituted into unrelated saved images. This is input replay plus actuator
execution testing, **not a closed-loop scene, dynamics simulation or task-success test**.
Saved observations do not establish physical tracking, camera alignment or manipulation.

`yamkit inference --backend lambda --policy pi05_base --task "…"` is an optional software-only
check, not a prerequisite. `rollout ... --dry-run` prepares but does not launch a rollout.
The existing UI does not yet launch official base; use the terminal command. `pi05` and
`pi05_yam` retain their separate fine-tuned YAM behavior and are not substitutes.

## What remains genuine

| Component | Fixed identity / behavior |
| --- | --- |
| Weights | `gs://openpi-assets/checkpoints/pi05_base`, generation-pinned and hash-verified |
| Runtime | `Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479` |
| Model | Native `Pi0Config(pi05=True)`: bf16, 10 flow steps, 50×32 output, 200 tokens |
| Native transforms | Original padded image resize, prompt/state tokenizer, state padding and native `Policy.infer` |
| Manifest SHA-256 | `439423350a9160e2157291aec1a48e8454193c7126e24fda670df39bb7c503db` |

There is no fine-tuning, learned adapter, LoRA, checkpoint conversion/substitution,
scripted task trajectory, changed sampling algorithm, or model-output clipping. Native
finite, correctly shaped 50×32 outputs delivered by the authenticated transport are
retained before YAM decoding/admission, including values outside normalized `[-1,1]`.
Malformed/nonfinite native outputs are rejected at the service or transport boundary;
their arrays are not available to client recording. Those failures retain sanitized
failure types/status only, not a claim to preserve every server-side anomaly.
Model weights and the native policy are unchanged. JAX bulk preallocation is disabled
to share the existing GPU; that changes allocation, not inference mathematics.
See [pinned model/assets and native parity](OPENPI_REFERENCE.md).

## The non-learned YAM interface

These are explicitly engineered deployment choices, not recovered official YAM training
metadata. They define the interface being qualified, not the model's competence.

1. **State and images.** Measured state is left joints 1–6 (radians), left gripper opening,
   then right joints 1–6 and right gripper. No joint sign flip, mirror, zero adjustment or
   ALOHA linkage conversion is invented. Closure is `1-opening` for each gripper. Top,
   left wrist and right wrist RGB map to the corresponding native image roles. Full
   640×480 uint8 RGB reaches native padded resizing; there is no center crop. Normalize
   the fourteen state values **before** native tokenization, then let native transforms
   pad to 32; prepadding would change the discrete state tokens.
2. **Statistics.** State/action q01/q99 are newly computed, non-learned YAM deployment
   statistics. They use existing sent-action recordings plus causally paired measured
   states and complete bimanual SDK receipts from the retained moving-right-arm trace.
   Complete windows are resampled onto the chosen 50 Hz command timebase by zero-order
   hold, without inventing intervening measurements. Anchors remain measured. The fixed
   selection uses at most 10 Hz anchors and equal activity-stratum/episode mass; weighted
   midpoint-CDF 1%/99% quantiles are computed before new model outputs are inspected.
   No statistics are tuned to force predicted bounds to pass. Historical dataset timing
   and provenance limitations remain in the corpus report.
3. **Joint/action reconstruction.** Native quantile arithmetic, including `1e-6`, is
   unchanged and unclipped. Unnormalize the first fourteen output channels. Every joint
   delta is added to the **same measured state at that request's origin**; it is not
   integrated from adjacent rows or multiplied by a timestep. Grippers remain absolute
   closure and convert back to opening. The remaining eighteen raw dimensions are
   retained as intentionally unconsumed model channels. Their exclusion is an explicit
   experimental embodiment choice, not evidence that the checkpoint was trained on YAM.
4. **Mechanical gripper endpoints.** A finite requested opening below zero requests the
   calibrated closed endpoint; above one requests the calibrated open endpoint. Interior
   values are unchanged. This is an explicit actuator saturation rule, **not native
   OpenPI output behavior** and not a silently widened anomaly tolerance. Every conversion
   records raw and executed values, delta, row/channel and whether it was in the committed
   prefix. Raw model/decoded values remain available. No percentages are fitted to output
   extrema; nonfinite/malformed values still fault. Arm joints are never projected.
5. **Commitment and transitions.** The model predicts fifty rows; commit rows 0–24 FIFO,
   then reobserve/replan. Rows 25–49 are intentionally unused, following the pinned
   official ALOHA client's 25-row/max-50-Hz convention. Before any prefix dispatch, all
   committed endpoints must satisfy the configured robot's bounds. Large initial,
   per-row and cross-chunk changes use synchronized linear substeps constrained by the
   existing ordinary YAM speed caps and its 0.01-second command cap. Endpoints are retained
   exactly; there is no averaging, prefix dropping, catch-up or task-specific easing.
   A subdivided row takes longer and that **time dilation is measured and logged**.

The first transition uses fresh measured feedback when the last command is stale;
ordinary SDK speed limiting stays enabled. A stalled transition or an unexpectedly
modified SDK receipt faults. The executor issues no policy sends or new observation calls
during the synchronous model RPC. Model requests have a two-second deadline, responses bind
the exact measured anchor/session/source/statistics, and Stop invalidates late results.
No automatic retry is performed. The existing joint/gripper bounds, startup/home limits
and 400 ms motor firmware protection are not disabled.

These are faithful native **inference** semantics with a documented experimental
**actuator** adaptation. YAM morphology, gripper geometry, camera placement, data support
and time-dilated actions differ from an ALOHA deployment. Neither numeric roundtrips nor
software qualification establishes zero-shot task success or collision safety.

The current statistics identity is
`fb8f3c40fe2cfca66b4530819964113bab57c0efeea004b39d347bf98cdc3949`, paired corpus
`dc0133ba26233745af8fa7c899916886cfa864470ffecd10a39fef3df83c7042`.
The immutable file is `data/openpi/yam/normalization.json` on both hosts. It remains
labeled experimental; its standalone metadata never grants physical readiness.

Sources: [native π0.5/ALOHA configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py),
[native quantile/delta transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py),
[ALOHA client](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/main.py),
[experimental YAM interface](../src/yamkit/openpi/interface.py),
[executor](../src/yamkit/openpi/executor.py).

## One-time installation configuration

The already configured Lenovo/Lambda installation uses a separate `pi05-base` entry in
private `data/inference/backends.json`, never the MA2/PI-YAM service identity or port:

```json
{
  "service": "lambda-openpi",
  "endpoint": "http://127.0.0.1:8767",
  "token_file": "data/inference/lambda-openpi.token",
  "saved_observations": [".context/saved-real-yam/observation-000.npz"],
  "remote": {
    "repo": "/absolute/path/to/gpu/yamkit",
    "token_file": "data/inference/lambda-openpi.token",
    "region": "your-existing-region",
    "session_seconds": 28800,
    "gpu": 0
  }
}
```

This object belongs under `backends.lambda.policies["pi05-base"]`; use the existing
verified SSH configuration described in [INFERENCE_CLI.md](INFERENCE_CLI.md). The example
paths are placeholders, not a request to create new keys, acquire observations or edit
the current installation manually. Each saved NPZ contains exactly `state`, `top`,
`left_wrist`, `right_wrist`. Missing saved inputs are never filled by live camera reads.

The GPU uses `data/openpi/venv/bin/python`, clean pinned `data/openpi/upstream` and verified
public assets under `data/openpi/cache`. The robot host uses its normal `.venv`. Source,
statistics, token files and environments must already be installed by the software
deployment; normal rollout does not require manual environment repair. Credentials remain
private repository-local files; configuration contains paths, never values. Automatic
startup never installs system packages, provisions VMs or changes SSH/network settings.

An existing authenticated model is reused without restart. Cold startup uses a finite
owned supervisor, repository-local locks and an 18,432 MiB free-GPU admission reserve.
Unknown listeners, source mismatches and authentication failures are not taken over.
Services and unrelated applications are not killed to make space. The model session can
expire; its existing VM remains billable. MA2 and `pi05_yam` environments/controllers remain
separate and unchanged.

## Qualification, recording and readiness boundary

Automatic qualification exercises 50 direct saved-real inputs and 50 integrated fake
prefixes (2,500 predicted / 1,250 committed rows), exact endpoint/substep receipts,
configured bounds and actual in-flight RPC Stop. It binds host, rig, task, source-content
builds, statistics and the exact service instance/expiry. Dynamic busy/warm-cache metadata
does not invalidate otherwise identical proof. Wrong/malformed/stale proof cannot reach
the hardware delegate. Evidence lives in `.context/openpi-qualification/<operation>/`;
the current pointer is `data/inference/openpi/lambda-openpi/qualification.json`.

For fake qualification and CLI replay, every RPC uses a complete recorded measured-state
and RGB pair. Fake SDK state independently follows actual command receipts for transition
checks. No new model request pretends those fake positions are the measured state of a
different saved scene. Physical mode continues to acquire current measured state and RGB.
The earlier integrated attempt that faulted after nineteen prefixes, including an
out-of-bound left-joint-4 target, remains negative evidence. Its synthetic state/scene
mismatch was a harness defect, but that does **not** prove the mismatch was the sole cause
of the model output or predict success in a real closed-loop trial. Hard bounds stay active.

With capture enabled, recording reserves memory before device construction. RGB follows
a nominal 30 Hz phase-bin capture cadence plus every actual policy-input image triplet;
state observations, native raw chunks, decoded targets, endpoint conversions, transition
points and timestamps are retained independently. Exported videos are variable-frame-rate:
they preserve the original monotonic receipt timestamps and hold the preceding image
across RPC gaps, without inventing frames. The first full 60-second fake run produced
three 60.005739-second videos with 976 original encoded frames each—not 1,800 constant-
30-FPS frames. Neither the nominal capture cadence nor video metadata claims 50 distinct
camera exposures per second. Saving/rendering/private-HF upload
begin only after confirmed release. Originals remain local on export/upload failure.
Files appear under `outputs/ui/deployments/`; fake runs are clearly labeled synthetic.

This document describes the implementation, not an issued readiness certificate.
The current command must produce matching passing software evidence before any supervised
trial. A software-ready result would still mean **hardware untested** for this adapter;
fresh on-site confirmation is mandatory and task success remains empirical.

The earlier [candidate investigation](OPENPI_YAM_EXPERIMENT_2026-09-13.md) and the
normalized-space [diagnostic report](OPENPI_REFERENCE.md) are historical evidence, retained
unchanged in scope. Their diagnostic-only `YAM_BLOCKERS` do not describe this separate
CLI's current host-bound qualification result and never authorize dispatch.
