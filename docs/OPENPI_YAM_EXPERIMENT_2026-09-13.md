# Official frozen π0.5 base with an experimental YAM adapter

Status: **PHYSICAL BLOCKED — raw target admission fails and temporal execution
remains unqualified.** No physical operations were authorized or
performed for this investigation. The working MolmoAct2 and `pi05_yam` paths
are outside the adapter experiment and must remain unchanged.

The intended command is:

```bash
yamkit rollout --backend lambda --policy pi05_base \
  --task "put the red cube into the black container" --duration 60
```

It remains blocked until the exact experimental interface and complete software
execution path pass the checks below. The blocker is **not simply the absence
of an official asset named YAM**. New non-learned statistics can define coherent
experimental coordinates; they do not automatically establish command safety or
the frozen model's ability to accomplish the task.

## Identities and scope

| Component | Identity |
| --- | --- |
| Model | Genuine `gs://openpi-assets/checkpoints/pi05_base` |
| Runtime | Official OpenPI JAX, `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Native configuration | `Pi0Config(pi05=True)`, bf16, 10 flow steps, 50×32 actions, discrete state, 200 tokens |
| Pinned acquisition manifest SHA-256 | `439423350a9160e2157291aec1a48e8454193c7126e24fda670df39bb7c503db` |
| Adapter candidate | `experimental_yam_chunk_delta_closure_quantiles_v1` |
| Expanded paired corpus SHA-256 | `a7410582f8043a32982518e39f500c2d25417ca6f0d79bf6fa290f99591a20d0` |
| Candidate statistics SHA-256 | `9b58b2d0a529bc162062ed6f0781a08991772044cdb01083acdcf651730639c8` |

Weights, architecture and native inference are not adapted or replaced. No
fine-tuning, LoRA, learned adapter, task-specific scripted action or borrowed
robot normalization is part of this candidate. The isolated official runtime
and generation-pinned asset acquisition are described in [OPENPI_REFERENCE.md](OPENPI_REFERENCE.md).
Candidate arithmetic is in [yam_candidate.py](../src/yamkit/openpi/yam_candidate.py).
All 30 acquired objects (29 checkpoint objects plus the public tokenizer) were
rehashed successfully during this mission, and again by the native loader.

The appropriate eventual label is **official frozen pi05_base + documented
experimental YAM adapter**, never “official pretrained YAM deployment.”

## What is established, and what is a deployment choice

| Item | Direct evidence | Experimental choice or remaining limitation |
| --- | --- | --- |
| YAM coordinates | Left six joints/gripper, then right six joints/gripper; radians and normalized opening | Retaining those coordinates in a newly normalized frozen-base input is a transfer hypothesis |
| Native π0.5 structure | Actual `pi05_aloha` selects joint-delta/absolute-gripper transforms and Trossen assets | ALOHA joint flips/linkage constants are not applied to YAM |
| Camera roles | Base view and corresponding left/right wrist slots exist in both π0.5 ALOHA and vendor YAM adapters | Similar role names do not calibrate optical geometry or remove morphology/domain shift |
| Gripper interface | YAM 0 closed / 1 open, calibrated SDK affine motor conversion | Candidate uses absolute closure `1-opening`; not a recovered Trossen angular representation |
| Normalization | Native π0.5 uses quantile arithmetic without clipping | These exact quantiles are newly computed from existing local paired data, not official assets |
| Horizon | Native model predicts 50 rows in 32 dimensions | Candidate interprets first 14 and retains the other 18; committed prefix is not yet qualified |
| Timebase | Paired recordings declare 30 FPS; official ALOHA example uses maximum 50 Hz | 30 Hz is a proposed YAM engineering timebase, not universal checkpoint metadata |

The YAM SDK maps normalized opening `g` to calibrated motor radians
`closed + g*(open-closed)`, including reversed motor direction. Its arm coordinates
follow motor/joint order 1–6; yamkit does not mirror the right arm automatically.
Rest poses do not redefine joint zero, and leader alignment offsets do not prove
another robot's joint-frame equivalence. [Pinned I2RT mapper](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robots/utils.py),
[YAM hardware configuration](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robots/config/yam_v1.yml).

### Evidence that actually applies to π0.5

The pinned official `pi05_aloha` config explicitly uses `Pi0Config(pi05=True)`
with `LeRobotAlohaDataConfig`. That factory applies deltas to twelve arm channels
and leaves grippers absolute. This is a genuine π0.5 precedent, not a π0-only
example, but its Trossen embodiment conversion remains robot-specific.
[Official configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py),
[ALOHA conversion](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/aloha_policy.py).

I2RT also publishes a YAM OpenPI fork with an actual `pi05_yam` config. Its
14D inputs retain YAM coordinates and its arm actions become chunk-origin
deltas, but it computes `assets/yam` statistics and performs **30,000 fine-tuning
steps** from the official base. Its schema is useful interface evidence; its
training results are not evidence for unchanged-base zero-shot behavior. The
reviewed vendor repository is pinned to `7d9f9d135a2b949de54a349856b65863f81319e1`.
[Vendor YAM adapter](https://github.com/i2rt-robotics/yam-abc-reproduce/blob/7d9f9d135a2b949de54a349856b65863f81319e1/third_party/policy/openpi/src/openpi/policies/yam_policy.py),
[vendor training configuration](https://github.com/i2rt-robotics/yam-abc-reproduce/blob/7d9f9d135a2b949de54a349856b65863f81319e1/third_party/policy/openpi/src/openpi/training/config.py).

The older official normalization guide labels its action definitions for
`pi0_base` / `pi0_fast_base`. It cannot alone establish π0.5 gripper conventions,
especially where actual ALOHA state and action conversions differ. Similarly,
old control-mode prompt strings are not added to this ordinary π0.5 task prompt.
[Normalization guide](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/docs/norm_stats.md),
[maintainer representation discussion](https://github.com/Physical-Intelligence/openpi/discussions/302).

## Exact candidate specification

The 14 channels are `left_joint_1.pos` through `left_joint_6.pos`,
`left_gripper.pos`, then the corresponding seven `right_` channels. Let
`J={0,1,2,3,4,5,7,8,9,10,11,12}` and `G={6,13}`.

1. Use **measured** YAM observation state `s`, not the MA2 policy's cached last
   command. Require finite exact shape and valid measured grippers.
2. Candidate state is `x[J]=s[J]`, `x[G]=1-s[G]`.
3. For each complete 50-row paired command window beginning at observation `s`,
   compute `a_candidate[t,J]=a_sent[t,J]-s[J]` and
   `a_candidate[t,G]=1-a_sent[t,G]`. Every row uses the same measured origin;
   there is no successive-row integration or multiplication by a timestep.
4. Estimate separate state and transformed-action q01/q99 from the development
   estimating split, using exact NumPy linear quantiles. State samples are
   complete-window anchors; action samples are all rows of those windows.
   This is not upstream's histogram estimator or a pretrained statistic.
5. Normalize `z=2*(x-q01)/(q99-q01+1e-6)-1`, without clipping. Tokenize **14
   normalized state values before padding the continuous state to 32**. Retain
   native task formatting, tokenizer, image handling and ten-step inference.
6. Map `top`, `left_wrist`, `right_wrist` RGB uint8 into `base_0_rgb`,
   `left_wrist_0_rgb`, `right_wrist_0_rgb`, all masks true. Use native 224×224
   padded resizing and native image scaling; no crop, mirroring or recoloring.
7. Preserve every raw 50×32 output. Unnormalize the first 14 using
   `x=(z+1)/2*(q99-q01+1e-6)+q01`; retain dimensions 14–31 unchanged in evidence.
8. Reconstruct `a_proposed[t,J]=x[t,J]+s[J]` and
   `a_proposed[t,G]=1-x[t,G]`. Proposed out-of-range values are retained and
   reported, never silently clipped. The pure decoder does not dispatch or
   claim that hard joint/rate bounds were checked.

The native transform/tokenizer ordering and absolute reconstruction follow the
pinned π0.5 path. The *selection of YAM coordinates, closure convention, newly
computed statistics and first-14 output interpretation* remains this explicit
experimental schema. [Native transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py),
[native tokenizer](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py),
[policy composition](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy_config.py).

The numerical minimum quantile span is `1e-4` (100 times native epsilon). It is
an arithmetic audit choice, **not a movement-coverage threshold or safety limit**.
Changing it or widening quantiles cannot be used to hide a failed audit.

### Timing remains to be qualified

Fifty rows at the proposed 30 Hz describe a nominal 1.667-second prediction
horizon. The pure arithmetic candidate has no executor and establishes neither
the committed prefix nor replanning/Stop behavior. The official ALOHA client,
for comparison, commits 25 rows at a maximum 50 Hz before another synchronous
request; predicting 50 rows does not require committing all 50. A future YAM
executor must declare its choice and account for every unused row.
[Official client](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/main.py),
[chunk broker](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py).

## Existing paired-data audit

No demonstrations or camera frames were collected. The expanded corpus contains
**16,486 paired rows in ten episodes across seven existing LeRobot datasets**:

| Dataset | Frames | Episodes |
| --- | ---: | ---: |
| integration_record_ui_02 | 2,771 | 2 |
| pick_red_cube_2demo_dummy | 1,495 | 2 |
| moveblackbintwo | 897 | 1 |
| ui_autostart_01 | 3,085 | 1 |
| ui_autostart_02 | 4,292 | 1 |
| integration_record_stop_01 | 2,154 | 2 |
| integration_record_ui_03 | 1,792 | 1 |

The final episode of each multi-episode dataset is held out **before** window
construction. This yields seven estimating episodes with **13,605 complete
windows**, and three held-out episodes with **2,391 complete windows**. The
corresponding action-row counts are 680,250 and 119,550. Windows overlap within
each split, but do not cross episode boundaries or train/held-out membership.
“Estimating” refers to statistics only; no model training occurs.

Both original and expanded NPZ hashes were independently verified against their
manifests. Export checks exact names/order, finite 14D shapes, metadata robot
type, episode/frame sequence and nominal 30 FPS. Metadata alone cannot certify
physical provenance of every historical recording. Uniform dataset timestamps
also do not prove exact physical dispatch rate. This is a development audit,
not independent task or dynamics evaluation.

Held-out normalization round-trip maximum errors are **4.44e-16** for state and
**1.11e-16** for transformed actions. These establish numerical consistency,
not robot compatibility or satisfactory distribution coverage.

### Coverage is still uneven

The original five-dataset corpus had almost no right J2/J3 movement; expansion
improves its scales but does not cover the existing moving-right-arm fixtures:

| Channel | Expanded state q01 / q99 | Saved50 normalized min / max | Outside [-1,1] |
| --- | --- | --- | ---: |
| Right J2 | 0.00171664 / 0.19855802 rad | −0.992 / **20.349** | 33/50 |
| Right J3 | 0.00438697 / 0.23441672 rad | −1.000000001 / **15.816** | 37/50 |
| Right gripper closure | 0.00768679 / 0.10382761 | −1.103 / **17.524** | 28/50 |

Right J2 and J3 produce only five and seven distinct native state-bin values,
respectively. The 50 saved RGB/state fixtures come from the previously retained
green-bowl run `20260912-033824-rollout-7500a8d1`; they are genuine recorded
measured feedback, not model-state substitutes. They informed development and
must not subsequently be labeled an independent generalization benchmark. Their
green-bowl images do not match the requested black-container scene.

The corpus also includes 2,771 frames in `integration_record_ui_02` with exactly
one distinct commanded target. Counts therefore overstate independent motion
coverage. Native state binning loses distinctions far beyond its upper range;
finite values and a nonzero quantile width do not fix that limitation. Conversely,
quantile extrapolation alone is not an algebra bug or automatic safety fault:
q01/q99 intentionally leave tails outside the interval. It must be measured and
assessed against the experiment's intended operating domain, not concealed.

Exact numerical evidence and per-source hashes are retained under
`.context/openpi-yam-adapter/inputs-expanded/` (`corpus-manifest.json`,
`paired-recordings.npz`, `candidate-audit.json`), with the statistics receipt at
`.context/openpi-yam-adapter/candidate-statistics.json`.

## Real-GPU parity and command-bound results

The unchanged official model processed **50 saved real RGB/measured-state
observations** through the expanded candidate. Five additional identical-noise
native comparisons made **55 model calls / 2,750 predicted rows** in total;
2,500 primary rows are retained for the command audit. No rows were dispatched,
no session was created, and no hardware was tested or activated.
All 50 primary raw arrays and their noise are retained. The five additional
native calls were asserted array-equal in the running probe and their zero
differences recorded; separate duplicate-output files were not written, so
they cannot be independently re-compared as two stored arrays after the fact.

| Check | Result |
| --- | --- |
| Native normalization, token and image-input parity | Exact for all 50 cases |
| Native unnormalization and absolute-action reconstruction | Exact for all 50 cases |
| Identical-noise native model comparisons | Five cases, maximum absolute difference **0** |
| Warm local inference, 49 primary calls | p50 **51.18 ms**, p95 **52.96 ms**, maximum **54.44 ms** |
| First cold primary call | **2.0454 s** |
| Joint hard-bound violations | **0** scalars / rows / chunks |
| Gripper scalars outside [0,1] | **125** in **6/50 chunks**, maximum **1.03326256** |
| Gripper scalars exceeding 1.01 | **96**; even the existing fine-tuned path's one-percent guard would not admit these |
| Measured anchor to first-row joint jump | All **50/50** chunks exceed ordinary **0.03 rad** command cap; maximum **0.36724678 rad** |
| Hypothetical first-step speed at 30 Hz | Maximum **11.0174 rad/s**, versus configured **3.0 rad/s** |
| Adjacent-row joint step | Maximum **0.0239691 rad**, or **0.7191 rad/s** at hypothetical 30 Hz |

Hard bounds were derived with a pure XML parse and the existing SDK's 0.15 rad
buffer; the captured follower configuration has no joint offsets. The audit
identifies rig SHA-256
`6e8cbc785c94a48672404edc9d124734d4df93bd696b6b840c4d6cd325c75785`
and XML SHA-256
`49cf63ee0f354210dac52e66f0eb87244569a3ac7ca04a17be82cbf477441ac4`.
Grippers retain their decoded values without clipping. The 0.03 rad comparison
comes from ordinary control's maximum 0.01-second command interval and configured
3.0 rad/s cap. **It is a diagnostic comparison, not a claim that an existing
native-base executor enforces this cap.** The separate, qualified MA2 reference
send deliberately bypasses the ordinary clamp; no reuse of that bypass is
authorized or qualified for this candidate.

Thus native parity passes, but **this candidate fails raw target admission** and
does not yet define a safe anchor-to-first-row or cross-chunk execution mapping.
Small adjacent-row changes do not resolve the large initial jumps. Inter-chunk
transitions, FIFO/committed-prefix accounting, integrated scheduling and full
Stop/in-flight invalidation/fault/release behavior have not been qualified.
Local GPU timing is not network, CLI or integrated observation-age qualification.
The CLI remains blocked before hardware, rather than silently selecting this
experimental decoder for a physical rollout.

The evidence files are
`.context/openpi-yam-adapter/real-state-probe-result.json` and
`.context/openpi-yam-adapter/command-bound-audit.json`, with individually hashed
inputs and raw/decoded outputs under Lambda repo
`.context/openpi-yam-adapter/real-state-probe-v2/`; verified output copies are
under cloud `.context/openpi-yam-adapter/raw-probe/`. Original saved RGB/state
inputs remain at the individually hashed paths in the result. The result's
`normalization_assets_used: []` belongs to the bare official diagnostic identity:
**no official robot asset was used, but the candidate statistics identified
above were used.** Its inherited generic missing-contract notes are not a claim
that the candidate's explicit experimental coordinates are undefined.

These findings are concrete failures of the current candidate and incomplete
execution qualification, not proof that frozen-base experimentation can never
work. Changing normalization, clipping grippers, or interpolating/rate-limiting
outputs would create a revised, explicitly documented experiment requiring its
own audit; none was silently applied to these results.

The first candidate probe exposed premature float64 promotion in our inverse
normalization. It was corrected to preserve native float32 `(x+1)/2`
intermediates before multiplication by float64 quantile spans. Exact dtype
regressions now cover float16/32/64; the successful probe above used the fix.
This changed only candidate arithmetic, not native model/runtime code or weights.

## Protected paths and regression status

The experiment branched from `70a2c4a1fc097acaf9e2fbe6af3435afbeb79cd3`, protected
by pushed tag `openpi-yam-baseline-20260913-70a2c4a`. No working MA2/PI-YAM
runtime, executor, robot plugin, checkpoint or dependency file was modified.
The shared policy-selection edit only replaces the base policy's explanatory
blocker; rejection still precedes backend/rig/hardware access.

**526 focused tests passed**, covering the new candidate, immutable OpenPI assets,
diagnostics, exact requested CLI rejection, UI no-device preparation guard,
PI-YAM parity/workflow/qualification, and MA2 reference interpolation/rate/
dispatch/Stop/qualification. Ruff passes on changed Python files. This is not
a passing base-model end-to-end fake rollout: that path remains unimplemented
and blocked after candidate admission failed.

- MA2 build remains `5128c7b55b6f9a62c109b581f98912de717c7d3746e8c6374341e91687d481fe`.
- PI-YAM build remains `ebec79a9a5ff27333a572193df378359391a86707cc852d8527dad48665e81ec`.
- No Lenovo UI, tunnel or model service was restarted. Only repo-local offline
  evidence was written on remote hosts; the candidate was not deployed as a service.

## Alternative considered: virtual ALOHA/Trossen retargeting

A different non-learned hypothesis maps measured YAM joints through FK to a
declared common tool frame, solves virtual VX300S IK, applies the actual native
`pi05_aloha` pipeline with its matching Trossen statistics, then retargets native
absolute output through VX300S FK and YAM IK. That would use Trossen statistics
**as Trossen statistics**, not silently attach them to raw YAM coordinates.

It cannot be reduced to per-joint sign/zero changes: YAM joints 2–4 are a
consecutive parallel-axis chain, while VX300S elbow/forearm-roll axes are
orthogonal. Geometry, joint ranges and gripper linkages differ. The reviewed
VX300S reference is Interbotix `0bb2b0e6d0e619bff02cf74dbd5af5681dcf80c9`.
[YAM URDF](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robot_models/arm/yam/v1/yam.urdf),
[VX300S model](https://github.com/Interbotix/interbotix_ros_manipulators/blob/0bb2b0e6d0e619bff02cf74dbd5af5681dcf80c9/interbotix_ros_xsarms/interbotix_xsarm_descriptions/urdf/vx300s.urdf.xacro).

A bounded CPU-only probe used six saved times per arm and two deliberately
uncalibrated frame hypotheses. Same-base placement produced five of twelve
IK fits within 1 mm/1°; home-aligned placement produced eight. The same near-home
state could map to approximately −π versus −0.034 rad virtual forearm roll.
Single-seed numerical failures do not prove unreachability; good residuals do
not select the correct frame or branch. This is an ambiguity demonstration,
not a qualification score.

This route requires actual base/tool/jaw geometry, a declared virtual placement,
deterministic IK branch continuity and singularity/reachability/rate/collision
checks. The current rig schema does not supply those base/TCP extrinsics.
Camera/morphology differences remain empirical generalization uncertainty, not
an automatically fatal coordinate error. This retargeter was not implemented
or selected for physical use.

## Smallest useful next steps and readiness boundary

The present candidate fails the output-domain check above. The smallest next
offline step is a causally verified extraction of the **existing** moving-right-arm
reference trace, retaining actual timing and calibration provenance. Determine
whether it supplies a defensible 50-row action distribution in a declared
timebase; do not collect demonstrations or infer actions from sparse states.
That may justify testing revised statistics, but it is not guaranteed to remove
the output failures. An independently specified gripper endpoint and initial/
cross-chunk transition policy also needs native-fidelity and safety review;
simply widening bounds or reusing the reference-send bypass is not a fix.

Safe trace extraction must use the latest complete measured `observation` or
`reference_row_observation` **before** a bimanual send, followed by both successful
`send_end.postclamp` receipts and their matching `reference_dispatch` commit.
Reject partial sends, missing events and wrong arm ordering. MA2 `policy_state`
is the last commanded target, not measured state. Do not pair a post-send
measurement retroactively as the same command's origin. Preserve hashes and
original timestamps. [Trace instrumentation](../scripts/trace_rollout.py),
[reference execution order](../src/yamkit/reference_strategy.py).

Reference traces contain interpolation points and RPC pauses, so consecutive
commands are not automatically a regular 30 FPS sequence. Do not build windows
across run/episode/fault/timing gaps. Any zero-order-held resampling must be a
separately documented experiment, not fabricated measured/sent pairs.

The intended command is **BLOCKED** until the selected adapter and execution
contract have verified provenance, correct native transforms, decoded-bound
handling, measured scheduling and complete fake-device CLI/Stop/fault/release
qualification. Undefined scales, wrong units/origins, invalid executable targets
or failed release/deadline safeguards are concrete blockers. Absence of an
official YAM label and unproven task success alone are not.

After software qualification, fresh exact-command approval, verified mounts,
clear workspace and on-site stopping arrangements remain mandatory. The first
supervised experiment tests physical compatibility and task competence; neither
is established by numerical parity, invertibility or finite output.
