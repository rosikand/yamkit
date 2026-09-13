# YAM π0.5: independent native reference execution

Status on 2026-09-13: the pinned tokenizer and 9.35 GB YAM weights load on the
existing GPU. Fresh contract-version-3 qualification passed on Lenovo with saved
observations and fake arms: 50 direct calls and 50 complete integrated FIFO
chunks / 1,500 executed rows, with zero unexpected drops, SDK modifications,
coherence violations or faults. A real-GPU five-second fake CLI run also passed
capture, three-video playback and full private-HF hash verification. This is
software evidence, not a physical trial, manipulation success or authorization
to move the arms. The final idle-host sign-off is software READY for fresh
supervised approval; see the [overnight handoff](OVERNIGHT_HANDOFF_2026-09-13.md)
for the dated deployment evidence and distinct official-base blocker.

The earlier two attempts stopped during direct warm sampling on a native gripper
prediction outside [0,1] (`left_gripper.pos=1.000895619392395`) and never reached
integrated execution. Their failed evidence remains in the
[authorization follow-up](PI05_QUALIFICATION_2026-09-13.md); it is not reused as
passing evidence. The new qualification exercises the explicit SDK endpoint
adapter and bounded-phase tail rule described below; historical version-2 proof
does not qualify this newer build.
The working MolmoAct2 path remains separate and unchanged.
See [the range review](PI05_YAM_RANGE_REVIEW.md) for primary sources, the raw
anomaly guard, exact raw/executed accounting and force-limiter caveat.

## Immutable source contract

Checkpoint: [Jiafei1224/molmoact2-yam-pi05](https://huggingface.co/Jiafei1224/molmoact2-yam-pi05/tree/51ab2720d7e56d51410407f98ea64bbea97feb2e),
revision `51ab2720d7e56d51410407f98ea64bbea97feb2e`.
Weights SHA-256: `a777861c627234f9aa54a1bb7bdee29101ee6513f4773ef0a581d9c5527981e4`.
Profile ID `pi05-yam`; execution contract `pi05_reference` version 3, with
`i2rt_gripper_endpoint_projection_v1` and `completed_fifo_phase_tail_v1`. The old low-level
`pi05` profile remains the unrelated, unmapped `lerobot/pi05_base` fixture.

| Boundary | Pinned native convention |
| --- | --- |
| Cameras | top, left, right, in checkpoint feature insertion order. Local wrist names map explicitly to left/right. |
| Saved input shape | Three RGB cameras, each 3×360×640. Full current 640×480 RGB is accepted by the native image transform without an added crop. |
| State/action order | Left joints 0–5, left gripper, right joints 0–5, right gripper. Maps exactly to yamkit joints 1–6. |
| Physical values | Absolute joint targets in radians; continuous grippers in [0,1], closed→open. No relative/delta action conversion. |
| Prediction | 30×14 rows, ten denoising steps, 30 Hz dataset/control frequency; `n_action_steps=30`. |
| Native execution | One FIFO row at a time, all 30 rows before replanning. RTC disabled by saved configuration. No Molmo interpolation or generic asynchronous queue. |
| Model precision | Native `bfloat16` configuration with upstream mixed-precision layers retained, not a blanket cast of all parameters. |

Feature/order/action settings come from the pinned checkpoint's `config.json`,
model card and saved processors. The associated public dataset's
[pinned metadata](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/blob/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c/meta/info.json)
confirms the same 14 feature names and 30 Hz. Its training config labels the
training dataset revision `local`; it does **not** identify an immutable training
dataset snapshot. Matching public metadata is mapping evidence, not proof of
the exact training-data revision or physical calibration.

## Native processing stays with the policy

The actual saved preprocessor sequence is rename → batch → disabled relative
actions → saved QUANTILES normalizer → PI05 state-to-prompt → PaliGemma tokenizer
→ device. State/action `q01`/`q99` statistics are loaded from the checkpoint;
there are no invented YAM statistics. Native state tokenization discretizes the
normalized state into 256 bins and appends it to the task. Native image processing
resizes with padding to 224×224 and maps [0,1] pixels to [-1,1].

The saved postprocessor applies QUANTILES unnormalization, disabled absolute
conversion, then CPU transfer. There is no saved output clamp. A separate
robot-host adapter projects only gripper values within the conservative raw
[-0.01,1.01] envelope to physical [0,1] endpoints, with raw/executed values logged.
Larger gripper excursions, nonfinite/shape errors and out-of-bound joints still
cause rejection. The envelope is a yamkit engineering guard, not an upstream
tolerance; it is never automatically widened.
Whole-chunk postprocessing is tested bit-for-bit against the same native
per-row postprocessor. Both relative conversions are disabled, so no moving
state anchor can alter queued rows.

References are LeRobot 0.6.1, commit
`7e241bd630a3719a56157a497ce5d08f244784f1`:
[PI05 model and FIFO](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/policies/pi05/modeling_pi05.py),
[PI05 processors](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/policies/pi05/processor_pi05.py),
[quantile arithmetic](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/processor/normalize_processor.py).

Native `select_action` consumes its FIFO before calling `predict_action_chunk`
again. The independent remote executor preserves those rows and queue semantics.
It observes measured state/cameras every tick; only the empty-FIFO observation
reaches the model, not Molmo's last-command state.
It does not infer the next chunk early or drop a prefix to conceal latency.
The pinned [LeRobot BaseStrategy loop](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/rollout/strategies/base.py)
starts each tick **before** observation, inference and sending, then sleeps only
the remainder of 1/30 second. The adapter follows this ordering exactly. A slow
inference tick has no added post-send wait: the next queued row follows the next
observation and may be less than 1/30 second after the first row. There is no
accumulated global-deadline catch-up loop. Fake-clock tests invoke the actual
native runner and PI05 FIFO methods, comparing every observation/send timestamp
and row for ordinary ticks, inference overruns, camera overruns and send overruns.
This proves control-cadence parity for identical supplied operation costs, not
equal real GPU/network latency. Network and safety adaptations are explicit: ≤2 s RPC, bounded
session, Stop invalidation, configured bounds and release. Normal duration/Stop
can leave an unexecuted tail; every such row is reported, not counted completed.

### Bounded-phase tail admission, version 1

A concrete version-2 software-only 5-second run completed three full chunks
(90 rows), then started another RPC with less than its fixed 2-second request
budget remaining. That request reached the shortened phase deadline and failed;
release, recording and private upload succeeded, but the nonzero rollout result
was correct. It is not reclassified as success. Preserved run:
`outputs/ui/deployments/software-fake-f73fc0645ff0463eb7393eb7006dce43`, private-HF
revision `8d5292f1e01d17fe298519821488caa82b6973a5`.

Contract version 3 explicitly admits a new RPC only when its existing fixed
`rpc_timeout_s` budget still fits, **after at least one complete FIFO chunk**.
At a completed chunk with less time remaining, the existing 30 Hz loop continues
observing and checking Stop/session state until normal phase completion, but
does not request a new chunk or send another policy target. Equality still admits
the RPC. The previous validated target is untouched; no new state/target checks,
extra observations, repeated sends, joint shaping or reduced RPC budgets are
introduced. Each observation consumes the existing tick budget, so a slow read
has no added sleep or catch-up burst. Existing observation exceptions and Stop
or session failures retain their existing semantics; healthy completion follows
the normal home/release path, while Stop/fault releases without home.

This is an intentional yamkit bounded-RPC admission difference from the unbounded
native BaseStrategy near the end of a phase, not a native model or FIFO optimization.
No existing queued row is dropped or reordered by this rule. Initial short runs
retain their original first-RPC behavior, and **every actual RPC timeout remains
a fault**, including one after previously completed chunks. A partial first FIFO
at duration/Stop is still explicitly accounted rather than called a full chunk.

Metrics and traces identify `bounded_phase_tail=completed_fifo_phase_tail_v1`:
`phase_tail_wait_ticks` counts existing observation ticks admitted to the tail,
`phase_tail_wait_s` measures actual time in the existing tick wait (including its
checks, excluding observation time), and `phase_tail_requests_avoided` records
the single terminal admission transition, not one invented RPC per waiting tick.
`phase_tail_wait_started` includes the decision time and original tick-start time.
Unchanged full-chunk cases retain exact native row/observation/send timing parity
and zero tail counters. Fake boundary/overrun/Stop/expiry tests are software
evidence only. The current version-3 build received the fresh qualification
reported below; future identity changes still require renewed qualification.

## Concrete native-loader hardening

The upstream PI05 `from_pretrained(strict=True)` catches weight-loading exceptions
and can return an unloaded or partially restored model. This adapter instead uses
the same native constructor and key-remapping helper, then calls strict
`load_state_dict` with failure propagation. It never silently uses random weights.
An explicit eager variant disables only the optional native `torch.compile`
setting; denoising count, preprocessing, normalization, dtype configuration and
raw model rows are unchanged. Earlier eager latency samples are retained in the
authorization follow-up, but do not constitute a passing qualification.

The PaliGemma tokenizer is pinned at
`google/paligemma-3b-pt-224@35e4f46485b4d07967e7e9935bc3786aad50687c`.
The user must accept the publisher terms with the downloading Hugging Face
account. The current deployment used the existing authorized Lenovo login, then
privately transferred and SHA-256-verified only the nine pinned tokenizer/config
assets into the GPU checkout's repository-local cache. Its normal pinned snapshot
loader succeeds from that cache; no HF token was transferred. Alternatively,
authenticate the GPU checkout using `yamkit hub login`. Never paste tokens into
chat, source, a rig file or a command argument. No alternate tokenizer is
substituted and no access gate is bypassed.

## Software qualification and future physical admission

`yamkit.pi05.qualification.collect_qualification` takes saved observations and
fake arms only. Each NPZ contains measured `state[14]` and original HWC uint8
`top`, `left_wrist`, `right_wrist` RGB arrays; loading disables pickle. It collects
50 direct warm calls, 50 complete fake FIFO chunks and a Stop-during-request
proof. It retains every direct native chunk for replay, all row counts, faults,
modified-command/coherence counts, execution rate, RPC p50/p95/max and observation
age. Qualification uses the actual robot host's rig and saved source hashes;
every saved RGB payload must match the configured per-camera dimensions before
any qualification RPC. Tiny fixtures cannot qualify full-resolution transport.
The exact camera set/rate and reviewed YAM/LINEAR_4310 mapping are checked
passively. Qualification never captures a new camera frame. Runtime/session/task/rig/source changes
invalidate physical admission. The p95 RPC budget is 1.6 s (20% margin on the
2 s request limit), explicitly separate from the one-second nominal action chunk.

The passing version-3 Lenovo record is
`.context/pi05-qualification/ebbb44c231ee4704a801204bb46894d1/qualification.json`
under `/home/andre/rohan-new`. It binds runtime build
`ebec79a9a5ff27333a572193df378359391a86707cc852d8527dad48665e81ec`, instance
`9d8df83c-9bd3-4547-9cc2-9a787f81446a`, and session expiry
`1789330739.4388595` (2026-09-13 20:18:59 UTC). Actual UI software preparation
completed in 212.24 s with exit 0 / ready true; motion approvals remained false.
Direct warm p50/p95/max were 0.4358435749891214 / 0.5070293031225447 /
0.50993867701618 s; one direct gripper scalar projected by 0.0018082857131958008.
All 50 integrated chunks / 1,500 rows completed at
29.924748810782212 Hz, with zero errors, unexpected drops, SDK modifications,
coherence violations or interpolation. One integrated gripper scalar projected
by 0.00006818771362304688. Integrated RPC p95 was 0.699393955245614 s and
observation-age p95 1.5480936542036943 s. In the separate Stop-in-flight probe, zero commands
were sent after Stop, all fake arms released and the 30 returned rows were
accounted as intentionally unused.

The successful five-second fake CLI run saved
`outputs/ui/deployments/software-fake-1644f54e4c5945cdbd39d7ac5fd64b72` and
completed its full capture/export/private-upload pipeline in 48.99 s. Its two
chunks / 60 rows and 107 observations had zero errors, drops or gripper
projections. The bounded tail recorded 47 observation ticks, 1.53967 s of existing
tick wait and one avoided terminal RPC. Fake resources released before export;
all three videos (107 original frames each, 5.00802 s), timelines and preserved
originals were checked, and all 335
files at private-HF revision `9e2f1741a4fc02818c792a2803ea6c5f81361500` were
downloaded and hash-verified. Lenovo verification:
`.context/overnight/pi05-artifact-verification-20260913/result.json`. Actual UI
recording/video header and byte-range routes also passed safe GET checks in
`.context/overnight/ui-playback-20260913T122814Z-2d20788e/result.json`; this is
saved-media verification, not a hardware-camera test.

The historical pre-tail-guard version-2 Lenovo record remains at
`.context/pi05-qualification/17c40a159fe64227aba5942c3272ae68/qualification.json`
under `/home/andre/rohan-new`, with `qualified=true` and `reasons=[]`. It binds
runtime build
`8bd160030ed010a9614063d8ed946982c0ce72e4297cb67a2eff76684229b921`
and instance `2e6e30a8-6d8b-4c1a-835a-405e5bb6cccf`. Direct warm latency was
p50 0.41705183550948277 s, p95 0.428424281970365 s and max
0.48977454798296094 s. The 50 integrated chunks completed all 1,500 rows at
29.925063124292105 Hz, with zero unexpected drops, SDK modifications, coherence
violations or faults. Direct samples required three explicit gripper-value
projections (maximum 0.001585245132446289); integrated samples required 31
(maximum 0.0032303333282470703). These documented endpoint projections are
accounted separately from unexpected command modifications; raw model rows are
retained. This record remains bound to its runtime/session/task/rig/source and
expiry, not permanent readiness.

The independent physical adapter is implemented but only fake-tested. It checks
current native qualification and explicit mapping/supervision flags before
constructing a robot, including a fresh remaining-lease margin for bounded
startup/return home after any delayed terminal confirmation. Existing rig
validation and enabled startup home must pass before qualification. It reuses the existing YAM plugin's cooperative ownership,
two-arm bounds/measurement checks, startup home/open, direct target dispatch and
release. Healthy completion homes preserving the final measured gripper opening;
Stop/fault releases without home or retries. JSON trace/report saving happens
after release. Native local recordings, three-video playback, UI history and
private-HF packaging/upload are implemented and verified through the native
real-GPU, fake-hardware recorded run above, independently of MA2 recording.
Reports distinguish attempted rows from complete receipts: a failed bimanual
SDK call records an unknown partial dispatch, because one arm may have received
its target even if the other arm failed. Zero completed rows is not proof of
zero hardware commands.

Failed qualification retains bounded, exact numeric output plus the phase,
zero-based sample/request indices, column ranges and gripper-bound violations.
Diagnostic copying occurs only after failure, outside measured execution. No
arbitrary response/error strings are copied; nonfinite values have JSON-safe
markers rather than fabricated finite actions. The CLI identifies the offending
gripper and saved report path. Diagnostics do not change acceptance, rows or timing.

Fake/native tests do not validate dynamics, calibration, camera placement or task
success. Version-3 real-GPU qualification and recorded fake-run artifact
verification passed with complete row accounting and explicit raw-versus-projected
action evidence. `hardware_tested=false` and physical task success remains
unproven. Qualification is bound to runtime/session/task/rig/source and expiry;
failed or stale qualification must remain visible. No software result replaces
fresh on-site supervision,
mount/stop verification and exact-command approval before any physical run.
