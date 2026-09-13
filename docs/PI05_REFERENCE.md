# YAM π0.5: independent native reference execution

Status on 2026-09-13: source contract reviewed and fake/native-method tests pass.
The existing GPU download obtained the 9.35 GB YAM weights, but model startup is
blocked by publisher-gated access to the pinned PaliGemma tokenizer. No π0.5
GPU forward pass, real-model qualification, physical trial or manipulation success
is claimed. Do not launch the physical entrypoint until qualification passes.
The working MolmoAct2 path remains separate and unchanged.

## Immutable source contract

Checkpoint: [Jiafei1224/molmoact2-yam-pi05](https://huggingface.co/Jiafei1224/molmoact2-yam-pi05/tree/51ab2720d7e56d51410407f98ea64bbea97feb2e),
revision `51ab2720d7e56d51410407f98ea64bbea97feb2e`.
Weights SHA-256: `a777861c627234f9aa54a1bb7bdee29101ee6513f4773ef0a581d9c5527981e4`.
Profile ID `pi05-yam`; execution contract `pi05_reference`. The old low-level
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
conversion, then CPU transfer. There is no saved output clamp. Native values
outside physical joint/gripper bounds cause rejection; they are not clipped.
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
It samples measured state at the chunk boundary, not Molmo's last-command state.
It does not infer the next chunk early or drop a prefix to conceal latency.
Its interruptible per-row 30 Hz waits never catch up with command bursts after
an overrun. Network and safety adaptations are explicit: ≤2 s RPC, bounded
session, Stop invalidation, configured bounds and release. Normal duration/Stop
can leave an unexecuted tail; every such row is reported, not counted completed.

## Concrete native-loader hardening

The upstream PI05 `from_pretrained(strict=True)` catches weight-loading exceptions
and can return an unloaded or partially restored model. This adapter instead uses
the same native constructor and key-remapping helper, then calls strict
`load_state_dict` with failure propagation. It never silently uses random weights.
An explicit eager variant disables only the optional native `torch.compile`
setting; denoising count, preprocessing, normalization, dtype configuration and
rows are unchanged. Eager performance is not yet measured.

The PaliGemma tokenizer is pinned at
`google/paligemma-3b-pt-224@35e4f46485b4d07967e7e9935bc3786aad50687c`.
The user must accept the publisher terms with the runtime's Hugging Face account
and authenticate that GPU checkout using `yamkit hub login`. Never paste tokens
into chat, source, a rig file or a command argument. No alternate tokenizer is
substituted and no access gate is bypassed.

## Software qualification and future physical admission

`yamkit.pi05.qualification.collect_qualification` takes saved observations and
fake arms only. Each NPZ contains measured `state[14]` and original HWC uint8
`top`, `left_wrist`, `right_wrist` RGB arrays; loading disables pickle. It collects
50 direct warm calls, 50 complete fake FIFO chunks and a Stop-during-request
proof. It retains every direct native chunk for replay, all row counts, faults,
modified-command/coherence counts, execution rate, RPC p50/p95/max and observation
age. Qualification uses the actual robot host's rig and saved source hashes;
it never captures a new camera frame. Runtime/session/task/rig/source changes
invalidate physical admission. The p95 RPC budget is 1.6 s (20% margin on the
2 s request limit), explicitly separate from the one-second nominal action chunk.

The independent physical adapter is implemented but only fake-tested. It checks
current native qualification and explicit mapping/supervision flags before
constructing a robot. It reuses the existing YAM plugin's cooperative ownership,
two-arm bounds/measurement checks, startup home/open, direct target dispatch and
release. Healthy completion homes preserving the final measured gripper opening;
Stop/fault releases without home or retries. JSON trace/report saving happens
after release. PI video capture/HF playback integration has not been qualified
and must not be presented as ready merely because MA2 recording works.

Fake/native tests do not validate real-model output, dynamics, calibration,
camera placement or task success. GPU latency and real-model row accounting are
currently unavailable because startup is gated. The user-facing workflow must
show that blocker instead of offering an apparently ready physical PI run.
