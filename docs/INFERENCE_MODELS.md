# Pinned inference models and mapping evidence

The catalog is offline. Merely listing models does not download weights, contact Modal,
open cameras, or connect arms. The catalog distinguishes a successful native forward
pass from a physical YAM mapping. Molmo's source conventions were reviewed; no supervised
physical mapping, calibration or camera validation was performed. **Physical Modal rollout
requires a current passing qualification on the actual robot host**, mapping acceptance and
supervised confirmation. No passing qualification has been demonstrated; cloud workspaces
and browser Modal Start remain blocked. A probe or ready service cannot replace these checks.
See [current latency and qualification results](MODAL_LATENCY.md) and
[the performance gate](REMOTE_PERFORMANCE.md).
See [staged acceptance](acceptance-test.md) for available checks and command effects.

| Profile | Checkpoint revision (model and saved processors) | Native state/action | YAM deployment |
|---|---|---|---|
| `smolvla` | [`c83c3163b8ca9b7e67c509fffd9121e66cb96205`](https://huggingface.co/lerobot/smolvla_base/tree/c83c3163b8ca9b7e67c509fffd9121e66cb96205) | 6 / 6; chunk 50 | No verified physical mapping; diagnostic inference only |
| `molmoact2` | [`fdade02d1f1c1dd819114b0478f735072fb6b212`](https://huggingface.co/lerobot/MolmoAct2-BimanualYAM-LeRobot/tree/fdade02d1f1c1dd819114b0478f735072fb6b212) | 14 / 14; chunk 30 | Source mapping reviewed; local sync physically unvalidated; Modal requires passing host qualification and supervised mapping acceptance |
| `pi05` | [`b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba`](https://huggingface.co/lerobot/pi05_base/tree/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba) | 32 / 32; chunk 50 | No physical YAM mapping or saved robot statistics; diagnostic inference only |

No physical vector is padded or truncated. The native fixture keeps the checkpoint's
original feature dimensions, camera names and saved statistics. Where the base checkpoint
provides no joint names, `native_state_N` and `native_action_N` are diagnostic position
labels, not physical joint identities. Base SmolVLA and pi0.5 outputs
are labeled `checkpoint_native`, never radians or YAM gripper units. Custom checkpoints
remain supported by the existing local path; the Modal catalog accepts reviewed profiles.

Nested assets are pinned too: SmolVLM2 processor/config
`7b375e1b73b11138ff12fe22c8f2822d8fe03467`, MolmoAct2-BimanualYAM backbone/processor
`8dcbed66f2380e4393189c303ea72488eb9e63c2`, and PaliGemma tokenizer
`35e4f46485b4d07967e7e9935bc3786aad50687c`. They are resolved to immutable local
snapshot paths before LeRobot constructs the model/processors. Remote code from model
repositories is not executed: LeRobot's packaged model implementation is used.

## MolmoAct2-YAM schema

The inspected dataset is `allenai/MolmoAct2-BimanualYAM-Dataset` at
[`e9f21ae15074330839f2ac25ed4b49d76dfa1f9c`](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/tree/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c).
Its [info.json](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/blob/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c/meta/info.json)
declares `bi_yam_follower`, 30 fps, three 640×360 RGB cameras, and identical ordered
state/action names:

```
left_joint_0.pos … left_joint_5.pos, left_gripper.pos,
right_joint_0.pos … right_joint_5.pos, right_gripper.pos
```

The explicit bidirectional map changes each dataset `joint_0..5` to yamkit
`joint_1..6`. It retains left-then-right vector positions. Neither vectors nor names
are alphabetically sorted. `gripper` remains element 6 and 13 (zero-based).

The [checkpoint config](https://huggingface.co/lerobot/MolmoAct2-BimanualYAM-LeRobot/blob/fdade02d1f1c1dd819114b0478f735072fb6b212/config.json)
declares absolute joint pose control, no joint sign/offset transforms and
`normalize_gripper=false`. The collection implementation linked by Ai2 uses
[`YAMRobot`](https://github.com/williamtsai726/YAM/blob/9a46f908d5dfb8a999f9a2c236681bf2af74f4da/gello_software/gello/robots/yam.py),
which reads `get_joint_pos()` and submits `command_joint_pos()` directly to I2RT with
`GripperType.LINEAR_4310`. The same adapter exists in
[Ai2's inference example](https://github.com/allenai/molmoact2/blob/66b87e64efd99dfd103241418113955cf64dfa9c/examples/yam/gello_min/yam.py).
These calls carry six absolute angular joint positions and a normalized gripper value.

Gripper direction is established by the SDK implementation, rather than inferred from
dataset extrema. The pinned
[I2RT motor-chain source](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robots/motor_chain_robot.py)
defines gripper limits as `[closed, open]` and initializes command `1` as fully open.
Its [JointMapper](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robots/utils.py)
converts command `x` to `closed + x * (open - closed)`. Thus the documented adapter
contract is 0 closed / 1 open, matching yamkit. Different gripper mechanisms or
unverified motor zero frames require a separately reviewed mapping. Collection source
does not prove that this user's individual calibration or camera geometry matches.

The [dataset stats](https://huggingface.co/datasets/allenai/MolmoAct2-BimanualYAM-Dataset/blob/e9f21ae15074330839f2ac25ed4b49d76dfa1f9c/meta/stats.json)
have action gripper extrema 0–1. Joint extrema describe recorded samples, not physical
joint limits. The saved pre/postprocessor safetensors have identical 14-element masks
`[1,1,1,1,1,1,0,1,1,1,1,1,1,0]`; joint quantile normalization excludes both grippers.
The inspected action and state q01/q99 values match the dataset metadata. Serving uses
the checkpoint's saved state, not a fresh statistics calculation or synthetic override.

## Camera and numerical processing

The single LeRobot rename processor runs on the server:

| Rig/recording name | Model camera |
|---|---|
| `top` | `top` |
| `left_wrist` | `left` |
| `right_wrist` | `right` |

Recordings and the rig retain their names. The client sends RGB arrays with the original
rig names. Native fixtures already use model names and use a separate saved preprocessor
with its original empty rename map.

The Molmo nested [saved image processor](https://huggingface.co/allenai/MolmoAct2-BimanualYAM/blob/8dcbed66f2380e4393189c303ea72488eb9e63c2/processor_config.json)
specifies `crop_mode=resize`, bilinear sampling and 378×378 images with channel
mean/std 0.5. This differs from the generic 224×224 feature placeholder in the converted
policy config. By default a 640×480 rig frame goes through that saved processor intact.
The optional `center_16_9` transform crops rows 60–419 of a 640×480 frame, producing
640×360 before saved resizing. That removes the top and bottom 60 rows and changes the
vertical field of view; it does not recover the training cameras' intrinsics, extrinsics,
placement or perspective. It is disabled by default and logged identically for checks,
probes and rollout. It never changes recording settings.

Serving performs saved preprocessor → `predict_action_chunk` → saved postprocessor.
It never calls cached `select_action` pops. Returned processed chunks are explicitly
`robot` units for the source-mapped profile; those units do not authorize execution.
The remote policy's client processors are numerical
identity with the required batch/device/schema handling; there is no second normalization.

Molmo's saved postprocessor clamps normalized actions to [-1,1] before masked quantile
unnormalization. Probes additionally return `unclipped_chunk`, constructed by a separate
LeRobot pipeline containing the same saved numerical/frame/device transformations while
omitting only that saved clamp. These diagnostic robot-unit targets can expose excursions
hidden by the saved clamp. They are never selected for execution. Probe extrema are
measured before that clamp and before any local robot speed or gripper clipping.

## Service and support boundaries

The CUDA dependency graph is separately pinned in `configs/modal-requirements.txt`;
the local CPU `uv.lock` is unchanged by the CUDA build. It uses LeRobot 0.6.1,
PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and Transformers 5.5.4. Regenerate with:

```bash
UV_CACHE_DIR="$PWD/.uv-cache" .tools/uv pip compile --no-config --python-version 3.12 \
  --python-platform x86_64-unknown-linux-gnu --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 --no-annotate --no-header \
  --emit-index-url -o configs/modal-requirements.txt configs/modal-requirements.in
```

One fixed-profile Modal class owns one L40S pool. It has no parameterized constructors,
minimum 0, maximum 1, buffer 0, and one concurrent input. Default allocated CPU/RAM is
4 cores/64 GiB to accommodate upstream Molmo loading its nested backbone followed by
converted LeRobot weights. This upstream constructor entails downloading both weight
sets; failed loading is reported as a dependency/access/resource failure, not successful
inference. Model weights persist in a dedicated HF cache volume; no observations are
written there. Production idle scale-down is configurable from 300–600 seconds. Tests
cap idle at 15 seconds, startup at 240 seconds, requests at 90 seconds and retries at 0.
Every RPC is authenticated through the outbound SDK. Only HF_TOKEN is forwarded using
Modal Secrets; no public inference endpoint, robot server, VPN or QUIC transport exists.

Every chunk starts from reset policy/processor state under a lock. IDs and ordered schema
are checked; duplicate/out-of-order and retired sessions are rejected. Session bookkeeping
is bounded; exhaustion requires a pool restart. Local monotonic timing is carried opaquely
and compared only on the client host. Server timing records local durations separately.
Saved observations carry their age diagnostically; live/motion requests must be fresh.

The protocol bounds RGB to three images, each at most 1280×720, with exactly three bytes
per pixel and at most 8,294,400 RGB bytes per request. A typical three-camera 640×480
payload is 2,764,800 bytes. Modal's [data residency documentation](https://modal.com/docs/guide/data-residency)
states that all spawned SDK payloads, regardless of size, use centralized US storage;
synchronous payloads over 2 MiB do too. The v0 cancellable spawned SDK calls therefore
retain `routing_region=us-east`. Reducing image size alone cannot change that storage
behavior. The [region documentation](https://modal.com/docs/guide/region-selection)
also restricts non-default routing to synchronous `remote`/`map` calls.
Compute region is separately configurable and returned in metadata. Specifying a compute
region adds a published multiplier (1.15× broad, 1.75× narrow); measure total latency from
the actual robot host before drawing conclusions about queue coverage.

Guided RTC is deliberately unsupported in this transport. In the actual LeRobot 0.6.1
wheel, native SmolVLA and pi0.5 declare RTC support, and native MolmoAct2 declares it
when `inference_action_mode` is `continuous`. Molmo's continuous implementation applies
`rtc_processor.denoise_step` inside its flow-matching loop; discrete mode does not
support that path. The remote proxy's `supports_rtc=False` and rejected continuation
describe this integration's unvalidated guidance boundary, not the native models' capability.

For the pinned absolute-action Molmo checkpoint, a robot-unit prefix would need the
same joint frame and masked quantile normalization as model actions. Its saved
preprocessor contains no relative-action step. Checkpoints with relative actions would
additionally require conversion to the current observation's anchor. Those conversions,
end-to-end delay guidance and their interactions have not been validated across this
transport. The service rejects continuation rather than sending raw robot units into a
denoiser. An unguided background chunk worker is not advertised as guided RTC.
No inference-mode wrapper is imposed around denoising; future verified guidance must retain
upstream gradient requirements. There are no pause heartbeats or permanent keepalives.

[Historical C validation](MODAL_VALIDATION.md) records actual model forward passes and
two warm RPC samples per model. [Integrated performance measurements](REMOTE_PERFORMANCE.md)
use fake hardware/RPC and preserve saved processors; their larger synthetic sample
count is not additional real-service evidence. Native bilateral teleop remains available;
recording and LeRobot teleoperation require yamkit's [operator wrappers](OPERATOR_PARITY.md)
and reject nonzero bilateral feedback.
