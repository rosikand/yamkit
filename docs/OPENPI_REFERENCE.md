# Official OpenPI π0.5 base: normalized-space diagnostics, YAM motion blocked

`pi05_base` means the genuine Physical Intelligence checkpoint—not `lerobot/pi05_base`
and not the separately supported `Jiafei1224/molmoact2-yam-pi05` fine-tune. This path
preserves official weights and the original JAX implementation. It cannot currently
produce justified physical YAM commands. **Loading, finite outputs, native-array parity,
or fake replay must not be reported as YAM qualification or manipulation success.**

The official README identifies the base checkpoint as a pretraining/fine-tuning model;
the provided directly deployable examples use robot-specific configurations. There is
no named `get_config("pi05_base")` in the pinned configuration registry.
[Official checkpoint catalog](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#model-checkpoints),
[official configurations](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py).

## Pinned identity and isolation

| Component | Fixed identity |
| --- | --- |
| Runtime | `Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479` |
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_base` |
| Checkpoint acquisition | 29 generation-pinned GCS objects; 12,441,749,581 bytes |
| Tokenizer | Official `gs://big_vision/paligemma_tokenizer.model`, generation `1711547605575873` |
| Tokenizer acquisition | 4,264,023 bytes; published MD5 `FCCtyYVnIKVZ6KhyhLGV4g==` |
| Manifest | [pi05_base_manifest.json](../src/yamkit/openpi/pi05_base_manifest.json), SHA-256 `439423350a9160e2157291aec1a48e8454193c7126e24fda670df39bb7c503db` |
| Upstream lock | SHA-256 `793488b5a55bb87200db90a61fd0af51922b686d94e1da4f4c587ab119b37d74` |
| Interpreter | Repository-local Python 3.11.13 |
| Environment | `data/openpi/venv`, upstream frozen `uv.lock`, separate `data/openpi/uv-cache` |
| Source/cache | `data/openpi/upstream`, `data/openpi/cache`; no MA2 dependency changes |

Each object GET explicitly selects its immutable GCS generation. The downloader checks
the returned generation, byte length, and published MD5 where supplied, then records
SHA-256 for every object—including composite objects without an MD5. It rehashes the
complete cache before loading. A changed/unreceipted file or failed partial is preserved
and rejected, never silently overwritten. Public GCS assets require no HF token.
The full manifest and per-object SHA-256 acquisition receipt, rather than the mutable
bucket prefix alone, identify an exact checkpoint. Runtime imports must resolve to the
clean, pinned isolated checkout.

`setup_openpi_inference.sh` uses upstream's locked JAX environment, including JAX 0.5.3,
Flax 0.10.2 and Orbax 0.11.13. It does not convert weights to PyTorch, patch transformers,
modify `.venv-inference`, start a server, or open a device. JAX's bulk GPU-memory
preallocation is disabled for diagnostics; this changes allocation, not model math.
[Official environment](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/pyproject.toml).

## Exact native inference being exercised

The diagnostic creates **`Pi0Config(pi05=True)` with its original defaults**:
`gemma_2b` vision-language model, `gemma_300m` action expert, bfloat16, 32 action/state
dimensions, 50 predicted rows, 200 prompt tokens and discrete state input. Parameters
are restored with the official Orbax loader; its native tree/shape validation must pass.
The flow matcher uses its unchanged default ten denoising steps. The original JAX
`Policy` and JIT wrapper perform inference; no LeRobot π0.5 implementation participates.
[Native model configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py),
[native sampling](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py),
[native loader](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/model.py),
[native policy](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/policy.py).

Saved camera RGB is assigned explicitly for **image-only diagnostics**:

| Saved key | Native image role |
| --- | --- |
| `top` | `base_0_rgb` |
| `left_wrist` | `left_wrist_0_rgb` |
| `right_wrist` | `right_wrist_0_rgb` |

This is a declared view-role correspondence, not evidence of matching training camera
extrinsics. Full HWC uint8 RGB reaches native `ResizeImages(224,224)`, which resizes with
padding. `Observation.from_dict` performs the original conversion to float RGB in
`[-1,1]`; all three image masks are true. Native `ModelTransformFactory` supplies prompt
handling, SentencePiece tokenization and state padding. Prompt text is cleaned and state
discretized by the original tokenizer, including its original 200-token limit.
[Model transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py),
[image transforms](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/transforms.py),
[tokenizer](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py).

**Recorded YAM joint/gripper states are deliberately not passed to the model.** The
diagnostic substitutes a clearly labeled synthetic 32D normalized zero vector, because
there is no justified YAM normalization or coordinate mapping. No normalization assets
are used and no output unnormalization occurs. All 50×32 raw model values are retained,
including finite values outside `[-1,1]`; none are clipped, projected or labeled radians.
There is no action dispatcher and no fake or real YAM rollout.

The report records native configuration, actual installed package versions, input/output
transforms, cold latency and warm p50/p95/max. Two identical-noise calls compare the thin
wrapper with direct native `Policy.infer`, preserving both entire chunks and input noise.
This proves the diagnostic wrapper does not alter native arrays; it does **not** prove
an embodiment adapter or task competence. Subsequent calls retain the official RNG path.

## Specific blockers to official base on YAM

The checkpoint's actual asset listing contains only `arx`, `arx_mobile`, `droid`,
`fibocom_mobile`, `franka`, `trossen`, `trossen_mobile`, `ur5e` and `ur5e_dual` normalization
assets. **There is no YAM asset.** Reusing an asset requires a documented correspondence
to its original robot/action space; matching a vector's dimension is insufficient.
[Pinned asset manifest](../src/yamkit/openpi/pi05_base_manifest.json),
[official normalization guidance](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/docs/norm_stats.md).

The supplied `pi05_aloha` configuration is not a generic dual-arm/YAM configuration. It
uses Trossen assets, ALOHA-specific joint sign changes and gripper linkage/angle
conversions, delta joint preprocessing and absolute-action reconstruction. Its output
adapter's first-14 slicing is justified for that embodiment; copying it alone would not
justify YAM semantics. YAM's current recorded gripper convention is 0 closed / 1 open;
the upstream generic Pi conventions describe 0 open / 1 closed, and the ALOHA adapter
contains additional device-specific transformations. These cannot be replaced with a
guessed sign flip or borrowed YAM-finetune statistics.
[ALOHA adapter](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/policies/aloha_policy.py),
[configuration](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/training/config.py).

Execution is also embodiment-specific. The official ALOHA example commits 25 rows at
a 50 Hz maximum loop rate before replanning, although the selected model predicts 50.
Its broker intentionally leaves the other horizon rows unused. Other examples use
different horizons/rates. Neither MA2's 30-row literal interpolation nor the YAM
LeRobot fine-tune's 30-row FIFO establishes the official base YAM contract. No execution
rate, chunk commitment, interpolation or physical absolute/delta interpretation is
invented for this diagnostic.
[ALOHA runtime selection](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/aloha_real/main.py),
[native chunk broker](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/packages/openpi-client/src/openpi_client/action_chunk_broker.py).

Before physical use, evidence is required for YAM normalization, training-coordinate
joint zero/sign/order, gripper units/direction/range, absolute/delta reconstruction,
camera roles and execution timing/commitment. Zero-shot use would remain experimental
even after that evidence and software qualification. The physical CLI must fail through
`yamkit.openpi.contract.require_yam_contract()` before hardware is opened; task success
and repeatability can only be assessed in later explicitly approved physical trials.

## Software-only commands

Run on the already authorized GPU checkout, with no other OpenPI process using this
environment while installation is in progress. These commands do not start robot
hardware or a model server:

```bash
bash scripts/setup_openpi_inference.sh
PYTHONPATH=src data/openpi/venv/bin/python -m yamkit.openpi.assets
PYTHONPATH=src data/openpi/venv/bin/python -m yamkit.openpi.diagnostic \
  --saved-observation .context/saved-observations/observation-000.npz \
  --task 'put the red cube into the black container' \
  --requests 5 --output .context/openpi-base-diagnostic-unique
```

Supply existing repo-local NPZ paths; repeat `--saved-observation` for additional saved
RGB triplets. Each NPZ contains `top`, `left_wrist`, `right_wrist`; an optional recorded
`state` is intentionally ignored. Calls are bounded to 1–50; output directories must be
new. Do not collect new camera frames for this workflow. On Ctrl-C, diagnostic evidence
is finalized where possible and interruption propagates. A successful process exit
means only **official normalized-space inference passed**, never physical readiness.

Hardware-free tests cover immutable assets, cache/sidecar symlink containment, no cache
or historical evidence overwrite, strict native shape/finite values, no output clipping,
image-only saved observations, native-wrapper parity, budget limits, error sanitization,
Ctrl-C evidence, and an unconditional physical block. Actual GPU timings belong in the
dated deployment evidence; unit tests alone do not claim that the official model loaded.
