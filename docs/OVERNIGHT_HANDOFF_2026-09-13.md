# Overnight handoff — 2026-09-13

Real-GPU qualification, recorded fake CLI/private-HF/playback verification,
policy switching, the full regression suite and the final backend negative/recovery
checks passed. The final read-only Lenovo report at **2026-09-13 12:31:15 UTC**
confirms software readiness. MA2 and YAM π0.5 are READY for a fresh supervised
trial; official π0.5 base remains BLOCKED for YAM execution. This is not a physical
task-success or bug-free guarantee.
No physical robot operation occurred overnight. Previous GOs are historical and
consumed; there is no current motion approval.

## What to run when back at the Lenovo

The existing host/backend installation is configured. In a new Lenovo terminal,
activate its repository-local environment once for that shell:

```bash
cd /home/andre/rohan-new
source scripts/env.sh
```

No Mac, new VM, separate prepare command, manual model-server startup or per-run
environment repair is required. Existing authorized credentials stay in private
repo-local files. Do not print or copy their contents into commands or reports.

| Requested command | Software status | Remaining requirement |
| --- | --- | --- |
| `molmoact2` | READY — software-qualified | Fresh on-site verification and one explicit confirmation; current admission must still pass. |
| `pi05_yam` | READY — software-qualified | Same supervision/admission requirements; first physical validation of this native path remains outstanding. |
| `pi05_base` | BLOCKED for YAM execution | Official inference works; justified YAM normalization, action/coordinate and execution-timing contracts are missing. |

For a READY policy, run one of these commands. The CLI
prepares or reuses the matching service, qualifies the exact task as needed, then
requests one fresh on-site confirmation before any motor activation:

```bash
yamkit rollout --backend lambda --policy molmoact2 \
  --task 'put the red cube into the black container' --duration 60

yamkit rollout --backend lambda --policy pi05_yam \
  --task 'put the red cube into the black container' --duration 60

# Intentionally BLOCKED before hardware until the official model's YAM contract exists:
yamkit rollout --backend lambda --policy pi05_base \
  --task 'put the red cube into the black container' --duration 60
```

Optional `--capture-trace` saves local video/trace/report artifacts. Adding
`--upload-repo-id rohanlux/yamkit-rollouts` also uploads a private archive after
release; local originals remain intact. The Inference UI at
`http://127.0.0.1:8400` on Lenovo shares backend preparation and execution APIs,
with sparse policy/task/duration fields and advanced settings collapsed. Choose
UI Start or terminal rollout, not both simultaneously; close UI camera previews
before terminal physical execution. See [the CLI guide](INFERENCE_CLI.md).

READY means software-qualified for a supervised trial, not guaranteed placement
or proof that no bugs exist. Verify mounts, arm/camera mapping, empty grippers,
clear movement area and reachable Stop/power cutoff; stay at the arms. Ctrl-C is
terminal Stop. Healthy completion returns home preserving final gripper opening,
then releases; Stop/fault releases without home or automatic retry. Export/upload
follows release. Never prefill approval flags in unattended commands or weaken a
failed memory, freshness, bounds or ownership admission check.

## Protected baseline and deployment identity

| Item | Recorded identity |
| --- | --- |
| Protected baseline tag | `overnight-baseline-20260913-6b348a8` |
| Baseline source SHA | `6b348a8673320b44899973d671ffca2c0c311652` |
| Work branch | `codex/overnight-three-policy-20260913` |
| Phase-tail implementation SHA | `1cacee93edcb827afe189fe02e0c3c7dad2125f7` |
| Final tested/deployed runtime source SHA | `078795acb018cae47e67a8dc66d3f577258dbb8a` |
| Frozen MA2 runtime build | `5128c7b55b6f9a62c109b581f98912de717c7d3746e8c6374341e91687d481fe` |
| Final PI contract-v3 build | `ebec79a9a5ff27333a572193df378359391a86707cc852d8527dad48665e81ec` |
| Final PI instance | `9d8df83c-9bd3-4547-9cc2-9a787f81446a` |
| PI session expiry | `1789330739.4388595` — 2026-09-13 20:18:59 UTC |
| MA2 instance | `2e3cac0b-0d6e-4704-92e7-60989b2dbd50` |
| MA2 session expiry | `1789326910.2363605` — 2026-09-13 19:15:10 UTC |

The tested runtime is followed by a documentation-only handoff commit; this
report does not use its own future commit hash as a runtime identity.

The initial remote snapshot is `.context/overnight/baseline-remote-state.json` in
the cloud workspace. Lenovo's Git HEAD was
`c87266e77c220799e0c02fa70a05682e2cdfaa00`; Lambda's was
`61388d942cd461d762a4bb4adecf26e161fdbd93`. Remote deployment overlays are tracked
by per-file manifests: remote Git HEAD alone is not deployed runtime identity.
The baseline Lenovo rig SHA-256 was
`6e8cbc785c94a48672404edc9d124734d4df93bd696b6b840c4d6cd325c75785`, and backend
configuration SHA-256 was
`3d31517d6ebd54b1003010aae94703b6554f3937be5505c6d3bfe6a7f7c4526c`.
These are baseline fingerprints, not claims that later additive backend
configuration has the same hash. Original configuration evidence remains private.

## Policy fidelity and results

### MolmoAct2: working model and controller preserved

Checkpoint `lerobot/MolmoAct2-BimanualYAM-LeRobot` stays pinned to
`fdade02d1f1c1dd819114b0478f735072fb6b212`, with dependency
`allenai/MolmoAct2-BimanualYAM@8dcbed66f2380e4393189c303ea72488eb9e63c2`.
Its existing LeRobot 0.6.1 environment, bf16 / ten-step `cuda_graph10` runtime,
three full 640×480 RGB images and no-crop input are unchanged.

`yam_upstream_literal_v1` still commits all 30 absolute 14-D model rows in order.
Each row uses inclusive full-14-D interpolation with
`n = min(int(max(abs(target-start)) / 0.01), 100)`; `n <= 1` sends the target
once. The order remains send → rate sleep → observe, with the existing extra
1 ms multipoint-row sleep and monotonic overrun reset. Replanning uses the last
committed 14-D command only after a complete chunk; measured-state monitoring is
separate. No prefix dropping, shaping, early inference or controller replacement
was introduced. Modal support remains separate and retained.

Real-GPU seeded comparisons against the protected baseline passed for seeds
17, 1337 and 20260913: every full native 30×14 chunk was bitwise equal, maximum
absolute difference 0, and runtime build remained the frozen value above.
Lenovo evidence:

- Baseline: `.context/day-mission/20260913/ma2-seeded-baseline/proof.json`.
- Comparison: `.context/overnight/ma2-seeded-final/proof.json`.
- Fresh qualification: `.context/inference-preparation/8486664244cf4ebe9282123c292f4cfe/qualification.json`;
  actual preparation CLI exit 0, 107.45 s. Direct warm p50/p95/max were
  0.352804964 / 0.414360184 / 0.472943454 s; integrated observation-age p95
  0.601141213 s. All 51 integrated chunks / 1,530 rows / 1,638 literal
  interpolation points completed with zero errors or drops. The separate Stop
  probe produced its one expected invalidated request, zero commands after Stop
  and release of all fake arms.
- Real-GPU five-second fake CLI recording:
  `outputs/ui/deployments/software-fake-1a06ddbca5104c689e6a5c19a55a26d0`;
  original trace `.context/rollout-traces/fe477f4b27ed417e9f42b8cc5f6525a2`.
- Private HF revision `bbb56ecdc79a88d081ed425a99490da3dfd1d8b0`: all 211
  allowlisted files downloaded and hash-verified, original files preserved, all
  three videos decoded (65 original frames per camera; 5.000074 s) and reference
  dispatches independently checked. Evidence:
  `.context/overnight/ma2-artifact-verification-20260913-v2/result.json`.

Fake SDK receipts and replayed scene images prove neither dynamics nor physical
task success. The recording uses the unchanged production recorder inside an
explicit fake-device context; its bounded observation-copy overhead is labeled.

### YAM π0.5: separate native controller, contract-v3 software evidence passed

Checkpoint `Jiafei1224/molmoact2-yam-pi05` is pinned to
`51ab2720d7e56d51410407f98ea64bbea97feb2e`; weights SHA-256 is
`a777861c627234f9aa54a1bb7bdee29101ee6513f4773ef0a581d9c5527981e4`.
Runtime is LeRobot 0.6.1 at `7e241bd630a3719a56157a497ce5d08f244784f1`.
The authorized PaliGemma tokenizer remains pinned to
`35e4f46485b4d07967e7e9935bc3786aad50687c`; there is no substitute tokenizer.

Native saved QUANTILES normalization, bf16 mixed-precision layers, ten denoising
steps and resize-with-padding to 224×224 remain intact. Measured state reaches
the model at an empty-FIFO boundary. Thirty absolute 14-D rows execute once each
at the native 30 Hz tick budget before replanning; there is no MA2 interpolation,
RTC or speculative prefix handling. The explicit endpoint adapter projects only
gripper columns 6/13 inside the conservative raw `[-0.01, 1.01]` envelope to
`[0, 1]`. Raw rows and every projection are retained separately from unexpected
SDK modifications; larger excursions and joint/finite/shape violations fail.

Contract-v2 qualification passed 50 direct calls and 50 integrated fake chunks:
1,500 executed rows, zero unexpected drops/modifications/coherence violations or
faults. Direct p50/p95/max were 0.417052 / 0.428424 / 0.489775 s; integrated rate
29.925063 Hz. Three direct gripper scalars projected by at most 0.0015852451 and
31 integrated scalars by at most 0.0032303333. This is retained historical evidence,
**not qualification of the newer contract**.

The subsequent real-GPU five-second fake recording exposed a bounded-phase
admission bug: after 90 correct rows, a fourth RPC started without its full two
seconds remaining and faulted at the phase limit. Release succeeded; fault did
not home. Three videos and its private archive were preserved at HF revision
`8d5292f1e01d17fe298519821488caa82b6973a5`. The failure remains a failure:
`.context/overnight/pi05-duration-boundary-failure.json` records it.

Version 3 adds the explicit `completed_fifo_phase_tail_v1` admission rule only
after a full FIFO: no new RPC starts when its full existing two-second budget
cannot fit. Existing 30 Hz observation/Stop checks continue to the phase end;
there are no new commands, discarded model rows or relabeled timeouts. Genuine
RPC failures still fault. The new build received its own renewed qualification.

Final Lenovo evidence is
`.context/pi05-qualification/ebbb44c231ee4704a801204bb46894d1/qualification.json`,
bound to the version-3 build/instance/expiry above. Actual UI software preparation
completed in 212.24 s with exit 0 and ready true; motion approvals remained false.
Its 50 direct calls had p50 0.4358435749891214 s, p95 0.5070293031225447 s and
max 0.50993867701618 s. One direct gripper scalar projected by
0.0018082857131958008. All 50 integrated chunks / 1,500 rows completed at
29.924748810782212 Hz, with zero errors, unexpected drops, SDK modifications,
coherence violations or interpolation. One integrated gripper scalar projected
by 0.00006818771362304688; raw/executed accounting is explicit. Integrated RPC
p95 was 0.699393955245614 s and observation-age p95 1.5480936542036943 s. The separate
Stop-in-flight probe sent zero commands after Stop and released all fake arms;
its 30 returned rows are intentionally unused, not falsely counted completed.

The successful five-second fake CLI completed its whole capture/export/private
upload pipeline in 48.99 s. It saved
`outputs/ui/deployments/software-fake-1644f54e4c5945cdbd39d7ac5fd64b72`, completed
2 chunks / 60 rows with 107 observations, and recorded 47 tail ticks / 1.53967 s
of existing tick wait / one avoided terminal RPC. There were no errors, drops or
gripper projections in this run. Fake release was confirmed before export/upload;
all three videos decoded (107 original frames per camera; 5.00802 s), their
timestamps and preserved original files checked,
and all 335 private HF files downloaded and hash-verified at revision
`9e2f1741a4fc02818c792a2803ea6c5f81361500`. Verification evidence:
`.context/overnight/pi05-artifact-verification-20260913/result.json`.
This is PI-specific end-to-end software evidence, not inference from MA2's
successful archive or proof of physical success. See [PI reference](PI05_REFERENCE.md),
[range review](PI05_YAM_RANGE_REVIEW.md) and [artifacts](PI05_ARTIFACTS.md).

### Official π0.5 base: genuine model, useful diagnostics, YAM blocked

Official JAX OpenPI is pinned to
`Physical-Intelligence/openpi@215abfb217dbac7d5f1273282331b9b1866c0479` in isolated
`data/openpi/venv`, with the upstream lock and no MA2 dependency upgrades.
The genuine `gs://openpi-assets/checkpoints/pi05_base` checkpoint comprises 29
generation-pinned objects / 12,441,749,581 bytes. Manifest SHA-256:
`439423350a9160e2157291aec1a48e8454193c7126e24fda670df39bb7c503db`.
Per-object receipts preserve full checksums outside the native Orbax tree.

Real-GPU diagnostics completed 50 native calls / 2,500 raw 32-D rows, with
bitwise wrapper-versus-native parity for identical noise. These use saved real
RGB but a clearly labeled synthetic normalized-zero 32-D state, not unnormalized
YAM joints. Original 50×32 outputs are retained without clipping, first-14
slicing, action dispatch or a fake YAM rollout. The Lambda record is
`.context/overnight/openpi-base-real-gpu-sidecar-fixed/diagnostic.json`. The 50-call
p50/p95/max were 0.06035815752693452 / 0.06107487933477387 /
4.56622357602464 s; the maximum includes the first unseeded JIT specialization,
not an omitted outlier. Model load took 9.417 s and the seeded cold call 11.465 s.
Seeded native-wrapper parity was bitwise equal with maximum absolute difference 0.

The official assets have no YAM normalization/embodiment configuration. Joint
zero/sign/order, absolute-versus-delta meaning, gripper units/direction and native
chunk-commitment/rate must be justified before physical use. Another robot's
normalization or ALOHA first-14 slicing is not that evidence. No fine-tuning,
checkpoint substitution or invented controller semantics were used to claim
success. `pi05_base` therefore intentionally blocks before hardware; see
[the official runtime report](OPENPI_REFERENCE.md).

## Reliability work and final acceptance

Additive CLI/UI work shares policy-specific preparation, exact identity/expiry
checks and cooperative workflow ownership. Managed cold starts use per-GPU locks
and explicit owned-process/listener checks; unrelated processes are not killed.
New policies stay isolated from the frozen MA2 controller. Strict native weight
loading cannot silently return an unloaded model. Orbax receipt placement no
longer pollutes checkpoint metadata. Recording/render/upload failures now produce
truthful nonzero CLI and UI outcomes while preserving release evidence and local
originals; no failed physical work is retried automatically.

The browser's intercepted-network checks passed sparse/mobile layout, collapsed
advanced settings, approval-free preparation, cancellation without rollout,
single confirmed native Start payload, official-base blocking and return to MA2
with approval cleared. They requested no real camera endpoints. Evidence:
`.context/overnight/browser-native/result.json`; this is not a live physical UI
test. Separately, 21 safe GET requests verified actual MA2 and PI recording
playback routes, including the three videos' headers and byte-range seeks for
each run. Evidence:
`.context/overnight/ui-playback-20260913T122814Z-2d20788e/result.json`. No rollout
POST or real camera endpoint was used for that playback check.

The full suite passed **3,214 tests, with 2 skips and 9 passing subtests**, in
302.49 s; report `.context/overnight/final-suite-v3.xml`. It covers frozen MA2,
native parity, failure/Stop paths, artifacts and shared workflow/UI regressions.
The subsequent explicit-missing-config fix passed another 124 targeted
backend/lifecycle/attachment tests, including four new regressions. These counts
are separate overlapping suites, not a sum of distinct test cases.

The actual CLI policy switch sequence passed: MA2 dry-run (0.982 s, ready/reused)
→ PI recorded fake run → official-base exact rollout command (exit 2 in 0.136 s,
blocked before hardware) → MA2 dry-run (0.949 s, ready/reused). MA2 returned with
unchanged selection key `2f59eafe2c797df523f6`, same instance and same expiry.
This is software switching, never an unattended physical policy sequence.

Explicitly missing backend configuration now fails before attachment fallback;
the default-missing-config compatibility path remains. Actual fresh CLI checks
confirmed missing explicit config exits 2 (0.1808 s), an unavailable loopback-only
config without SSH/remote startup exits 2 (0.2078 s), and restoring normal
configuration returns ready/reused with exit 0 (0.9242 s). No service or tunnel
changed. Lenovo evidence: `.context/overnight/cli-recovery-v2/result.json`.

Final read-only Lenovo state at 12:31:15 UTC is
`.context/overnight/final-state-v3/result.json`, with `software_ready=true`:

- The tested source and both model identities match the table. UI process
  PID 721257 / start identity `29292303` serves the current static UI on loopback
  8400; the session is inactive, owns no cameras and has no direct cameras open.
  Existing inference forward PID 47435 / start identity `2053106` is retained.
- Both actual UI preflights passed for the canonical task, 60-second capture and
  private-HF upload: MA2 selection `93a7e4d13a01c10a7d89`, native PI selection
  `1a0f47f7877b1bf79709`. Neither selection is motion approval.
- Available memory was 8,585,220,096 bytes, above the unchanged 90-second
  admission threshold of 8,010,125,312 bytes; no reservation was performed during
  this check. Disk free was 813,827,112,960 bytes. The actual command must recheck
  current memory; this snapshot is not a future capacity guarantee.
- The rig retains its baseline hash. The old pending 90-second request remains
  unchanged at SHA-256
  `885c41ceed56f35ea377ab76f1afb01df074260d26948431e9bb09b36100aa9e` and remains
  unapproved. Backend configuration is now
  `84f9f34598cb738bae4fb20b5f8762184f8de7162cc56a47c1c871e3bc3cd883`; only saved
  MA2 fake-input paths were added, with the original private backup retained.

The final Lambda process snapshot, `.context/overnight/policy-switch/final.json`,
records the two exact owned services: MA2 PID 33333 on 8765 and PI PID 36316 on
8766. The 16 sanitized reports are also preserved in the cloud workspace under
`.context/overnight/evidence/{lenovo,lambda}/`; every copy matches the SHA-256 in
`manifest.json`. No credentials were transferred into this evidence collection.
All readiness is session/identity-bound; after expiry the command must prepare
and qualify again rather than reuse stale proof.

## Non-destructive rollback

The protected tag provides a separate source checkout without altering the
current working tree or historical evidence. From the cloud repository root,
only if this destination does not already exist:

```bash
git worktree add --detach .context/rollback/overnight-baseline \
  overnight-baseline-20260913-6b348a8
```

This is a source reference, not a command to activate hardware, copy credentials
or reuse stale qualification. No automatic service or physical rollback runs.

Remote source overlays have per-file deployment manifests and sibling
`originals/<relative-path>` backups. Recorded starting points for review are:

- Lenovo: `.context/overnight/deployments/20260913T121859Z-lenovo/deployment.json`.
- Final Lenovo runtime update: `.context/overnight/deployments/20260913T123024Z-lenovo/deployment.json`.
- Lambda: `.context/overnight/deployments/20260913T121855Z-lambda-pi05/deployment.json`.

All earlier manifests under each host's `.context/overnight/deployments/` remain
necessary to reconstruct the complete rollback chain, not just the final update.

Restore only reviewed files from the appropriate backup chain while exact owned
services are idle and with separate authorization for any service changes.
Verify current hashes before restoring; if later user edits overlap, preserve
them and stop for review. Never `git reset --hard`, overwrite a dirty remote tree,
or remove old recordings/configuration/credentials. Recheck source/model identity
and renew software qualification after restoration; old proof and GOs do not
authorize the restored deployment. Existing user-funded VM billing continues;
this handoff neither provisions nor terminates a machine.
