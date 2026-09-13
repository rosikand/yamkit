# Software-only day mission — 2026-09-13

Historical mission-close record below; its readiness statements describe that
earlier deployment and are superseded by the
[overnight handoff](OVERNIGHT_HANDOFF_2026-09-13.md) and
[current YAM π0.5 reference evidence](PI05_REFERENCE.md). The intermediate
[authorization follow-up](PI05_QUALIFICATION_2026-09-13.md) retains the tokenizer
resolution and native gripper-range failures. Those failures and the original
report below are preserved, not reclassified as successful trials.

MolmoAct2 is software-qualified and its UI/CLI handoff is verified. π0.5 is
implemented and fake-tested, but **not GPU-qualified or ready for physical use**:
its publisher-gated tokenizer is unavailable to the configured accounts.
No motors, active encoder reads, homing, real camera acquisition, or physical
rollouts were performed. Software evidence cannot guarantee physical task success.

## Protected baseline and tested implementation

- Baseline: `e8f43fed3b04ea1ff2da0a6df19d1c2367574fe6`.
- Pushed protected tag: `ma2-known-good-20260913-e8f43fe`.
- Final tested software: `da0a16de98cd40e8537ca89db95182c47395dec5`.
- MolmoAct2 build, unchanged before/after:
  `5128c7b55b6f9a62c109b581f98912de717c7d3746e8c6374341e91687d481fe`.
- Branch: `conductor/yamkit-andre-tailnet`; implementation milestones were pushed
  individually. This report is a documentation-only follow-up to the tested SHA.

The model runtime, checkpoint, preprocessing, reference controller, interpolation,
SDK dispatch, hardware guards, capture/export runner and qualification code have
no diff against the protected baseline. Three seeded 30×14 predictions were
bit-for-bit identical across the original and recovered model instances; maximum
absolute difference was zero in all three cases. This is fixture parity plus
unchanged-source evidence, not a claim that every possible physical scene was tested.

Robot-host changes were deployed as checked, backed-up patches. Its pre-existing
worktree, branch and Git index were preserved; the remote Git HEAD alone therefore
does not identify the deployed tree. Deployment manifests record per-file hashes.
The GPU's original MolmoAct2 source was retained, with the new PI package additive.

## Bugs fixed

1. Non-finite native diagnostics could make strict JSON status/Stop responses fail.
   Display-only invalid numbers now become null, never invented joint readings.
2. Invalid UTF-8 child output could terminate the status reader and strand run
   finalization/ownership. Replacement decoding preserves lifecycle handling.
3. An unwritable historical upload receipt could prevent UI startup. Recovery now
   retains a conservative interrupted state without overwriting the original.
4. Missing recording summaries and artifact/receipt write failures could be shown
   as success or leave saving permanently active. Physical and saving outcomes
   are kept distinct, with actionable failures and no automatic upload retry.
5. Cached UI readiness could bypass a dead model service. Explicit Start now
   checks/restarts the configured software service and reuses current qualification
   when possible. Page loading and form changes do not start GPU work or motion.
6. CLI/UI preparation could collide with a running child. A shared repo-local lock
   is retained through managed child lifetime, including parent-process loss.
7. A long terminal confirmation delay could leave insufficient session lifetime.
   Exact local qualification and startup/run/home time margin are rechecked before
   hardware construction, without another model qualification call.
8. Existing UI camera previews could collide with terminal rollout, and delayed
   browser subscriptions could open cameras after terminal ownership began.
   Passive preview status and a lock around actual capture-thread creation close
   that boundary for the normal UI port. Software-only preparation still permits
   existing previews; no user UI or application is automatically stopped.
9. Native PI strict loading could silently return unloaded weights through the
   upstream exception handler. The independent loader propagates failure.
10. PI timing initially differed from the pinned native loop. Tests against the
    actual LeRobot runner now enforce per-tick observation and whole-tick timing,
    including RPC, camera and send overruns.

## Verification

Final full hardware-free suite: **2,927 passed, 2 skipped**, plus nine passing
subtests, in 261.73 seconds. Ruff and JavaScript syntax checks passed. The two
optional skips and existing deprecation warnings are retained, not hidden.

Real Chrome, with synthetic/intercepted devices and services only:

- Multiple windows maintain at most three managed preview streams; third-window
  navigation completed in 34 ms.
- Legacy Inference hash navigation opens no camera streams.
- Status recovers after an intentional six-socket HTTP stress case.
- All 22 preparation/confirmation checks passed, including live-check failure and
  declined confirmation with no rollout launch.
- All six camera-boundary regressions passed; a response created before a terminal
  lock cannot subsequently start capture while that lock is held.
- Test browsers and synthetic listeners were closed afterward.

### Real MolmoAct2 service, generated RGB and fake arms

Exact task: **put the red cube into the black container**. Each qualification used
50 direct warm requests and 50 integrated warm samples; integrated accounting
includes the first chunk, hence 51 completed chunks.

| Metric | Baseline | Final deployed version |
| --- | ---: | ---: |
| Qualified | yes | yes |
| Direct warm p50 | 0.356276 s | 0.371566 s |
| Direct warm p95 | 0.359309 s | 0.433061 s |
| Direct warm maximum | 0.532563 s | 0.630051 s |
| Integrated observation-age p95 | 0.638996 s | 0.620204 s |
| Predicted / completed rows | 1530 / 1530 | 1530 / 1530 |
| Completed chunks | 51 | 51 |
| Dropped / uncompleted rows | 0 / 0 | 0 / 0 |
| Coherence violations | 0 | 0 |
| Literal interpolation dispatches | 1665 | 1700 |
| Commands after simulated Stop | 0 | 0 |
| All fake arms released | yes | yes |

Final post-clamp modifications: **0**. SDK sends during completed synchronous RPC:
**0**. One intentional in-flight Stop invalidation is expected; neither run
reported an unexpected rollout fault. Configured control rate remains 30 Hz;
final measured dispatch rate including synchronous RPC gaps was 20.8071 Hz.
Unseeded prediction values determine interpolation counts, so those counts need
not match between qualifications. The separate seeded comparison was exactly equal.

An actual idle model-service outage was exercised through UI preparation. The
workflow restored the service on the existing GPU, retained the existing SSH
tunnel, requalified using fake arms, and required fresh confirmation afterward.
Reusing that UI proof took 2.61 seconds. Final real-host CLI `inference` and
`rollout --dry-run` both exited 0, reused current proof in about 0.8 seconds, and
opened no hardware. Final UI preparation also reused proof and ended inactive.

At the final check, the 90-second recording admission had 11,975,266,304 available
bytes against the unchanged 8,010,125,312-byte requirement, with approximately
818.8 GB free disk. No frame reservation was made. Admission remains dynamic and
is checked again for a real recording; this snapshot is not future permission.

### π0.5 native contract and blocker

Checkpoint:
[`Jiafei1224/molmoact2-yam-pi05@51ab2720d7e56d51410407f98ea64bbea97feb2e`](https://huggingface.co/Jiafei1224/molmoact2-yam-pi05/tree/51ab2720d7e56d51410407f98ea64bbea97feb2e).
Independent contract: `pi05_reference`; native build:
`b2c80d8f1b9ee6fb5dc29628d66aa401fc392aae86d4ee39af66b700af81c8ce`.

Camera order top/left/right; 14-D left six joints and continuous gripper, then
right; absolute actions; saved quantile processing exactly once; native 224²
letterboxing; bf16 configuration; ten denoising steps; 30-row FIFO at nominal
30 Hz. Observe each tick, replan only when FIFO empties, and sleep only the
remaining tick budget after observation/RPC/send. No RTC, Molmo interpolation,
prefix dropping or output clipping. Full details and pinned sources are in
[PI05_REFERENCE.md](PI05_REFERENCE.md).

79 native/fake tests passed. Four fake-clock cases call the actual pinned LeRobot
runner and policy FIFO methods, matching all 60 observations, sends and row values
per case (timestamp tolerance 1e-10 s). Saved native processing, strict loading,
bounds, full row accounting, Stop/fault/release, trace/report replay and admission
are tested without real devices.

Fifty paired observations were prepared from an existing successful recording:
original full-resolution RGB plus its corresponding measured 14-D state, not
MolmoAct2's last-command state. Original files were preserved. The pinned 9.35 GB
weights were downloaded, but both configured accounts were denied the pinned
[`google/paligemma-3b-pt-224` tokenizer](https://huggingface.co/google/paligemma-3b-pt-224).
The user must review and obtain publisher access; no gate or license was bypassed.
No alternative tokenizer/model was substituted.

Consequently **real PI p50/p95/max, integrated rate, model-output row accounting
and real-model qualification are unavailable**, not zero and not passing. Its
guarded physical entrypoint remains unqualified. PI artifacts currently cover
JSON trace/report replay; UI policy selection, video playback and HF upload are
not integrated for PI. MolmoAct2's existing recording/upload path is unaffected.

## Commands for the on-site operator

From the Lenovo checkout, activate its existing environment:

```bash
source scripts/env.sh
```

Software-only inference readiness (optional; rollout already includes it):

```bash
yamkit inference --backend lambda --policy molmoact2 \
  --task 'put the red cube into the black container'
```

One future supervised MolmoAct2 run:

```bash
yamkit rollout --backend lambda --policy molmoact2 \
  --task 'put the red cube into the black container' --duration 60
```

The terminal asks for one on-site confirmation before hardware. Verify mapping,
mounts, empty grippers, clear movement area and the physical cutoff. **Ctrl-C in
that terminal is Stop for terminal runs**; the dashboard Stop controls dashboard
runs. Healthy completion homes preserving final gripper opening then releases;
Stop/fault releases without home. No physical retry is automatic. Use the UI's
Recording/Upload controls when videos/HF archives are wanted.

The UI is left running at `http://127.0.0.1:8400/#/inference` on Lenovo. An existing
UI at that port is reused, not silently killed. Close direct previews before a
terminal physical run, or use UI Start for its normal camera handoff. Existing
previews on custom UI ports must also be closed manually; passive terminal
preview discovery covers the standard port only. New same-repo capture starts
are fenced across ports. See [UI.md](UI.md) and [INFERENCE_CLI.md](INFERENCE_CLI.md).

Only after publisher access, separate native backend configuration, successful
saved-observation GPU qualification and fresh on-site verification:

```bash
yamkit inference --backend lambda --policy pi05 \
  --task 'put the red cube into the black container'
yamkit rollout --backend lambda --policy pi05 \
  --task 'put the red cube into the black container' --duration 5
```

These PI commands are conditional instructions, **not currently ready to run**.
The native service requires its own configured port/token-file paths; no token
values belong in command arguments, chat, source or rig files.

## Retained evidence

Cloud checkout: `.context/day-mission/` (release JUnit, baseline summaries and PI
contract/source evidence), `.context/dh/final-browser-validation.json`.

Robot checkout: `.context/day-mission/20260913/ma2-baseline/`, `ma2-final/`,
`ma2-seeded-baseline/`, `ma2-seeded-final/`, `ui-recovery/`, `ui-final/`,
`saved-observations/`, `final-software-check/`, plus source-hash deployment backups
under `.context/day-mission/deployments/`. Historical readiness/captures were not
overwritten. The old pending 90-second request remains unchanged; its embedded
confirmation flags are not approval. No motion approval is active.
