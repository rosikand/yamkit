# π0.5 authorization and real-model qualification follow-up

**PaliGemma authorization is resolved. π0.5 is not qualified for physical use.**
No real cameras, CAN, encoder reads, homing or motor commands were used. The
working MolmoAct2 source, runtime and controller were not changed.

## Authorized assets and exact runtime

The existing HF login on Lenovo successfully fetched
`google/paligemma-3b-pt-224@35e4f46485b4d07967e7e9935bc3786aad50687c`.
Nine tokenizer/config files (21,918,641 bytes) were securely transferred into the
existing Lambda checkout's repository-local cache and SHA-256 verified. **No HF
token was transferred.** The standard pinned snapshot loader then succeeded on
Lambda without HF login. No alternative model/tokenizer or permission bypass.

The independent `lambda-pi05` service loaded the pinned
`Jiafei1224/molmoact2-yam-pi05@51ab2720d7e56d51410407f98ea64bbea97feb2e`
checkpoint on the existing GPU, using port 8766 separately from MolmoAct2's 8765.
The existing MolmoAct2 backend configuration was preserved. Native build actually
tested: `b2c80d8f1b9ee6fb5dc29628d66aa401fc392aae86d4ee39af66b700af81c8ce`;
instance: `3a6b805e-6915-4bb4-9d1e-767edc04a64c`. This is the pre-diagnostics build,
not a claim that the later reporting-only patch was deployed or GPU-qualified.

## Observed failure

Task: **put the red cube into the green bowl**. Inputs were 50 saved,
full-resolution RGB triplets paired with recorded measured 14-D state from an
existing successful MolmoAct2 recording. No new frames were acquired.

| Partial direct warm samples | CLI attempt | Instrumented diagnostic attempt |
| --- | ---: | ---: |
| Valid samples before failure | 38 / 50 | 34 / 50 |
| p50 | 0.515499 s | 0.449911 s |
| p95 | 0.524570 s | 0.520022 s |
| Maximum | 0.525785 s | 0.526625 s |
| Qualified | no | no |

These are **partial** sample statistics, not the complete qualification result.
Neither attempt reached integrated fake-arm execution or Stop-during-RPC proof.
Those real-model results are unavailable, not zero and not passing. No current
qualification record was promoted.

The instrumented attempt retained each native numeric response unchanged. Its
36th response (request sequence 35) was finite and correctly shaped 30×14, but
zero-based row 0, column 6 (`left_gripper.pos`) was **1.000895619392395**. The
allowed range is [0,1]. The saved native postprocessor has no output clamp;
yamkit rejected this value without clipping or dispatching it. Three separate
predictions of the first attempt's failing observation passed, demonstrating
that a successful retry does not establish validity of every stochastic output.

This follow-up improves failure reporting and regression coverage only. It does
not relax bounds, change denoising/normalization or substitute controller semantics.
The native-output issue remains unresolved; no physical trial is approved.

## Evidence and tests

On Lenovo, under `/home/andre/rohan-new`:

- First result:
  `.context/pi05-qualification/5d27fe830b624cabbfc31c58d6709c99/qualification.json`.
- Instrumented result and all 36 native responses:
  `.context/day-mission/pi05-authorization/qualification-diagnostic-20260913T075153Z/`.
  `native-response-036.json` retains the exact rejected chunk.
- `corrected-summary.json` supersedes an indexing error in the diagnostic
  display's original `summary.json`; the raw response was never changed.
- Backend backup and configuration metadata:
  `.context/day-mission/pi05-authorization/configure-20260913T074525Z/`.

Cloud evidence: `.context/day-mission/pi05-authorized-20260913/`. The final focused
diagnostics/workflow/native/MolmoAct2 reference suite passed **331 tests** in
21.03 seconds; Ruff and diff checks passed.
Tests include the exact observed overshoot, unchanged retained rows, no current
qualification promotion, fault cleanup, failed-RPC stale-response prevention,
and safe handling of nonfinite, oversized and credential-like error values.
These hardware-free regressions do not replace successful real-model qualification.

The final passive check at approximately 08:02 UTC found the existing Lenovo UI
idle, with no owned or direct cameras open. Original UI/tunnel process identities
and MolmoAct2 build were unchanged. Both independent inference listeners remained
up; π0.5 had no current qualification pointer. The reporting-only patch is pushed
to GitHub but is **not deployed** to either live checkout in this follow-up, so
neither service requires a restart for these diagnostics.
