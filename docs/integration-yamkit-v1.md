# yamkit v1 integration record

This branch integrates committed source revisions only; `main` and source branches are unchanged.
No arms or physical cameras are used during integration. This record separates software verification
from supervised hardware acceptance; see [acceptance-test.md](acceptance-test.md) for operator steps.

## Verified sources

Fetched `origin` before any code changes. The clean workspace HEAD exactly matched current
`origin/main`: `dbd01f1e70be66bb0e639789d0d37d7ecb5bd166`.

| Order | Source | Exact included SHA | Verification |
|---|---|---|---|
| A | `origin/seattle` | `686b95e2da7a68b7dc79832ea080efbfc9921c4b` | Exact expected SHA |
| P | `origin/charlotte` | `8900a984dbcdf38177ee8e0f4bd3fe4537c3acdb` | Pushed HEAD matches reported `8900a984...` |
| B | `origin/oslo` | `ad52a189816dcb1e346a1da8f94accb19c239427` | Matches expected `ad52a18` |
| C | `origin/bridgetown` | `f3efcd1194620b938d9aa4750bbd895c9eb96c4c` | Exact expected SHA |

Merged with ordinary non-fast-forward merge commits in that order. No source workspace state,
rebase, squash, reset, force push, or whole-file ours/theirs conflict resolution was used.

## Conflicts and semantic resolutions

A and B had no textual conflicts. P conflicted in the follower plugin. C conflicted in the seven
files listed below (the follower plugin also contains the P resolution).

| File | Resolution |
|---|---|
| `plugins/lerobot_robot_yamkit/lerobot_robot_yamkit/yam_follower.py` | Retain A validation, bimanual prevalidation, retryable arm close, partial startup cleanup and `disconnect(home=False)`. Use P's recorder-owned preview and explicit camera ownership handshake. Camera teardown is attempted before optional homing; a camera fault skips homing but still releases every arm. Keep C stop checks and transient runtime handle; `disconnect_no_home()` delegates to the same hardened teardown. |
| `src/yamkit/arm.py` | Keep A raw/aligned limits, prevalidation and start barrier. Add C's optional external stop event to `go_home_all`, without losing the barrier or per-arm checks. |
| `src/yamkit/ui/server.py` | Combine lifespan cleanup with inference endpoints/validation. Use P's explicit acquisition/release protocol instead of C's older mode-name suspension and upload-log-based resume, preventing premature direct capture. |
| `src/yamkit/ui/sessions.py` | Preserve P authenticated preview registration, stdin acknowledgement, descendant/output drain and ownership retention. Keep C credential filtering, structured inference results and sanitized spawn failures. |
| `tests/conftest.py` | Combine A fake-robot compatibility with C isolation of all service credentials and Hub caches. Ensure the repo-local pytest temporary parent exists on a fresh clone. |
| `pyproject.toml` | Preserve agent, Modal, optional Molmo and development extras; retain CPU torch/torchvision sources. Consolidate uv settings here because a separate `uv.toml` caused modern uv to ignore the package index and build constraints. |
| `uv.lock` | Reconcile both sets of extras, then regenerate with `uv lock`; install with the resolver. Local torch remains `2.11.0+cpu`, torchvision `0.26.0+cpu`. Modal CUDA requirements stay in `configs/modal-requirements.*`. |

Additional baseline integration fixes:

- Active-read probes use P's camera lease before any camera/arm acquisition and retain it if
  release cannot be confirmed.
- Preview publisher startup and close handle cancellation, attempt every acquired resource and
  retain failed cleanup for an explicit retry. Nested startup handlers no longer immediately
  retry camera teardown and risk premature ownership release.
- B acknowledges the available no-home cleanup but keeps live execution blocked: cached SDK state
  and camera reads do not prove sensor acquisition after a command. Fixture freshness checks use
  the time **after** `send_action()` returns.
- C/P lifecycle assertions now test explicit camera acquisition and process/output drain rather
  than superseded command-name ownership. The real tiny ACT regression produces deterministic
  in-bounds outputs; A safety bounds were not relaxed.

## Verification evidence

Detailed machine-generated logs and JSON live under `.context/integration/` (gitignored). Results
and milestone commits are recorded below as the integration stages complete.

The initial full run exposed a fresh-clone pytest temp-directory bug and stale lifecycle assertions;
these failures are preserved in `baseline-pytest.log`, not counted as a passing run.

Real SmolVLA CPU baseline: pinned checkpoint loaded; three fresh calls produced finite `[50, 6]`
chunks. Readiness 35.24 s; RPC-equivalent local calls 3.784 / 3.677 / 4.497 s on this 8-vCPU VM while
offline tests also ran. These timings do not establish local real-time control or physical mapping.
No OpenAI/Modal paid retest was required for the baseline.

### Passing integrated baseline

- Full hardware-free suite: **864 passed**, 4 existing deprecation warnings, 109.18 s.
- Additional cancellation tests: the focused publisher/plugin/agent checks passed (193 tests;
  strengthened publisher subset 46 passed). No safety test was removed or weakened.
- Ruff (`src plugins tests scripts/browser_smoke.py`) and `git diff --check`: passed.
- Actual Chrome: **23 checks passed**, zero JavaScript exceptions; real hardware and paid services
  trapped. The baseline Inference page still requested direct camera previews on load; Stage 5
  will make those previews opt-in.
- Preview benchmark: 3 synthetic 640×480 RGB cameras maintained 30 Hz for all four 10-second
  scenarios. Enabled handoff p99 162.08 µs; slow viewers 165.16 µs; nine viewers 166.16 µs.
  No acquisition/encoding errors or surviving viewer threads. Values describe this VM only.
- Agent mocked tests: 179 passed. Modal/local/probe subset: 169 passed, including the real tiny
  ACT forward pass through the upstream local strategy and fake YAM plugin.
- Real SmolVLA CPU three-fresh-chunk check: passed as detailed above.

Committed artifacts: [CPU](integration-results/baseline-smolvla.json),
[preview benchmark](integration-results/baseline-preview-benchmark.json),
[browser](integration-results/baseline-browser.json).

## Integration milestones

| Milestone | Commit |
|---|---|
| Passing A/P/B/C baseline, before new behavior | `f4039738c880506a4b2ee84a98c29a58ef179905` |
| Shared operator semantics and executed recording labels | `fb00094f8ded24c07e98b768efadcb1de5ab1d13` |
| Remote timing, expired-prefix correction and physical performance gate | `1ea171d6234a6c43eebb3d9cf307a0ec042a89a6` |
| Inference opt-in previews and confirmed probe cleanup | `bd92418b40231e463ad5d5dd1c53dafdfd19bfbc` |
| CLI/UI support and A–L acceptance documentation | `c7550eb4b76347401f662d7e24412de3d8b20cd4` |
| Credential-safe SDK failures, commands and session results | `bb7a1de09a207daff2de89d1194e70844090fe8a` |
| Preserve camera readers through failed warmup and cleanup retries | `6a076005250354d75f5452eadf020a0085afd07c` |

The parity change uses pinned LeRobot's public recording processor hook and its unchanged
recording loop. The processed action object is acknowledged with the actual bounded command
before the upstream dataset frame is built. Native teleop and LeRobot retain separate loops;
the shared per-pair state handles buttons, bounded synchronization, mappings and measured hold.
Native bilateral feedback remains supported; recording/LeRobot teleoperation reject nonzero
bilateral feedback before connecting. Raw LeRobot YAM leader commands require the yamkit wrapper.
See [OPERATOR_PARITY.md](OPERATOR_PARITY.md) for the pinned-hook contract and limitations.

The remote queue previously appended whole chunks without removing elapsed action prefixes.
It now discards expired/overlapping prefixes, preserves each action's observation-relative
deadline through concurrent merges, and rechecks expiry immediately before `Robot.send_action`.
No processor, transport, model, region, FPS, speed limit or firmware timeout changed.
The fake benchmark exercises genuine upstream execution components and the YAM plugin;
it is not additional real-service evidence. See [REMOTE_PERFORMANCE.md](REMOTE_PERFORMANCE.md).

Active-read cleanup additionally detects camera backends that return successfully while still
connected or while an acquisition thread survives. Every remaining resource is attempted and
the camera lease stays held unless release is confirmed. Probe results now explicitly distinguish
source mapping review from supervised physical validation.

Final review also found that raw SDK exception messages could reach CLI/browser logs.
Modal lookup, call, construction and deployment errors now suppress those messages while
preserving actionable failure categories and cancellation cleanup. Session output redacts
known environment and local HF-file credentials, including escaped/multiline forms, before
command display or structured-result parsing. Tests use dummy credentials and verify the
parent environment remains unchanged. No credential values were included in review artifacts.

Pinned OpenCV/RealSense cleanup can clear a thread reference after a bounded join times out,
including inside failed warmup. The final camera fix observes that existing instance cleanup
method during connect/disconnect, restores it afterward, and keeps every observed reader until
it terminates. A surviving detached reader prevents successful camera startup and lease release,
including automatic cleanup retries; all other resources are still attempted. Probe camera
startup remains before arm activation. The existing follower startup/home order is unchanged;
camera startup failure causes no additional home move. No camera implementation was copied or
installed LeRobot file patched. The focused pinned-camera/fake-device suite passed 195 tests.

## Final combined verification

The full combined run used frozen executable source
`6a076005250354d75f5452eadf020a0085afd07c`; only documentation and result artifacts changed
subsequently. CPU inference, preview timing and remote timing ran separately after the full
suite, on this 8-vCPU cloud VM. Main still matched the original base at final remote verification.

| Check | Final result |
|---|---|
| Complete hardware-free `make test` | **925 passed**, 4 existing Starlette/fork deprecation warnings, 129.66 s |
| Lint | `make lint` and Ruff on all three diagnostic scripts passed |
| Resolver and CPU environment | Offline lock check passed, 175 packages; torch `2.11.0+cpu`, torchvision `0.26.0+cpu` |
| Whitespace | Working diff and complete `origin/main...HEAD` passed |
| Actual Chrome UI/HTTP/session/preview | **25 checks passed**, zero JavaScript exceptions or forbidden real hardware/service attempts |
| Preview benchmark | Four 10-second scenarios, three synthetic 640×480 RGB feeds at 30 Hz; p99 handoff 155.88 µs enabled, 128.88 µs slow viewers, 164.94 µs nine viewers; zero errors/surviving viewer threads |
| Actual SmolVLA CPU | Three fresh finite `[50, 6]` chunks; readiness 23.15 s; calls 3.835 / 3.762 / 3.833 s; no physical mapping or continuous-control qualification |
| Integrated remote execution benchmark | Each healthy 32-second scenario: 69 requests, 68 warm samples, 955 successful bimanual sends, zero underruns. A 700 ms spike safely underruns after 55 sends. Injected 1.48 / 2.378 / 3.166 s cases send zero actions. All fake robots released. |

The full suite includes mocked LLM/provider tests, actual hardened fake-plugin cleanup,
state/image freshness rejection, native/recording parity and exact dataset action labels,
record/reset/preview ownership, Modal saved processors/probes/Stop/deadlines, credential
redaction, partial camera warmup and cleanup retries, and the genuine tiny ACT forward pass
through the existing local LeRobot strategy. No paid real-service retest was needed. The real
Modal history remains source C's evidence, with two warm samples per model, not an integrated
p95/p99 qualification. The larger remote benchmark sample count describes injected delays only.

Committed final evidence: [verification summary](integration-results/final-verification.json),
[source ancestry](integration-results/final-provenance.json),
[Chrome](integration-results/final-browser.json),
[preview benchmark](integration-results/final-preview-benchmark.json),
[real CPU chunks](integration-results/final-smolvla.json), and
[remote synthetic results](remote-performance.json). Full logs/traces stay in ignored
`.context/integration/`. The final documentation commit and draft PR identify the published
integration HEAD; the source commit above identifies the exact executable code that was tested.

## Acceptance restrictions

- Live B execution remains disabled. A supplies no-home cleanup, but SDK cached state and
  `camera.read_latest()` do not expose sufficient acquisition evidence to prove every state/image
  was acquired after a physical action completed. Fixture tests reject stale feedback and stop
  further decisions/actions; they do not establish physical sensor freshness.
- Physical Modal rollout is unconditionally blocked in shared CLI/UI validation, the runner and
  direct remote-policy construction. Readiness, a successful probe or operator confirmation
  cannot override it. Real integrated warm tail latency and sustained queue supply remain unmeasured.
- No rig passed physical mapping/calibration/camera or operator acceptance during this task.
  MolmoAct2's source mapping is reviewed and its local sync implementation remains available;
  local Molmo compute and physical behavior still require separate acceptance. SmolVLA/pi05 base
  physical mapping remains blocked; guided remote RTC and local Molmo guidance are unsupported.
- No real arms/cameras were accessed, no paid OpenAI/Modal calls or deployments were made, and
  no source branch or `main` was modified. Local CPU torch and isolated Modal CUDA dependencies
  are preserved. No database integration was added.

The [A–L acceptance checklist](acceptance-test.md) gives actual CLI/UI operations, all six
command effects, Stop/rollback procedures and the complete model/backend support matrix.

## Independent software review

From a clean checkout of `origin/integration/yamkit-v1`, confirm the local HEAD matches
the published integration ref before running tests. These commands activate no motors,
perform no calibration/homing, capture no physical cameras, spend no inference/API money
and send no physical policy actions. Dependency/checkpoint downloads stay inside the repo.
The browser command requires the Chrome already installed in this cloud workspace.

```bash
git fetch origin
git rev-parse HEAD origin/integration/yamkit-v1
git diff --stat origin/main...origin/integration/yamkit-v1
git diff --check origin/main...origin/integration/yamkit-v1
source scripts/env.sh
mkdir -p .context/review .context/tmp
export TMPDIR="$PWD/.context/tmp" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
uv sync --extra dev --extra agent --extra modal
HF_HUB_OFFLINE=1 make test
make lint
.venv/bin/ruff check scripts/browser_smoke.py scripts/benchmark_remote.py scripts/benchmark_preview.py
.venv/bin/python scripts/browser_smoke.py --output .context/review/browser.json
.venv/bin/python scripts/benchmark_preview.py --duration 10 --warmup 1 --output .context/review/preview.json
.venv/bin/python -m scripts.benchmark_remote --output .context/review/remote.json
yamkit policy-check --policy smolvla --backend local --device cpu --steps 3
```

Inspect `git diff origin/main...origin/integration/yamkit-v1` alongside this record's conflict
table. Review `OPERATOR_PARITY.md`, `REMOTE_PERFORMANCE.md` and the A–L checklist before any
separately authorized camera/motor work. Do not deploy, bypass either motion gate or merge this
draft PR as part of software verification.
