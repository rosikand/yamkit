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
