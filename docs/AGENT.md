# Multimodal LLM episodes

`yamkit agent` runs a small synchronous controller alongside the existing LeRobot VLA `rollout`.
The available modes use **labeled synthetic robot state and RGB images**. Live execution is
disabled because the current robot interface cannot verify state and image acquisition
freshness. Hardware hardening supplies public cleanup without homing; the remaining blocker
is described below.

## Install and run without hardware or API calls

From the repository, after the normal setup:

```bash
source scripts/env.sh
uv sync --extra agent --extra dev
yamkit agent --model YOUR_MODEL_ID --task "inspect the fixture and finish" \
  --arm left_follower --rig configs/rig.example.yaml --dry-run --offline \
  --max-steps 5 --settle-s 0 --log-path outputs/agent/offline.jsonl
```

The OpenAI SDK is an optional `agent` extra and is imported only when constructing the real
provider. The offline mode works without that extra and without credentials. It runs a
deterministic mocked provider; the requested model ID is recorded but never contacted.
The example rig is read only to validate the named follower. Neither dry-run mode constructs a
physical plugin, opens a CAN bus or camera, energises/calibrates/homes a motor, sends a physical
action, or invokes physical teardown. Simulated actions update only the fixture.

## Paid API use with fixtures

Supply an API key to the command's environment using your usual secret manager:

1. `YAMKIT_OPENAI_API_KEY` takes precedence whenever it is present.
2. `OPENAI_API_KEY` is used only when the namespaced variable is absent.
3. An empty namespaced variable means **MISSING**; it does not fall back to the other key.

The selected value is passed explicitly to the OpenAI client. The command reports only
**SET** or **MISSING**, never a value or prefix. Keys do not belong in `configs/rig.yaml`, source
files, logs, or committed shell scripts. This feature does not change Codex/Conductor credentials.
The client endpoint is fixed to `https://api.openai.com/v1`. Nonempty `OPENAI_CUSTOM_HEADERS`
is rejected so custom headers cannot override the explicitly selected credential.

Choose a model available to your account that supports images, strict function calling, and the
provider's `low` reasoning setting. Model access is specific to the requested ID: testing a
different model does not establish access to your deployment model.

```bash
yamkit agent --model YOUR_MODEL_ID --task "describe the fixture through observe, then finish" \
  --arm left_follower --rig configs/rig.example.yaml --dry-run \
  --max-steps 3 --api-timeout-s 30 --episode-timeout-s 90 \
  --log-path outputs/agent/api-fixture.jsonl
```

**`--dry-run` prevents hardware use, but API requests still incur charges unless `--offline` is
also supplied.** Named fixture RGB images, joint/gripper state, the task, and bounded conversation
history are transferred to OpenAI. The provider uses the Responses API with `store=False`,
`parallel_tool_calls=False`, strict tool schemas, a 512-token output limit, and no SDK retries.
The output cap includes reasoning tokens for reasoning models. Low-detail JPEG image inputs
are resized to at most 512 pixels per side, with at most four cameras. A request keeps at most
three complete previous turns plus the current observation; historical images are removed,
and pruning preserves function call/output pairs. These bounds limit
usage per request; they are not a monetary budget. Start with a small step count and check the
[current API pricing](https://developers.openai.com/api/docs/pricing) before paid episodes.

## Options and execution

`--model`, `--task`, and one `--arm` are required. The arm must identify a follower with a motor
gripper, and the rig must contain at least one camera configuration. `--rig` defaults to
`configs/rig.yaml`. Select exactly one of `--dry-run` or `--execute`;
`--offline` requires `--dry-run`. Configuration is validated before robot/provider construction.
`--execute` currently exits with the integration blockers before importing or opening hardware
and before making an API request.

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-steps` | `50` | Total decisions, including malformed, empty, duplicate, and observe-only replies. |
| `--settle-s` | `0.5` | Delay after each action, followed by a fresh observation. |
| `--max-joint-delta` | `0.10` | Per-joint delta clamp, in radians; can be lowered, never raised above `0.10`. |
| `--motion-timeout-s` | `5` | Deadline for each fixed-target operation. |
| `--api-timeout-s` | `30` | Deadline for each API request. |
| `--episode-timeout-s` | `300` | Overall episode deadline, in seconds. |
| `--log-path` | `outputs/agent/episode-<time>.jsonl` | A new JSONL file inside the repository; existing files are rejected. |

The model can call only these tools:

| Tool | Behavior |
| --- | --- |
| `observe()` | Return measured joint/gripper state and named RGB images as multimodal inputs. |
| `move_joints(delta)` | Require exactly six finite numbers, clamp each delta, and add it once to fresh measured joints. Preserve the starting gripper target. |
| `open_gripper()` | Target gripper `1`, preserving starting joints. |
| `close_gripper()` | Target gripper `0`, preserving starting joints. |
| `finish(success, reason)` | Stop and record **model-declared** success or failure, never independently verified success. |

Unknown fields, booleans in numeric deltas, malformed arguments, and unknown tools are rejected
locally. A response containing multiple tool calls executes none of them. Call IDs are preserved
and deduplicated; retries never replay actions. Scene text is observation data and cannot grant
new tools, change limits, or override instructions.

Robot interaction stays in one adapter using `Robot.get_observation()` and `Robot.send_action()`.
The intended physical path remains `YamFollower.send_action()` → `YamArm.command()`, including
the existing speed clamps and firmware timeout. An operation computes **one absolute target**
from its initial observation, then repeatedly submits that same target at a bounded cadence.
It does not repeatedly add a relative delta. The returned command describes what was sent;
completion is established separately from measured feedback. Timeout, cancellation, invalid or
stale feedback, and excessive tracking error stop the operation. After an action, the controller
settles and reacquires state/images before asking the model for another decision.

The operation submits at most 10 times per second. Joint arrival tolerance is 0.01 rad and
gripper tolerance is 0.03; tracking error over 0.35 rad (or 0.35 gripper units) aborts, including
the observation after settling. `--max-joint-delta` can lower the 0.10 rad cap, but cannot raise it.
API/motion/episode deadlines are checked around synchronous calls; late API responses cannot
execute actions. The SDK timeout limits transport inactivity, not total wall time. A blocked
native robot call or continuously trickling HTTP response cannot be preempted by this loop.
This is an additional limitation to resolve before enabling hardware execution.

Logs are capped at 2 MiB, with 64 KiB per event and reserved space for termination. They include
task/model, timing, decision/tool IDs, camera names,
arguments, state, requested/bounded/sent/measured actions, errors, usage, and termination.
They contain no base64 image dumps or API keys. Task text, scene descriptions, and joint state
can still be private: keep generated logs under the ignored `outputs/` directory. A `finished`
episode exits successfully even when the model declares failure; inspect `success` and
`success_basis` in the termination record. Deadline/error/budget exhaustion returns a nonzero
exit status, and cancellation returns `130`.

## Why live execution is disabled

The integrated [`YamFollower`](../plugins/lerobot_robot_yamkit/lerobot_robot_yamkit/yam_follower.py)
now exposes `disconnect(home=False)`. It attempts camera/preview cleanup and every arm release
even when another resource raises an exception or cancellation. Partial startup failure uses
this no-home path. Fake-resource integration tests cover single and bimanual cleanup after
camera faults and cancellation; neither path adds a home command.

The remaining requirement is acquisition evidence. `YamFollower.get_observation()` still
returns camera `read_latest()` arrays without capture timestamps or frame sequence IDs in the
observation. Preview timestamps are optional diagnostics, not a freshness contract for the
controller. Joint observations also lack sensor acquisition metadata: the timestamp in
[`YamArm.read()`](../src/yamkit/arm.py) is generated when cached SDK state is read. Re-reading
state or an image after settling cannot prove that either was acquired after the action.

Accordingly, `make_live_robot()` still fails before plugin construction, hardware activation,
or an API call. Live support requires acquisition metadata for state and every image, with
strictly advancing sequences and acquisition times after the command completes and after
settling. Fixture tests prove that each completed tool action supplies new state/images to the
next decision, and that stale, invalid, or cached feedback stops further commands. They do not
establish physical sensor freshness.

Future live construction must also account for the ordinary plugin startup effects:
`YamFollower.connect()` enables and normally homes the arm;
[`YamArm.connect()`](../src/yamkit/arm.py) may auto-calibrate a gripper without saved limits.
Its public `calibrate` argument does not suppress those effects. The LLM gate prevents these
paths from being reached. Speed limits and the 400 ms motor firmware timeout remain unchanged.
The agent does not provide collision checking, IK, Cartesian tools, bimanual control, or
independently verified task success. Fixtures verify software behavior only.

## Development checks

```bash
make test
make lint
```

Use `HF_HUB_OFFLINE=1 make test` to prevent existing Hub/UI tests from accessing your Hub account.
Pytest temporary files stay under the repository's ignored `.pytest_cache/tmp` directory.

The agent tests mock provider responses and use fixture/fake robots. They make no paid API calls
and cover schemas, targets and readback, freshness, deadlines, duplicate/multiple calls,
cleanup, credential precedence, and CLI mode isolation. Real API verification is a separate,
explicitly budgeted development activity using fixtures only.

### Feature verification (2026-09-05)

Starting revision: `dbd01f1e70be66bb0e639789d0d37d7ecb5bd166`. The first implementation milestone
was committed and pushed as `700e23d` on `oslo`. No physical arms/cameras were opened, and no
CAN/system settings were changed. Final review added rejection of excessive post-settle drift.

The intended test model was **`gpt-5.4`**, using OpenAI SDK **2.54.0**, Responses, `low` reasoning,
and at most 512 output tokens per request. One real three-decision fixture episode called
`observe` → `move_joints` → `finish`. It correctly described the red shape on the left and blue
shape on the right, and reported measured joint 1 at 0.01 rad after one simulated command.
The fixture closed successfully. The final declaration was `success=false`, explicitly limited
to software verification; this does not validate physical manipulation.

Paid requests: **3 of 10**. Usage: **2,774 input + 312 output tokens**, including **195 reasoning
tokens** within the output count; no cached input tokens. At the checked
[official GPT-5.4 prices](https://developers.openai.com/api/docs/models/gpt-5.4) of $2.50/M input
and $15/M output, estimated cost is **$0.011615**, leaving **$1.988385** of the $2 allowance and
seven request slots unused. No further paid testing is needed or scheduled.

Before each call, the ignored persistent ledger `.context/agent/paid-usage.json` reserved
$0.173448: 60,000 input tokens, 512 maximum output/reasoning tokens, and a 10% surcharge reserve.
The input bound includes the 32,768-character text/context cap, framing, opaque conversation
items, and up to four bounded images under the official
[image token rules](https://developers.openai.com/api/docs/guides/images-vision). Total reserved
cost was $0.520344, leaving $1.479656 conservatively unreserved. Reservations and request counts
were written before calls, and the ledger is now marked complete/halted. The ledger and raw
JSONL records are development artifacts and are not committed.

Hardware-free validation: one full run had **265 tests passed** with Hub networking disabled;
`make lint` and `git diff --check` passed. After configuring repo-local pytest temporary files,
the final full repeat had **264 passed and one existing intermittent failure** at
`tests/test_arm.py:148` (`test_go_home_all_runs_arms_together_and_ctrl_c_releases_all`); it passed
again in isolation. Its arm code and test are unchanged from `origin/main`. All new agent tests
passed in both runs. An initial run also encountered a UI assertion influenced by live Hub
entries; `HF_HUB_OFFLINE=1` prevents that network dependence. A pre-existing Starlette/httpx
deprecation warning remains. Mocked tests include actual SDK request serialization over an
in-memory HTTP transport.
