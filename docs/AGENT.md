# Multimodal LLM episodes

`yamkit agent` runs a small synchronous controller alongside the existing LeRobot VLA `rollout`.
The available modes use **labeled synthetic robot state and RGB images**. Live execution is
disabled because the current robot interface cannot provide the required fault cleanup and
capture freshness guarantees; the exact blockers are below.

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
are resized to at most 512 pixels per side, with at most four cameras. Recent conversation
history is capped at four complete turns; historical images are removed. These bounds limit
usage per request; they are not a monetary budget. Start with a small step count and check the
[current API pricing](https://openai.com/api/pricing/) before paid episodes.

## Options and execution

`--model`, `--task`, and one `--arm` are required. The arm must identify a follower with a motor
gripper. `--rig` defaults to `configs/rig.yaml`. Select exactly one of `--dry-run` or `--execute`;
`--offline` requires `--dry-run`. Configuration is validated before robot/provider construction.
`--execute` currently exits with the integration blockers before importing or opening hardware
and before making an API request.

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-steps` | `50` | Total decisions, including malformed, empty, duplicate, and observe-only replies. |
| `--settle-s` | `0.5` | Delay after each action, followed by a fresh observation. |
| `--max-joint-delta` | `0.10` | Per-joint delta clamp, in radians. |
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

Logs have bounded events/size and include task/model, timing, decision/tool IDs, camera names,
arguments, state, requested/bounded/sent/measured actions, errors, usage, and termination.
They contain no base64 image dumps or API keys. Task text, scene descriptions, and joint state
can still be private: keep generated logs under the ignored `outputs/` directory. A `finished`
episode exits successfully even when the model declares failure; inspect `success` and
`success_basis` in the termination record. Deadline/error/budget exhaustion returns a nonzero
exit status, and cancellation returns `130`.

## Why live execution is disabled

These are existing integration gaps, not issues a CLI flag can safely bypass:

* [`YamFollower.disconnect()`](../plugins/lerobot_robot_yamkit/lerobot_robot_yamkit/yam_follower.py#L163)
  has no public no-home option. It disconnects cameras before releasing the arm, so a camera
  teardown exception skips arm cleanup. `_FollowerHandle.disconnect()` defaults to starting a
  home trajectory. The private handle's `home=False` parameter does not satisfy a public,
  verified fault-cleanup contract.
* [`YamFollower.connect()`](../plugins/lerobot_robot_yamkit/lerobot_robot_yamkit/yam_follower.py#L131)
  enables the arm and normally homes it before camera startup. Camera startup failure calls
  the same homing cleanup. [`YamArm.connect()`](../src/yamkit/arm.py#L83) can also perform a
  gripper open/close auto-calibration when limits are absent. The public `calibrate` argument
  does not suppress these effects.
* [`YamFollower.get_observation()`](../plugins/lerobot_robot_yamkit/lerobot_robot_yamkit/yam_follower.py#L152)
  returns camera `read_latest()` arrays without capture timestamps or frame sequence IDs.
  Joint observations likewise lack sensor acquisition metadata: the timestamp in
  [`YamArm.read()`](../src/yamkit/arm.py#L128) is generated when cached SDK state is read, not
  when hardware feedback was acquired. Re-reading an array after settling cannot prove freshness.

Live support needs public cleanup that always releases connected resources without starting a
home move, plus verifiable state/frame acquisition freshness through the observation boundary.
This feature leaves plugins, `control.home_speed`, speed limits, and the 400 ms motor firmware
timeout unchanged. It does not provide collision checking, IK, Cartesian tools, bimanual
control, or independently verified task success. Fixtures verify software behavior only.

## Development checks

```bash
make test
make lint
```

The agent tests mock provider responses and use fixture/fake robots. They make no paid API calls
and cover schemas, targets and readback, freshness, deadlines, duplicate/multiple calls,
cleanup, credential precedence, and CLI mode isolation. Real API verification is a separate,
explicitly budgeted development activity using fixtures only.
