# yamkit agent guide

## Purpose and architecture

yamkit is the self-contained YAM-specific layer between I2RT robot hardware and LeRobot workflows. It provides SocketCAN discovery, rig configuration, guarded arm commands, direct leader-to-follower teleoperation, and thin wrappers for LeRobot recording, training, and rollout.

The control path is `configs/rig.yaml` → `RigConfig` → `YamArm` → vendored I2RT `MotorChainRobot` → CAN hardware. Direct teleop uses `TeleopSession`; LeRobot reaches the same `YamArm` boundary through the robot and teleoperator plugins. I2RT owns motor drivers, gravity compensation, robot construction, and models. LeRobot owns cameras, datasets, training, and policy execution.

## Important locations

- `src/yamkit/arm.py`: sole hardware/control abstraction; state normalization, speed clamps, teardown.
- `src/yamkit/config.py`, `configs/rig.yaml`: rig schema and source of hardware identity/control limits.
- `src/yamkit/teleop.py`: engagement, synchronization, tracking, bilateral feedback, shutdown.
- `src/yamkit/cli.py`: direct commands and thin LeRobot wrappers.
- `plugins/`: LeRobot `Robot` and `Teleoperator` adapters; they must delegate hardware access to `YamArm`.
- `tests/`: hardware-free tests using `FakeRobot` in `tests/conftest.py`.
- `third_party/i2rt/`: pinned editable SDK; revision and local patch ledger are in `third_party/i2rt.VERSION`.
- `docs/`, `mkdocs.yml`: Material for MkDocs documentation.

## Setup and checks

```bash
./setup.sh
source scripts/env.sh
make test
make lint
uv sync --extra dev --extra docs
uv run --extra docs mkdocs build --strict
```

The project requires Python 3.12 and keeps its interpreter, environment, uv cache, datasets, model caches, and outputs inside the repository. Do not install system packages or systemd/udev templates unless explicitly requested. Default tests must remain hardware-free.

## Development rules

- Use typed Python, dataclasses for configuration, Typer for CLI commands, and Rich for operator output.
- Resolve repository-owned paths through `yamkit.paths` and preserve the self-contained cache layout.
- Keep LeRobot wrappers thin and pass unknown framework flags through.
- Do not duplicate `YamArm`, I2RT's motor/control layer, or the existing LeRobot plugin abstractions. Extend those boundaries directly.
- Preserve existing hardware behavior unless the user explicitly requests a behavior change.
- Treat vendored I2RT changes as exceptional: follow `third_party/i2rt/AGENTS.md`, update `third_party/i2rt.VERSION`, and add relevant guards.

## Working with the API

- Load hardware and control settings through `RigConfig`; never hard-code adapter identity, calibration, or speed limits.
- Resolve hardware with `resolve_channel()` and manage `YamArm` with a context manager or equivalent `try/finally` teardown.
- Retain rig-derived `max_joint_speed` and `max_gripper_speed`; ordinary calls to `command()` must keep `limit_speed=True`.
- Prefer the registered LeRobot robot/teleoperator plugin APIs for dataset or policy workflows rather than rebuilding their feature mapping.
- Add hardware-free fake-backed tests for any new state, action, or API behavior.

## Robot safety invariants

- Only `yamkit can`, `yamkit discover`, and `yamkit policy-check` are guaranteed not to energize motors. Assume other hardware commands energize or move the robot.
- Never bypass or weaken `YamArm.command()` speed clamps in normal control or policy paths.
- Preserve follower synchronization on teleop engage and orderly gravity-idle/hold teardown.
- Keep the motor firmware timeout at its 400 ms factory default; never disable or extend it.
- Do not change joint/gripper conventions, gains, calibration, CAN identity, movement timing, or shutdown behavior without an explicit hardware-behavior request.
- Tests and documentation examples must not imply that physical safety is guaranteed by software. Require a clear, supervised workspace and no competing CAN controller.
