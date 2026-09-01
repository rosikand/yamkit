# Development

## Repository map

```text
src/yamkit/                            core package and CLI
plugins/lerobot_robot_yamkit/          LeRobot follower Robot plugin
plugins/lerobot_teleoperator_yamkit/   LeRobot leader Teleoperator plugin
configs/rig.yaml                       hardware identity and control settings
scripts/                               environment and CAN setup helpers
system/                                opt-in systemd and udev templates
tests/                                 hardware-free yamkit tests
third_party/i2rt/                      pinned editable I2RT SDK
data/                                  local caches and datasets
outputs/                               training outputs and checkpoints
```

## Set up a development environment

```bash
./setup.sh
source scripts/env.sh
make test
make lint
```

After dependency or plugin metadata changes:

```bash
make sync
```

The root package requires Python 3.12. `uv.toml` keeps the managed interpreter and cache under this repository.

## Test and lint

```bash
uv run pytest -q
uv run pytest tests/test_arm.py -q
uv run ruff check src plugins tests
```

The yamkit test suite replaces hardware connections with `FakeRobot`. Keep default tests hardware-free: a developer or CI runner must not need SocketCAN devices or energize an arm.

The vendored I2RT tree is a separate project with its own instructions in `third_party/i2rt/AGENTS.md`. Only run or change its wider suite when work actually touches the SDK.

## Conventions

- Use Python 3.12 type hints and the existing dataclass-based configuration model.
- Keep the CLI in Typer and use Rich for operator-facing tables and status.
- Resolve repository paths through `yamkit.paths`; do not redirect project state to user-global caches.
- Keep I2RT calls inside `YamArm` and adapt that object in the existing LeRobot plugins.
- Preserve hardware behavior unless the task explicitly requests a behavior change.
- Record every local vendor change in `third_party/i2rt.VERSION` and protect it with a test where practical.
- Keep wrapper commands thin; unknown LeRobot options should remain pass-through options.

## Documentation

Install the documentation dependency and start a live-reloading local server:

```bash
uv sync --extra dev --extra docs
uv run --extra docs mkdocs serve
```

Open <http://127.0.0.1:8000>. Build the exact static site with strict link and configuration checks:

```bash
uv run --extra docs mkdocs build --strict
```

The generated `site/` directory is build output and should not be committed.

When changing runtime behavior, update the smallest relevant guide or reference page and verify every copied command against `yamkit <command> --help`.
