# Installation

yamkit targets Linux with SocketCAN and Python 3.12. The bootstrap script installs `uv`, Python, the virtual environment, and Python dependencies inside the repository.

## Prerequisites

- Linux with SocketCAN support
- `gcc`, `g++`, `make`, and `curl`
- Internet access for the initial dependency install
- `sudo` when bringing CAN interfaces up
- Optional: `can-utils` for `candump` and other bus diagnostics

On Ubuntu, install the system prerequisites before running the project bootstrap:

```bash
sudo apt install build-essential curl can-utils
```

## Bootstrap

```bash
git clone <repo-url> yamkit
cd yamkit
./setup.sh
source scripts/env.sh
```

`scripts/env.sh` activates `.venv` and directs Hugging Face, LeRobot, Torch, and Weights & Biases data into `data/` and `outputs/`. Python launched from `.venv` also loads this cache configuration through `yamkit_env.pth`.

Run commands without activating the environment when preferred:

```bash
uv run yamkit --help
.venv/bin/yamkit --help
```

## Validate the install

```bash
yamkit doctor
make test
make lint
```

`doctor` checks the interpreter, Torch, I2RT, LeRobot plugin registration, environment variables, SocketCAN interfaces, cameras, and the rig file. Unit tests use a fake robot and do not require attached hardware.

## Bring up CAN

Plug in the CAN adapters, then inspect them before changing interface state:

```bash
yamkit can
scripts/can_up.sh
yamkit can
```

The script configures every detected SocketCAN interface at **1 Mbit/s**. Use the reset mode only to recover an unresponsive or erroring bus:

```bash
scripts/can_up.sh --reset
```

!!! note "Boot-time setup is deliberate"

    `system/80-yam-can.network`, `system/yamkit-can.service`, and `system/90-yam-can.rules` are templates only. They are not installed by `setup.sh`. Review paths and adapter identities before making system changes.

## Optional dependencies

Intel RealSense support is opt-in:

```bash
uv sync --extra dev --extra realsense
```

After changing the root project or either plugin's `pyproject.toml`, refresh the workspace environment with `make sync`.
