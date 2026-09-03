# yamkit — notes for AI assistants working in this repo

Read `README.md` first. Hard rules for this repository:

- **Self-contained.** Interpreter, venv, uv cache, vendored SDK, datasets, models, checkpoints all live
  under this directory (`.uv-python/`, `.venv/`, `.uv-cache/`, `third_party/`, `data/`, `outputs/`).
  Never install system packages or write outside the repo. The single exception is the boot-time CAN
  bring-up (`system/80-yam-can.network`), installed only by `scripts/install_system.sh`, which
  `setup.sh` offers interactively; never add other system-level steps.
- **It must just work for a new user.** `git clone` + `./setup.sh` on a fresh machine with a new
  bimanual rig has to end with a working `configs/rig.yaml` (machine-specific, git-ignored; the
  format is `configs/rig.example.yaml`). Prefer auto-discovery (`yamkit discover --write` for arms
  AND cameras, idempotent: keeps names/calibration by serial) over hand-edited config; keep the rig
  file readable by non-programmers (`RigConfig.save` writes the commented layout — extend the
  comments in `config.py` when adding fields). Left/right of arms and wrist cameras is a physical
  check the user does once (`yamkit read`, `yamkit swap`); do not try to auto-detect it.
- **Do not reuse code from other folders on the machine.** Re-download / vendor instead.
- **LeRobot is the infrastructure** (datasets, cameras, `lerobot-record/teleoperate/rollout/train`).
  `yamkit` only adds the YAM-specific hardware layer and thin CLI wrappers; the LeRobot plugins in
  `plugins/` are the integration point.
- **Hardware safety.** `yamkit can` / `yamkit discover` / `yamkit policy-check` never energise a motor.
  Everything else does. Followers move to the leader pose on engage; keep speed clamps
  (`control.max_joint_speed`) and the motors' 400 ms firmware timeout as they are.
- **Vendored SDK patches** are listed in `third_party/i2rt.VERSION`; keep that list current.

Commands: `./setup.sh` (bootstrap), `source scripts/env.sh`, `make test`, `make lint`, `yamkit doctor`.
Hardware-free tests use the fake robot in `tests/conftest.py`.
