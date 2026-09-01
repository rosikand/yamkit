# yamkit — notes for AI assistants working in this repo

Read `README.md` first. Hard rules for this repository:

- **Self-contained.** Interpreter, venv, uv cache, vendored SDK, datasets, models, checkpoints all live
  under this directory (`.uv-python/`, `.venv/`, `.uv-cache/`, `third_party/`, `data/`, `outputs/`).
  Never install system packages or write outside the repo; anything system-level (udev, systemd)
  lives as templates in `system/` and is installed only on explicit request.
- **Do not reuse code from other folders on the machine.** Re-download / vendor instead.
- **LeRobot is the infrastructure** (datasets, cameras, `lerobot-record/teleoperate/rollout/train`).
  `yamkit` only adds the YAM-specific hardware layer and thin CLI wrappers; the LeRobot plugins in
  `plugins/` are the integration point.
- **Hardware safety.** `yamkit can` / `yamkit discover` / `yamkit policy-check` and the storage
  commands (`yamkit dataset|model|storage`) never energise a motor.
  Everything else does. Followers move to the leader pose on engage; keep speed clamps
  (`control.max_joint_speed`) and the motors' 400 ms firmware timeout as they are.
- **Vendored SDK patches** are listed in `third_party/i2rt.VERSION`; keep that list current.

Commands: `./setup.sh` (bootstrap), `source scripts/env.sh`, `make test`, `make lint`, `yamkit doctor`.
Hardware-free tests use the fake robot in `tests/conftest.py`.
