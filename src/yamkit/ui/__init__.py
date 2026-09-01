"""Local web UI for yamkit (`yamkit ui`).

The UI is a thin viewer/launcher on top of the existing toolkit: read-only pages read the rig
file, sysfs (CAN) and the filesystem (datasets, checkpoints); hardware actions run the unmodified
`yamkit read/teleop/record/rollout` CLI as child processes (`yamkit.ui.sessions`). Importing or
serving the UI never connects to an arm.
"""
