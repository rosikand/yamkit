# Safety

yamkit controls physical robot arms. Software limits reduce risk; they do not make the workspace intrinsically safe.

## Command risk levels

| Category | Commands | Hardware effect |
| --- | --- | --- |
| Passive | `can`, `discover`, `policy-check`, `doctor`, `env`, `version` | `can`, `discover`, and `policy-check` do not energize motors. `doctor` inspects devices and configuration. |
| Connects to arms | `read`, `set-rest`, `calibrate-gripper` | Enables motors in I2RT gravity-compensation mode. Gripper calibration can move the gripper. |
| Commands motion | `teleop`, `teleoperate`, `rest`, `record`, `rollout` | Sends position targets; teleop engagement first moves each follower toward its leader pose. |
| Device configuration | `zero-handle` | Writes the passive encoder's zero offset to EEPROM. |

Training does not connect to robot hardware, but it can consume significant compute resources.

## Before enabling hardware

1. Clear people, cables, tools, and loose objects from the full arm and gripper workspace.
2. Confirm the correct CAN serial-to-arm mapping with `yamkit can` and the rig file.
3. Ensure no other controller is transmitting on the same buses. In particular, stop or verify idle any `ctrl_pi` Docker container. `candump can0` can help reveal unexpected traffic.
4. Verify the leader and follower names physically before engaging a pair.
5. Keep a fast, known way to remove power or stop the process.

## During teleoperation and rollout

- Expect a follower to move to the leader pose over `control.sync_seconds` when engaged.
- Start with conservative poses and only one pair when validating new hardware or configuration.
- Do not bypass `YamArm.command()`. Its `max_joint_speed` and `max_gripper_speed` clamps bound each position-target change.
- Treat policy output as untrusted. Use `yamkit policy-check` before a rollout, keep the area clear, and supervise the full run.
- Stop immediately on unexpected identity, direction, speed, bus errors, or tracking behavior.

## Safety mechanisms that must remain intact

- `YamArm.command()` starts a stale command stream from the measured pose and clamps target steps by elapsed time.
- The configured follower speed limits apply to direct teleop and LeRobot actions.
- On normal shutdown, leaders return to gravity compensation and followers hold before connections close.
- The motor firmware timeout remains at its **400 ms factory default**. Do not disable or extend it.

!!! warning "Bilateral feedback changes leader behavior"

    A positive `bilateral_kp` commands the leader toward the measured follower pose. Leave it at `0.0` until ordinary leader-to-follower teleoperation is verified. The repository notes `0.1–0.2` as the intended range when it is deliberately enabled.
