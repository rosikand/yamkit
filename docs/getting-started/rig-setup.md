# Configure a rig

`configs/rig.yaml` is the shared source of truth for arm identity, leader–follower pairs, cameras, calibration values, rest poses, and control limits.

## Discover adapters and arms

With each SocketCAN interface up:

```bash
yamkit discover
```

Discovery passively probes buses and classifies the attached device:

- a leader has motors 1–6 and a teaching-handle encoder;
- a follower has motors 1–7;
- an arm without a detected gripper is reported separately.

No motor is enabled by discovery. Review the proposal, then save it:

```bash
yamkit discover --write
```

Discovery assigns left and right in probe order. That order is not a physical guarantee.

## Verify physical identity

Clear the robot workspace, then connect one arm at a time:

```bash
yamkit read left_leader
yamkit read left_follower
```

Move the connected arm gently and confirm that its reported state matches the intended name. If two leaders or two followers were reversed, swap their physical mappings without changing pair names:

```bash
yamkit swap left_leader right_leader
yamkit swap left_follower right_follower
```

`swap` exchanges adapter identity, calibration, rest pose, and notes. It only accepts arms with the same role.

## Calibrate and prepare

A follower without `gripper_limits` auto-calibrates its linear gripper when it connects. To force calibration and persist the returned limits:

```bash
yamkit calibrate-gripper left_follower
```

Store the current joint pose for controlled parking:

```bash
yamkit set-rest left_follower
yamkit rest left_follower --duration 4
```

If a released teaching-handle trigger does not read close to `1.0`, release it fully and re-zero the encoder:

```bash
yamkit zero-handle right_leader
```

This writes the encoder zero offset to device EEPROM and prompts before doing so.

## Add cameras

OpenCV cameras and optional Intel RealSense cameras are declared under `cameras`. Find OpenCV devices with:

```bash
lerobot-find-cameras opencv
```

Then add verified device paths or serials to the rig. See [Rig configuration](../reference/rig-config.md#cameras) for the schema.

Finish with:

```bash
yamkit doctor
```
