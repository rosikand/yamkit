# Rig configuration

The default rig is `configs/rig.yaml`. Every hardware command accepts `--rig` to select another file.

```yaml
version: 1
arms:
  left_leader:
    role: leader
    side: left
    arm_type: yam
    gripper: yam_teaching_handle
    can_serial: "<adapter-serial>"
  left_follower:
    role: follower
    side: left
    arm_type: yam
    gripper: linear_4310
    can_serial: "<adapter-serial>"
    gripper_limits: [6.47, 1.16]
    rest_pose: [0.0, 0.2, -0.4, 0.0, 0.2, 0.0]
pairs:
  - leader: left_leader
    follower: left_follower
cameras: {}
control:
  teleop_hz: 100.0
  sync_seconds: 3.0
  bilateral_kp: 0.0
  engage_button: 0
  max_joint_speed: 3.0
  max_gripper_speed: 3.0
```

The numeric calibration values above only illustrate the field shapes. Keep the values measured for your hardware; do not copy calibration from another arm.

## Arms

| Field | Meaning |
| --- | --- |
| `role` | Required: `leader` or `follower`. |
| `side` | Informational physical side such as `left` or `right`; wrappers can select pairs by side. |
| `arm_type` | `yam`, `yam_pro`, `yam_ultra`, `yam_ultra_2`, or `big_yam`. Defaults to `yam`. |
| `gripper` | `crank_4310`, `linear_3507`, `linear_4310`, `flexible_4310`, `yam_teaching_handle`, or `no_gripper`. |
| `can_serial` | Preferred USB serial identity for the CAN adapter. |
| `can_iface` | Explicit interface name; when present it takes precedence over `can_serial`. |
| `gripper_limits` | Closed and open motor angles returned by I2RT calibration. Presence skips auto-calibration. |
| `rest_pose` | Six joint angles in radians used by `yamkit rest`. |
| `notes` | Operator context with no runtime meaning. |

Each arm needs `can_serial` or `can_iface`. A motorized gripper adds a seventh normalized degree of freedom; the teaching handle exposes its trigger as a normalized gripper action.

## Pairs

Each pair references one arm whose role is `leader` and one whose role is `follower`. The direct teleop loop supports multiple configured pairs. LeRobot wrappers select one or two pairs.

## Cameras

Camera entries are passed into LeRobot camera configuration objects. The adapter recognizes the types implemented in `yamkit.cameras`:

=== "OpenCV"

    ```yaml
    cameras:
      top:
        type: opencv
        index_or_path: /dev/video0
        width: 640
        height: 480
        fps: 30
    ```

=== "Intel RealSense"

    ```yaml
    cameras:
      wrist:
        type: intelrealsense
        serial_number_or_name: "<camera-serial>"
        width: 640
        height: 480
        fps: 30
    ```

RealSense requires `uv sync --extra realsense`. Unknown camera types raise an error.

## Control

| Field | Default | Effect |
| --- | ---: | --- |
| `teleop_hz` | `100.0` | Direct teleoperation loop target frequency. |
| `sync_seconds` | `3.0` | Duration of the follower's initial interpolation to leader pose. |
| `bilateral_kp` | `0.0` | Leader feedback gain scale; zero disables bilateral feedback. |
| `engage_button` | `0` | Teaching-handle button index used for edge-triggered engage/disengage. |
| `max_joint_speed` | `3.0` | Maximum follower target change in radians per second. |
| `max_gripper_speed` | `3.0` | Maximum normalized gripper target change per second. |

Control limits apply when `YamArm.command()` is called with its default `limit_speed=True`. Preserve that path for all normal control and policy integrations.
