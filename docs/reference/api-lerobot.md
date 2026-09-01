# LeRobot plugin API

yamkit's two plugin packages adapt `YamArm` to LeRobot. They do not implement a second controller: every follower action still passes through `YamArm.command()` and its speed clamps.

## Registered types

| Package | Configuration | Runtime class | Registered type |
| --- | --- | --- | --- |
| `lerobot_robot_yamkit` | `YamFollowerConfig` | `YamFollower` | `yam_follower` |
| `lerobot_robot_yamkit` | `BiYamFollowerConfig` | `BiYamFollower` | `bi_yam_follower` |
| `lerobot_teleoperator_yamkit` | `YamLeaderConfig` | `YamLeader` | `yam_leader` |
| `lerobot_teleoperator_yamkit` | `BiYamLeaderConfig` | `BiYamLeader` | `bi_yam_leader` |

The workspace packages export both configuration and runtime classes from their top-level modules.

## Follower configuration

```python
from lerobot_robot_yamkit import BiYamFollowerConfig, YamFollowerConfig
```

Common configuration fields:

| Field | Default | Behavior |
| --- | --- | --- |
| `rig` | `configs/rig.yaml` | Rig used for hardware identity, cameras, and control defaults. |
| `id` | type-specific | LeRobot robot identifier. |
| `cameras` | `{}` | Explicit LeRobot `CameraConfig` objects. Non-empty values override rig cameras. |
| `use_rig_cameras` | `True` | Decode `rig.cameras` when explicit cameras are empty. |
| `max_joint_speed` | `None` | Per-plugin override; `None` uses the rig control value. |
| `max_gripper_speed` | `None` | Per-plugin override; `None` uses the rig control value. |

`YamFollowerConfig` adds `arm`, defaulting to `right_follower`. `BiYamFollowerConfig` adds `left` and `right`, defaulting to the corresponding follower names.

Construction loads and validates the named arm role but does not connect hardware. `connect()` opens the arm before its cameras and rolls the arm connection back if a camera fails. `disconnect()` closes cameras before the arm.

The plugin reports `is_calibrated=True`; its `calibrate()` and `configure()` methods are no-ops because gripper limits live in the rig and motor calibration/offsets live in hardware.

## Follower feature contracts

### Single arm

For a follower with a motorized gripper:

```python
{
    "joint_1.pos": float,
    "joint_2.pos": float,
    "joint_3.pos": float,
    "joint_4.pos": float,
    "joint_5.pos": float,
    "joint_6.pos": float,
    "gripper.pos": float,
}
```

Joint values are radians. The gripper is normalized to `0` closed and `1` open. A `no_gripper` follower omits `gripper.pos`.

`action_features` contains motor features. `observation_features` contains the same motor features plus camera entries whose values are `(height, width, 3)` tuples. `get_observation()` returns named scalar state plus the latest image from each camera. `send_action()` requires all six joint keys and returns the actual, potentially rate-limited target sent.

### Bimanual

The bimanual plugin prefixes motor keys with its logical side:

```python
{
    "left_joint_1.pos": float,
    # ...
    "left_gripper.pos": float,
    "right_joint_1.pos": float,
    # ...
    "right_gripper.pos": float,
}
```

Camera names remain unprefixed and are shared by the bimanual robot. `send_action()` partitions the mapping by `left_` and `right_`, commands each follower, and returns prefixed actual targets.

## Leader configuration and features

```python
from lerobot_teleoperator_yamkit import BiYamLeaderConfig, YamLeaderConfig
```

`YamLeaderConfig` has `rig`, `id`, and `arm`. `BiYamLeaderConfig` has `rig`, `id`, `left`, and `right`.

The single leader exposes the same six joint keys and, for a teaching handle, `gripper.pos`. `get_action()` reads the leader; it does not command it. Bimanual output uses the same `left_`/`right_` prefix convention as the follower, so its action dictionary can be passed directly to the matching bimanual robot.

Leader `feedback_features` is empty and `send_feedback()` is a no-op. Bilateral force feedback exists only in yamkit's direct `TeleopSession`, not the LeRobot teleoperator plugin.

## Construct through LeRobot factories

```python
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot_robot_yamkit import YamFollowerConfig
from lerobot_teleoperator_yamkit import YamLeaderConfig

robot = make_robot_from_config(
    YamFollowerConfig(
        id="bench_follower",
        rig="configs/rig.yaml",
        arm="left_follower",
    )
)
teleoperator = make_teleoperator_from_config(
    YamLeaderConfig(
        id="bench_leader",
        rig="configs/rig.yaml",
        arm="left_leader",
    )
)
```

At this point neither object is connected. A complete lifecycle must unwind partial failures:

```python
robot_connected = False
teleoperator_connected = False

try:
    teleoperator.connect()
    teleoperator_connected = True
    robot.connect()
    robot_connected = True

    observation = robot.get_observation()
    action = teleoperator.get_action()
    actual = robot.send_action(action)
finally:
    if robot_connected:
        robot.disconnect()
    if teleoperator_connected:
        teleoperator.disconnect()
```

!!! danger

    Both `connect()` calls energize their arms. The final `send_action()` commands the follower toward the leader without the direct teleop session's synchronization phase. For ordinary manual operation, use `yamkit teleop` or `TeleopSession` so engagement includes the controlled synchronization move.

## Override cameras programmatically

Explicit camera objects take precedence over the rig:

```python
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot_robot_yamkit import YamFollowerConfig

config = YamFollowerConfig(
    rig="configs/rig.yaml",
    arm="left_follower",
    cameras={
        "top": OpenCVCameraConfig(
            index_or_path="/dev/video0",
            width=640,
            height=480,
            fps=30,
        )
    },
)
```

Set `use_rig_cameras=False` with an empty `cameras` mapping for a state-only integration. This affects the constructed plugin; it does not mutate `configs/rig.yaml`.

## Discover registered types

LeRobot's CLIs register installed third-party plugins automatically. An embedding application can request registration before inspecting choices:

```python
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.import_utils import register_third_party_plugins

register_third_party_plugins()

assert "yam_follower" in RobotConfig.get_known_choices()
assert "bi_yam_follower" in RobotConfig.get_known_choices()
assert "yam_leader" in TeleoperatorConfig.get_known_choices()
assert "bi_yam_leader" in TeleoperatorConfig.get_known_choices()
```

`yamkit doctor` performs the same registration check as part of environment validation.

## Dataset naming boundary

The plugin operates on flat named scalar and camera mappings. LeRobot's processors and dataset writer convert those into dataset-level fields such as `observation.state`, `action`, and `observation.images.<camera>`. Keep that conversion in LeRobot instead of adding dataset logic to the plugins.
