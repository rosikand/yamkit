# Working with the API

Use yamkit's Python API when a workflow cannot be expressed through the CLI or the installed LeRobot plugins. Keep `YamArm` as the single hardware boundary so speed limiting and shutdown behavior remain centralized.

## API map

<div class="grid cards" markdown>

-   :material-language-python: **Core Python API**

    Rig dataclasses, passive CAN discovery, arm state and command contracts, lifecycle methods, teleoperation, and offline policy checks.

    [Core API reference →](api-core.md)

-   :material-robot-industrial: **LeRobot plugin API**

    Robot and teleoperator configuration classes, feature dictionaries, cameras, overrides, and connection behavior.

    [LeRobot plugin reference →](api-lerobot.md)

-   :material-code-braces: **Programmatic recipes**

    Passive inventory, telemetry, bounded control loops, session callbacks, checkpoint checks, and fake-backed tests.

    [Programmatic recipes →](api-recipes.md)

</div>

The package is intentionally import-light at `import yamkit`. Application code imports concrete building blocks from `yamkit.config`, `yamkit.arm`, `yamkit.can`, `yamkit.discovery`, `yamkit.teleop`, and `yamkit.policy_check`.

## Load and inspect a rig

Configuration objects are dataclasses. Loading resolves relative paths against the repository root.

```python
from yamkit.config import RigConfig

rig = RigConfig.load("configs/rig.yaml")

problems = rig.validate()
if problems:
    raise RuntimeError("; ".join(problems))

for pair in rig.pairs:
    leader = rig.arm(pair.leader)
    follower = rig.arm(pair.follower)
    print(leader.name, "->", follower.name)
```

`RigConfig.save()` writes the supported version-1 YAML representation. Preserve hardware-specific calibration values when changing other fields.

## Read an arm safely

`resolve_channel()` maps an arm's configured interface or USB serial to a live, UP SocketCAN interface. `YamArm.connect()` energizes the arm in gravity-compensation mode.

```python
from yamkit.arm import YamArm, resolve_channel
from yamkit.config import RigConfig

rig = RigConfig.load()
spec = rig.arm("left_leader")

with YamArm.connect(spec, resolve_channel(spec)) as arm:
    state = arm.read()
    print(state.q)        # six joint positions, radians
    print(state.gripper)  # normalized 0..1, when present
    print(state.buttons)  # tuple for a teaching handle
```

The context manager returns the arm to gravity compensation and closes the I2RT connection on exit.

## Command a follower

Pass the rig limits into the connection and leave speed limiting enabled:

```python
import numpy as np

from yamkit.arm import YamArm, resolve_channel
from yamkit.config import RigConfig

rig = RigConfig.load()
spec = rig.arm("left_follower")
control = rig.control

with YamArm.connect(
    spec,
    resolve_channel(spec),
    max_joint_speed=control.max_joint_speed,
    max_gripper_speed=control.max_gripper_speed,
) as arm:
    measured = arm.read()
    requested = measured.q + np.array([0.05, 0, 0, 0, 0, 0])
    sent = arm.command(requested, measured.gripper)
    print(sent)  # actual rate-limited target sent to I2RT
```

!!! danger

    This example commands physical motion. Clear and supervise the workspace. Do not use `limit_speed=False` for ordinary control or policy integrations.

`ArmState.vector()` returns six joint values plus a gripper value when the arm has one. Gripper position is normalized as `0` closed and `1` open.

## Run a teleoperation session

`TeleopSession.from_rig()` connects the selected pair or pairs and adopts control defaults from the rig:

```python
from yamkit.config import RigConfig
from yamkit.teleop import TeleopSession

rig = RigConfig.load()
session = TeleopSession.from_rig(
    rig,
    pair_names=["left_follower"],
    auto_engage=False,
)
stats = session.run(duration=30)
print(stats.ticks, stats.rate_hz, stats.overruns)
```

Without `auto_engage`, the configured teaching-handle button toggles the pair. `run()` shuts the session down in a `finally` block.

## Use the LeRobot plugin API

The plugins are workspace packages and register their configuration classes with LeRobot.

```python
from lerobot.robots.utils import make_robot_from_config
from lerobot_robot_yamkit import YamFollowerConfig

config = YamFollowerConfig(
    rig="configs/rig.yaml",
    arm="left_follower",
    id="bench_follower",
)
robot = make_robot_from_config(config)

try:
    robot.connect()
    observation = robot.get_observation()
    action = {
        "joint_1.pos": float(observation["joint_1.pos"]),
        "joint_2.pos": float(observation["joint_2.pos"]),
        "joint_3.pos": float(observation["joint_3.pos"]),
        "joint_4.pos": float(observation["joint_4.pos"]),
        "joint_5.pos": float(observation["joint_5.pos"]),
        "joint_6.pos": float(observation["joint_6.pos"]),
        "gripper.pos": float(observation["gripper.pos"]),
    }
    actual = robot.send_action(action)
finally:
    if robot.is_connected:
        robot.disconnect()
```

For two followers, use `BiYamFollowerConfig`; action and state keys gain `left_` and `right_` prefixes. `YamLeaderConfig` and `BiYamLeaderConfig` expose matching action features through LeRobot's teleoperator API.

## Extension rules

- Route hardware reads and commands through `YamArm`; do not wrap `MotorChainRobot` elsewhere.
- Route LeRobot integration through the existing plugin packages rather than duplicating recording or policy infrastructure.
- Preserve rig-derived speed clamps, normal shutdown, and the firmware timeout.
- Add hardware-free tests using `tests/conftest.py` fakes for new state or action behavior.
