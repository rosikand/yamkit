# Programmatic recipes

These recipes compose the existing API without adding another hardware abstraction. Hardware examples are explicitly marked and retain rig-derived limits and deterministic teardown.

## Inventory CAN without changing state

```python
from yamkit.can import list_can_interfaces

for interface in list_can_interfaces():
    print(
        {
            "name": interface.name,
            "state": interface.state,
            "bitrate": interface.bitrate,
            "serial": interface.serial,
            "rx": interface.rx_packets,
            "tx": interface.tx_packets,
            "errors": interface.bus_errors,
        }
    )
```

This only reads sysfs and `ip` output. Use it in health checks before opening an arm.

## Produce a discovery report without writing a rig

```python
from yamkit.can import list_can_interfaces
from yamkit.discovery import probe_all, suggest_rig

interfaces = list_can_interfaces()
probes = probe_all(interfaces)
draft = suggest_rig(probes, interfaces)

report = [
    {
        "interface": probe.iface,
        "classification": probe.classification,
        "motor_ids": sorted(probe.motor_ids),
        "encoder_versions": probe.encoder_versions,
        "type_mismatches": probe.type_mismatches,
        "error": probe.error,
    }
    for probe in probes
]

print(report)
print(draft.to_dict())
```

The CAN probes are passive and `suggest_rig()` stays in memory. Do not call `draft.save()` until an operator has verified physical left/right identity and reviewed which existing calibration values should be retained.

## Capture bounded telemetry

!!! warning "Hardware activation"

    `YamArm.connect()` energizes the selected arm in gravity-compensation mode.

```python
import time

from yamkit.arm import YamArm, resolve_channel
from yamkit.config import RigConfig

rig = RigConfig.load()
spec = rig.arm("left_leader")

samples = []
with YamArm.connect(spec, resolve_channel(spec)) as arm:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = arm.read()
        samples.append(
            {
                "timestamp": state.t,
                "q": state.q.tolist(),
                "qd": state.qd.tolist(),
                "tau": state.tau.tolist(),
                "gripper": state.gripper,
                "buttons": state.buttons,
            }
        )
        time.sleep(0.02)

print(f"captured {len(samples)} samples")
```

The context manager performs normal teardown even when sampling raises.

## Build a speed-clamped command loop

!!! danger "Commands physical motion"

    Clear and supervise the full workspace, verify adapter identity, and ensure no competing CAN controller is active.

```python
import time

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
    start = arm.read()
    requested = start.q + np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        actual = arm.command(requested, start.gripper)
        time.sleep(1.0 / control.teleop_hz)

    arm.hold()
```

The returned `actual` array is the target sent after rate limiting. Never assume it equals `requested` on the first call. Keep `limit_speed=True`, use a monotonic deadline, and avoid blocking work inside the command loop.

## Observe a direct teleop session

`on_tick` supports lightweight metrics without replacing `TeleopSession`'s control behavior:

```python
import math

from yamkit.config import RigConfig
from yamkit.teleop import TeleopSession


def report(session: TeleopSession) -> None:
    for pair in session.pairs:
        if not math.isnan(pair.tracking_error):
            print(pair.name, pair.engaged, pair.tracking_error)


rig = RigConfig.load()
session = TeleopSession.from_rig(
    rig,
    pair_names=["left_follower"],
    on_tick=report,
)
session.run(duration=30)
```

The callback runs synchronously in the control loop. Queue data to another thread or process for file, network, visualization, or expensive computation; a slow callback increases loop overruns.

From another thread, request normal shutdown with:

```python
session.stop_event.set()
```

## Check policy compatibility without hardware

```python
from yamkit.policy_check import run_policy_check

result = run_policy_check(
    "outputs/train/smolvla_pick_cube/checkpoints/last/pretrained_model",
    rig_path="configs/rig.yaml",
    arms=["left"],
    task="pick up the red cube",
    device="cpu",
    n_steps=3,
    fake_camera=(480, 640),
    use_robot_features=True,
)

print(
    {
        "policy_type": result.policy_type,
        "state_dim": result.state_dim,
        "action_dim": result.action_dim,
        "image_keys": result.image_keys,
        "first_call_s": result.first_call_s,
        "next_call_s": result.step_call_s,
        "sample_action": result.action,
    }
)
```

This can download or load a large checkpoint and allocate substantial CPU/GPU memory, but it does not connect to the robot. With `use_robot_features=True`, yamkit derives policy features from the selected rig as the training workflow does.

## Validate feature compatibility before connecting

Plugin construction is hardware-free. Compare leader output and follower input schemas before either `connect()` call:

```python
from lerobot.robots.utils import make_robot_from_config
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot_robot_yamkit import BiYamFollowerConfig
from lerobot_teleoperator_yamkit import BiYamLeaderConfig

robot = make_robot_from_config(
    BiYamFollowerConfig(rig="configs/rig.yaml")
)
teleoperator = make_teleoperator_from_config(
    BiYamLeaderConfig(rig="configs/rig.yaml")
)

if teleoperator.action_features != robot.action_features:
    raise RuntimeError("leader/follower feature mismatch")
```

Camera features appear only in `robot.observation_features`, not actions.

## Test control code without hardware

The repository's `FakeRobot` implements the I2RT methods consumed by `YamArm`. Inject it through the ordinary constructor and leave `command()`'s clamp enabled:

```python
import numpy as np

from tests.conftest import FakeRobot
from yamkit.arm import YamArm
from yamkit.config import ArmSpec


def test_first_command_is_clamped() -> None:
    spec = ArmSpec(
        name="test_follower",
        role="follower",
        gripper="linear_4310",
        can_serial="fake",
    )
    fake = FakeRobot(7, gripper=True)
    arm = YamArm(
        spec,
        "can-test",
        fake,
        max_joint_speed=1.0,
        max_gripper_speed=2.0,
    )

    actual = arm.command(np.ones(6), gripper=1.0)

    assert np.all(np.abs(actual[:6]) <= 0.01 + 1e-9)
    assert actual[-1] <= 0.02 + 1e-9
```

Use the existing `rig` and `fake_connect` pytest fixtures when testing `TeleopSession` or plugins. Do not add tests that look for real SocketCAN interfaces or energize hardware by default.

## Avoid private wrapper APIs

Functions in `yamkit.cli` whose names begin with `_` assemble command-line arguments and replace the current process with LeRobot. They are implementation details, not an embedding API. For programmatic dataset or rollout behavior, use LeRobot's public Python interfaces with the yamkit configuration classes; for a subprocess workflow, invoke the documented `yamkit ... --dry-run` wrappers.
