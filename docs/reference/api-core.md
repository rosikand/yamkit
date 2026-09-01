# Core Python API

This page documents yamkit's programmatic building blocks and their runtime contracts. Import objects from their defining modules; the top-level `yamkit` package only exposes lightweight package metadata.

## Safety and lifecycle at a glance

| API | Opens CAN | Energizes motors | Commands motion |
| --- | --- | --- | --- |
| `list_can_interfaces()` | No | No | No |
| `probe_channel()` / `probe_all()` | Yes | No; register reads only | No |
| `suggest_rig()` / `RigConfig.load()` | No | No | No |
| `robot_features_from_rig()` | No | No | No |
| `run_policy_check()` | No | No | No |
| `YamArm.connect()` | Yes | Yes, in gravity compensation | Connection may calibrate an uncalibrated motorized gripper |
| `YamArm.command()` / `move_to()` | Uses an open connection | Already energized | Yes |
| `TeleopSession.from_rig()` | Yes | Yes | Connects both sides; motion starts on engagement |

!!! danger

    Constructors such as `YamArm(...)` are used for dependency injection in tests. Real integrations should use `YamArm.connect()` and must treat the call as hardware activation.

## Configuration objects

```python
from yamkit.config import ArmSpec, ControlSpec, PairSpec, RigConfig
```

### `RigConfig`

`RigConfig` is the in-memory representation of a version-1 rig file.

```python
rig = RigConfig.load("configs/rig.yaml")
problems = rig.validate()

leaders = rig.leaders()
followers = rig.followers()
left = rig.arm("left_follower")
pair = rig.pair_for(left.name)
```

| Member | Contract |
| --- | --- |
| `load(path=None)` | Load YAML from `path`; with no path, use `configs/rig.yaml`. Relative paths resolve from the repository root. |
| `save(path=None)` | Serialize the current dataclasses as version-1 YAML and update `rig.path`. This changes a repository file. |
| `arm(name)` | Return an `ArmSpec`; raise `KeyError` with available names when absent. |
| `leaders()` / `followers()` | Return arms in YAML insertion order filtered by role. |
| `pair_for(name)` | Return the first pair containing an arm or `None`. |
| `validate()` | Return all pair-reference, role, and missing-CAN-identity problems without raising. |
| `to_dict()` / `from_dict()` | Convert between supported dataclasses and plain mappings. |

`ArmSpec` validates its `role`, `arm_type`, `gripper`, and six-element `rest_pose` when constructed. Its convenience properties are `has_motor_gripper`, `has_handle`, and `n_dofs`.

`ControlSpec` contains direct teleop frequency, synchronization duration, bilateral gain, engage-button index, and maximum follower joint/gripper target speeds. See [Rig configuration](rig-config.md) for every field.

## Repository paths

```python
from yamkit.paths import DATASETS_DIR, DEFAULT_RIG, OUTPUT_DIR, ROOT, resolve

checkpoint = resolve("outputs/train/demo/checkpoints/last/pretrained_model")
```

`resolve()` returns absolute paths unchanged and resolves relative paths below the detected repository root. `ROOT` is found from the installed yamkit location or `YAMKIT_ROOT`, falling back to the process working directory.

## Passive CAN inventory

```python
from yamkit.can import (
    CanIface,
    bringup_commands,
    find_by_name,
    find_by_serial,
    list_can_interfaces,
    udev_rules_text,
)

interfaces = list_can_interfaces()
for interface in interfaces:
    print(
        interface.name,
        interface.state,
        interface.bitrate,
        interface.serial,
        interface.rx_packets,
        interface.tx_packets,
        interface.bus_errors,
    )

adapter = find_by_serial("<adapter-serial>", interfaces)
commands = bringup_commands([i.name for i in interfaces if not i.up])
```

`list_can_interfaces()` combines `/sys/class/net` with `ip -j -d link` output. It returns only interfaces whose kernel hardware type is SocketCAN. `find_by_serial()` and `find_by_name()` return `None` when no match exists.

`bringup_commands()` and `udev_rules_text()` return text only; they do not execute commands or install rules. The default generated bitrate is `1_000_000`.

### `CanIface`

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | `str` | Current kernel interface name. |
| `up` / `state` | `bool` / `str` | Whether both `UP` and `LOWER_UP` flags are present; `state` renders `UP` or `DOWN`. |
| `bitrate` | `int \| None` | Configured CAN bitrate when reported by `ip`. |
| `serial` | `str \| None` | USB adapter serial, preferred for rig identity. |
| `product`, `manufacturer`, `usb_path` | optional strings | USB sysfs metadata. |
| `rx_packets`, `tx_packets`, `bus_errors` | `int` | Interface counters; `bus_errors` combines RX and TX errors. |

## Passive arm discovery

```python
from yamkit.can import list_can_interfaces
from yamkit.discovery import probe_all, suggest_rig

interfaces = list_can_interfaces()
probes = probe_all(interfaces)
draft = suggest_rig(probes, interfaces)

for probe in probes:
    print(probe.iface, probe.classification, probe.type_mismatches)
```

`probe_channel()` sends Damiao float-register reads for motor IDs 1–7 and optionally requests teaching-handle encoder versions. These requests do not enable motors. `probe_all()` returns an error-valued `ChannelProbe` for a down interface rather than opening it.

`ChannelProbe.classification` is one of `error`, `empty`, `follower`, `leader`, `arm_no_gripper`, or `partial`. `MotorProbe.motor_type` maps known gear ratios to DM motor types.

`suggest_rig(probes, interfaces, existing=None)` builds a `RigConfig` in memory. Names and sides follow discovery order and remain provisional. When `existing` is supplied, matching adapter serials retain rest poses, gripper limits, gripper/arm types, cameras, and control settings. The function does not save the draft.

## Arm state

```python
from yamkit.arm import ArmState
```

An `ArmState` contains:

- `t`: wall-clock sample timestamp;
- `q`: six joint positions in radians;
- `qd`: six joint velocities in radians per second;
- `tau`: six reported motor efforts in Nm;
- `gripper`: normalized `0..1`, or `None` without a gripper/handle;
- `buttons`: teaching-handle inputs, or `None` on followers.

`state.vector()` returns a copy of `q` and appends the gripper when present. A leader trigger is inverted so released maps to `1`/open and squeezed maps to `0`/closed.

## `YamArm`

### Connect and close

```python
from yamkit.arm import YamArm, resolve_channel

spec = rig.arm("left_follower")
channel = resolve_channel(spec)

with YamArm.connect(
    spec,
    channel,
    max_joint_speed=rig.control.max_joint_speed,
    max_gripper_speed=rig.control.max_gripper_speed,
) as arm:
    state = arm.read()
```

`resolve_channel(spec)` prefers `can_iface` when configured; otherwise it matches `can_serial`. It raises `RuntimeError` for missing identity, absent adapters, or a down interface.

`YamArm.connect()` constructs I2RT's robot in zero-gravity mode by default. A follower without saved `gripper_limits` may perform I2RT's short gripper calibration during connection. For leaders, the call waits up to `encoder_timeout_s` for the teaching handle and closes before raising `TimeoutError`.

`close()` is idempotent. It requests gravity-compensation idle, stops the I2RT server and CAN loop in order, then closes the underlying robot. Prefer a context manager; otherwise close in `finally`.

### Read and command

```python
state = arm.read()
actual_target = arm.command(state.q, state.gripper)
```

`command(q, gripper=None, *, limit_speed=True)` requires six joint values, clips a motorized gripper to `0..1`, and returns the actual target sent. When no gripper target is provided, it retains the last commanded value or uses the measured value.

With the default speed limit:

- a fresh or more-than-`0.5 s` stale stream starts from measured state with a `0.01 s` step allowance;
- subsequent steps use elapsed monotonic time;
- each joint delta is bounded by `max_joint_speed × dt`;
- the gripper delta is bounded independently by `max_gripper_speed × dt`.

`move_to(q, gripper=None, duration=3.0, hz=100.0)` linearly interpolates from measured state. Its interpolation calls `command(..., limit_speed=False)`, so the caller is responsible for choosing a safe duration and rate. `hold()` commands the current measured pose without rate limiting.

### Gains and modes

| Method | Effect |
| --- | --- |
| `set_gains(kp, kd)` | Write explicit I2RT gain arrays. |
| `scale_gains(kp_scale, kd_scale=0.0)` | Scale gains captured at connection. Direct teleop uses this for optional bilateral feedback. |
| `restore_gains()` | Restore gains captured at connection. |
| `gravity_idle()` | Enter compliant gravity compensation and clear command history. |
| `zero_torque()` | Enter zero-torque mode, zero gains, and clear command history. |
| `info()` | Return I2RT's robot information mapping. |

These are low-level hardware operations. Do not change gains or modes in a general integration unless the desired physical behavior is explicit and tested.

## Direct teleoperation objects

```python
from yamkit.teleop import TeleopPair, TeleopSession, TeleopStats
```

`TeleopSession.from_rig(rig, pair_names=None, **overrides)` connects every configured pair, or pairs containing any requested leader/follower name. Defaults come from `rig.control`.

| Member | Contract |
| --- | --- |
| `engage(pair)` | Read the leader, interpolate the follower over `sync_seconds`, optionally scale leader gains, then mark engaged. |
| `disengage(pair)` | Hold the follower, return the leader to gravity idle, restore gains. |
| `step()` | Read each pair, handle engage-button edges, and send tracking/feedback commands. |
| `run(duration=None)` | Execute the timed loop until duration, stop event, or `Ctrl-C`; always call `shutdown()`. |
| `shutdown()` | Disengage active pairs and close leaders and followers. |
| `stop_event` | A `threading.Event`; set it from another thread to request loop termination. |
| `on_tick` | Optional callback receiving the session after each completed step. Keep it non-blocking. |

`TeleopPair.tracking_error` is the maximum absolute six-joint position difference, or `NaN` until both states have been read. `TeleopStats` reports ticks, overruns, and the measured average rate.

## Offline policy API

```python
from yamkit.policy_check import robot_features_from_rig, run_policy_check

observation_features, action_features, robot_type = robot_features_from_rig(
    "configs/rig.yaml",
    arms=["left"],
    fake_camera=(480, 640),
)

result = run_policy_check(
    "lerobot/smolvla_base",
    rig_path="configs/rig.yaml",
    arms=["left"],
    task="pick up the object",
    device="cpu",
    n_steps=3,
)
```

Neither function connects to hardware. `robot_features_from_rig()` instantiates a follower plugin without calling `connect()`; when the rig has no cameras, `fake_camera` adds a synthetic OpenCV feature shape.

`run_policy_check()` loads a checkpoint and LeRobot processors, constructs synthetic state/images, performs one fresh-chunk inference plus `n_steps` subsequent calls, and returns `PolicyCheckResult` with dimensions, image keys, timing, chunk size, and a sample named action.
