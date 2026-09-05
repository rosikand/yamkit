# Hardware command and ownership guarantees

The command path remains `Robot.send_action → YamArm.command → I2RT SDK`.
Native read, teleop, alignment, rest and calibration also open arms through
`YamArm.connect`. Discovery and policy-check do not acquire an arm or activate motors.

## Validation and configured bounds

Joint targets must be flat vectors of exactly six finite numbers. Motor gripper
commands must be finite scalars in `[0, 1]`. Malformed values, wrong dimensions,
nonfinite measurements or previous targets, invalid gains, and invalid control
values raise before a position command or gain change. `limit_speed=False` skips
only the target speed limiter; it still performs validation and bound checks.

The LeRobot followers require exactly their advertised action keys, including the
gripper when present. A bimanual action is parsed and both arms' targets, measured
state and previous commands are checked before either side is commanded. Native
teleop also prevalidates the complete tick. This is preflight, not an atomic
transaction: a new hardware fault during execution can still leave only one side
commanded.

Bounds come from the pinned SDK's `get_yam_joint_limits`: the arm/gripper XML
ranges selected by `ArmType`/`GripperType`, the hardware-configured number of arm
motors, and the SDK's **existing 0.15 rad outward allowance**. The same helper is
used by SDK construction and yamkit preflight. These are the vendor's configured
software bounds, not newly inferred physical limits. No dataset extrema are used.
The wrapper checks the SDK's reported bounds against that configuration.

For stock `yam`, the configured raw-coordinate ranges in radians are:

| Joint | Lower | Upper |
| --- | ---: | ---: |
| 1 | -2.76799 | 3.29159 |
| 2 | -0.15000 | 3.81519 |
| 3 | -0.15000 | 3.29159 |
| 4 | -1.84297 | 1.72080 |
| 5 | -1.72080 | 1.72080 |
| 6 | -2.24440 | 2.24440 |

Other variants use their own vendored model and configuration. With calibration,
`aligned = raw + joint_offsets`; targets and home poses are checked after undoing
those offsets. The aligned bounds shift with the offsets. Existing rig fields and
format remain unchanged; invalid old values now produce a validation error. The
home pose must remain valid after alignment. Gripper calibration endpoints can
have either ordering, but must be two distinct finite values.

Out-of-bounds joint targets are **rejected**, with the offending joint numbers in
the error. Measured out-of-bounds positions are also rejected before commanding,
including during homing: yamkit does not snap them into bounds. Stop the session
and have the operator inspect the physical pose, zero calibration and configured
arm type before recovery. A startup error may occur after the SDK has enabled
motors to obtain feedback; cleanup then stops its transmitters. An uncalibrated
motor gripper may still run the SDK's existing calibration during connection.

## Timing, hold and cleanup

Normal commands return the target actually sent after speed limiting. The first
command and a command after more than 0.5 s without application commands ramp
from measured state with a 10 ms target-step budget. Consecutive commands use
actual elapsed time capped at 10 ms; there is no minimum elapsed-time credit or
accumulated catch-up budget. Calling slower than 100 Hz can therefore make the
ramp slower than the configured maximum.

`move_to`, engage synchronization and homing extend their requested duration when
necessary to respect both configured joint and gripper target speeds. Helper
interpolation advances by at most one period per wakeup, even after a scheduling
stall. Homing also respects its requested speed. Holding replaces an obsolete
target with the measured pose before restoring zeroed gains. Compliant homing
installs its hold target with low gains, then moves; the gripper stays at its
measured opening.

Startup tracks every successfully opened arm and every attempted camera open.
Failure or cancellation attempts all cleanup operations, without allowing one
cleanup error to skip later resources. Concurrent homing checks all jobs before
starting movement and waits for active workers after cancellation. Successful
teardown is idempotent. Failed SDK close keeps its ownership lease and can be
retried; a failed SDK construction whose cleanup cannot finish keeps the lease
until process exit.

Default teardown still homes the arms. New callers that need release and close
without a return-home move can use:

```python
session.shutdown(home=False)  # skips hold and home, then closes all arms
robot.disconnect(home=False)  # single/bimanual LeRobot follower
teleop.disconnect(home=False) # single/bimanual LeRobot leader
```

These options do not implement an emergency stop. Close requests gravity idle
and stops the SDK threads/socket. Failed teardown resources stay tracked for a
retry and prevent reconnect while still owned.

## Cooperative ownership across processes

All yamkit worktrees use one Linux runtime directory: **`/tmp/yamkit-arm-locks`**.
This is an intentional runtime exception to repo-local application files. It
requires no daemon, sudo or setup script. Lock files are persistent shared inodes;
**do not delete them to clear a conflict**, since that can create two independent
locks for one adapter. An error names the conflicting adapter and lock path.

`flock` is acquired before SDK construction. Its key uses the live adapter's USB
serial, or canonical sysfs device path when no serial exists. A virtual CAN
interface uses its network namespace and interface index. Rig names, repo paths,
configured aliases and working directories do not affect ownership. Repeated
connections within one process conflict too. When a later arm fails to open,
startup closes earlier arms; their locks release only when SDK shutdown succeeds.

The owning process keeps the descriptor until close, including while SDK threads
run. Process death releases it automatically. Descriptors are close-on-exec:
replacing the process also replaces its SDK threads and releases the lease.
Fork children close their inherited lock copies without unlocking the parent;
inherited `YamArm` operations are rejected. A child must open its own arm after
the parent has released it. Garbage collection cannot release an active lease.

These locks are cooperative and local to one Linux host sharing the runtime
filesystem. Unrelated drivers, old yamkit checkouts, raw SDK consumers, and
containers with a different `/tmp` mount can ignore or bypass them. They must
still be kept away from an active rig. No mechanism here arbitrates such drivers.

## Limits and supervised acceptance

These changes constrain **command targets**. They do not guarantee measured
joint velocity, avoid collisions, or provide a safety-rated emergency stop. The
motors' 400 ms firmware timeout is unchanged. The 0.5 s stale-command ramp reset
is **not a watchdog**: SDK background threads may continue transmitting the last
command while the application stalls. No application-stall watchdog was added.

Automated acceptance uses fake robots/cameras and real subprocess/descriptor
lock tests. Real hardware and cameras were not opened for this change. Before
operational use, a trained operator should perform these checks on a clear rig:

1. Verify physical left/right, arm types, calibration and home poses. Confirm
   valid measured raw positions and alignment under supervision.
2. Start one arm owner, then attempt an overlapping rig from another checkout;
   confirm rejection before the second SDK activates. Close normally, then
   confirm another owner can acquire it. Test process death separately with
   the arm supported and the existing stop procedure ready.
3. With conservative speed settings, observe engage, hold and home. Check that
   a short synchronization duration extends and that hold does not resume an
   earlier target. Confirm compliant leader homing and gripper preservation.
4. Confirm Stop during synchronization and a second Stop during home release
   the arms as expected. Exercise default and no-home teardown, followed by
   a clean reconnect. Test startup/cleanup faults with fakes before any supervised
   unplug/reconnect exercise.

Do not intentionally drive hardware beyond bounds to test rejection; those cases
are covered with injected measurements and targets in the hardware-free suite.
