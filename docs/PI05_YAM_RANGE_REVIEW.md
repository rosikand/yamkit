# π0.5 YAM gripper range review — 2026-09-13

The native model and its saved normalization are unchanged. An explicitly
versioned **robot-host gripper endpoint adapter** now handles small extrapolations
after native postprocessing. This is not a claim that the checkpoint itself
clips outputs, nor that an overshoot is floating-point roundoff. Fresh real-GPU
qualification of the changed execution identity is required. No physical test
or manipulation success is implied.

## What failed and what the sources establish

The retained real-GPU response contained 30 finite 14-D rows. Its first left
gripper value was **1.000895619392395**, outside the executable [0,1] range.
Before this adapter, strict qualification correctly rejected that response.
The raw result remains evidence; it is not rewritten as a successful historical
qualification.

The pinned [YAM checkpoint](https://huggingface.co/Jiafei1224/molmoact2-yam-pi05/tree/51ab2720d7e56d51410407f98ea64bbea97feb2e)
describes 14 absolute joint/gripper values, continuous [0,1] grippers, and
30-row chunks. Its saved postprocessor contains QUANTILES unnormalization,
disabled absolute-action conversion, then CPU transfer. There is no saved
gripper clamp. The [native quantile implementation](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/processor/normalize_processor.py)
uses the checkpoint's q01/q99 affine inverse without saturation. Thus normalized
regression outputs can extrapolate beyond the quantiles and physical endpoints.

For this checkpoint, the left gripper q01/q99 are
0.05101636052131653 / 0.9870786666870117; the right values are
0.03335051238536835 / 0.9953596591949463. Consequently, normalized values above
approximately 1.027608 on the left or 1.009647 on the right map above one.
This explains why modest native extrapolation can trigger the boundary. It
does not establish the cause of every model error or justify changing statistics.
Saved statistics hashes and camera/state/action ordering remain pinned.

## Why endpoint projection is an embodiment adapter

The [pinned I2RT JointMapper](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robots/utils.py)
maps normalized aperture `u` to `closed + u * (open - closed)`. The
[pinned SDK update](https://github.com/i2rt-robotics/i2rt/blob/47fee5e7dec4e30ca054f798bda1c8894b465ed2/i2rt/robots/motor_chain_robot.py)
saturates the resulting gripper target to the calibrated physical endpoints.
This remains true when the motor's open/closed numerical direction is reversed.

The official MolmoAct2 repository links the YAM submodule at
`9f06bba2a36dd84fb36d0c31337c85d4bf1cea22`. Its
[PI evaluator](https://github.com/williamtsai726/YAM/blob/9f06bba2a36dd84fb36d0c31337c85d4bf1cea22/gello_software/experiments/launch_yaml_eval.py)
applies native postprocessing and sends the resulting rows directly to its
robot environment. Its [SDK copy](https://github.com/williamtsai726/YAM/blob/9f06bba2a36dd84fb36d0c31337c85d4bf1cea22/i2rt/i2rt/robots/motor_chain_robot.py)
also contains calibrated gripper endpoint saturation. This is evidence for the
embodiment's target-range handling, not grounds to copy that launcher's camera
names, resizing, cleanup, or timing into the separately pinned LeRobot runner.

Projecting normalized aperture to [0,1] gives the same requested physical endpoint
as affine mapping followed by endpoint saturation. However, the SDK may also run
an effort-dependent force limiter before its final saturation. Moving a small
projection to the host boundary does **not** prove identical force-limiter state
or contact dynamics at an endpoint. The existing limiter and all hardware guards
remain enabled; physical behavior still needs a supervised trial.

## Explicit, bounded contract

`pi05_reference` contract version **2** includes action transform
`i2rt_gripper_endpoint_projection_v1`:

1. Require exactly 30×14 finite native postprocessed values. No padding or
   truncation; keep all raw rows unchanged in evidence.
2. Require both raw grippers to lie within **[-0.01,1.01]**. This one-percent
   normalized-travel envelope is a conservative **yamkit engineering anomaly
   guard**, not an upstream tolerance, learned guarantee, or roundoff threshold.
   Larger excursions remain errors; the envelope must not auto-widen.
3. Project only columns 6 and 13 onto [0,1]. All twelve joint values remain
   exactly unchanged. No temporal smoothing, thresholded open/close, release
   script, interpolation, or altered policy inference is introduced.
4. Apply the existing strict whole-chunk joint and gripper bounds before any
   dispatch. SDK-vs-request coherence, measured-state guards, Stop, session/RPC
   deadlines and hardware ownership are unchanged.

The runtime still returns raw native `chunk` values. Its identity advertises the
required host adapter; the algorithm, constants and contract participate in the
PI source build and qualification identity. Old proof cannot qualify version 2.
MolmoAct2's runtime, controller and qualification path are not changed.

## Evidence and regression checks

`chunk_admitted` records `raw_rows`, executed `rows`, `action_transform` and every
`gripper_projections` entry (row/column, raw/executed value, signed delta).
Dispatch events retain `raw_requested`, bounded `requested`, and the actual
returned `sent` mapping. Direct qualification retains `direct_chunks` as raw
outputs alongside `direct_executed_chunks` and `direct_gripper_projections`.

Metrics distinguish projected rows/scalars, actually dispatched projected
rows/scalars, maximum projection, and unexpected SDK command modifications.
Planned-but-unused rows at normal duration/Stop are separately identified from
unexpected drops. Legitimate projection is not mislabeled zero modification of
the raw policy output; zero coherence violations means the SDK receipt matches
the explicitly transformed target.

Tests cover the exact observed overshoot; positive and negative raw boundaries;
adjacent floating values just outside the anomaly guard; NaN/Infinity/shape
failures; unchanged joints; full-chunk rejection; raw/executed trace accounting;
and Stop/fault behavior. They invoke the actual pinned SDK affine mapper and
update-method body extracted by AST with hash verification, using only fake
state and an inert command sink. No SDK module, hardware constructor, CAN bus,
encoder, camera or acquisition thread is opened. Both calibration directions
match physical target results within 1e-12 radians with the force limiter inactive;
the force-limiter caveat above is explicit.

Native checkpoint postprocessing remains separately tested against the actual
LeRobot processor, and FIFO/timing parity remains tested against its pinned
BaseStrategy. These tests establish software boundaries, not collision safety,
dynamic tracking, camera alignment, or task completion.
