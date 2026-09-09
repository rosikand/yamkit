# MolmoAct2 execution diagnosis

The checkpoint stays fixed at
`lerobot/MolmoAct2-BimanualYAM-LeRobot@fdade02d1f1c1dd819114b0478f735072fb6b212`.
The task is **put the red cube into the black container**. No demonstration collection,
fine-tuning, sampler change, or training pipeline is part of this correction.

## Controller correction

The remote joint shaper previously limited speed and acceleration but only reserved braking
distance near physical joint limits. With a constant interior target of 0.2 rad from rest at
30 Hz, it overshot by 0.08 rad and repeatedly reversed before settling. This reproduced with
no model or hardware involved.

The asynchronous controller's shaper now slows its desired velocity as it approaches the requested pose. It uses the
existing conservative stopping envelope, `v*h + v²/(2*a) <= distance`, then projects that
soft desired velocity through the unchanged hard speed, acceleration, step, and joint-boundary
constraints. Here `a=1 rad/s²` accounts for the existing 50 ms acceleration budget and maximum
100 ms dispatch gap; `h=dt+100 ms` reserves discrete stopping room. A target that jumps inside
the stopping distance can still be overshot while the command decelerates. Snapping to it would
violate the acceleration constraint.

Gripper handling, queue behavior, observation/action expiry, Stop handling, and the 400 ms motor
timeout are unchanged. This is command shaping, not a guarantee about measured acceleration,
collision avoidance, grasping, or task success.

Hardware-free constant-target tests cover both directions, nonzero starts, tiny errors and
variable intervals. A saved-command replay also retains the three recorded trials' original
targets and dispatch timestamps. It initializes both implementations from the same two recorded
sends; the old implementation reproduces all subsequent joint commands exactly.

| Recorded run | Commands replayed | Requested-minus-sent joint RMS, before → after |
|---|---:|---:|
| `20260908-235721-rollout-a074d6a7` | 261 | 0.277 → 0.249 rad |
| `20260909-001841-rollout-a7fdf6f6` | 560 | 0.251 → 0.222 rad |
| `20260909-004446-rollout-12b67b72` | 560 | 0.202 → 0.171 rad |

The fixed-target overshoot disappears in these tests. Saved-command replay remains a
counterfactual: different physical motion would produce different observations and model
predictions. It does not establish improved grasping or physical smoothness.

## Reference comparison and remaining differences

[AllenAI's deployment links](https://github.com/allenai/molmoact2/blob/66b87e64efd99dfd103241418113955cf64dfa9c/README.md)
point to two different implementations. The
[linked YAM runner](https://github.com/williamtsai726/YAM/blob/9a46f908d5dfb8a999f9a2c236681bf2af74f4da/gello_software/experiments/launch_yaml_eval_molmoact.py#L175-L204)
executes complete chunks synchronously and interpolates all 14 values together, stretching
trajectory time. Its wrapper reports cached command positions after initialization. The
[linked Cortex client](https://github.com/SuveenE/lerobot/blob/e0bf4a54f600f5601e8753bf15ca1599ecf24f5d/src/lerobot/async_inference/robot_client.py#L443-L486)
uses executed action indices and incoming queue replacement with weighted overlap averaging.
Neither is identical to yamkit's async wall-clock expiry and append queue. Neither examined deployment
enables RTC guidance. Absolute joint units, left/right ordering and normalized gripper direction
agree with yamkit; physical calibration accuracy remains a separate hardware question.

Comparing predictions on their overlapping observation-relative 30 Hz intervals, rather than
matching equal row numbers, still shows substantial disagreement: future-overlap joint RMS is
0.311, 0.268 and 0.198 rad across the three trials. Queue joins also occasionally skip or overlap
about one interval. A simple absolute-tail splice correction underruns in two saved timing
replays; fixed-grid resampling also produces expiry faults. Those changes were rejected.

Those archived async trials differ from the linked YAM runner in joint/gripper coordination,
chunk consumption and measured versus cached state. Reference mode below follows that runner's
execution contract. The reference 640×360 versus current 640×480 camera geometry remains a
separate difference; physical calibration accuracy has not been inferred from software mapping.

## Reference controller mode

`--controller-mode reference` selects serial execution for the fixed MolmoAct2 checkpoint over
HTTP with `cuda_graph10`, raw RGB and no crop. `--controller-mode async` retains the experimental
asynchronous append queue and independent joint shaping; it remains the default. Qualification
is bound to the selected mode, with a separate `-reference.json` record. Async evidence cannot
qualify reference execution. These are implementation contracts, not a claim of a passing
reference qualification or successful physical placement.

The pinned YAM runner's spatial oracle takes a 14D start and target, computes
`n = min(int(max(abs(target - start)) / 0.01), 100)`, and sends the target directly when
`n <= 1`; otherwise it sends `np.linspace(start, target, n)`, including both endpoints.
The maximum includes both normalized grippers. Yamkit sends these exact points: no cubic easing,
extra ticks or terminal holds. The runner's unused `action_horizon=25` accessor does not truncate
the response: all 30 returned rows finish before the next RPC. There is no prefix dropping,
queue replacement, overlap blending or application command during inference.

For each point the order is **send → Rate.sleep → observe**, followed by a 1 ms sleep when the
row has multiple points. The nominal rate is 30 Hz. As upstream, the post-send rate clock resets
after an overrun; it imposes no minimum interval before the next send after a slow RPC. Yamkit
uses a monotonic clock and interruptible 100 µs polling for that wait. This preserves the rate
algorithm while making Stop responsive and avoiding wall-clock adjustments. It does not
guarantee 33 ms between commands or bound physical speed and acceleration.

Reference dispatch uses `send_reference_action` → `YamArm.command(limit_speed=False)`. Only this
mode bypasses yamkit's per-step joint/gripper clamps, stale ramp reset and async joint shaper;
it has no added 100 ms interpolation-stall limit. Finite 14D targets, joint bounds, normalized
gripper bounds, measured-state validation and exact returned-command coherence are still checked.
Stop/session cancellation, the 400 ms motor firmware timeout, accepted mapping and supervised
approval remain required. Teleop, recording, local/async rollout and normal homing retain their
existing command limits.

In a software-only replay of all 73 saved chunks from run `20260909-024246-rollout-bdf21f72`,
the literal oracle produces **6,999 points**, compared with **28,183** from the previous eased
planner, for the same 2,190 model rows. Every literal point and original row endpoint matches
the pinned formula. These frozen predictions establish implementation agreement, not a physical
trajectory or successful placement.

The initial model state and interpolation anchor use the first measured observation before
inference. Thereafter both use the last committed 14D command, matching the linked runner.
Measured encoders remain in monitoring and hardware guards. RGB comes from the initial or latest
post-step observation. The extra observation read at each model row matches upstream's row-anchor
read; its images are not selected for inference.

Response validation and plan admission must finish within the configured observation freshness
budget, at most two seconds, and the finite RPC timeout. Once admitted, the plan is bounded by
the existing phase/session deadline; it does not acquire async per-row expiry or an added timing
lease. `planned_duration_s = interpolation_points / 30` is nominal accounting, not a wall-time
prediction. Another RPC is skipped if its entire request budget cannot fit before the phase
ends. Healthy duration completion homes and then releases both followers; Stop, a fault or
session expiry releases without homing.

Reference qualification uses the same real model with generated images and fake arms: 50 warm
direct samples and 50 warm, fully consumed integrated chunks, with the first sample excluded.
The integrated diagnostic allows at most 1,800 seconds with the same literal rate algorithm;
incomplete evidence fails. It checks exact row counts, full chunk consumption, direct sends and
zero SDK sends during RPC against the versioned literal contract, so older reference evidence
cannot qualify this implementation.
Its latency margin uses the synchronous admission budget, while async uses the remaining queue
horizon. The diagnostic limit does not extend physical rollout or capture limits.

In saved artifacts, `metrics.json.reference_execution.dispatch_samples` provides `dispatch_index`,
`chunk_index`, `row_index`, `point_index`, scalar `progress` and `endpoint`. Chunk and row indices
refer directly to `trace.json.chunks`, joining a committed point to its original model row. In reference mode,
`command_shaping.samples[].requested` is that interpolated point, **not** the raw model row;
`shaped` and returned `sent` must preserve it. Bounded sample loss is counted explicitly. Use
`completed_steps`, `completed_chunks` and `partial_chunk_at_stop` to distinguish a full chunk
from one interrupted by duration or Stop. `trace.json.chunks[].policy_state` retains the exact
14D robot-unit input after client preprocessing; measured observation events are separate.
Historical traces can lack this model-state field.

`reference_row_observation` events retain timestamps and measured positions for the extra
row-anchor reads, with `rgb_retained=false`. Their unused RGB is intentionally omitted, as the
summary's `auxiliary_observation_scope` states. Initial and post-step RGB remain captured with
explicit missing-frame counts and original receipt timestamps. Video gaps preserve those
timestamps; footage and measured positions establish physical outcomes separately.

### Physical validation of literal execution

Run `20260909-050720-rollout-8c1d5dba` on commit `28012421` completed all 330 model
rows from 11 chunks as exactly 659 reference commands. Every recorded point matched
the literal formula and the requested/shaped/SDK-returned values were identical.
Median arm-send spacing was 33.42 ms; the expected short post-RPC intervals and
blocking inference gaps remain. The 30-second phase completed normally, followed
by 8.53 seconds of return home and release.

Top and left-wrist video show a grasp, lift and retained transport toward the
container at approximately 22–29 seconds. The cube was still gripped at the final
observation: policy-controlled release and placement are unproven. A 45-second
capture is supported for the next separately approved trial, using the same
controller/model with a larger bounded recording allocation. The
[original archive](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/41528fac2f660bacd38b3e740ee640da68328151/runs/20260909-050720-rollout-8c1d5dba)
retains the full evidence. Its report's generic easing/hold paragraph is stale;
the recorded literal contract, trace and bundle README describe the actual run.

## First supervised validation after target braking

Run `20260909-014413-rollout-1559398b` on source `929180c` shows the left gripper grasping,
lifting and retaining the cube from approximately 13.35 seconds through the 20-second end.
The cube is still being transported at the cutoff; placement and release in the container
were not demonstrated. Both followers subsequently homed and released normally. The
[complete private recording](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/02f7f45b99d81504431e6e5feefedb8a4fad34f9/runs/20260909-014413-rollout-1559398b)
contains all three videos and 599 original RGB frames per camera.

The next supported capture duration is 30 seconds, retaining the same controller/model and
the existing 120-second rollout and 90-second export wall limits. Its 903-slot frame pool needs
2,496,614,400 bytes plus the existing 512 MiB headroom before hardware can connect. The report
renderer accepts the collector's existing 8,192-event limit. A longer trial requires fresh
supervised approval; it is an experiment, not a claim that additional time guarantees placement.

## Gripper reconnect correction

The next attempt, `20260909-020727-rollout-ab339b57`, stopped during left-follower
connection because its measured gripper state was outside [0, 1]. The policy never started;
cleanup released the left follower before the right connected. Its
[failure archive](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/3a14ec9d0f59739abed5d40803e7e4e4ed77ddd7/runs/20260909-020727-rollout-ab339b57)
retains the error and explicitly flags absent policy frames, videos and report.

Inspection found the SDK startup loop applying arm angle wrapping to the gripper motor as
well, without shifting its saved calibration. With the left gripper's [6.44979782, 1.23235676]
raw endpoints, a valid normalized opening of 0.309 has raw angle 4.837608532 radians. Folding
it by −2π makes the reported opening 1.513265699. This is a hardware-free counterexample,
not the measured value from the failed attempt: that value was not logged.

Startup now corrects only the arm joints and preserves the gripper's raw calibrated frame.
All six joints remain eligible for correction on followers, teaching handles and bare arms.
Factory tests cover both signs of raw gripper coordinates, closed/open/intermediate positions,
joint wrapping and normalized observation/command round trips. Existing calibration, strict
measured-state rejection, command limits and motor timeout remain unchanged. The later open-gripper
read and bimanual rollout below connected and released normally; closed-aperture reconnect remains
physically unverified.

## First-dispatch pose correction

After the gripper fix, a supervised left-only read connected and released normally with the
gripper open. Both followers then connected and homed in run
`20260909-022529-rollout-3eea2c74`, but its first policy dispatch triggered the postclamp
acceleration guard. The left wrist was still settling when the shaper captured its initial
pose; it moved from approximately −0.051 to −0.014 rad while startup admission waited for
a sufficiently fresh chunk. The first shaped command still used the earlier pose, so the
arm's unchanged 0.03-rad clamp altered it. Both followers released after the fault.

The shaper now captures current joint/gripper state once, immediately before preparing the
first actual policy command, without acquiring camera frames. Later commands retain their
committed state. Initial command time, zero command velocity, target bounds, Stop/expiry checks,
all speed/acceleration limits and postclamp rejection remain unchanged. This snapshot does not
establish zero physical velocity. The initial joint positions are included in shaper metrics.

The [1.24-second failure recording](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/8effa656d266a01d11998554526aa0a1deb1f4df/runs/20260909-022529-rollout-3eea2c74)
contains 38 frames per camera and one bimanual send. Its full metrics were not exported, and
the historical report has a broken metrics link; the archive flags that omission. Future
fault traces preserve the metrics attached by runner cleanup, including command-shaping faults,
while retaining the original failure and release gate. The subsequent physical run below completed
the first-dispatch path without a clamp intervention. Its first-observation-to-dispatch joint change
was only 0.00038 rad: the larger startup-drift regression remains validated by hardware-free tests,
not reproduced on the rig in that run.

## Thirty-second validation and release diagnosis

Run `20260909-024246-rollout-bdf21f72` on source `ea0a447` completed 859 policy commands in
the 30-second phase, followed by normal home and release. There were no queue underruns,
expired dispatches, postclamp joint interventions or capture drops. The
[pinned private recording](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/e3b5bef2b19055b23c939031046ed96e1aeb9d8a/runs/20260909-024246-rollout-bdf21f72)
contains three videos, 898 original RGB frames per camera and complete metrics.

The task still failed. The left arm grasped and transported the cube twice, with losses visible
around 13.18–13.21 seconds and 24.08–24.15 seconds. Both followed partial model-requested
opening past the held-cube aperture of approximately 0.31, before placement over the bin.
The cube landed outside the container; a separate post-home image confirmed it beside the left
outer wall. An intermediate brief grip/contact also ended on the table. Saved measured joints
and forward kinematics quantify pose disagreement, but provide neither a calibrated bin pose
nor contact force. They cannot establish whether an unfiltered requested pose would have placed
the cube correctly, or which controller change would preserve the grasp.

An offline replay reproduced all 73 merges and 859 actual sent commands exactly, then applied
incoming queue replacement and a 0.3-old/0.7-new overlap blend to the complete saved history.
Both advanced the opening commands too: the first held-aperture crossing moved 167 ms earlier,
and the second 233 ms earlier. Nearest-time alignment removed positional blend mismatches of
up to 36.46 ms without changing those crossing times. Original deadlines, the dequeued prefix,
joint shaping and the 0.03 gripper step were preserved. These results reject queue replacement
or blending as an evidence-backed retention fix here. They are fixed-prediction counterfactuals;
changed physical motion would produce different observations and predictions, so no task-success
or physical-safety conclusion follows from their passing command-space checks.
