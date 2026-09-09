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

The shaper now slows its desired velocity as it approaches the requested pose. It uses the
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
Neither is identical to yamkit's wall-clock expiry and append queue. Neither examined deployment
enables RTC guidance. Absolute joint units, left/right ordering and normalized gripper direction
agree with yamkit; physical calibration accuracy remains a separate hardware question.

Comparing predictions on their overlapping observation-relative 30 Hz intervals, rather than
matching equal row numbers, still shows substantial disagreement: future-overlap joint RMS is
0.311, 0.268 and 0.198 rad across the three trials. Queue joins also occasionally skip or overlap
about one interval. A simple absolute-tail splice correction underruns in two saved timing
replays; fixed-grid resampling also produces expiry faults. Those changes were rejected.

Joint/gripper coordination, retained late chunk rows, measured versus cached state, and the
reference 640×360 versus current 640×480 camera geometry remain execution differences to
investigate. The small target-braking correction deliberately isolates one demonstrated defect.

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
