# Remote rollout integration audit

Audited against the actual PyPI `lerobot==0.6.1` wheel. The adapter depends on this
pin; upgrading LeRobot requires rerunning the integration and fault tests.

## Factories and activation

`yamkit.remote_policy` registers `YamkitRemoteConfig` as `yamkit_remote` through
`PreTrainedConfig.register_subclass`. LeRobot's real `make_policy_config`,
`get_policy_class`, `make_policy`, and `make_pre_post_processors` resolve the
configuration, model, and processor modules by upstream naming conventions.
The proxy has zero parameters. Its `from_pretrained` accepts the explicit remote
configuration and rejects weight paths. Client processors only retain batch and
CPU device handling, with an empty observation rename step and no numerical
normalization. Saved model processing and the camera rename happen on the server.

`run_remote_rollout` first inspects the unconnected YAM plugin schema, including
exact ordered names, calibration for every gripper, camera color mode, hardware
variant and action cadence. It then calls the real `build_rollout_context`.
The proxy's readiness request and metadata validation execute in the upstream
policy-loading phase, before `Robot.connect`. A transient shutdown event is passed
to the policy and YAM plugin so Stop during preparation prevents subsequent
activation. Initial homing retains the existing implementation and speed settings;
the same event interrupts it through `go_home_all`'s optional stop argument.

## Unguided background inference

The upstream `supports_rtc_inference` requires both `policy.supports_rtc()` and
the RTC argument signature. This gate applies even when `rtc.enabled=False`.
The pinned native `MolmoAct2Policy.supports_rtc()` returns
`config.inference_action_mode == "continuous"`; its `predict_action_chunk(**kwargs)`
passes the signature check. Its continuous path implements guidance inside the
flow-matching loop through `rtc_processor.denoise_step`. SmolVLA and pi0.5 also
declare upstream RTC support. Molmo's discrete inference mode does not.

The remote transport has not validated the model-space prefix conversions and
guidance behavior required to expose those native capabilities. Its lightweight
proxy therefore returns false, and the CLI rejects `--rtc` for remote profiles.
This is a boundary of the remote implementation, not missing upstream Molmo RTC
support. The pinned Molmo checkpoint uses absolute actions and its saved
preprocessor has no `RelativeActionsProcessorStep`; the stock sync engine's
enabled-relative-actions rejection is not a blocker for that checkpoint.

The upstream inference factory has a fixed sync/RTC dispatch with no external
engine-registration hook. The narrow adapter builds a genuine sync context, then
replaces its inference member with `UnguidedRemoteInferenceEngine`, a subclass of
`RTCInferenceEngine`. It reuses the **unchanged upstream background worker** with
`RTCConfig(enabled=False)`, its observation conversion, and its latency estimate.
The real upstream base strategy, action dispatcher, and robot action processor
perform execution: `Robot.send_action -> YamArm.command`. There is no copied
rollout loop or installed-package monkey patch.

In guided RTC, upstream `ActionQueue.original_queue` contains model-space actions
and `queue` contains postprocessed robot-space actions. Upstream relative prefix
handling reanchors processed leftovers against the current raw state and then
renormalizes them. This proxy returns robot-space actions, so **both queues contain
robot units**. Its unguided worker's leftover/delay arguments are never fed to a
denoiser. The wire continuation is explicitly `None`; requests carrying guided
continuation are rejected. Server-side normalized/relative prefix conversion,
anchor context and gradient-enabled guidance remain unsupported and unvalidated.
Buffering is not labeled RTC.

## Faults and Stop

Upstream normally retries worker errors until ten consecutive failures, returns `None` on underrun,
and tears down by returning to the initial pose and calling a plugin disconnect
that also homes. Those defaults are unsuitable for a remote fault.

The adapter invalidates the policy session and current queue on a fault. Queue
objects are permanently invalidated before replacement on pause/reset, so even
an in-flight worker retaining an old queue cannot merge a late result. The queue
has a finite capacity (one chunk plus a half-chunk prefetch threshold) and tracks
the local observation deadline of every action. Responses are checked for exact
session/sequence/revision/schema, finite values, units and local deadlines. RPC
requests are serialized, bounded, and made only in the background worker.
An instance-ID change after readiness also stops execution and requires fresh
preparation; a restarted container cannot silently inherit an active robot session.

The first chunk gets a bounded startup grace period. Once execution begins, an
empty or expired queue raises a fault before upstream interpolation can replay
an old action. Local Stop is also checked immediately before dispatch. Release
uses `disconnect_no_home`, before joining an outstanding RPC worker; it never
returns to start, homes, or falls back to CPU. A transient reference on the YAM
config enables release even if upstream context construction fails after connect.
Normal local plugin disconnection still homes as before.

Local observation timestamps represent receipt of the robot/camera snapshot;
they are not camera exposure timestamps. Server clock values are not subtracted
from client clock values. Bounded telemetry includes request count, first/warm
round-trip samples and warm p50/p95/p99, encoding and server time, payload size,
observation age, queue depth and underruns. The first request is not described as
a cold container measurement: readiness may already have warmed the container.

## Verification and remaining limits

`tests/test_remote_policy.py` exercises the actual policy factories, numerical
identity processors, three fresh mocked chunk calls, reset, `select_action`, real
rollout context, and real base strategy with fake YAM hardware. It also checks
preactivation failures, readiness/Stop races, partial-context release, response
validation, late replies, deadlines, queue limits, underrun and no post-stop
actions. A tiny local ACT performs a real forward pass through the unchanged
local sync context and base strategy, also using fake hardware.

Remote rollout currently supports only the base strategy, exact profile cadence,
no interpolation, and the CPU RPC proxy. Recording/DAgger/episodic remote
strategies and guided RTC are rejected. Physical operation still requires a
supervised probe and separate motion approval; passing unit tests or native
checkpoint inference does not validate a particular robot's calibration or
camera geometry. See the feature's main documentation for actual CPU/GPU test
results, model revisions and deployment procedures.
