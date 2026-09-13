# Native π0.5 recordings and private archives

The native YAM π0.5 runner records its own contract. It never uses the frozen
MolmoAct2 controller, collector or cached-command model state. Optional capture
retains only already acquired policy-phase observations; it opens no extra
cameras and performs no extra active reads. Software/fake-arm artifacts are
explicitly labeled and never establish physical task success.

## Entry points

`yamkit.pi05.rollout.run_rollout` accepts these additive options:

- `capture_trace=False`: retain native action/observation trace and lifecycle
  report; `True` additionally retains every existing full RGB observation.
- `upload_repo_id=None`: an explicit `namespace/dataset` forces RGB capture and
  requests one private upload after release and successful export. There is no
  automatic physical or upload retry.
- `artifact_metadata=None`: optional sanitized provenance, never credential values.

The destination must be a new directory inside the checkout with no symlink
components. CLI runs can select `outputs/ui/deployments/<run-id>` for immediate
history/playback visibility. Managed UI children can use their existing fresh
`.context/rollout-traces/<id>` directory; the parent copies bounded public
artifacts to the deployment record as usual.

For UI-owned uploads, do **not** also request child upload. The post-exit parent
uses `package_native_rollout(run_dir, trace_dir=original_trace_dir)` followed by
the existing `upload_rollout(bundle, repo_id=...)`. Original lossless frames stay
in their original trace directory, while the immutable bundle gets its own copy.
Private-repository enforcement, conflict-safe uploads and durable HF receipts
reuse the unchanged existing upload implementation. Authentication stays in the
existing local HF token store; tokens are neither arguments nor archived files.

## Captured evidence

`trace.json.native_responses` preserves untouched native postprocessed 30×14
responses before action-range handling. Invalid non-finite numbers become JSON
`null` with an explicit count, not invented finite values. Admitted chunks retain
`raw_actions`, executed `actions`, the named action transform and individual
gripper projection records. Events retain measured states and explicit
chunk/row-indexed attempted and completed dispatches. Incomplete tails,
interpolation-point counts, documented projections and unexpected SDK command
modifications remain separate in execution metrics.

`report.json` is the lifecycle report; the same data appears in
`metrics.json.native_pi05_rollout` for the existing archive allowlist.
`summary.json`, `meta.json`, and `run_metadata.json` keep UI history and capture
status separate from manipulation success. `task_success` remains unknown.

When capture is enabled, original RGB PNGs and three H.264 videos preserve the
existing observation receipt timestamps. `video_timeline.json` maps playback to
original observation indices and preserves real gaps; no frames are invented.
`report.html` offers local playback timers and native-specific trace links.
The report does not describe native π0.5 commands as MolmoAct2 interpolation.

## Bounds and ordering

The complete frame pool is admitted and prefaulted before any robot constructor.
At 90 seconds this uses the unchanged admission threshold of **8,010,125,312
bytes**, including 512 MiB headroom. Insufficient memory fails preparation;
capture duration and safety checks are not reduced automatically. Observations
only copy into reserved slots; all serialization, PNG/video encoding, report
rendering and upload occur after confirmed resource release. A failed release
saves diagnostic JSON only and prevents image export and upload.

Export retains the existing 240-second post-release wall bound. Export/upload
failure preserves local originals and cannot initiate another physical run.
Capture overflow/errors are counted and visible, never hidden by treating an
incomplete recording as complete.

The explicit fake seam requires **both** a supplied robot factory and a fake
home callback. It cannot fall back to a real home implementation. Tests use
saved/generated RGB arrays and an import guard that fails on hardware modules.
