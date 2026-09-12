# Inspect rollout bundles on another computer

Rollout archives are ordinary files in a **private Hugging Face dataset repo**.
They are debugging records, not a converted LeRobot training dataset. Each finalized
run occupies `runs/<run-id>/`. Local originals remain on the robot host.

The operator workflow is: run a supervised rollout, wait for its child process,
return home/release and artifact export to finish, then upload the finalized bundle.
Packaging and network transfer run in a separate worker after finalization. Upload
failure does not restart a rollout or change its physical outcome.
Keep the dashboard running until upload completes. If it restarts mid-upload,
the next startup marks the retained receipt interrupted and exposes a retry command.

## Enable post-run upload

In Inference, enable **Save recording locally** to retain camera videos, original RGB
frames and joint/timing traces without uploading. Enable **Also upload to Hugging Face** and
select a private dataset destination such as `rohanlux/yamkit-rollouts` to upload the
same recording; local originals are kept. Both options default off.

Capture supports the exact currently qualified MolmoAct2 task at 5, 10, 20, 30, 45, 60
or 90 seconds with all three cameras and both named followers. It preserves the selected
task, reference/async controller and raw-RGB HTTP graph settings; qualification, memory
admission and hardware confirmation remain required. Other saved run directories can be
bundled manually, with missing data explicitly listed. An unrecorded rollout has no
video to recover retroactively.

`hub.rollout_repo` in `configs/rig.yaml` stores only the optional default destination.
The HF token stays in HF's existing token file under `data/hf/`. A managed API request
can opt in with `upload_repo_id`; this is separate from motion confirmation.

Click a rollout in Inference history or Runs to play its local camera recordings after
export finishes. Runs show upload status and a private HF link when uploaded. A failed or interrupted
upload leaves the run and bundle available locally. Retry a finalized run without
opening hardware:

```bash
source scripts/env.sh
yamkit bundle-rollout outputs/ui/deployments/<run-id> \
  --trace-dir .context/rollout-traces/<trace-id> \
  --upload-to rohanlux/yamkit-rollouts
```

Omit `--upload-to` to package locally only. `--metadata <json-file>` supplies reviewed
provenance for older runs that predate automatic metadata snapshots. Unknown
historical values remain unknown; the current checkout is not substituted for the
commit that actually ran. The uploader refuses a public repository and conflicting
content already stored under the same run ID.

## Download and inspect

Sign into the same HF account, or an account granted access, on the inspection
machine. With the `hf` CLI installed:

```bash
hf auth login
hf download rohanlux/yamkit-rollouts --repo-type dataset \
  --include 'runs/<run-id>/*' --local-dir ./yamkit-rollouts
```

Open `yamkit-rollouts/runs/<run-id>/README.md` and `report.html`. The report's video,
plot and JSON links are relative, so the entire folder can be moved. Alternatively,
serve the downloaded folder with `python -m http.server 8000 --directory ./yamkit-rollouts`
and open the report in a browser. No robot software, GPU or Lenovo connection is
needed to inspect the files.

The Python API provides the same download independently of the CLI version:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('rohanlux/yamkit-rollouts', repo_type='dataset', allow_patterns='runs/<run-id>/*', local_dir='./yamkit-rollouts')"
```

For a reproducible download, add `revision='<HF commit hash>'`. On the tested Lenovo,
the installed `hf` CLI downloaded every file but then raised `click.exceptions.Exit: 0`;
the Python API completed cleanly. Automatic post-run upload uses the Python API.

The bundle includes videos, original RGB PNGs, joint plots, traces, timing and metric
JSONs, console logs, sanitized configuration/model metadata, and a SHA-256 manifest.
Its README documents the exact schema, timestamp relationships, and missing data.
Commands after clamping are distinguished from measured joint positions and from
model predictions; a completed process does not imply successful manipulation.

## First archived physical run

Run `20260908-184743-rollout-578b88cc` has these Lenovo originals:

```text
/home/andre/rohan-new/outputs/ui/deployments/20260908-184743-rollout-578b88cc/
/home/andre/rohan-new/.context/rollout-traces/93e72d9e8688467d8d9a2536cf1095c6/
```

Videos, plots and JSONs exist in both locations; console log and run metadata are
in the deployment directory. Original images are in the trace directory under
`frames/{top,left_wrist,right_wrist}/frame-000000.png` through `frame-000149.png`.
The five-second policy phase contains 150 observations and 132 bimanual dispatches.
The operator confirmed video playback and return home, but the lid-placement task
failed and motion was jittery. No motion correction is implied by archiving it.

The [first private archive](https://huggingface.co/datasets/rohanlux/yamkit-rollouts/tree/90aed52d118eb356c9149ba640385450f4951c00/runs/20260908-184743-rollout-578b88cc)
contains 467 files totaling 174,213,977 bytes. A separate HF download verified all
file checksums. Its manifest SHA-256 is
`944477b940481e10ffd55a648338b61cbf51afe4196952984b6c6a2d8aa7bd6e`.
The generated local bundle is the deployment directory's `bundle/` subdirectory.

Known gaps include camera exposure timestamps and startup/home imagery and joint
traces. The older run's INFO phase messages were suppressed at capture time and
cannot be reconstructed. Model chunks and full metrics are retained, but no model
weights or hidden activations are included. See [trial evidence](VLA_HTTP_ROLLOUT.md).
