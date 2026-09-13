# Inference without a separate preparation command

The simple workflow keeps **policy** and **compute backend** separate. It wraps the existing
MolmoAct2 reference qualification and rollout APIs; it does not replace the controller.

From the robot checkout:

```bash
source scripts/env.sh
yamkit inference --backend lambda --policy molmoact2 \
  --task 'put the red cube into the black container'
```

`inference` connects to the configured existing GPU, reuses a matching live service and current
host/task qualification when possible, or starts and qualifies that service. Its warmup and
50-sample qualification use generated images and fake arms. It never opens a camera or arm,
and its successful result is not motion approval. Add `--requalify` to collect fresh software
evidence deliberately. Evidence stays under `.context/inference-preparation/<operation>/`;
existing historical evidence is retained.

For a physical run, there is no prerequisite prepare command:

```bash
yamkit rollout --backend lambda --policy molmoact2 \
  --task 'put the red cube into the black container' --duration 60
```

After software preparation, the terminal shows the exact command and effects and asks for
one on-site confirmation. Verify arm/camera mapping, secure mounts, empty grippers, a clear
workspace, and accessible Stop/power cutoff. Stay with the arms. Only affirmative confirmation
starts that one run. The existing `--accept-mapping --confirm-supervised` flags remain available
for an independently approved exact command; do not put them into unattended scripts.

Both followers energize, home and open their grippers, then execute the fixed policy. Healthy
completion homes while preserving the final gripper opening and releases; Stop/fault releases
without home. Ctrl-C remains the terminal Stop. There are no automatic physical retries.
`rollout --backend lambda ... --dry-run` may prepare GPU software but never starts hardware.

The simple MolmoAct2 path fixes reference execution, 30 Hz, HTTP, `cuda_graph10`, full 640×480
raw RGB, no crop and no RTC. Legacy `--backend external`/`modal`/`local` commands retain their
existing defaults and behavior. Normal CLI rollout prints controller metrics; use the UI's
Recording/Upload choices for video capture, playback and HF archives.

## Configure an existing GPU once

An installation with exactly one matching saved Lambda attachment can reuse it without a new
configuration file. Automatic reconnect/start needs explicit local configuration. Store this
JSON at `data/inference/backends.json`, with mode `0600`. It is git-ignored. Replace all example
values with the existing installation's paths and service identity:

```json
{
  "version": 1,
  "backends": {
    "lambda": {
      "ssh": {
        "host": "your-gpu-ssh-alias"
      },
      "policies": {
        "molmoact2": {
          "service": "your-model-service",
          "endpoint": "http://127.0.0.1:8765",
          "token_file": "data/inference/your-model-service.token",
          "remote": {
            "repo": "/absolute/path/to/gpu/yamkit",
            "token_file": "data/inference/your-model-service.token",
            "region": "your-existing-region",
            "session_seconds": 28800,
            "gpu": 0
          }
        }
      }
    }
  }
}
```

The SSH host may be an existing SSH-config alias or `user@host`. Normal SSH configuration and
agent behavior are preserved. Optional `ssh.identity_file` and `ssh.known_hosts_file` specify
existing repository-local files; neither file is printed or copied. Host verification is always
strict and noninteractive. Verify unknown host keys independently; this workflow never accepts
them automatically or edits SSH settings. Existing legacy tunnels are reused only when their
owned listener, explicit forwarding arguments and SSH destination match the configuration.

The GPU checkout must already have the matching source, `.venv-inference`,
`data/inference/env.sh`, cached/downloadable pinned checkpoint and its private token file;
see [the Lambda setup guide](LAMBDA.md). Keep the same bearer token in the named private file
on the robot host, transferred securely outside this command. Tokens are **file paths only**
in configuration, never token values. The local and remote loopback port must match, because
runtime identity binds that exact origin.

Omit `remote` to manage only an existing forward/service. Omit both `remote` and `ssh` to use
an already working tunnel. There is no host, account, key or credential hardcoded in the CLI.
An unknown listener, authentication mismatch or source mismatch produces an actionable error;
the workflow never kills or takes over that listener. GPU startup uses a repository-local
cross-process lock and a bounded detached supervisor, so repeated commands reuse the loaded
model. Its log is `data/inference/managed/<service>/service.log` on the GPU.

This never provisions or terminates a VM. Model expiry stops only the bounded model process;
**VM billing continues**. Stop an owned service manually when appropriate using the existing
administrative workflow. A service approaching expiry is not silently interrupted to extend
its lease.

```bash
yamkit backend-status --backend lambda --policy molmoact2  # saved metadata only
yamkit external-status --service your-model-service      # existing admin interface
```

## Shared UI behavior

The Inference page uses the same prompt-preparation helper and backend connection API. Opening
the page, changing fields and checking preflight are local-only; pressing Start may launch an
explicit managed software-only preparation job. With matching backend configuration, that job
can recover an expired attachment or connect/start the existing GPU service. It then warms and
qualifies the exact task. The browser still requires its fresh foreground supervised confirmation
before motion; preparation never automatically starts the arms.

A repository-local cross-process lock excludes simultaneous CLI/UI model preparation and managed
Start. A user-active UI operation, including saving, blocks CLI preparation. The simple CLI
holds the same lock for its entire physical rollout, so another UI prompt cannot rewarm its
model mid-run. Nothing stops user applications or reduces capture memory admission thresholds.

## Native π0.5

The simple workflow maps `--policy pi05` to `pi05-yam`, the separate native checkpoint
`Jiafei1224/molmoact2-yam-pi05@51ab2720d7e56d51410407f98ea64bbea97feb2e`.
The legacy low-level `pi05` profile remains the unadapted base checkpoint; it is never selected
as a fallback by this workflow. Native π0.5 observes every control tick and executes all 30
absolute 14-D rows FIFO before replanning, without MolmoAct2 interpolation, prefix drops or RTC.
Its 30 Hz tick budget starts before observation/inference; an overrun adds no post-send wait. Saved native
normalization and 224×224 padded image preprocessing stay with the π0.5 runtime.

Add a separate `pi05-yam` entry under `backends.lambda.policies`, using a different service/port:

```json
{
  "service": "your-pi05-service",
  "endpoint": "http://127.0.0.1:8766",
  "token_file": "data/inference/your-pi05-service.token",
  "saved_observations": [".context/saved-real-yam/observation-000.npz"],
  "remote": {
    "repo": "/absolute/path/to/gpu/yamkit",
    "token_file": "data/inference/your-pi05-service.token",
    "region": "your-existing-region",
    "session_seconds": 28800,
    "gpu": 0
  }
}
```

List up to 50 existing real-recording NPZ paths. Each contains exactly `state` (14 finite values),
`top`, `left_wrist`, and `right_wrist` (RGB uint8 arrays matching the rig). Qualification cycles
through the listed observations, with no live capture. It checks 50 direct warm samples and
50 full fake-arm chunks, row accounting, configured bounds and Stop during RPC. Reports remain
under `.context/pi05-qualification/<operation>/`; failed evidence never becomes current proof.

The configured HF account must have accepted and received access to the gated
`google/paligemma-3b-pt-224` tokenizer dependency. Access denial is a blocker, not permission to
substitute another tokenizer, loosen checkpoint loading, or run an unqualified policy. No new
VM is provisioned and the workflow never stops MolmoAct2 to free GPU memory automatically.

```bash
yamkit inference --backend lambda --policy pi05 --task 'put the red cube into the black container'
# Only after successful native qualification and fresh on-site verification:
yamkit rollout --backend lambda --policy pi05 --task 'put the red cube into the black container' --duration 5
```

The rollout command still asks for one explicit terminal confirmation. Native π0.5 saves JSON
trace/report artifacts under `outputs/inference/pi05-<operation>/` after release. Native video,
HF upload and UI policy selection are not yet integrated; use MolmoAct2's established UI for
that workflow. Software qualification does not prove physical task success.
