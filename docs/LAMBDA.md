# Run MolmoAct2 on an existing Lambda GPU

Use an already running Linux x86_64 Lambda instance with a working NVIDIA driver. The robot
stays on the Lenovo; Conductor administers the machines over SSH. The inference route is direct:

```text
Lenovo cameras/state -> Lenovo localhost:8765 -> SSH -> Lambda localhost:8765 -> one GPU
Lenovo action queue  <- validated predictions <- SSH <- authenticated model service
```

These steps do not create, resize, or terminate a VM. Setup, model serving, attachment and
qualification do not activate robot motors. The rollout command at the end does, and needs the
operator's approval of that exact command before it is run.

## 1. Prepare the GPU checkout

Lambda's default SSH username is `ubuntu`. Verify the instance's SSH host-key fingerprint through
the cloud console or an already verified administrative connection, and keep the verified entry
in a repository-local `known_hosts` file. The examples use `-F /dev/null` to prevent unrelated
user SSH configuration from changing the connection. See Lambda's
[SSH connection guide](https://docs.lambda.ai/public-cloud/on-demand/connecting-instance/).

Clone the complete yamkit repository into `/home/ubuntu/yamkit` and check out the same commit as
the Lenovo. Retain `src/`, `configs/` and `plugins/`: source identity includes the follower's
validation/command-limit source even though the GPU process never imports the hardware plugin.

On Lambda:

```bash
cd /home/ubuntu/yamkit
nvidia-smi
./scripts/setup_inference.sh
source data/inference/env.sh
```

The script installs uv in `.tools/`, Python 3.12 in `.uv-python/`, and the CUDA environment in
`.venv-inference/`. Its generated `data/inference/env.sh` keeps Python/model/compiler caches and
temporary files under the checkout. It uses `configs/modal-requirements.txt`, plus the same
`uvicorn==0.52.4` / `h11==0.16.0` HTTP layer as the Modal image. The inference environment loads
yamkit through `PYTHONPATH=src`, without installing the local arm SDK or plugins.

Use this script on the GPU host instead of the robot's `setup.sh` or `uv sync`: the normal
project lock selects CPU PyTorch. `--no-config` prevents that CPU index from affecting the GPU
install. No system packages, driver changes, shell-profile changes, model loading or service
startup happen during setup. Stop an existing inference process before rerunning setup or
changing its checkout. In a new GPU shell, source `data/inference/env.sh` again.

Lambda's default image already includes NVIDIA drivers/CUDA and development tools, but alternative
images differ. Verify the actual machine before installing dependencies; the script does not repair
a missing driver. [Lambda image documentation](https://docs.lambda.ai/public-cloud/on-demand/).

## 2. Start one private, bounded model process

Create a dedicated bearer token on Lambda without displaying it. This refuses to overwrite an
existing token:

```bash
cd /home/ubuntu/yamkit
source data/inference/env.sh
.venv-inference/bin/python - <<'PY'
import os
import secrets
from pathlib import Path

path = Path('data/inference/lambda-georgia.token')
with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as output:
    output.write(secrets.token_urlsafe(48) + '\n')
PY
```

Start the service in a terminal that remains connected:

```bash
CUDA_VISIBLE_DEVICES=0 .venv-inference/bin/python -m yamkit.inference.standalone_service \
  --service-id lambda-georgia --provider lambda --region Georgia --port 8765 \
  --token-file data/inference/lambda-georgia.token --session-seconds 28800 \
  --task 'put the red cube into the black container'
```

The service binds only to `127.0.0.1`. It loads the pinned MolmoAct2 checkpoint/backbone, warms
the actual task and 640×480 raw RGB inputs, and retains the unchanged bfloat16, ten-step CUDA
graph and 30×14 action output. Downloads reuse `data/hf/`; no separate checkpoint copies are
needed. `CUDA_VISIBLE_DEVICES=0` selects one GPU even on an instance with several GPUs.

The supervisor bounds its owned model process to eight hours in this example, including startup.
Expiry or stopping the supervisor stops its model process. **This does not stop or terminate the
Lambda VM, and VM billing continues.** VM lifecycle remains the owner's responsibility. Do not
reuse a qualification after restarting the service: its process identity and expiry change.

## 3. Forward directly from Lenovo

On Lenovo, use the robot checkout's normal environment:

```bash
cd /home/andre/rohan-new
source scripts/env.sh
mkdir -p data/inference
```

Store the SSH private key at `data/inference/lambda.pem` with mode `0600`, and the independently
verified host-key entry at `data/inference/known_hosts`. Copy the bearer token directly over SSH;
replace `LAMBDA_IP` with the existing instance's address:

```bash
umask 077
scp -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=data/inference/known_hosts -i data/inference/lambda.pem \
  ubuntu@LAMBDA_IP:/home/ubuntu/yamkit/data/inference/lambda-georgia.token \
  data/inference/lambda-georgia.token
chmod 600 data/inference/lambda-georgia.token
```

Keep this tunnel running in a separate Lenovo terminal:

```bash
ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=data/inference/known_hosts -i data/inference/lambda.pem \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -N -L 127.0.0.1:8765:127.0.0.1:8765 ubuntu@LAMBDA_IP
```

Both ports must match the service's `--port`: readiness binds the literal origin
`http://127.0.0.1:8765`. SSH encrypts the network hop; the local RPC still requires its independent
bearer token. No public model port or Mac/Conductor inference relay is needed. Lambda's default
firewall permits SSH; retain that access and keep the model port closed. Known source addresses
can be restricted through the existing firewall. [Lambda firewall documentation](https://docs.lambda.ai/public-cloud/firewalls/).

## 4. Attach and qualify on Lenovo

These commands contact the model with hardware-free inputs and do not open arms:

```bash
yamkit external-attach --name lambda-georgia --endpoint http://127.0.0.1:8765 \
  --token-file data/inference/lambda-georgia.token
yamkit external-status --service lambda-georgia
yamkit external-qualify --service lambda-georgia \
  --task 'put the red cube into the black container' --requests 50
```

Qualification must pass on Lenovo through this exact tunnel and current source/model identity.
It measures the actual remote pipeline, including image transfer, inference and queue behavior;
GPU size and region alone do not establish usable latency. Keep the task, image shape and
execution settings unchanged. A failed or expired qualification cannot authorize motion.

## 5. Supervised physical rollout

Before running this command, obtain explicit operator approval for the exact command and its
effects: both followers energize and home, the policy runs for five seconds, a healthy completed
run returns home, then followers release. Stop, faults and expiry abort movement and release.

```bash
yamkit rollout --policy molmoact2 --backend external --external-service lambda-georgia \
  --call-mode http --execution-mode cuda_graph10 --image-encoding rgb8 \
  --task 'put the red cube into the black container' \
  --arms left_follower --arms right_follower --duration 5 --accept-mapping --confirm-supervised
```

An attached server or passing qualification does not prove the red-cube task will succeed.
Inspect observations, predictions, executed actions and video after each supervised trial.
Existing command shaping, original action deadlines and Stop/release behavior still apply.

When finished, stop the service supervisor on Lambda and close the Lenovo tunnel. Detaching a
local service registration is not VM shutdown; neither this guide's commands nor the model's
expiry terminate the user-owned Lambda instance.

```bash
yamkit external-detach --service lambda-georgia
```
