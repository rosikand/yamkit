# Source this to work in the yamkit environment:   source scripts/env.sh
# Everything (interpreter, packages, caches, datasets, checkpoints) stays inside this directory.
_yk_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export YAMKIT_ROOT="$_yk_root"
export UV_PYTHON_INSTALL_DIR="$_yk_root/.uv-python"
export HF_HOME="$_yk_root/data/hf"
export HF_LEROBOT_HOME="$_yk_root/data/lerobot"
export TORCH_HOME="$_yk_root/data/torch"
export WANDB_DIR="$_yk_root/outputs/wandb"
export HF_HUB_DISABLE_TELEMETRY=1
# uv installed by setup.sh (only if it was not already on the machine)
[ -x "$_yk_root/.tools/uv" ] && export PATH="$_yk_root/.tools:$PATH"
# shellcheck disable=SC1091
source "$_yk_root/.venv/bin/activate"
unset _yk_root
