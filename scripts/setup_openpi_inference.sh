#!/usr/bin/env bash
# Dedicated official JAX runtime. Never touches .venv or .venv-inference.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  printf '%s\n' 'Usage: bash scripts/setup_openpi_inference.sh' \
    'Installs the immutable official OpenPI runtime in data/openpi/venv.' \
    'Requires repository-local .tools/uv. No model, service, or hardware is started.'
  exit 0
fi
(( $# == 0 )) || { printf '%s\n' 'Unexpected arguments; use --help.' >&2; exit 2; }
OPENPI_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$OPENPI_ROOT"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || exit 2
[[ -x .tools/uv ]] || { printf '%s\n' 'Run the one-time yamkit inference installation to obtain repo-local uv first.' >&2; exit 2; }
for openpi_path in .tools/uv .uv-python data/openpi data/openpi/upstream data/openpi/venv \
  data/openpi/uv-cache data/openpi/cache data/openpi/tmp data/openpi/hf data/openpi/xdg; do
  case "$(realpath -m -- "$openpi_path")" in
    "$OPENPI_ROOT"/*) ;;
    *) printf '%s\n' 'Refusing an OpenPI path outside this checkout.' >&2; exit 2 ;;
  esac
done
mkdir -p data/openpi/tmp data/openpi/uv-cache data/openpi/hf data/openpi/xdg
export UV_PYTHON_INSTALL_DIR="$OPENPI_ROOT/.uv-python"
export UV_PYTHON_BIN_DIR="$OPENPI_ROOT/.tools"
export UV_CACHE_DIR="$OPENPI_ROOT/data/openpi/uv-cache"
export UV_CREDENTIALS_DIR="$OPENPI_ROOT/data/openpi/uv-credentials"
export UV_PROJECT_ENVIRONMENT="$OPENPI_ROOT/data/openpi/venv"
export UV_LINK_MODE=copy
export HF_HOME="$OPENPI_ROOT/data/openpi/hf"
export XDG_CACHE_HOME="$OPENPI_ROOT/data/openpi/xdg"
export TMPDIR="$OPENPI_ROOT/data/openpi/tmp"
export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1
OPENPI_REVISION=215abfb217dbac7d5f1273282331b9b1866c0479
if [[ ! -e data/openpi/upstream ]]; then
  git init --quiet data/openpi/upstream
  git -C data/openpi/upstream remote add origin https://github.com/Physical-Intelligence/openpi.git
  git -C data/openpi/upstream fetch --depth 1 origin "$OPENPI_REVISION"
  git -C data/openpi/upstream switch --detach --quiet "$OPENPI_REVISION"
fi
[[ "$(git -C data/openpi/upstream rev-parse HEAD)" == "$OPENPI_REVISION" ]] || {
  printf '%s\n' 'Existing OpenPI checkout has another revision; no overwrite attempted.' >&2; exit 2;
}
[[ -z "$(git -C data/openpi/upstream status --porcelain)" ]] || {
  printf '%s\n' 'Existing OpenPI checkout is dirty; no overwrite attempted.' >&2; exit 2;
}
# Use upstream's complete committed lock, not the MolmoAct2/LeRobot dependency environment.
.tools/uv --no-config python install 3.11.13
.tools/uv sync --project data/openpi/upstream --frozen --no-dev --python 3.11.13
.tools/uv pip check --python data/openpi/venv/bin/python
printf '%s\n' 'Official OpenPI JAX environment installed; physical pi05_base YAM rollout remains blocked.' \
  'Download pinned public assets with PYTHONPATH=src data/openpi/venv/bin/python -m yamkit.openpi.assets'
