# Everything lives in this directory: interpreter (.uv-python), venv (.venv), uv cache (.uv-cache).
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python

.PHONY: setup sync test lint doctor can discover

setup:            ## first-time bootstrap (uv, Python 3.12, deps) — all inside this directory
	./setup.sh

sync:             ## re-install after editing pyproject / plugins
	uv sync --extra dev

test:
	uv run pytest -q

lint:
	uv run ruff check src plugins tests

doctor:
	uv run yamkit doctor

can:
	uv run yamkit can

discover:
	uv run yamkit discover
