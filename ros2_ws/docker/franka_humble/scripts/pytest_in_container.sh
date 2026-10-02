#!/usr/bin/env bash
# Runs inside the Humble container, where the repository is /workspace/data_collect.
# Test paths are relative to the repository root; the default is the whole suite.
set -euo pipefail

cd /workspace/data_collect

# The image's own /opt/uv/venv is empty until it is synced from the mounted
# project. inference.py and inference_hitl.py need websockets and msgpack from
# that venv, so a successful sync here is also what proves they can run here.
uv sync --project "$UV_PROJECT" --locked

if (( $# == 0 )); then
  set -- tests/
fi

exec uv run --project "$UV_PROJECT" --locked --with pytest python -m pytest -q "$@"
