#!/usr/bin/env bash
set -euo pipefail
replay_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$replay_repo_root"
uv sync --project replay
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project replay pytest replay/tests/test_readers.py -q
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --project replay pytest replay/tests/test_api.py -q
env -u PYTHONPATH uv run --project replay ruff check replay
pnpm --dir replay/frontend install --frozen-lockfile
pnpm --dir replay/frontend build
