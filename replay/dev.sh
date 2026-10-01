#!/usr/bin/env bash
set -euo pipefail

replay_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$replay_repo_root"

for replay_tool in uv pnpm setsid; do
  if ! command -v "$replay_tool" >/dev/null 2>&1; then
    printf '缺少 %s，请先安装后再启动。\n' "$replay_tool" >&2
    exit 1
  fi
done

uv sync
pnpm --dir replay/frontend install --frozen-lockfile
if [[ -z "${REPLAY_DATA_ROOT:-}" ]] && {
  [[ ! -f replay/demo_data/demo_v21/meta/info.json ]] ||
  [[ ! -f replay/demo_data/demo_v30/meta/info.json ]]
}; then
  env -u PYTHONPATH uv run python -m replay.scripts.generate_demo
fi

replay_api_pid=''
replay_web_pid=''
replay_stop() {
  trap - EXIT INT TERM
  for replay_pid in "$replay_api_pid" "$replay_web_pid"; do
    if [[ -n "$replay_pid" ]]; then
      kill -- "-$replay_pid" 2>/dev/null || true
      wait "$replay_pid" 2>/dev/null || true
    fi
  done
}
trap replay_stop EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid env -u PYTHONPATH uv run uvicorn replay.backend.main:app \
  --host 127.0.0.1 --port 8000 &
replay_api_pid=$!
setsid pnpm --dir replay/frontend dev --host 127.0.0.1 --port 5173 --strictPort &
replay_web_pid=$!
printf '\n回放页面：http://127.0.0.1:5173\nAPI 文档：http://127.0.0.1:8000/docs\nCtrl+C 关闭前后端。\n\n'
wait -n "$replay_api_pid" "$replay_web_pid"
