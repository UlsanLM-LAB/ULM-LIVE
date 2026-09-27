#!/usr/bin/env bash
set -euo pipefail
ROOT="${LIVE_ROOT:-/home/ubuntu/ULM-LIVE}"
DIR="$ROOT/outputs/runtime"
for name in live text; do
  f="$DIR/$name.pid"
  if [[ -f "$f" ]]; then
    pid="$(cat "$f" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
    rm -f "$f"
  fi
done
for port in 8001 8000; do
  pids="$(lsof -t -iTCP:$port -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
  fi
done
sleep 1
for port in 8001 8000; do
  pids="$(lsof -t -iTCP:$port -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    kill -9 $pids 2>/dev/null || true
  fi
done
