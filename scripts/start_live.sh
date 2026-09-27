#!/usr/bin/env bash
set -euo pipefail

LIVE_ROOT="${LIVE_ROOT:-/home/ubuntu/ULM-LIVE}"
TEXT_ROOT="${TEXT_ROOT:-/home/ubuntu/ULM-4B-runtime}"
TEXT_PY="${TEXT_PY:-/home/ubuntu/ULM-1.7B/.venv/bin/python}"
TTS_PY="${TTS_PY:-$LIVE_ROOT/.venv-tts/bin/python}"
BASE_MODEL="${ULM_MODEL_PATH:-/home/ubuntu/models/Qwen3.8-4B-Distill}"
ULM_ADAPTER_PATH="${ULM_ADAPTER_PATH:-/home/ubuntu/models/ULM-4B-Arm-B}"
ULM_TTS_ADAPTER_PATH="${ULM_TTS_ADAPTER_PATH:-/home/ubuntu/models/ULM-Live-Ulsan-TTS-smoke-v1.pt}"
RUNTIME_DIR="$LIVE_ROOT/outputs/runtime"
mkdir -p "$RUNTIME_DIR"

"$LIVE_ROOT/scripts/stop_live.sh" >/dev/null 2>&1 || true

cd "$TEXT_ROOT"
export ULM_ADAPTER_PATH
nohup "$TEXT_PY" scripts/serve.py --model "$BASE_MODEL" --host 127.0.0.1 --port 8000 \
  > "$RUNTIME_DIR/text.log" 2>&1 < /dev/null &
echo $! > "$RUNTIME_DIR/text.pid"

for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS http://127.0.0.1:8000/health >/dev/null

cd "$LIVE_ROOT"
export ULM_BACKEND_URL="http://127.0.0.1:8000/v1/chat/completions"
export ULM_TTS_ADAPTER_PATH
export ULM_ASR_MODEL="${ULM_ASR_MODEL:-openai/whisper-small}"
export LD_LIBRARY_PATH="$LIVE_ROOT/.venv-tts/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
nohup "$TTS_PY" -m uvicorn ulm_live.tts_service:app --host 0.0.0.0 --port 8001 --workers 1 \
  > "$RUNTIME_DIR/live.log" 2>&1 < /dev/null &
echo $! > "$RUNTIME_DIR/live.pid"

for _ in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8001/health >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS http://127.0.0.1:8001/health
echo
echo "ULM Live ready on :8001"
