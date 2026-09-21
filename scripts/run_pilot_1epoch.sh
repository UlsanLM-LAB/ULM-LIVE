#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/home/ubuntu/ULM-LIVE/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
cd /home/ubuntu/ULM-LIVE
source .venv/bin/activate

OUTDIR="/home/ubuntu/ULM-LIVE/outputs/talker-pilot-1epoch"
mkdir -p "$OUTDIR"

echo "=== [1/2] Starting 1-Epoch Pilot Training on L40S ==="
echo "Timestamp: $(date -u)"
echo "Output Dir: $OUTDIR"

python3 -u scripts/train_talker.py \
    --config configs/talker.yaml \
    --manifest data/ulsan-full/train.cached.jsonl \
    --val-manifest data/ulsan-full/val.cached.jsonl \
    --thinker /home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged \
    --output-dir "$OUTDIR" \
    --epochs 1 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 0.0002 \
    --warmup-ratio 0.03 \
    --stop-pos-weight 5.0 \
    --bf16 \
    --num-workers 4 \
    --save-at-steps 346,692,1038,1385 \
    --eval-at-steps 346,692,1038,1385 \
    --save-steps 999999 \
    --eval-steps 999999

echo "=== [2/2] Training Complete. Starting Pilot Regression & Diagnostics ==="
echo "Timestamp: $(date -u)"

python3 -u scripts/evaluate_pilot_checkpoints.py \
    --thinker /home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged \
    --checkpoint-dir "$OUTDIR" \
    --steps 346,692,1038,1385 \
    --output-dir "$OUTDIR/eval" \
    --device cuda

echo "=== 1-Epoch Pilot Pipeline Successfully Finished ==="
echo "Timestamp: $(date -u)"
