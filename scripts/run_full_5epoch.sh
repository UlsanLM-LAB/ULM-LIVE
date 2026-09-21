#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH=/home/ubuntu/ULM-LIVE/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
cd /home/ubuntu/ULM-LIVE
source .venv/bin/activate

OUTDIR="/home/ubuntu/ULM-LIVE/outputs/talker-full-5epoch"
mkdir -p "$OUTDIR"

echo "=========================================================="
echo "=== Starting Production 5-Epoch Full Talker Training ==="
echo "Timestamp: $(date -u)"
echo "Output Directory: $OUTDIR"
echo "=========================================================="

python3 -u scripts/train_full_5epoch.py \
    --config configs/talker.yaml \
    --manifest data/ulsan-full/train.cached.jsonl \
    --val-manifest data/ulsan-full/val.cached.jsonl \
    --test-manifest data/ulsan-full/test.cached.jsonl \
    --thinker /home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged \
    --output-dir "$OUTDIR" \
    --epochs 5 \
    --batch-size 8 \
    --grad-accum 2 \
    --lr 0.0002 \
    --warmup-ratio 0.03 \
    --stop-pos-weight 5.0 \
    --stop-loss-weight 1.0 \
    --device cuda \
    --bf16 \
    --num-workers 4

echo "=========================================================="
echo "=== Full 5-Epoch Pipeline Successfully Completed ==="
echo "Timestamp: $(date -u)"
echo "=========================================================="
