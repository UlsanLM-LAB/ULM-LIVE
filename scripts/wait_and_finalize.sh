#!/bin/bash
set -euo pipefail
cd /home/ubuntu/ULM-LIVE

echo "=== Waiting for run_download.sh and curl to finish ==="
while pgrep -f "run_download.sh" > /dev/null || pgrep -f "fileSn=572714" > /dev/null; do
    CUR_SIZE=$(ls -lh /opt/dlami/nvme/aihub_raw/download.tar 2>/dev/null | awk '{print $5}' || echo "N/A")
    echo "[$(date '+%H:%M:%S UTC')] Download in progress... download.tar: $CUR_SIZE"
    sleep 30
done

echo "=== run_download.sh finished! Waiting 10 seconds for file system sync ==="
sleep 10

# Verify merged zip exists
ZIP_PATH="/opt/dlami/nvme/aihub_raw/(비식별화완료)경상도_2.zip"
if [ ! -f "$ZIP_PATH" ]; then
    echo "Searching for merged zip..."
    ZIP_PATH=$(find /opt/dlami/nvme -name "*(비식별화완료)경상도_2.zip" | head -n 1)
fi

if [ -z "$ZIP_PATH" ] || [ ! -f "$ZIP_PATH" ]; then
    echo "Error: Merged zip file not found!" >&2
    exit 1
fi

echo "Found zip file: $ZIP_PATH ($(du -sh "$ZIP_PATH"))"

# Run post_download_secure.py
export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=/home/ubuntu/ULM-LIVE/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
uv run python scripts/post_download_secure.py

echo "=== Post-download verification and extraction completed successfully ==="
touch /home/ubuntu/READY_TO_STOP
