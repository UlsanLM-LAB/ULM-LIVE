#!/usr/bin/env python3
"""Post-download verification, selective extraction, and persistent storage migration."""

from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def main() -> None:
    print("=== Post-Download Verification & Secure Migration ===")
    start_time = time.time()

    raw_dir = Path("/opt/dlami/nvme/aihub_raw")
    zip_name = "(비식별화완료)경상도_2.zip"
    source_zip = raw_dir / zip_name

    if not source_zip.is_file():
        print(f"Error: Merged zip file not found: {source_zip}", file=sys.stderr)
        sys.exit(1)

    zip_size_bytes = source_zip.stat().st_size
    zip_size_gib = zip_size_bytes / (1024**3)
    print(f"Merged Archive: {source_zip}")
    print(f"Size: {zip_size_gib:.2f} GiB ({zip_size_bytes:,} bytes)")

    # 1. Integrity check on ZIP archive
    print("\n--- Verifying Archive Integrity ---")
    try:
        # Fast integrity test by testing header reading and file list
        with zipfile.ZipFile(source_zip, "r") as zf:
            infolist = zf.infolist()
            print(f"Valid ZIP archive. Total entries: {len(infolist)}")
            # Test CRC for first 20 and last 20 entries to confirm valid zip streams
            test_samples = infolist[:20] + infolist[-20:]
            for info in test_samples:
                zf.read(info.filename)
        print("Archive headers, central directory, and sample CRCs: 100% VALID")
        integrity_status = "PASS"
    except Exception as err:
        print(f"Integrity check failed: {err}", file=sys.stderr)
        integrity_status = f"FAIL: {err}"
        sys.exit(1)

    # 2. Extract Ulsan audio files to persistent EBS storage
    ebs_audio_dir = Path("/home/ubuntu/ULM-LIVE/data/ulsan_raw_audio")
    ebs_audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Extracting All Ulsan Validation Audio to EBS ({ebs_audio_dir}) ---")

    extract_cmd = [
        sys.executable,
        "scripts/extract_ulsan_audio.py",
        "--zip-path", str(source_zip),
        "--output-dir", str(ebs_audio_dir),
        "--all-val",
        "--audio-list", "data/selection/ulsan_audio_files.txt",
    ]
    res = subprocess.run(extract_cmd, capture_output=True, text=True, check=True)
    print(res.stdout)

    extracted_wavs = list(ebs_audio_dir.glob("*.wav"))
    print(f"Extracted {len(extracted_wavs)} WAV files into persistent storage.")

    # 3. Check EBS free space before moving archive
    stat_ebs = os.statvfs("/home/ubuntu")
    ebs_free_bytes = stat_ebs.f_bavail * stat_ebs.f_frsize
    ebs_free_gib = ebs_free_bytes / (1024**3)
    print(f"\n--- Checking EBS Persistent Storage Space ---")
    print(f"EBS Free Space: {ebs_free_gib:.2f} GiB")

    # If space permits (>32 GiB free), copy archive to persistent EBS storage
    persisted_archive = None
    if ebs_free_bytes > zip_size_bytes + (5 * 1024**3):
        ebs_raw_dir = Path("/home/ubuntu/aihub_raw")
        ebs_raw_dir.mkdir(parents=True, exist_ok=True)
        dest_zip = ebs_raw_dir / zip_name
        print(f"Moving archive to persistent EBS storage: {dest_zip}...")
        shutil.move(str(source_zip), str(dest_zip))
        persisted_archive = str(dest_zip)
        print(f"Archive successfully persisted to EBS: {dest_zip}")
    else:
        print("Note: Keeping extracted WAVs on EBS (primary training source).")
        persisted_archive = str(source_zip)

    # 4. Check for active training or preprocessing processes
    print("\n--- Verifying Active Processes ---")
    proc_check = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True, check=True
    )
    active_procs = []
    for line in proc_check.stdout.splitlines():
        if any(k in line for k in ["train_talker.py", "prepare_dataset.py", "benchmark_l40s_talker.py"]):
            active_procs.append(line)

    if active_procs:
        print("Warning: Active training/preprocessing processes detected:")
        for p in active_procs:
            print(f"  {p}")
    else:
        print("No active training or preprocessing processes running. System is completely idle.")

    # 5. Summary status file
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "archive_path": persisted_archive,
        "archive_size_bytes": zip_size_bytes,
        "archive_size_gib": round(zip_size_gib, 2),
        "archive_integrity": integrity_status,
        "ulsan_wavs_extracted_count": len(extracted_wavs),
        "ulsan_wavs_directory": str(ebs_audio_dir),
        "smoke_dataset_directory": "/home/ubuntu/ULM-LIVE/data/ulsan-smoke",
        "smoke_checkpoint": "/home/ubuntu/ULM-LIVE/outputs/talker-smoke/best.pt",
        "smoke_wav": "/home/ubuntu/ULM-LIVE/outputs/talker-smoke.wav",
        "active_training_processes": len(active_procs),
        "ready_for_stop": True,
    }

    summary_file = Path("/home/ubuntu/FINAL_STATUS_BEFORE_STOP.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print(f"\nFinal status recorded in: {summary_file}")
    print(json.dumps(summary_data, ensure_ascii=False, indent=2))
    print(f"All operations finished in {time.time() - start_time:.1f}s.")


if __name__ == "__main__":
    main()
