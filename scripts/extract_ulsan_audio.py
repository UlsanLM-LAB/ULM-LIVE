#!/usr/bin/env python3
"""Extract selected Ulsan dialect WAV audio files from AI Hub shard ZIP archive."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import zipfile
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Selectively extract Ulsan dialect audio files from AI Hub zip archive."
    )
    parser.add_argument(
        "--zip-path",
        type=str,
        required=True,
        help="Path to merged AI Hub ZIP archive (e.g., (비식별화완료)경상도_2.zip).",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="data/selection/selected_sessions.json",
        help="Path to selected_sessions.json containing target session IDs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Target directory to extract audio files.",
    )
    parser.add_argument(
        "--all-val",
        action="store_true",
        help="Extract all 176 validation sessions instead of just selected 98 sessions.",
    )
    parser.add_argument(
        "--audio-list",
        type=str,
        default="data/selection/ulsan_audio_files.txt",
        help="Path to ulsan_audio_files.txt for all validation session lookup.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    if not zip_path.is_file():
        print(f"Error: ZIP file does not exist: {zip_path}", file=sys.stderr)
        sys.exit(1)

    target_session_ids = set()
    if args.all_val:
        audio_list_path = Path(args.audio_list)
        if not audio_list_path.is_file():
            print(f"Error: audio list file does not exist: {audio_list_path}", file=sys.stderr)
            sys.exit(1)
        with open(audio_list_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[1] == "val":
                    target_session_ids.add(parts[0])
        print(f"Targeting all {len(target_session_ids)} validation sessions.")
    else:
        selection_path = Path(args.selection)
        if not selection_path.is_file():
            print(f"Error: selection file does not exist: {selection_path}", file=sys.stderr)
            sys.exit(1)
        with open(selection_path, "r", encoding="utf-8") as f:
            selected_data = json.load(f)
        target_session_ids = {item["session_id"] for item in selected_data}
        print(f"Targeting {len(target_session_ids)} selected sessions (25~30h subset).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    print(f"Opening ZIP archive: {zip_path} ({zip_path.stat().st_size / (1024**3):.2f} GB)...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        print(f"Total entries in archive: {len(namelist)}")

        # Map stem to archive filename
        matched_entries = []
        for name in namelist:
            p = Path(name)
            if p.suffix.lower() == ".wav" and p.stem in target_session_ids:
                matched_entries.append(name)

        print(f"Matched {len(matched_entries)} target WAV files.")
        if not matched_entries:
            print("Warning: No matching WAV files found in archive!", file=sys.stderr)
            # Sample first 10 entries for debugging
            print("Sample entries in archive:")
            for n in namelist[:10]:
                print(f"  {n}")
            sys.exit(1)

        total_extracted_bytes = 0
        extracted_count = 0
        for idx, entry_name in enumerate(matched_entries, 1):
            dest_file = output_dir / Path(entry_name).name
            if not dest_file.exists() or dest_file.stat().st_size == 0:
                with zf.open(entry_name) as src, open(dest_file, "wb") as dst:
                    data = src.read()
                    dst.write(data)
                    total_extracted_bytes += len(data)
            else:
                total_extracted_bytes += dest_file.stat().st_size

            extracted_count += 1
            if idx % 20 == 0 or idx == len(matched_entries):
                print(f"  [{idx}/{len(matched_entries)}] Extracted {Path(entry_name).name}")

    duration = time.time() - start_time
    print(f"\n=== Extraction Complete in {duration:.1f}s ===")
    print(f"Extracted files: {extracted_count} WAVs")
    print(f"Total extracted audio: {total_extracted_bytes / (1024**3):.2f} GB ({total_extracted_bytes / (1024**2):.1f} MB)")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
