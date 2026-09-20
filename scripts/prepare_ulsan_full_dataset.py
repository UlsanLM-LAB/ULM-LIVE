#!/usr/bin/env python3
"""Automated pipeline to prepare the canonical Ulsan 25-30h speech dataset from AI Hub audio archive.

Workflow:
1. Load selected sessions from data/selection/selected_sessions.json (98 sessions, 159 speakers, ~26h).
2. Scan or extract source audio files (.wav) from raw archive / directory:
   - If raw zip: extract ONLY the 98 target sessions to scratch storage (e.g. NVMe scratch).
   - If already extracted directory, locate them directly.
3. For each session and utterance:
   - Match label JSON from data/labels/{session_id}.json.
   - Slices waveform, resamples to 24 kHz mono 16-bit PCM.
   - Validates audio & transcript quality (non-empty, finite, duration bounds).
   - Saves preprocessed WAV to data/ulsan-full/audio/{item_id}.wav.
   - Emits record to data/ulsan-full/manifest.jsonl with full metadata.
4. Generates comprehensive integrity report data/ulsan-full/dataset_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import soundfile as sf
import torch
import torchaudio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Ulsan speech dataset from AI Hub shard 572714.")
    parser.add_argument(
        "--raw-archive",
        type=str,
        default="/home/ubuntu/aihub_raw/(비식별화완료)경상도_2.zip",
        help="Path to raw audio zip archive or extracted audio directory.",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        help="Directory containing session JSON labels (176 sessions).",
    )
    parser.add_argument(
        "--selection-file",
        type=str,
        default="data/selection/selected_sessions.json",
        help="Path to selected sessions JSON (98 sessions).",
    )
    parser.add_argument(
        "--scratch-dir",
        type=str,
        default="/opt/dlami/nvme/scratch_audio",
        help="Temporary scratch directory for extracting session WAVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ulsan-full",
        help="Directory to save canonical processed WAVs and manifest.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=24000,
        help="Target sample rate in Hz (default: 24000).",
    )
    parser.add_argument(
        "--clean-scratch",
        action="store_true",
        default=True,
        help="Clean up scratch directory after processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_archive = Path(args.raw_archive)
    labels_dir = Path(args.labels_dir)
    selection_file = Path(args.selection_file)
    scratch_dir = Path(args.scratch_dir)
    output_dir = Path(args.output_dir)
    audio_out_dir = output_dir / "audio"
    manifest_path = output_dir / "manifest.jsonl"
    report_path = output_dir / "dataset_report.json"

    audio_out_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)

    print("=== [ULM-LIVE] Ulsan Speech Dataset Pipeline ===")
    print(f"Raw archive / dir: {raw_archive}")
    print(f"Labels dir:        {labels_dir}")
    print(f"Selection file:    {selection_file}")
    print(f"Scratch dir:       {scratch_dir}")
    print(f"Output dir:        {output_dir}")

    # 1. Load selected sessions
    if not selection_file.is_file():
        print(f"Error: selection file not found: {selection_file}", file=sys.stderr)
        sys.exit(1)

    with open(selection_file, "r", encoding="utf-8") as f:
        selected_sessions_data = json.load(f)

    target_session_ids = {s["session_id"] for s in selected_sessions_data}
    print(f"Target selected sessions: {len(target_session_ids)}")

    # 2. Extract or locate session WAVs
    session_audio_map: dict[str, Path] = {}

    if raw_archive.is_dir():
        print(f"Locating session WAVs in directory: {raw_archive}")
        for p in raw_archive.rglob("*.wav"):
            if p.stem in target_session_ids:
                session_audio_map[p.stem] = p
    elif raw_archive.is_file() and zipfile.is_zipfile(raw_archive):
        print(f"Extracting target sessions from zip archive: {raw_archive}")
        with zipfile.ZipFile(raw_archive, "r") as zf:
            namelist = zf.namelist()
            matching_names = [name for name in namelist if Path(name).stem in target_session_ids and name.lower().endswith(".wav")]
            print(f"Found {len(matching_names)} matching session audio files in zip.")
            for idx, name in enumerate(matching_names, 1):
                stem = Path(name).stem
                dest_path = scratch_dir / f"{stem}.wav"
                if not dest_path.is_file():
                    with zf.open(name) as src, open(dest_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                session_audio_map[stem] = dest_path
                if idx % 20 == 0 or idx == len(matching_names):
                    print(f"  Extracted: {idx}/{len(matching_names)} session WAVs")
    else:
        print(f"Error: Raw archive does not exist or is not a valid zip / dir: {raw_archive}", file=sys.stderr)
        sys.exit(1)

    print(f"Total session WAVs available: {len(session_audio_map)} / {len(target_session_ids)}")

    # 3. Process utterances into 24kHz mono WAVs and build manifest
    processed_items = []
    missing_source_audio = 0
    missing_processed_wav = 0
    invalid_wav = 0
    empty_transcript = 0
    duplicate_ids = 0
    duplicate_audio_path = 0

    seen_ids = set()
    seen_paths = set()
    speakers_seen = set()
    sessions_seen = set()

    gender_counter = Counter()
    age_counter = Counter()
    durations = []

    resamplers: dict[int, torchaudio.transforms.Resample] = {}

    item_counter = 0

    for s_info in selected_sessions_data:
        sess_id = s_info["session_id"]
        audio_src = session_audio_map.get(sess_id)
        if not audio_src or not audio_src.is_file():
            missing_source_audio += 1
            continue

        # Load session audio
        try:
            waveform, orig_sr = torchaudio.load(str(audio_src))
            # Convert to mono if multi-channel
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
        except Exception as err:
            print(f"Failed to load audio for {sess_id}: {err}")
            invalid_wav += 1
            continue

        # Load label JSON
        label_file = labels_dir / f"{sess_id}.json"
        if not label_file.is_file():
            print(f"Warning: label file missing: {label_file}")
            continue

        with open(label_file, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # Speaker metadata mapping
        spk_meta_map = {}
        for spk in label_data.get("speaker", []):
            spk_id = spk.get("id")
            spk_meta_map[spk_id] = spk

        utterances = label_data.get("utterance", [])
        for utt in utterances:
            utt_id = utt.get("id")
            text = utt.get("dialect_form") or utt.get("form") or ""
            text = text.strip()
            if not text:
                empty_transcript += 1
                continue

            local_spk_id = utt.get("speaker_id")
            global_spk_id = f"{sess_id}_{local_spk_id}"

            spk_meta = spk_meta_map.get(local_spk_id, {})
            gender = spk_meta.get("gender", "unknown")
            age = spk_meta.get("age", "unknown")

            start_sec = float(utt.get("start", 0.0))
            end_sec = float(utt.get("end", 0.0))
            dur = end_sec - start_sec
            if dur < 0.3 or dur > 30.0:
                continue

            start_idx = int(start_sec * orig_sr)
            end_idx = int(end_sec * orig_sr)
            if start_idx >= waveform.shape[-1] or end_idx <= start_idx:
                invalid_wav += 1
                continue

            seg = waveform[:, start_idx:min(end_idx, waveform.shape[-1])]
            if seg.shape[-1] == 0 or not torch.isfinite(seg).all():
                invalid_wav += 1
                continue

            # Resample to 24kHz if needed
            if orig_sr != args.target_sr:
                if orig_sr not in resamplers:
                    resamplers[orig_sr] = torchaudio.transforms.Resample(orig_sr, args.target_sr)
                seg_resampled = resamplers[orig_sr](seg)
            else:
                seg_resampled = seg

            # Peak normalize
            max_val = torch.max(torch.abs(seg_resampled))
            if max_val > 0:
                seg_resampled = seg_resampled * (0.95 / max_val)

            # Save WAV
            out_filename = f"{item_counter:06d}_{utt_id}.wav"
            out_filepath = audio_out_dir / out_filename
            rel_path = f"audio/{out_filename}"

            torchaudio.save(str(out_filepath), seg_resampled, args.target_sr, encoding="PCM_S", bits_per_sample=16)

            if not out_filepath.is_file():
                missing_processed_wav += 1
                continue

            final_dur = seg_resampled.shape[-1] / args.target_sr
            durations.append(final_dur)

            # Uniqueness check
            unique_id = f"ulsan_{item_counter:06d}"
            if unique_id in seen_ids:
                duplicate_ids += 1
            seen_ids.add(unique_id)

            if rel_path in seen_paths:
                duplicate_audio_path += 1
            seen_paths.add(rel_path)

            speakers_seen.add(global_spk_id)
            sessions_seen.add(sess_id)
            gender_counter[gender] += 1
            age_counter[age] += 1

            record = {
                "id": unique_id,
                "utterance_id": utt_id,
                "session_id": sess_id,
                "speaker_id": global_spk_id,
                "dialect": "ulsan",
                "region": "ulsan",
                "text": text,
                "standard_text": utt.get("standard_form", text),
                "audio_path": rel_path,
                "duration": round(final_dur, 3),
                "sample_rate": args.target_sr,
                "channels": 1,
                "gender": gender,
                "age": age,
            }
            processed_items.append(record)
            item_counter += 1

        print(f"Processed session {sess_id}: cumulative items = {len(processed_items)}")

    # 4. Write manifest.jsonl
    print(f"\nWriting canonical manifest: {manifest_path}")
    with open(manifest_path, "w", encoding="utf-8") as f:
        for it in processed_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # 5. Clean scratch if requested
    if args.clean_scratch and scratch_dir.exists():
        print(f"Cleaning scratch audio files from {scratch_dir}...")
        shutil.rmtree(scratch_dir, ignore_errors=True)

    # 6. Generate dataset_report.json
    total_hours = sum(durations) / 3600.0 if durations else 0.0
    report = {
        "dataset_name": "ULM-LIVE-Ulsan-Full-26h",
        "sample_rate": args.target_sr,
        "channels": 1,
        "total_sessions": len(sessions_seen),
        "total_utterances": len(processed_items),
        "total_speakers": len(speakers_seen),
        "total_hours": round(total_hours, 2),
        "gender_distribution": dict(gender_counter),
        "age_distribution": dict(age_counter),
        "duration": {
            "min": round(min(durations), 3) if durations else 0.0,
            "mean": round(sum(durations) / len(durations), 3) if durations else 0.0,
            "median": round(float(torch.median(torch.tensor(durations)).item()), 3) if durations else 0.0,
            "max": round(max(durations), 3) if durations else 0.0,
        },
        "integrity": {
            "missing_source_audio": missing_source_audio,
            "missing_processed_wav": missing_processed_wav,
            "invalid_wav": invalid_wav,
            "empty_transcript": empty_transcript,
            "duplicate_ids": duplicate_ids,
            "duplicate_audio_path": duplicate_audio_path,
            "critical_errors": missing_source_audio + missing_processed_wav + invalid_wav + empty_transcript + duplicate_ids + duplicate_audio_path,
        },
        "mimi_full_encoding": "NOT STARTED",
        "talker_full_training": "NOT STARTED",
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==========================================")
    print("ULM-LIVE DATASET PREPARATION SUMMARY:")
    print(f"Total Sessions:   {report['total_sessions']}")
    print(f"Total Utterances: {report['total_utterances']}")
    print(f"Total Speakers:   {report['total_speakers']}")
    print(f"Total Hours:      {report['total_hours']}h")
    print(f"Critical Errors:  {report['integrity']['critical_errors']}")
    print(f"Report saved to:  {report_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
