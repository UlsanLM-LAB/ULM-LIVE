#!/usr/bin/env python3
"""Comprehensive integrity verification for ULM-LIVE processed datasets and codec tokens."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify dataset integrity, codec tokens, audio files, and speaker disjointness."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to processed dataset directory (e.g., data/ulsan-smoke or data/ulsan).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Enforce zero missing files, valid shapes, and zero speaker leakage (default: True).",
    )
    return parser.parse_args()


def verify_dataset(dataset_dir: Path, strict: bool = True) -> dict:
    manifest_path = dataset_dir / "manifest.jsonl"
    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "val.jsonl"
    test_path = dataset_dir / "test.jsonl"
    summary_path = dataset_dir / "summary.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    print(f"=== Verifying Dataset Integrity: {dataset_dir} ===")

    # 1. Read manifest items
    items = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    total_samples = len(items)
    print(f"Total manifest samples: {total_samples}")

    # Track errors & metrics
    missing_wav = 0
    missing_codec = 0
    invalid_codec = 0
    invalid_audio = 0
    empty_text = 0
    total_audio_seconds = 0.0

    speakers_all = set()
    codebook_counts = set()
    token_min_val = float("inf")
    token_max_val = float("-inf")

    for idx, item in enumerate(items, 1):
        spk = item.get("speaker_id", "")
        speakers_all.add(spk)

        text = item.get("text", "")
        if not text or not str(text).strip():
            empty_text += 1

        # Check audio
        audio_rel = item.get("audio_path")
        if not audio_rel:
            missing_wav += 1
        else:
            audio_full = dataset_dir / audio_rel if not Path(audio_rel).is_absolute() else Path(audio_rel)
            if not audio_full.is_file():
                missing_wav += 1
            else:
                try:
                    info = sf.info(str(audio_full))
                    if info.samplerate != 24000:
                        invalid_audio += 1
                    total_audio_seconds += info.duration
                except Exception:
                    invalid_audio += 1

        # Check codec
        codec_rel = item.get("codec_path")
        if not codec_rel:
            missing_codec += 1
        else:
            codec_full = dataset_dir / codec_rel if not Path(codec_rel).is_absolute() else Path(codec_rel)
            if not codec_full.is_file():
                missing_codec += 1
            else:
                try:
                    tokens = torch.load(codec_full, map_location="cpu", weights_only=True)
                    if not isinstance(tokens, torch.Tensor):
                        invalid_codec += 1
                        continue

                    # Squeeze batch if rank 3
                    if tokens.dim() == 3:
                        tokens = tokens.squeeze(0)

                    if tokens.dim() != 2:
                        invalid_codec += 1
                        continue

                    k, t = tokens.shape
                    codebook_counts.add(k)
                    if k != 32 or t <= 0:
                        invalid_codec += 1
                        continue

                    # Check NaN / Inf
                    if torch.isnan(tokens).any() or torch.isinf(tokens).any():
                        invalid_codec += 1
                        continue

                    # Check token range (0 <= token < 2048)
                    t_min = int(tokens.min().item())
                    t_max = int(tokens.max().item())
                    token_min_val = min(token_min_val, t_min)
                    token_max_val = max(token_max_val, t_max)

                    if t_min < 0 or t_max >= 2048:
                        invalid_codec += 1
                        continue

                except Exception as err:
                    print(f"Error loading codec token {codec_full}: {err}", file=sys.stderr)
                    invalid_codec += 1

        if idx % 500 == 0 or idx == total_samples:
            print(f"  Verified [{idx}/{total_samples}] items...")

    # 2. Check splits and speaker disjointness
    def get_speakers_from_split(path: Path) -> tuple[set[str], int, float]:
        if not path.is_file():
            return set(), 0, 0.0
        spks = set()
        count = 0
        dur = 0.0
        with open(path, "r", encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    obj = json.loads(l)
                    spks.add(obj.get("speaker_id", ""))
                    count += 1
                    dur += float(obj.get("duration", 0.0))
        return spks, count, dur

    train_spks, train_cnt, train_dur = get_speakers_from_split(train_path)
    val_spks, val_cnt, val_dur = get_speakers_from_split(val_path)
    test_spks, test_cnt, test_dur = get_speakers_from_split(test_path)

    leak_train_val = train_spks & val_spks
    leak_train_test = train_spks & test_spks
    leak_val_test = val_spks & test_spks
    total_leakage = len(leak_train_val) + len(leak_train_test) + len(leak_val_test)

    results = {
        "dataset_dir": str(dataset_dir),
        "total_samples": total_samples,
        "total_speakers": len(speakers_all),
        "total_hours": round(total_audio_seconds / 3600.0, 3),
        "missing_wav": missing_wav,
        "missing_codec": missing_codec,
        "invalid_codec": invalid_codec,
        "invalid_audio": invalid_audio,
        "empty_text": empty_text,
        "codebook_dimensions": list(codebook_counts),
        "token_min": token_min_val if token_min_val != float("inf") else None,
        "token_max": token_max_val if token_max_val != float("-inf") else None,
        "speaker_leakage": total_leakage,
        "splits": {
            "train": {"samples": train_cnt, "speakers": len(train_spks), "hours": round(train_dur / 3600.0, 3)},
            "val": {"samples": val_cnt, "speakers": len(val_spks), "hours": round(val_dur / 3600.0, 3)},
            "test": {"samples": test_cnt, "speakers": len(test_spks), "hours": round(test_dur / 3600.0, 3)},
        },
    }

    print("\n=== Verification Summary ===")
    print(f"Total samples:     {results['total_samples']}")
    print(f"Total speakers:    {results['total_speakers']}")
    print(f"Total hours:       {results['total_hours']} h")
    print(f"Missing WAV:       {missing_wav}")
    print(f"Missing Codec:     {missing_codec}")
    print(f"Invalid Codec:     {invalid_codec}")
    print(f"Invalid Audio:     {invalid_audio}")
    print(f"Empty Text:        {empty_text}")
    print(f"Codebook Dims:     {results['codebook_dimensions']}")
    print(f"Token Range:       [{results['token_min']}, {results['token_max']}]")
    print(f"Speaker Leakage:   {total_leakage} (train∩val: {len(leak_train_val)}, train∩test: {len(leak_train_test)}, val∩test: {len(leak_val_test)})")
    print(f"Splits:")
    for s_name, s_info in results["splits"].items():
        print(f"  - {s_name:5s}: {s_info['samples']:5d} samples | {s_info['speakers']:3d} speakers | {s_info['hours']:.3f} h")

    if strict:
        errors = []
        if missing_wav > 0:
            errors.append(f"Missing WAV files: {missing_wav}")
        if missing_codec > 0:
            errors.append(f"Missing Codec files: {missing_codec}")
        if invalid_codec > 0:
            errors.append(f"Invalid Codec tensors: {invalid_codec}")
        if invalid_audio > 0:
            errors.append(f"Invalid Audio files: {invalid_audio}")
        if empty_text > 0:
            errors.append(f"Empty text samples: {empty_text}")
        if total_leakage > 0:
            errors.append(f"Speaker leakage across splits: {total_leakage}")
        if results["codebook_dimensions"] != [32]:
            errors.append(f"Expected codebooks [32], got {results['codebook_dimensions']}")
        if results["token_min"] is None or results["token_min"] < 0 or results["token_max"] >= 2048:
            errors.append(f"Token values out of [0, 2047] range: [{results['token_min']}, {results['token_max']}]")

        if errors:
            print("\n❌ STRICT VERIFICATION FAILED:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)
        else:
            print("\n✅ STRICT VERIFICATION PASSED: All integrity checks succeeded with 0 defects.")

    return results


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.is_dir():
        print(f"Error: dataset directory not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    verify_dataset(dataset_path, strict=args.strict)


if __name__ == "__main__":
    main()
