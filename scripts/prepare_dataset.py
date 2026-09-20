import argparse
import json
from pathlib import Path
import sys

# Ensure repo root is on sys.path for direct script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.codec import build_codec
from ulm_live.data.aihub import (
    AIHubParser,
    load_dataset_config,
    scan_aihub_directory,
)
from ulm_live.data.filters import QualityFilter
from ulm_live.data.manifest import (
    compute_dataset_summary,
    split_dataset,
    write_jsonl,
)
from ulm_live.data.schema import DatasetItem
from ulm_live.data.segment import AudioSegmenter, match_audio_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated pipeline to prepare Ulsan dialect speech dataset from AI Hub raw data."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to AI Hub raw dataset directory containing JSON and audio files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory to save processed dataset and manifests.",
    )
    parser.add_argument(
        "--region",
        "-r",
        type=str,
        default="ulsan",
        help="Target dialect region to filter (default: 'ulsan').",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML file.",
    )
    parser.add_argument(
        "--encode-codec",
        action="store_true",
        help="Also encode processed audio segments into discrete neural codec tokens (.pt).",
    )
    parser.add_argument(
        "--codec-config",
        type=str,
        default="configs/codec.yaml",
        help="Path to codec YAML configuration file for --encode-codec.",
    )
    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["session", "speaker", "utterance"],
        default=None,
        help="Derived split strategy ('session' recommended; also 'utterance' or legacy 'speaker').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum samples to process (default: 0 for all). Useful for testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device for codec encoding ('auto', 'cuda', 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.is_dir():
        print(
            f"Error: Input dataset directory does not exist: {input_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load configuration
    config = load_dataset_config(args.config if Path(args.config).is_file() else None)
    filter_cfg = config.get("filters", {})
    split_cfg = config.get("split", {})

    target_sample_rate = int(filter_cfg.get("target_sample_rate", 24000))
    quality_filter = QualityFilter.from_config(config)

    # 2. Build Codec if requested
    codec_instance = None
    if args.encode_codec:
        print("Initializing Neural Audio Codec for token encoding...")
        try:
            codec_instance = build_codec(
                config_path=args.codec_config
                if Path(args.codec_config).is_file()
                else None,
                device=args.device or "auto",
            )
            print(
                f"Codec initialized: backend={codec_instance.backend_name}, device={codec_instance.device}"
            )
        except Exception as err:
            print(f"Error initializing codec backend: {err}", file=sys.stderr)
            sys.exit(1)

    # 3. Scan directory
    print(f"Scanning AI Hub dataset directory: {input_path}")
    json_files, audio_files = scan_aihub_directory(input_path)
    print(f"Found {len(json_files)} JSON files and {len(audio_files)} audio files.")

    # Index audio files for fast lookup
    audio_by_stem: dict[str, Path] = {p.stem: p for p in audio_files}
    audio_by_name: dict[str, Path] = {p.name: p for p in audio_files}

    parser = AIHubParser(config)
    segmenter = AudioSegmenter(
        output_dir=output_path,
        target_sample_rate=target_sample_rate,
        quality_filter=quality_filter,
        codec=codec_instance,
    )

    processed_items: list[DatasetItem] = []
    rejections: dict[str, int] = {}
    target_region = args.region.strip()

    item_idx = 0
    per_session_max = (
        max(20, args.max_samples // 4) if args.max_samples > 0 else 999999999
    )

    for jf in json_files:
        try:
            speakers, utterances, audio_hint = parser.parse_file(jf)
        except Exception as err:
            rejections["json_parse_error"] = rejections.get("json_parse_error", 0) + 1
            continue

        matched_audio = match_audio_file(audio_hint, jf, audio_by_stem, audio_by_name)
        if not matched_audio:
            rejections["audio_file_not_found"] = rejections.get(
                "audio_file_not_found", 0
            ) + len(utterances)
            continue

        session_processed_count = 0
        # Filter utterances by target speaker region
        for utt in utterances:
            speaker_info = speakers.get(utt.speaker_id)
            if speaker_info is not None:
                if not parser.region_classifier.is_target_region(
                    speaker_info, target_region
                ):
                    rejections["non_target_region"] = (
                        rejections.get("non_target_region", 0) + 1
                    )
                    continue
            else:
                # If speaker info wasn't in file, check if file or utterance mentions target region
                continue

            # Ensure speaker_id is unique across sessions by prefixing with session ID if it is a local index
            local_spk = utt.speaker_id
            global_spk = (
                f"{jf.stem}_{local_spk}"
                if not local_spk.startswith(jf.stem)
                else local_spk
            )
            utt.speaker_id = global_spk

            item_id = f"{target_region}_{item_idx:06d}"
            item, reason = segmenter.process_utterance(
                item_id=item_id,
                utterance=utt,
                source_audio_path=matched_audio,
                dialect=target_region,
            )

            if item is not None:
                processed_items.append(item)
                item_idx += 1
                session_processed_count += 1
                if args.max_samples > 0 and len(processed_items) >= args.max_samples:
                    break
                if session_processed_count >= per_session_max:
                    break
            else:
                key = reason or "unknown_rejection"
                rejections[key] = rejections.get(key, 0) + 1

        if args.max_samples > 0 and len(processed_items) >= args.max_samples:
            break

    if not processed_items:
        print("\nNo samples met the criteria. Rejection statistics:")
        for r, cnt in sorted(rejections.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {r}: {cnt}")
        print("\nAborting manifest generation since 0 valid samples were created.")
        sys.exit(0)

    # 4. Manifest generation and splits
    split_strategy = args.split_strategy or split_cfg.get("strategy", "session")
    train_ratio = float(split_cfg.get("train_ratio", 0.8))
    val_ratio = float(split_cfg.get("val_ratio", 0.1))
    test_ratio = float(split_cfg.get("test_ratio", 0.1))
    seed = int(split_cfg.get("seed", 42))

    train_items, val_items, test_items = split_dataset(
        processed_items,
        strategy=split_strategy,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    # Write manifests
    write_jsonl(processed_items, output_path / "manifest.jsonl")
    write_jsonl(train_items, output_path / "train.jsonl")
    write_jsonl(val_items, output_path / "val.jsonl")
    write_jsonl(test_items, output_path / "test.jsonl")

    # Summary
    summary = compute_dataset_summary(processed_items)
    with open(output_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)

    print("\n=== Dataset Preparation Complete ===")
    print(f"Target Region:     {target_region}")
    print(f"Total processed:   {summary.num_samples} samples")
    print(f"Total speakers:    {summary.num_speakers}")
    print(f"Total audio hours: {summary.total_hours:.3f} h")
    print(f"Mean duration:     {summary.mean_duration:.2f} s")
    print(
        f"Split strategy:    {split_strategy} (train={len(train_items)}, val={len(val_items)}, test={len(test_items)})"
    )
    print()

    print("Speaker breakdown:")
    for spk, s_stat in sorted(
        summary.speaker_stats.items(), key=lambda x: x[1]["samples"], reverse=True
    )[:10]:
        print(
            f"  {spk:15s}  {s_stat['samples']:4d} samples  {s_stat['total_hours']:.4f} h"
        )
    if len(summary.speaker_stats) > 10:
        print(f"  ... and {len(summary.speaker_stats) - 10} more speakers.")
    print()

    print(f"Outputs written to: {output_path}")
    print("  - manifest.jsonl")
    print("  - train.jsonl")
    print("  - val.jsonl")
    print("  - test.jsonl")
    print("  - summary.json")
    print("  - audio/*.wav")
    if args.encode_codec:
        print("  - codec/*.pt")


if __name__ == "__main__":
    main()
