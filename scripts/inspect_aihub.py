import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path for direct script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.data.aihub import inspect_aihub_dataset, load_dataset_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Non-destructively inspect AI Hub dialect dataset structure and metadata fields."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to AI Hub raw dataset root directory.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset configuration YAML file.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=500,
        help="Maximum JSON files to scan for metadata field inspection (default: 500, 0 for all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input dataset directory does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    config = load_dataset_config(args.config if Path(args.config).is_file() else None)

    try:
        results = inspect_aihub_dataset(
            input_path,
            config=config,
            max_scan_files=args.max_files if args.max_files > 0 else 0,
        )
    except Exception as err:
        print(f"Error inspecting dataset: {err}", file=sys.stderr)
        sys.exit(1)

    print("=== AI Hub Dataset Inspection ===")
    print("Files discovered:")
    print(f"  JSON files:  {results['num_json_files']}")
    print(f"  Audio files: {results['num_audio_files']}")
    print()

    print("Detected metadata fields:")
    speaker_fields = ", ".join(results["detected_speaker_fields"]) or "None detected"
    region_fields = ", ".join(results["detected_region_fields"]) or "None detected"
    transcript_fields = ", ".join(results["detected_text_fields"]) or "None detected"
    print(f"  Speaker fields:    {speaker_fields}")
    print(f"  Region fields:     {region_fields}")
    print(f"  Transcript fields: {transcript_fields}")
    print()

    print("Regions found:")
    regions = results["regions_found"]
    if regions:
        for reg, count in sorted(regions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reg}: {count} speaker records")
    else:
        print("  No region metadata matched with current aliases/fields.")
    print()

    print(f"Scanned {results['scanned_json_files']} JSON files: {results['total_speakers_scanned']} unique speakers, {results['total_utterances_scanned']} utterances.")


if __name__ == "__main__":
    main()
