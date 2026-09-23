#!/usr/bin/env python3
"""Summarize canonical Talker audio levels, silence, and speaker distribution."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import wave

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def spread_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        raise ValueError("length and count must be positive")
    count = min(length, count)
    return [round(i * (length - 1) / (count - 1)) for i in range(count)] if count > 1 else [length // 2]


def audio_stats(path: Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != 24000:
            raise ValueError(f"unexpected canonical audio format: {path}")
        frames = wav.getnframes()
        samples = np.frombuffer(wav.readframes(frames), dtype="<i2").astype(np.float32) / 32768
    if samples.size != frames or not np.isfinite(samples).all():
        raise ValueError(f"invalid PCM data: {path}")
    energy = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
    return {"duration": frames / 24000, "rms": float(energy),
            "silence_fraction": float(np.mean(np.abs(samples) < .005)),
            "clip_fraction": float(np.mean(np.abs(samples) >= .99))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/ulsan-full"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/talker-ar-diagnostics-v3/data_quality"))
    p.add_argument("--train-samples", type=int, default=2000)
    args = p.parse_args()
    records = []
    split_counts = {}
    speaker_counts = Counter()
    for split in ("train", "val", "test"):
        items = [json.loads(line) for line in (args.data_dir / f"{split}.cached.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        split_counts[split] = len(items)
        speaker_counts.update(str(item["speaker_id"]) for item in items)
        indices = spread_indices(len(items), args.train_samples) if split == "train" else list(range(len(items)))
        for n, idx in enumerate(indices, 1):
            item = items[idx]
            path = Path(item["audio_path"])
            if not path.is_absolute():
                path = args.data_dir / path
            records.append({"split": split, "id": item["id"], "speaker": item["speaker_id"],
                            "text_chars": len(str(item["text"])), **audio_stats(path)})
            if n % 500 == 0:
                print(f"{split}: {n}/{len(indices)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audio_stats.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {"canonical_splits": split_counts, "speaker_count": len(speaker_counts),
               "sampled_audio_count": len(records),
               "speaker_size_quantiles": np.quantile(list(speaker_counts.values()), [0, .25, .5, .75, 1]).tolist(),
               "duration_quantiles": np.quantile([r["duration"] for r in records], [0, .25, .5, .75, .95, .99, 1]).tolist(),
               "rms_quantiles": np.quantile([r["rms"] for r in records], [0, .25, .5, .75, .95, .99, 1]).tolist(),
               "silence_fraction_quantiles": np.quantile([r["silence_fraction"] for r in records], [0, .25, .5, .75, .95, .99, 1]).tolist(),
               "near_silent_count": sum(r["rms"] < .005 for r in records),
               "high_silence_count": sum(r["silence_fraction"] > .8 for r in records),
               "clipped_count": sum(r["clip_fraction"] > .01 for r in records)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
