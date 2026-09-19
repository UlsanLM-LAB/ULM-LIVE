import json
from pathlib import Path
import random
from typing import Any

from ulm_live.data.schema import DatasetItem, DatasetSummary


def write_jsonl(items: list[DatasetItem], output_file: Path) -> None:
    """Write dataset items to a JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")


def compute_dataset_summary(items: list[DatasetItem]) -> DatasetSummary:
    """Calculate overall dataset summary and per-speaker breakdown."""
    num_samples = len(items)
    if num_samples == 0:
        return DatasetSummary(
            num_samples=0,
            num_speakers=0,
            total_hours=0.0,
            mean_duration=0.0,
            speaker_stats={},
        )

    speakers: set[str] = set()
    total_duration = 0.0
    speaker_durations: dict[str, float] = {}
    speaker_counts: dict[str, int] = {}

    for item in items:
        spk = item.speaker_id
        speakers.add(spk)
        dur = item.duration
        total_duration += dur
        speaker_durations[spk] = speaker_durations.get(spk, 0.0) + dur
        speaker_counts[spk] = speaker_counts.get(spk, 0) + 1

    total_hours = total_duration / 3600.0
    mean_duration = total_duration / float(num_samples)

    speaker_stats: dict[str, dict[str, Any]] = {}
    for spk in sorted(list(speakers)):
        spk_dur = speaker_durations[spk]
        speaker_stats[spk] = {
            "samples": speaker_counts[spk],
            "total_seconds": round(spk_dur, 2),
            "total_hours": round(spk_dur / 3600.0, 4),
        }

    return DatasetSummary(
        num_samples=num_samples,
        num_speakers=len(speakers),
        total_hours=round(total_hours, 3),
        mean_duration=round(mean_duration, 3),
        speaker_stats=speaker_stats,
    )


def split_dataset(
    items: list[DatasetItem],
    strategy: str = "speaker",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[DatasetItem], list[DatasetItem], list[DatasetItem]]:
    """Split dataset into train, validation, and test subsets.

    Args:
        items: List of DatasetItem.
        strategy: 'speaker' (speaker-disjoint split, default) or 'utterance' (random split).
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        seed: Random seed for reproducibility.

    Returns:
        (train_items, val_items, test_items)
    """
    if not items:
        return [], [], []

    rng = random.Random(seed)
    total_ratio = train_ratio + val_ratio + test_ratio
    train_norm = train_ratio / total_ratio
    val_norm = val_ratio / total_ratio

    if strategy == "speaker":
        # Group items by speaker_id
        speaker_to_items: dict[str, list[DatasetItem]] = {}
        for item in items:
            speaker_to_items.setdefault(item.speaker_id, []).append(item)

        unique_speakers = sorted(list(speaker_to_items.keys()))
        rng.shuffle(unique_speakers)

        n_spk = len(unique_speakers)
        n_train = max(1, int(round(n_spk * train_norm))) if n_spk >= 3 else n_spk
        n_val = 1 if n_spk >= 3 else 0
        if n_train + n_val >= n_spk and n_spk >= 3:
            n_train = n_spk - 2
            n_val = 1

        train_speakers = set(unique_speakers[:n_train])
        val_speakers = set(unique_speakers[n_train : n_train + n_val])
        test_speakers = set(unique_speakers[n_train + n_val :])

        train_items = [it for it in items if it.speaker_id in train_speakers]
        val_items = [it for it in items if it.speaker_id in val_speakers]
        test_items = [it for it in items if it.speaker_id in test_speakers]

        # If test or val ended up empty due to very small speaker count (< 3),
        # fallback to utterance split for remainder
        if n_spk < 3:
            if not val_items and len(items) >= 2:
                val_items.append(train_items.pop())
            if not test_items and len(items) >= 3:
                test_items.append(train_items.pop())

        return train_items, val_items, test_items

    elif strategy == "utterance":
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(round(n_total * train_norm))
        n_val = int(round(n_total * val_norm))

        train_items = shuffled[:n_train]
        val_items = shuffled[n_train : n_train + n_val]
        test_items = shuffled[n_train + n_val :]
        return train_items, val_items, test_items

    else:
        raise ValueError(f"Unknown split strategy: '{strategy}'. Choose 'speaker' or 'utterance'.")
