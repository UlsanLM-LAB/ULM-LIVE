#!/usr/bin/env python3
"""Prepares a balanced 1.5 - 2.5 hour smoke subset from Tier 1 Ulsan speech data.

Criteria:
1. Pure Tier 1 Ulsan native speakers only (exclude Tier 2/3).
2. Minimum 20 speakers with balanced gender and age distribution.
3. Capped duration per speaker (max ~5 min) to prevent single-speaker dominance.
4. Utterance duration bounded to 1.0s - 8.0s (natural conversation).
5. Validation subset: ~15 min.
6. Test subset: 25 unseen samples.
7. Zero speaker/session leakage guaranteed.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/ulm-live-tts-v2")
    parser.add_argument("--output-dir", type=str, default="data/ulm-live-tts-v2/smoke")
    parser.add_argument("--target-train-hours", type=float, default=2.0)
    parser.add_argument("--target-val-minutes", type=float, default=15.0)
    parser.add_argument("--num-test-samples", type=int, default=25)
    parser.add_argument("--max-speaker-minutes", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== [ULM-LIVE] Preparing Tier 1 Ulsan Smoke Adaptation Subset ===")

    # 1. Load and filter train candidates
    train_candidates = []
    with open(data_dir / "train.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("is_ulsan_tier1") and 1.0 <= item["duration"] <= 8.0:
                train_candidates.append(item)

    print(f"Total Tier 1 1-8s train candidates: {len(train_candidates)}")

    spk_to_utts = defaultdict(list)
    for it in train_candidates:
        spk_to_utts[it["speaker_id"]].append(it)

    # Group speakers by demographic bin: (gender, age)
    demog_bins = defaultdict(list)
    for sid, utts in spk_to_utts.items():
        demog_bins[(utts[0]["gender"], utts[0]["age"])].append(sid)

    print("\nDemographic distribution of available Tier 1 speakers:")
    for bin_key, sids in sorted(demog_bins.items()):
        print(f"  {bin_key}: {len(sids)} speakers")

    # Select >= 20 speakers across bins
    selected_speakers = []
    for bin_key, sids in sorted(demog_bins.items()):
        shuffled = list(sids)
        random.shuffle(shuffled)
        # Take up to 3 for male/older or 4 for common female bins
        pick_count = min(len(shuffled), 3 if "남성" in bin_key[0] or bin_key[1] in ("40대", "50대", "60대 이상") else 4)
        selected_speakers.extend(shuffled[:pick_count])

    print(f"\nSelected {len(selected_speakers)} balanced Tier 1 speakers for smoke training.")

    # Sample utterances per speaker up to max_speaker_minutes
    smoke_train_items = []
    speaker_stats = []
    max_sec = args.max_speaker_minutes * 60.0

    for sid in selected_speakers:
        utts = list(spk_to_utts[sid])
        random.shuffle(utts)
        cum_dur = 0.0
        chosen = []
        for u in utts:
            if cum_dur + u["duration"] <= max_sec:
                chosen.append(u)
                cum_dur += u["duration"]
        smoke_train_items.extend(chosen)
        meta = chosen[0]
        speaker_stats.append({
            "speaker_id": sid,
            "session_id": meta["session_id"],
            "gender": meta["gender"],
            "age": meta["age"],
            "tier": meta["tier"],
            "utterances": len(chosen),
            "duration_seconds": round(cum_dur, 2),
            "duration_minutes": round(cum_dur / 60.0, 2),
        })

    smoke_train_items.sort(key=lambda x: x["utterance_id"])
    total_train_sec = sum(x["duration"] for x in smoke_train_items)
    print(f"Smoke train set: {len(smoke_train_items)} utterances, {total_train_sec/3600.0:.2f} hours ({total_train_sec:.1f}s)")

    # 2. Validation subset (~15 min from validation.jsonl Tier 1)
    val_candidates = []
    with open(data_dir / "validation.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("is_ulsan_tier1") and 1.0 <= item["duration"] <= 8.0:
                val_candidates.append(item)

    val_spks = defaultdict(list)
    for it in val_candidates:
        val_spks[it["speaker_id"]].append(it)

    target_val_sec = args.target_val_minutes * 60.0
    val_items = []
    val_dur = 0.0
    # Sample proportionally across validation speakers
    val_spk_list = list(val_spks.keys())
    random.shuffle(val_spk_list)
    idx = 0
    while val_dur < target_val_sec and val_spk_list:
        sid = val_spk_list[idx % len(val_spk_list)]
        if val_spks[sid]:
            chosen_u = val_spks[sid].pop(0)
            val_items.append(chosen_u)
            val_dur += chosen_u["duration"]
        else:
            val_spk_list.remove(sid)
            if not val_spk_list:
                break
        idx += 1

    val_items.sort(key=lambda x: x["utterance_id"])
    print(f"Smoke validation set: {len(val_items)} utterances, {val_dur/60.0:.1f} minutes ({val_dur:.1f}s)")

    # 3. Test subset (25 unseen samples from test.jsonl Tier 1)
    test_candidates = []
    with open(data_dir / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("is_ulsan_tier1") and 1.5 <= item["duration"] <= 7.0:
                test_candidates.append(item)

    random.shuffle(test_candidates)
    test_items = test_candidates[: args.num_test_samples]
    test_items.sort(key=lambda x: x["utterance_id"])
    test_dur = sum(x["duration"] for x in test_items)
    print(f"Smoke test set: {len(test_items)} utterances, {test_dur/60.0:.1f} minutes ({test_dur:.1f}s)")

    # 4. Zero leakage verification
    tr_spks = {x["speaker_id"] for x in smoke_train_items}
    v_spks = {x["speaker_id"] for x in val_items}
    te_spks = {x["speaker_id"] for x in test_items}

    assert len(tr_spks & v_spks) == 0, "Speaker leak between train and val!"
    assert len(tr_spks & te_spks) == 0, "Speaker leak between train and test!"
    assert len(v_spks & te_spks) == 0, "Speaker leak between val and test!"

    tr_sess = {x["session_id"] for x in smoke_train_items}
    v_sess = {x["session_id"] for x in val_items}
    te_sess = {x["session_id"] for x in test_items}

    assert len(tr_sess & v_sess) == 0, "Session leak between train and val!"
    assert len(tr_sess & te_sess) == 0, "Session leak between train and test!"
    assert len(v_sess & te_sess) == 0, "Session leak between val and test!"

    print("\n>>> Zero speaker and session leakage verified! <<<")

    # 5. Save jsonl and stats
    def write_jsonl(items: list[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(smoke_train_items, output_dir / "smoke_train.jsonl")
    write_jsonl(val_items, output_dir / "smoke_validation.jsonl")
    write_jsonl(test_items, output_dir / "smoke_test.jsonl")

    with open(output_dir / "smoke_speakers.json", "w", encoding="utf-8") as f:
        json.dump(speaker_stats, f, ensure_ascii=False, indent=2)

    stats = {
        "dataset_name": "ULM-LIVE-TTS-V2-SMOKE",
        "description": "Tier 1 Ulsan dialect smoke adaptation subset (1.5-2.5h)",
        "train": {
            "speakers": len(selected_speakers),
            "utterances": len(smoke_train_items),
            "duration_seconds": round(total_train_sec, 2),
            "duration_hours": round(total_train_sec / 3600.0, 2),
            "gender_distribution": dict(Counter(x["gender"] for x in smoke_train_items)),
            "age_distribution": dict(Counter(x["age"] for x in smoke_train_items)),
        },
        "validation": {
            "speakers": len(v_spks),
            "utterances": len(val_items),
            "duration_seconds": round(val_dur, 2),
            "duration_minutes": round(val_dur / 60.0, 2),
        },
        "test": {
            "speakers": len(te_spks),
            "utterances": len(test_items),
            "duration_seconds": round(test_dur, 2),
            "duration_minutes": round(test_dur / 60.0, 2),
        },
    }

    with open(output_dir / "smoke_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nSaved smoke subset manifests and stats to {output_dir}/")


if __name__ == "__main__":
    main()
