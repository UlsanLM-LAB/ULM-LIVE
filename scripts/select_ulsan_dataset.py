#!/usr/bin/env python3
"""Speaker-balanced selection of 25-30 hours Ulsan dialect speech dataset."""

from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path

def select_dataset(target_min_hours: float = 26.0, target_max_hours: float = 28.5) -> None:
    speakers_file = Path("data/selection/ulsan_speakers.json")
    audio_files = Path("data/selection/ulsan_audio_files.txt")

    with open(speakers_file, "r", encoding="utf-8") as f:
        speakers_data = json.load(f)

    spk_meta = {it["speaker_id"]: it for it in speakers_data}

    # Load all val sessions
    val_sessions = []
    with open(audio_files, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 5 and parts[1] == "val":
                sess_id = parts[0]
                utterances = int(parts[2])
                duration_sec = float(parts[3])
                local_spks = parts[4].split(",")
                matched_spks = [spk_meta[k] for k in spk_meta if spk_meta[k]["session_id"] == sess_id]
                genders = list(set(sp["gender"] for sp in matched_spks if "gender" in sp))
                ages = list(set(sp["age"] for sp in matched_spks if "age" in sp))
                val_sessions.append({
                    "session_id": sess_id,
                    "utterances": utterances,
                    "duration_sec": duration_sec,
                    "hours": duration_sec / 3600.0,
                    "local_speakers": local_spks,
                    "speakers": matched_spks,
                    "genders": genders,
                    "ages": ages,
                })

    # Sort sessions deterministically by session_id
    val_sessions.sort(key=lambda s: s["session_id"])

    # Strategy:
    # 1. Prioritize all male sessions (scarce)
    # 2. Prioritize rare ages (60+, 50s, 40s, 10s)
    # 3. Add 20s and 30s to reach ~27-28 hours with speaker balance
    selected_sessions = []
    selected_sess_ids = set()
    total_hours = 0.0

    # Group 1: Male or older ages
    for s in val_sessions:
        is_male = "남성" in s["genders"]
        is_older = any(a in ("40대", "50대", "60대 이상") for a in s["ages"])
        is_teen = "10대" in s["ages"]
        if is_male or is_older:
            selected_sessions.append(s)
            selected_sess_ids.add(s["session_id"])
            total_hours += s["hours"]

    # Group 2: Teens
    for s in val_sessions:
        if s["session_id"] in selected_sess_ids:
            continue
        if "10대" in s["ages"]:
            selected_sessions.append(s)
            selected_sess_ids.add(s["session_id"])
            total_hours += s["hours"]

    # Group 3: 20s and 30s balanced until target_min_hours <= total_hours <= target_max_hours
    remaining = [s for s in val_sessions if s["session_id"] not in selected_sess_ids]
    # alternate between 30s and 20s
    thirties = [s for s in remaining if "30대" in s["ages"]]
    twenties = [s for s in remaining if "20대" in s["ages"] and "30대" not in s["ages"]]

    i3, i2 = 0, 0
    while total_hours < target_min_hours and (i3 < len(thirties) or i2 < len(twenties)):
        if i3 < len(thirties):
            s = thirties[i3]
            i3 += 1
            if s["session_id"] not in selected_sess_ids:
                selected_sessions.append(s)
                selected_sess_ids.add(s["session_id"])
                total_hours += s["hours"]
                if total_hours >= target_min_hours:
                    break
        if i2 < len(twenties):
            s = twenties[i2]
            i2 += 1
            if s["session_id"] not in selected_sess_ids:
                selected_sessions.append(s)
                selected_sess_ids.add(s["session_id"])
                total_hours += s["hours"]

    # Calculate statistics
    total_utts = sum(s["utterances"] for s in selected_sessions)
    selected_speakers_list = []
    seen_speakers = set()
    for s in selected_sessions:
        for sp in s["speakers"]:
            sp_id = sp["speaker_id"]
            if sp_id not in seen_speakers:
                seen_speakers.add(sp_id)
                selected_speakers_list.append(sp)

    gender_counts = Counter()
    age_counts = Counter()
    for sp in selected_speakers_list:
        gender_counts[sp.get("gender", "unknown")] += 1
        age_counts[sp.get("age", "unknown")] += 1

    summary = {
        "target_hours_range": [target_min_hours, target_max_hours],
        "total_selected_sessions": len(selected_sessions),
        "total_selected_speakers": len(selected_speakers_list),
        "total_selected_utterances": total_utts,
        "total_selected_hours": round(total_hours, 2),
        "gender_distribution_speakers": dict(gender_counts),
        "age_distribution_speakers": dict(age_counts),
    }

    out_dir = Path("data/selection")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "selected_sessions.json", "w", encoding="utf-8") as f:
        json.dump(selected_sessions, f, ensure_ascii=False, indent=2)

    with open(out_dir / "selected_speakers.json", "w", encoding="utf-8") as f:
        json.dump(selected_speakers_list, f, ensure_ascii=False, indent=2)

    with open(out_dir / "selection_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Selection Completed Successfully ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    select_dataset()
