#!/usr/bin/env python3
# ruff: noqa: E402
"""Measure same-speaker semantic sensitivity of a completed fresh Talker pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.diagnose_talker_ar_v3 import distribution_metrics, semantic_variant, teacher_prediction
from ulm_live.talker import TalkerCollator, TalkerConfig, TalkerDataset, ULMTalker, build_id_mappings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("baseline", "scheduled"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ulsan-full"))
    parser.add_argument("--pilots-dir", type=Path, default=Path("outputs/talker-ar-diagnostics-v3/pilots-postfix"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/talker-ar-diagnostics-v3/semantic_ablation"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    pilot_dir = args.pilots_dir / f"2048-multi-{args.variant}"
    selection = [row["id"] for row in json.loads((pilot_dir / "selection.json").read_text())]
    payload = torch.load(pilot_dir / "final.pt", map_location="cpu", weights_only=False)
    if payload["selection_ids"] != selection:
        raise ValueError("checkpoint and selection differ")
    model = ULMTalker(TalkerConfig(**payload["config"])).to(args.device).eval()
    model.load_state_dict(payload["model_state_dict"])
    train = args.data_dir / "train.cached.jsonl"
    speaker2id, dialect2id = build_id_mappings([train, args.data_dir / "val.cached.jsonl", args.data_dir / "test.cached.jsonl"])
    dataset = TalkerDataset(train, speaker2id=speaker2id, dialect2id=dialect2id,
                            num_quantizers=16, cache_only=True)
    index = {row["id"]: i for i, row in enumerate(dataset.items)}
    chosen = [selection[round(i * (len(selection) - 1) / 19)] for i in range(20)]
    by_speaker = {}
    for ident in selection:
        row = dataset.items[index[ident]]
        by_speaker.setdefault(row["speaker_id"], []).append(ident)
    collator = TalkerCollator(pad_token_id=model.config.pad_token_id,
                             bos_token_id=model.config.bos_token_id,
                             max_audio_len=model.config.max_seq_len,
                             acoustic_delay_frames=model.config.acoustic_delay_frames)
    rng = torch.Generator().manual_seed(20260923)
    rows = []
    for ident in chosen:
        sample = dataset[index[ident]]
        speaker = dataset.items[index[ident]]["speaker_id"]
        other_id = next(i for i in by_speaker[speaker] if i != ident)
        other = dataset[index[other_id]]
        baseline = None
        for mode in ("correct", "zero", "random", "shuffled"):
            variant = dict(sample)
            variant["semantic_hidden_states"] = semantic_variant(
                sample["semantic_hidden_states"], other["semantic_hidden_states"], mode, rng
            )
            output, batch = teacher_prediction(model, collator, variant, torch.device(args.device))
            logits = output.logits.float()
            metrics = distribution_metrics(logits, batch["targets"])
            if baseline is None:
                baseline = logits
                metrics["mean_abs_logit_diff"] = 0.0
                metrics["kl_from_correct"] = 0.0
            else:
                log_p = F.log_softmax(baseline, -1)
                metrics["mean_abs_logit_diff"] = float((logits - baseline).abs().mean())
                metrics["kl_from_correct"] = float(F.kl_div(
                    F.log_softmax(logits, -1), log_p.exp(), reduction="batchmean"
                ) / (logits.shape[1] * logits.shape[2]))
            rows.append({"sample_id": ident, "speaker": speaker, "other_id": other_id,
                         "mode": mode, **metrics})
        print(f"ablation {args.variant} {ident}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / f"2048-{args.variant}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {mode: {key: float(np.mean([r[key] for r in rows if r["mode"] == mode]))
                      for key in ("codec_ce", "mean_accuracy", "q0_accuracy", "residual_accuracy",
                                  "mean_abs_logit_diff", "kl_from_correct")}
               for mode in ("correct", "zero", "random", "shuffled")}
    (args.output_dir / f"2048-{args.variant}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
