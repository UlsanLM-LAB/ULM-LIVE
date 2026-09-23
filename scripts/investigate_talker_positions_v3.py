#!/usr/bin/env python3
# ruff: noqa: E402
"""Paired old-checkpoint probes of semantic padding and batch dependence."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.diagnose_talker_ar_v3 import (
    OUTPUT_ROOT, distribution_metrics, load_model_and_data, rollout, selected_indices,
    teacher_prediction, to_device,
)


def shifted(sample, shift):
    result = dict(sample)
    result["semantic_hidden_states"] = F.pad(sample["semantic_hidden_states"], (0, 0, 0, shift))
    result["semantic_attention_mask"] = F.pad(sample["semantic_attention_mask"], (0, shift))
    return result


def prediction(model, collator, samples, device):
    batch = to_device(collator(samples), device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        out = model(batch["semantic_hidden_states"], batch["audio_codes"],
                    batch["speaker_ids"], batch["dialect_ids"], targets=batch["targets"],
                    semantic_attention_mask=batch["semantic_attention_mask"],
                    audio_attention_mask=batch["audio_attention_mask"])
    length = collator([samples[0]])["targets"].shape[-1]
    return out.logits[0:1, :, :length].float().cpu(), batch["targets"][0:1, :, :length].cpu()


def main():
    from argparse import Namespace
    args = Namespace(device="cuda", checkpoint_dir=Path("outputs/talker-full-retrain-v2"),
                     data_dir=Path("data/ulsan-full"), output_dir=OUTPUT_ROOT)
    device = torch.device(args.device)
    model, dataset, collator = load_model_and_data(args, 10)
    candidates = [dataset[i] for i in selected_indices(dataset, 100)]
    candidates.sort(key=lambda s: s["semantic_hidden_states"].shape[0])
    a = next(s for s in candidates if s["semantic_hidden_states"].shape[0] >= 8)
    b = min((s for s in candidates if s["id"] != a["id"]),
            key=lambda s: abs(s["semantic_hidden_states"].shape[0] - a["semantic_hidden_states"].shape[0] - 16))
    c = candidates[-1]
    base, target = prediction(model, collator, [a], device)
    cases = []
    for name, mate in (("alone", None), ("medium_mate", b), ("long_mate", c)):
        logits, actual = prediction(model, collator, [a] if mate is None else [a, mate], device)
        assert torch.equal(target, actual)
        diff = (logits - base).abs()
        pred = logits.argmax(-1)
        cases.append({"case": name, "mate_id": mate["id"] if mate else None,
                      "mate_semantic_length": int(mate["semantic_hidden_states"].shape[0]) if mate else None,
                      "sample_semantic_length": int(a["semantic_hidden_states"].shape[0]),
                      "max_abs_logit_difference": float(diff.max()),
                      "mean_abs_logit_difference": float(diff.mean()),
                      "argmax_agreement": float((pred == base.argmax(-1)).float().mean()),
                      "q0_agreement": float((pred[:, 0] == base.argmax(-1)[:, 0]).float().mean()),
                      **distribution_metrics(logits, target)})
    samples = [a, candidates[len(candidates)//3], candidates[2*len(candidates)//3], c]
    rows = []
    wav_codes = []
    for sample in samples:
        true = sample["audio_codes"]
        for shift in (0, 4, 8, 16, 24, 32):
            variant = shifted(sample, shift)
            output, batch = teacher_prediction(model, collator, variant, device)
            metrics = distribution_metrics(output.logits.float(), batch["targets"])
            generated, _ = rollout(model, variant, 0, true.shape[-1], device)
            same = generated == true
            counts = torch.bincount(generated.flatten(), minlength=model.config.codebook_size).float()
            probabilities = counts[counts > 0] / counts.sum()
            rows.append({"sample_id": sample["id"], "semantic_length": int(sample["semantic_hidden_states"].shape[0]),
                         "audio_frames": int(true.shape[-1]), "shift": shift, **metrics,
                         "ar_mean_accuracy": float(same.float().mean()),
                         "ar_q0_accuracy": float(same[0].float().mean()),
                         "ar_residual_accuracy": float(same[1:].float().mean()),
                         "ar_unique_codes": int((counts > 0).sum()),
                         "ar_code_entropy": float(-(probabilities * probabilities.log()).sum()),
                         "ar_sha256": hashlib.sha256(generated.numpy().tobytes()).hexdigest()})
            if sample is a and shift in (0, 16, 32):
                wav_codes.append((shift, generated))
            print(f"position sample={sample['id']} shift={shift}", flush=True)
    dest = OUTPUT_ROOT / "position_alignment"
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / "fixed_shifts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {str(shift): {key: float(np.mean([r[key] for r in rows if r["shift"] == shift]))
                            for key in ("codec_ce", "mean_accuracy", "q0_accuracy", "residual_accuracy",
                                        "ar_mean_accuracy", "ar_q0_accuracy", "ar_residual_accuracy",
                                        "ar_unique_codes", "ar_code_entropy")}
               for shift in (0, 4, 8, 16, 24, 32)}
    (dest / "results.json").write_text(json.dumps({"batch_invariance": cases, "shift_summary": summary}, indent=2) + "\n")
    print(json.dumps({"batch_invariance": cases, "shift_summary": summary}), flush=True)
    from ulm_live.codec import EncodedAudio, build_codec
    from ulm_live.utils.audio import save_wav
    codec = build_codec(backend="mimi", device="cuda", num_quantizers=16)
    for shift, codes in wav_codes:
        waveform, sr = codec.decode(EncodedAudio(codes=codes.to(device), sample_rate=24000,
                                                  frame_rate=12.5, metadata={"backend": "mimi", "num_quantizers": 16}))
        save_wav(dest / f"{a['id']}-shift-{shift:02d}.wav", waveform.squeeze(0).cpu(), sr)


if __name__ == "__main__":
    main()
