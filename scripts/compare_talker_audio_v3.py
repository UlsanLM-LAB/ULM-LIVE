#!/usr/bin/env python3
# ruff: noqa: E402
"""Decode matched 2048-sample baseline and scheduled Talker outputs for listening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.pilot_talker_ar_v3 import forward, measure_ar
from ulm_live.codec import EncodedAudio, build_codec
from ulm_live.talker import TalkerCollator, TalkerConfig, TalkerDataset, ULMTalker, build_id_mappings
from ulm_live.utils.audio import save_wav


def decode_save(codec, codes, path):
    audio, sr = codec.decode(EncodedAudio(
        codes=codes, sample_rate=24000, frame_rate=12.5,
        metadata={"backend": "mimi", "num_quantizers": 16},
    ))
    save_wav(path, audio.squeeze(0).cpu(), sr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/ulsan-full"))
    parser.add_argument("--pilots-dir", type=Path, default=Path("outputs/talker-ar-diagnostics-v3/pilots-postfix"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/talker-ar-diagnostics-v3/audio_compare"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--models", nargs="+", choices=("baseline", "scheduled"), default=("baseline", "scheduled"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.data_dir / "train.cached.jsonl"
    speaker2id, dialect2id = build_id_mappings([manifest, args.data_dir / "val.cached.jsonl", args.data_dir / "test.cached.jsonl"])
    dataset = TalkerDataset(manifest, speaker2id=speaker2id, dialect2id=dialect2id,
                            num_quantizers=16, cache_only=True)
    index_by_id = {row["id"]: i for i, row in enumerate(dataset.items)}
    baseline_dir = args.pilots_dir / "2048-multi-baseline"
    scheduled_dir = args.pilots_dir / "2048-multi-scheduled"
    baseline_ids = [r["id"] for r in json.loads((baseline_dir / "selection.json").read_text())]
    scheduled_ids = [r["id"] for r in json.loads((scheduled_dir / "selection.json").read_text())]
    if baseline_ids != scheduled_ids:
        raise ValueError("baseline and scheduled sample selections differ")
    chosen_ids = [baseline_ids[round(i * (len(baseline_ids) - 1) / 19)] for i in range(20)]
    samples = [dataset[index_by_id[ident]] for ident in chosen_ids]
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=16)
    for ordinal, (ident, sample) in enumerate(zip(chosen_ids, samples, strict=True), 1):
        dest = args.output_dir / f"sample_{ordinal:03d}"
        dest.mkdir(parents=True, exist_ok=True)
        row = dataset.items[index_by_id[ident]]
        source = dataset._path(row["audio_path"])
        if not (dest / "original.wav").exists():
            shutil.copy2(source, dest / "original.wav")
        if not (dest / "mimi_gt.wav").exists():
            decode_save(codec, sample["audio_codes"], dest / "mimi_gt.wav")
        (dest / "text.txt").write_text(f"{ident}\n{sample['text']}\n{row['speaker_id']}\n{row['dialect']}\n", encoding="utf-8")
    device = torch.device(args.device)
    for label, pilot_dir in (("baseline", baseline_dir), ("scheduled", scheduled_dir)):
        if label not in args.models:
            continue
        payload = torch.load(pilot_dir / "final.pt", map_location="cpu", weights_only=False)
        if payload["selection_ids"] != baseline_ids:
            raise ValueError(f"{label} checkpoint selection differs")
        model = ULMTalker(TalkerConfig(**payload["config"])).to(device).eval()
        model.load_state_dict(payload["model_state_dict"])
        collator = TalkerCollator(pad_token_id=model.config.pad_token_id,
                                 bos_token_id=model.config.bos_token_id,
                                 max_audio_len=model.config.max_seq_len,
                                 acoustic_delay_frames=model.config.acoustic_delay_frames)
        for ordinal, sample in enumerate(samples, 1):
            dest = args.output_dir / f"sample_{ordinal:03d}"
            teacher_path, ar_path = dest / f"{label}_teacher_forced.wav", dest / f"{label}_ar.wav"
            if teacher_path.exists() and ar_path.exists():
                continue
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in collator([sample]).items()}
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                teacher = forward(model, batch).logits.argmax(-1)[0].cpu()
            _, generated = measure_ar(model, [sample], device, normal=True)
            if not teacher_path.exists():
                decode_save(codec, teacher, teacher_path)
            if not ar_path.exists():
                decode_save(codec, generated[0]["codes"], ar_path)
            (dest / f"{label}_metrics.json").write_text(json.dumps({
                key: value for key, value in generated[0].items() if key != "codes"
            }, indent=2) + "\n")
            print(f"audio {label} {ordinal}/20 {sample['id']}", flush=True)
        del model
        torch.cuda.empty_cache()
    (args.output_dir / "selection.json").write_text(json.dumps(chosen_ids, indent=2) + "\n")


if __name__ == "__main__":
    main()
