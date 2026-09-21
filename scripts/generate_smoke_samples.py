#!/usr/bin/env python3
"""End-to-end inference and speech generation for 100-step Smoke Talker checkpoint.
Text -> Thinker (Phase 4 best merged) -> Smoke Talker -> Mimi 16q -> 24kHz WAV.
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import soundfile as sf
import torch
from ulm_live.codec import build_codec
from ulm_live.talker import (
    SpeechSynthesizer,
    TalkerGenerationConfig,
    load_talker_checkpoint,
)
from ulm_live.thinker import ULMThinker

PROMPTS = [
    "안녕",
    "밥 묵었나?",
    "오늘 뭐 하고 있었노?",
    "오늘 날씨 좋네.",
    "울산에서 놀러 갈 만한 데 추천해줘.",
    "나는 오늘 학교 끝나고 친구 만나러 간다.",
    "1 더하기 1은 2다.",
    "하늘이 파란 이유를 간단하게 설명해줘.",
    "Python에서 리스트를 정렬하는 방법 알려줘.",
    "오늘 기분이 좀 안 좋다.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate WAVs from trained smoke checkpoint.")
    p.add_argument("--thinker", required=True, help="Path to Thinker model.")
    p.add_argument("--checkpoint", required=True, help="Path to Talker checkpoint (.pt).")
    p.add_argument("--output-dir", default="outputs/talker-smoke-phase4/audio", help="Output directory for WAVs.")
    p.add_argument("--device", default="cuda", help="Execution device.")
    p.add_argument("--max-frames", type=int, default=750, help="Safety cap on generated frames.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    print("=== [ULM-LIVE] End-to-End Speech Synthesis Pipeline ===")
    print(f"Thinker:    {args.thinker}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {out_dir}")
    print(f"Device:     {device}")

    # 1. Load Thinker
    print("\n1. Loading Phase 4 Thinker model...")
    thinker = ULMThinker(args.thinker, device=device, torch_dtype="bf16", freeze=True)

    # 2. Load Talker
    print("2. Loading Smoke Talker checkpoint...")
    talker, checkpoint_meta = load_talker_checkpoint(args.checkpoint, device=device)
    talker.eval()

    # 3. Load Codec
    print("3. Loading Mimi Codec (device=cuda)...")
    codec = build_codec(backend="mimi", device=device)

    # 4. Build Synthesizer
    synthesizer = SpeechSynthesizer(
        thinker=thinker,
        talker=talker,
        codec=codec,
        device=device,
    )

    from collections import Counter
    gen_cfg = TalkerGenerationConfig(
        max_new_tokens=args.max_frames,
        do_sample=False,
        stop_threshold=0.5,
        return_details=True,
    )

    report_items = []
    print(f"\n4. Generating speech for {len(PROMPTS)} prompts...")

    for idx, prompt in enumerate(PROMPTS, 1):
        slug = f"sample_{idx:02d}"
        out_wav = out_dir / f"{slug}.wav"

        t_start = time.perf_counter()
        res = synthesizer.synthesize(
            prompt,
            speaker="speaker_001",
            dialect="ulsan",
            generation_config=gen_cfg,
        )
        total_time = time.perf_counter() - t_start

        # Save audio
        res.save(out_wav)

        # Integrity checks
        waveform = res.waveform.detach().cpu().squeeze()
        is_finite = bool(torch.isfinite(waveform).all().item())
        is_non_empty = bool(waveform.numel() > 0 and torch.max(torch.abs(waveform)).item() > 1e-6)
        clipping_ratio = float((torch.abs(waveform) >= 0.999).sum().item()) / waveform.numel() if waveform.numel() > 0 else 0.0

        # Codec token checks
        tokens = res.codec_tokens.detach().cpu()
        if tokens.dim() == 3:
            tokens = tokens.squeeze(0)
        t_min = int(tokens.min().item()) if tokens.numel() > 0 else -1
        t_max = int(tokens.max().item()) if tokens.numel() > 0 else -1
        valid_tokens = bool(0 <= t_min and t_max < 2048)

        # Entropy and unique tokens
        ent_list = []
        unique_list = []
        if tokens.ndim == 2:
            for q in range(tokens.shape[0]):
                q_toks = tokens[q].tolist()
                unique_list.append(len(set(q_toks)))
                cnts = Counter(q_toks)
                tot = len(q_toks)
                e = -sum((c / tot) * math.log2(c / tot) for c in cnts.values())
                ent_list.append(round(e, 3))
        mean_ent = round(sum(ent_list) / len(ent_list), 3) if ent_list else 0.0

        passed = bool(
            is_finite
            and is_non_empty
            and valid_tokens
            and clipping_ratio < 0.05
            and res.termination_reason == "STOP_PREDICTED"
            and mean_ent > 1.0
        )

        item_report = {
            "index": idx,
            "text": prompt,
            "speaker": "speaker_001",
            "frames": int(tokens.shape[1]) if tokens.ndim == 2 else 0,
            "stop_position": int(tokens.shape[1]) if tokens.ndim == 2 else 0,
            "duration": round(res.duration, 3),
            "generation_time_sec": round(total_time, 3),
            "talker_time_sec": round(res.timings.get("talker_time", 0.0), 3),
            "decode_time_sec": round(res.timings.get("decode_time", 0.0), 3),
            "rtf": round(res.rtf, 3),
            "wav_path": str(out_wav),
            "is_finite": is_finite,
            "is_non_empty": is_non_empty,
            "valid_codec_range": valid_tokens,
            "token_range": [t_min, t_max],
            "clipping_ratio": round(clipping_ratio, 4),
            "termination_reason": res.termination_reason,
            "max_stop_prob": res.max_stop_prob,
            "mean_entropy": mean_ent,
            "entropy_q0_q15": ent_list,
            "unique_tokens_q0_q15": unique_list,
            "status": "PASS" if passed else "FAIL",
        }
        report_items.append(item_report)

        print(
            f"  [{idx:02d}/10] '{prompt[:15]:<15}' -> {res.duration:5.2f}s | "
            f"Frames: {item_report['frames']:3d} | Reason: {res.termination_reason:<14} | "
            f"MaxStopP: {res.max_stop_prob:.3f} | Ent: {mean_ent:4.2f} | Status: {item_report['status']}"
        )

    # Save summary report
    report_file = out_dir / "generation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_items, f, ensure_ascii=False, indent=2)

    print("\n=== Generation Verification Completed ===")
    print(f"Report saved to: {report_file}")
    all_passed = all(x["status"] == "PASS" for x in report_items)
    print(f"All 10 samples status: {'ALL PASS' if all_passed else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
