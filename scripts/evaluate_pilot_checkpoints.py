#!/usr/bin/env python3
"""Comprehensive evaluation script for 1-Epoch Pilot checkpoints.
Evaluates:
- 10 fixed prompts per checkpoint
- Greedy decoding (do_sample=False, stop_threshold=0.50)
- Entropy, unique tokens, top-1 token ratio vs Mimi cache baseline
- Learned stop behavior & termination reasons
- Diagnostic stop threshold sweep on final checkpoint (0.50, 0.45, 0.40, 0.35)
- Sampling comparison on final checkpoint (temperature=0.8, top_p=0.9)
"""

from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

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
    p = argparse.ArgumentParser(description="Evaluate 1-epoch pilot checkpoints")
    p.add_argument("--thinker", required=True, help="Path to Thinker model directory")
    p.add_argument("--checkpoint-dir", required=True, help="Directory containing pilot checkpoints")
    p.add_argument("--steps", default="346,692,1038,1385", help="Comma-separated steps to evaluate")
    p.add_argument("--output-dir", default="outputs/talker-pilot-1epoch/eval", help="Output directory")
    p.add_argument("--device", default="cuda", help="Inference device")
    return p.parse_args()


def compute_token_stats(tokens: torch.Tensor):
    """Compute entropy, unique tokens, top-1 ratio for all codebooks."""
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)
    num_q = tokens.shape[0] if tokens.ndim == 2 else 0
    total_len = tokens.shape[1] if tokens.ndim == 2 else 0
    
    ent_list = []
    unique_list = []
    top1_ratio_list = []
    
    for q in range(num_q):
        q_toks = tokens[q].tolist()
        unique_list.append(len(set(q_toks)))
        cnts = Counter(q_toks)
        tot = len(q_toks)
        if tot > 0:
            e = -sum((c / tot) * math.log2(c / tot) for c in cnts.values())
            top1 = cnts.most_common(1)[0][1] / tot
        else:
            e = 0.0
            top1 = 0.0
        ent_list.append(round(e, 3))
        top1_ratio_list.append(round(top1, 4))
        
    mean_ent = round(sum(ent_list) / len(ent_list), 3) if ent_list else 0.0
    mean_unique = round(sum(unique_list) / len(unique_list), 1) if unique_list else 0.0
    mean_top1 = round(sum(top1_ratio_list) / len(top1_ratio_list), 4) if top1_ratio_list else 0.0
    
    return {
        "num_frames": total_len,
        "q0_entropy": ent_list[0] if ent_list else 0.0,
        "mean_entropy": mean_ent,
        "entropy_by_q": ent_list,
        "q0_unique": unique_list[0] if unique_list else 0,
        "mean_unique": mean_unique,
        "unique_by_q": unique_list,
        "q0_top1_ratio": top1_ratio_list[0] if top1_ratio_list else 0.0,
        "mean_top1_ratio": mean_top1,
        "top1_ratio_by_q": top1_ratio_list,
    }


def evaluate_checkpoint(
    synthesizer: SpeechSynthesizer,
    ckpt_path: Path,
    step_name: str,
    out_dir: Path,
    gen_config: TalkerGenerationConfig,
    speaker: str = "speaker_001",
    dialect: str = "ulsan",
):
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    results = []
    
    print(f"\n--- Evaluating {step_name} ({ckpt_path.name}) ---")
    for idx, prompt in enumerate(PROMPTS, 1):
        slug = f"sample_{idx:02d}"
        wav_path = wav_dir / f"{slug}.wav"
        
        t0 = time.perf_counter()
        res = synthesizer.synthesize(
            prompt,
            speaker=speaker,
            dialect=dialect,
            generation_config=gen_config,
        )
        total_time = time.perf_counter() - t0
        
        res.save(wav_path)
        
        waveform = res.waveform.detach().cpu().squeeze()
        is_finite = bool(torch.isfinite(waveform).all().item())
        is_non_empty = bool(waveform.numel() > 0 and torch.max(torch.abs(waveform)).item() > 1e-6)
        clipping_ratio = float((torch.abs(waveform) >= 0.999).sum().item()) / waveform.numel() if waveform.numel() > 0 else 0.0
        
        tokens = res.codec_tokens.detach().cpu()
        stats = compute_token_stats(tokens)
        
        passed = bool(
            is_finite
            and is_non_empty
            and clipping_ratio < 0.05
            and res.termination_reason == "STOP_PREDICTED"
            and stats["mean_entropy"] > 1.0
        )
        
        item = {
            "index": idx,
            "text": prompt,
            "duration": round(res.duration, 3),
            "frames": stats["num_frames"],
            "termination_reason": res.termination_reason,
            "max_stop_prob": round(res.max_stop_prob, 4),
            "q0_entropy": stats["q0_entropy"],
            "mean_entropy": stats["mean_entropy"],
            "q0_unique": stats["q0_unique"],
            "mean_unique": stats["mean_unique"],
            "q0_top1_ratio": stats["q0_top1_ratio"],
            "mean_top1_ratio": stats["mean_top1_ratio"],
            "generation_time_sec": round(total_time, 3),
            "rtf": round(res.rtf, 3),
            "clipping_ratio": round(clipping_ratio, 4),
            "wav_path": str(wav_path),
            "status": "PASS" if passed else "FAIL",
        }
        results.append(item)
        print(
            f"  [{idx:02d}/10] '{prompt[:12]:<12}' -> {res.duration:5.2f}s | "
            f"Frames: {stats['num_frames']:3d} | Reason: {res.termination_reason:<14} | "
            f"MaxStopP: {res.max_stop_prob:.3f} | MeanEnt: {stats['mean_entropy']:4.2f} | "
            f"Q0Top1: {stats['q0_top1_ratio']:.2f} | Status: {item['status']}"
        )
        
    summary = {
        "step_name": step_name,
        "checkpoint": str(ckpt_path),
        "total_samples": len(results),
        "num_pass": sum(1 for r in results if r["status"] == "PASS"),
        "num_stop_predicted": sum(1 for r in results if r["termination_reason"] == "STOP_PREDICTED"),
        "num_max_frames": sum(1 for r in results if r["termination_reason"] == "MAX_FRAMES"),
        "avg_duration": round(sum(r["duration"] for r in results) / len(results), 2),
        "avg_max_stop_prob": round(sum(r["max_stop_prob"] for r in results) / len(results), 4),
        "avg_q0_entropy": round(sum(r["q0_entropy"] for r in results) / len(results), 2),
        "avg_mean_entropy": round(sum(r["mean_entropy"] for r in results) / len(results), 2),
        "avg_q0_top1_ratio": round(sum(r["q0_top1_ratio"] for r in results) / len(results), 4),
        "avg_mean_top1_ratio": round(sum(r["mean_top1_ratio"] for r in results) / len(results), 4),
        "avg_rtf": round(sum(r["rtf"] for r in results) / len(results), 3),
        "samples": results,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    return summary


def main():
    args = parse_args()
    device = args.device
    ckpt_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    steps = [int(x.strip()) for x in args.steps.split(",") if x.strip()]
    
    print("=== [ULM-LIVE] 1-Epoch Pilot Regression & Diagnostics ===")
    print(f"Thinker:    {args.thinker}")
    print(f"Ckpt dir:   {ckpt_dir}")
    print(f"Steps:      {steps}")
    print(f"Output dir: {out_dir}")
    print(f"Device:     {device}")
    
    print("\n1. Loading Thinker...")
    thinker = ULMThinker(args.thinker, device=device, torch_dtype="bf16", freeze=True)
    
    print("2. Loading Mimi Codec...")
    codec = build_codec(backend="mimi", device=device)
    
    overall_report = {
        "checkpoints": {},
        "diagnostic_threshold_sweep": {},
        "sampling_comparison": {},
    }
    
    # 1. Run greedy evaluation across all specified checkpoints
    final_talker = None
    final_ckpt_path = None
    
    for step in steps:
        step_name = f"step_{step}"
        # Locate checkpoint
        ckpt_path = ckpt_dir / f"checkpoint-{step}.pt"
        if not ckpt_path.exists():
            # Try final.pt if this is the last step
            if step == steps[-1] and (ckpt_dir / "final.pt").exists():
                ckpt_path = ckpt_dir / "final.pt"
            else:
                print(f"Warning: Checkpoint {ckpt_path} not found, skipping.")
                continue
                
        talker, _ = load_talker_checkpoint(str(ckpt_path), device=device)
        talker.eval()
        
        synthesizer = SpeechSynthesizer(
            thinker=thinker,
            talker=talker,
            codec=codec,
            device=device,
        )
        
        gen_cfg = TalkerGenerationConfig(
            max_new_tokens=750,
            do_sample=False,
            stop_threshold=0.50,
            return_details=True,
        )
        
        step_out = out_dir / step_name
        step_summary = evaluate_checkpoint(
            synthesizer, ckpt_path, step_name, step_out, gen_cfg
        )
        overall_report["checkpoints"][step_name] = step_summary
        
        final_talker = talker
        final_ckpt_path = ckpt_path

    # 2. On final checkpoint, run Diagnostic Stop Threshold Sweep
    if final_talker is not None:
        print("\n=== Running Stop Threshold Sweep on Final Checkpoint ===")
        thresholds = [0.50, 0.45, 0.40, 0.35]
        sweep_out = out_dir / "threshold_sweep"
        sweep_out.mkdir(parents=True, exist_ok=True)
        
        sweep_results = {}
        for th in thresholds:
            print(f"\n--- Threshold {th:.2f} ---")
            th_out = sweep_out / f"th_{int(th*100)}"
            cfg = TalkerGenerationConfig(
                max_new_tokens=750,
                do_sample=False,
                stop_threshold=th,
                return_details=True,
            )
            synth = SpeechSynthesizer(
                thinker=thinker,
                talker=final_talker,
                codec=codec,
                device=device,
            )
            summary = evaluate_checkpoint(
                synth, final_ckpt_path, f"threshold_{th:.2f}", th_out, cfg
            )
            sweep_results[f"{th:.2f}"] = {
                "threshold": th,
                "num_stop_predicted": summary["num_stop_predicted"],
                "avg_duration": summary["avg_duration"],
                "avg_max_stop_prob": summary["avg_max_stop_prob"],
                "avg_mean_entropy": summary["avg_mean_entropy"],
            }
        overall_report["diagnostic_threshold_sweep"] = sweep_results
        
        # 3. Sampling comparison (temperature=0.8, top_p=0.9)
        print("\n=== Running Sampling Comparison (T=0.8, top_p=0.9) on Final Checkpoint ===")
        samp_out = out_dir / "sampling_comparison"
        samp_out.mkdir(parents=True, exist_ok=True)
        samp_cfg = TalkerGenerationConfig(
            max_new_tokens=750,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            stop_threshold=0.50,
            return_details=True,
        )
        synth_samp = SpeechSynthesizer(
            thinker=thinker,
            talker=final_talker,
            codec=codec,
            device=device,
        )
        samp_summary = evaluate_checkpoint(
            synth_samp, final_ckpt_path, "sampling_t08_p09", samp_out, samp_cfg
        )
        overall_report["sampling_comparison"] = {
            "num_stop_predicted": samp_summary["num_stop_predicted"],
            "avg_duration": samp_summary["avg_duration"],
            "avg_max_stop_prob": samp_summary["avg_max_stop_prob"],
            "avg_mean_entropy": samp_summary["avg_mean_entropy"],
            "avg_q0_top1_ratio": samp_summary["avg_q0_top1_ratio"],
            "avg_mean_top1_ratio": samp_summary["avg_mean_top1_ratio"],
        }
        
    final_report_file = out_dir / "pilot_report.json"
    with open(final_report_file, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, ensure_ascii=False, indent=2)
        
    print(f"\n=======================================================")
    print(f"PILOT REPORT SAVED: {final_report_file}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
