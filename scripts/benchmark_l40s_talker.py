#!/usr/bin/env python3
"""L40S batch throughput and VRAM benchmark for Talker training."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Talker training batch sizes on NVIDIA L40S GPU."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/ulsan/train.jsonl",
        help="Path to training manifest JSONL.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="data/ulsan/val.jsonl",
        help="Path to validation manifest JSONL.",
    )
    parser.add_argument(
        "--thinker",
        type=str,
        default="/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged",
        help="Path to merged Thinker model.",
    )
    parser.add_argument(
        "--semantic-cache-thinker",
        type=str,
        default=None,
        help="Bind semantic cache identity without loading live Thinker into VRAM.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of optimizer steps to benchmark per candidate (default: 20).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/benchmark_results.json",
        help="Path to save benchmark JSON output.",
    )
    return parser.parse_args()


def get_gpu_telemetry() -> dict[str, float]:
    """Query nvidia-smi for current utilization, temp, power, and memory."""
    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        line = res.stdout.strip().splitlines()[0]
        parts = [float(x.strip()) for x in line.split(",")]
        return {
            "gpu_util": parts[0],
            "temp": parts[1],
            "power": parts[2],
            "mem_used_mib": parts[3],
            "mem_total_mib": parts[4],
        }
    except Exception:
        return {
            "gpu_util": 0.0,
            "temp": 0.0,
            "power": 0.0,
            "mem_used_mib": 0.0,
            "mem_total_mib": 0.0,
        }


def run_candidate(
    candidate_id: str,
    batch_size: int,
    grad_accum: int,
    steps: int,
    manifest: str,
    val_manifest: str,
    thinker: str | None = None,
    semantic_cache_thinker: str | None = None,
) -> dict:
    print(f"\n=======================================================")
    print(
        f"Running Benchmark Candidate {candidate_id}: batch_size={batch_size}, grad_accum={grad_accum} (eff_batch={batch_size * grad_accum})"
    )
    print(f"=======================================================")

    cmd = [
        sys.executable,
        "scripts/train_talker.py",
        "--config",
        "configs/talker.yaml",
        "--manifest",
        manifest,
        "--val-manifest",
        val_manifest,
        "--batch-size",
        str(batch_size),
        "--gradient-accumulation-steps",
        str(grad_accum),
        "--max-steps",
        str(steps),
        "--eval-steps",
        str(steps),
        "--bf16",
        "--strict-codec",
        "--max-eval-batches",
        "10",
        "--output-dir",
        f"outputs/bench_{candidate_id}",
    ]
    if semantic_cache_thinker:
        cmd.extend(["--semantic-cache-thinker", semantic_cache_thinker])
    elif thinker:
        cmd.extend(["--thinker", thinker])

    telemetry_samples = []
    start_time = time.time()

    # Reset max memory before launch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    oom = False
    output_lines = []

    # Monitor process while running
    while proc.poll() is None:
        telem = get_gpu_telemetry()
        if telem["gpu_util"] > 0 or telem["mem_used_mib"] > 0:
            telemetry_samples.append(telem)
        time.sleep(1.0)

    stdout, _ = proc.communicate()
    total_time = time.time() - start_time
    output_lines = stdout.splitlines()

    for line in output_lines:
        if "OutOfMemoryError" in line or "CUDA out of memory" in line:
            oom = True
            break

    if oom or proc.returncode != 0:
        print(
            f"Candidate {candidate_id} FAILED (OOM={oom}, returncode={proc.returncode})"
        )
        return {
            "candidate": candidate_id,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "effective_batch": batch_size * grad_accum,
            "status": "OOM" if oom else "ERROR",
            "runtime_sec": round(total_time, 2),
            "sec_per_step": None,
            "steps_per_sec": None,
            "peak_vram_gib": None,
            "vram_util_pct": None,
            "avg_gpu_util_pct": None,
            "max_gpu_temp_c": None,
            "max_power_w": None,
            "train_loss": None,
            "val_loss": None,
        }

    # Parse train loss, val loss, and steps from output
    train_loss = None
    val_loss = None
    for line in reversed(output_lines):
        stripped = line.strip()
        if stripped.startswith('"last_loss":'):
            try:
                train_loss = float(stripped.split(":", 1)[1].rstrip(","))
            except ValueError:
                pass
        if stripped.startswith("{") and '"validation"' in stripped:
            try:
                val_loss = float(json.loads(stripped)["validation"]["total_loss"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass
        if "Step" in line and "Loss" in line:
            # e.g., Step 20/20 | Train Loss: 3.456 | Val Loss: 3.512
            parts = line.split("|")
            for p in parts:
                if "Train Loss:" in p:
                    try:
                        train_loss = float(p.split(":")[-1].strip())
                    except ValueError:
                        pass
                if "Val Loss:" in p:
                    try:
                        val_loss = float(p.split(":")[-1].strip())
                    except ValueError:
                        pass
            if train_loss is not None:
                break

    sec_per_step = round(total_time / steps, 3)
    steps_per_sec = round(steps / total_time, 3)

    # Compute telemetry aggregates
    max_mem_mib = (
        max([s["mem_used_mib"] for s in telemetry_samples])
        if telemetry_samples
        else 0.0
    )
    total_mem_mib = (
        telemetry_samples[0]["mem_total_mib"] if telemetry_samples else 46000.0
    )
    avg_util = (
        sum([s["gpu_util"] for s in telemetry_samples]) / len(telemetry_samples)
        if telemetry_samples
        else 0.0
    )
    max_temp = max([s["temp"] for s in telemetry_samples]) if telemetry_samples else 0.0
    max_power = (
        max([s["power"] for s in telemetry_samples]) if telemetry_samples else 0.0
    )

    peak_vram_gib = round(max_mem_mib / 1024.0, 2)
    vram_util_pct = round((max_mem_mib / total_mem_mib) * 100.0, 1)

    result = {
        "candidate": candidate_id,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "effective_batch": batch_size * grad_accum,
        "status": "PASS",
        "runtime_sec": round(total_time, 2),
        "sec_per_step": sec_per_step,
        "steps_per_sec": steps_per_sec,
        "peak_vram_gib": peak_vram_gib,
        "vram_util_pct": vram_util_pct,
        "avg_gpu_util_pct": round(avg_util, 1),
        "max_gpu_temp_c": round(max_temp, 1),
        "max_power_w": round(max_power, 1),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }

    print(
        f"Candidate {candidate_id} Result: {sec_per_step} s/step | Peak VRAM: {peak_vram_gib} GiB ({vram_util_pct}%) | Loss: {train_loss}"
    )
    return result


def main() -> None:
    args = parse_args()
    candidates = [
        ("A", 4, 4),  # batch=4, accum=4 -> eff=16
        ("B", 8, 2),  # batch=8, accum=2 -> eff=16
        ("C", 16, 1),  # batch=16, accum=1 -> eff=16
    ]

    results = []
    for cid, bs, ga in candidates:
        res = run_candidate(
            candidate_id=cid,
            batch_size=bs,
            grad_accum=ga,
            steps=args.steps,
            manifest=args.manifest,
            val_manifest=args.val_manifest,
            thinker=args.thinker,
            semantic_cache_thinker=args.semantic_cache_thinker,
        )
        results.append(res)
        if res["status"] == "OOM":
            print(f"Stopping benchmark early due to OOM on candidate {cid}.")
            break

    # Save results
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print markdown table
    print("\n" + "=" * 60)
    print("### L40S Batch Benchmark Comparison Table")
    print("=" * 60)
    print(
        "| Candidate | Batch | Grad Accum | Eff Batch | Sec/Step | Steps/Sec | Peak VRAM (GiB) | VRAM % | Avg Util % | Max Temp | Train Loss | Status |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        vram_str = f"{r['peak_vram_gib']:.2f}" if r["peak_vram_gib"] else "N/A"
        pct_str = f"{r['vram_util_pct']:.1f}%" if r["vram_util_pct"] else "N/A"
        sec_str = f"{r['sec_per_step']:.3f}" if r["sec_per_step"] else "N/A"
        steps_str = f"{r['steps_per_sec']:.3f}" if r["steps_per_sec"] else "N/A"
        util_str = f"{r['avg_gpu_util_pct']:.1f}%" if r["avg_gpu_util_pct"] else "N/A"
        temp_str = f"{r['max_gpu_temp_c']:.0f}°C" if r["max_gpu_temp_c"] else "N/A"
        loss_str = f"{r['train_loss']:.4f}" if r["train_loss"] else "N/A"
        print(
            f"| {r['candidate']} | {r['batch_size']} | {r['grad_accum']} | {r['effective_batch']} | {sec_str} | {steps_str} | {vram_str} | {pct_str} | {util_str} | {temp_str} | {loss_str} | {r['status']} |"
        )


if __name__ == "__main__":
    main()
