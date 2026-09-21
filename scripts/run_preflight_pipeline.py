#!/usr/bin/env python3
"""Automated pipeline runner for ULM-LIVE L40S Preflight steps:
1. 16q Mimi Codec Cache (val, test, train)
2. Strict Cache & Split Verification
3. Real-data Batch Benchmark (Candidate A, B, C)
4. 100-step Smoke Training
5. End-to-End Speech Synthesis (10 test prompts)
6. Comprehensive JSON summary export
"""

from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

PYTHON_EXEC = sys.executable
DATASET_DIR = Path("/home/ubuntu/ULM-LIVE/data/ulsan-full")
THINKER_PATH = "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged"
OUTPUT_DIR = Path("/home/ubuntu/ULM-LIVE/outputs")


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n>>> [PIPELINE] {desc}")
    print(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/home/ubuntu/ULM-LIVE/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:{env.get('LD_LIBRARY_PATH', '')}"
    res = subprocess.run(cmd, env=env, check=True)
    print(f">>> [DONE] {desc} in {time.time() - t0:.2f}s (Exit code: {res.returncode})")


def main():
    t_pipeline_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    codec_dir = DATASET_DIR / "codec"
    codec_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Step 1: Precompute 16q Mimi Codec Tokens
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 1: Precomputing 16q Mimi Codec Tokens")
    print("=======================================================")
    codec_stats = {}
    t_codec_start = time.time()
    for split in ["val", "test", "train"]:
        in_manifest = DATASET_DIR / f"{split}.cached.jsonl"
        # If .cached.jsonl doesn't exist yet, fallback to .jsonl
        if not in_manifest.is_file():
            in_manifest = DATASET_DIR / f"{split}.jsonl"
        out_manifest = DATASET_DIR / f"{split}.cached.jsonl"
        
        cmd = [
            PYTHON_EXEC,
            "scripts/cache_mimi_tokens.py",
            "--manifest", str(in_manifest),
            "--output-dir", str(codec_dir),
            "--output-manifest", str(out_manifest),
            "--num-quantizers", "16",
            "--device", "cuda",
        ]
        run_cmd(cmd, f"Mimi 16q encoding for {split} split")

    total_codec_time = time.time() - t_codec_start
    total_codec_files = len(list(codec_dir.glob("*.pt")))
    codec_size_mb = sum(f.stat().st_size for f in codec_dir.glob("*.pt")) / (1024 * 1024)

    codec_summary = {
        "num_quantizers": 16,
        "total_files": total_codec_files,
        "runtime_sec": round(total_codec_time, 2),
        "samples_per_sec": round(total_codec_files / total_codec_time, 1) if total_codec_time > 0 else 0,
        "cache_size_mb": round(codec_size_mb, 2),
    }

    # -------------------------------------------------------------
    # Step 2: Strict Cache & Split Integrity Verification
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 2: Verifying Cache & Split Integrity")
    print("=======================================================")
    from scripts.verify_dataset_integrity import verify_dataset
    verify_res = verify_dataset(DATASET_DIR, strict=True, num_quantizers=16)

    # -------------------------------------------------------------
    # Step 3: L40S Real-data Batch Benchmark (Candidate A, B, C)
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 3: Running L40S Real-data Batch Benchmark (A, B, C)")
    print("=======================================================")
    bench_out_json = OUTPUT_DIR / "l40s_benchmark_results.json"
    cmd_bench = [
        PYTHON_EXEC,
        "scripts/benchmark_l40s_talker.py",
        "--manifest", str(DATASET_DIR / "train.cached.jsonl"),
        "--val-manifest", str(DATASET_DIR / "val.cached.jsonl"),
        "--semantic-cache-thinker", THINKER_PATH,
        "--steps", "20",
        "--output-json", str(bench_out_json),
    ]
    run_cmd(cmd_bench, "L40S Batch Benchmark")

    with open(bench_out_json, "r", encoding="utf-8") as f:
        bench_results = json.load(f)

    # Select best candidate
    passed_candidates = [c for c in bench_results if c.get("status") == "PASS"]
    if not passed_candidates:
        raise RuntimeError("All benchmark candidates failed!")
    
    # Sort by sec_per_step (fastest)
    best_candidate = min(passed_candidates, key=lambda c: c.get("sec_per_step", 999.0))
    print(f"\nSelected Optimal Candidate: {best_candidate['candidate']} (batch={best_candidate['batch_size']}, accum={best_candidate['grad_accum']}, {best_candidate['sec_per_step']} s/step)")

    # -------------------------------------------------------------
    # Step 4: 100-step Smoke Training
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 4: Executing 100-step Real Talker Smoke Training")
    print("=======================================================")
    smoke_out_dir = OUTPUT_DIR / "talker-smoke-phase4"
    smoke_out_dir.mkdir(parents=True, exist_ok=True)
    smoke_log_file = smoke_out_dir / "smoke_training.log"

    cmd_smoke = [
        PYTHON_EXEC,
        "scripts/train_talker.py",
        "--config", "configs/talker.yaml",
        "--manifest", str(DATASET_DIR / "train.cached.jsonl"),
        "--val-manifest", str(DATASET_DIR / "val.cached.jsonl"),
        "--semantic-cache-thinker", THINKER_PATH,
        "--batch-size", str(best_candidate["batch_size"]),
        "--gradient-accumulation-steps", str(best_candidate["grad_accum"]),
        "--max-steps", "100",
        "--eval-steps", "25",
        "--save-steps", "25",
        "--bf16",
        "--strict-codec",
        "--lr", "0.0002",
        "--warmup-ratio", "0.03",
        "--output-dir", str(smoke_out_dir),
    ]

    t_smoke_start = time.time()
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"/home/ubuntu/ULM-LIVE/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:{env.get('LD_LIBRARY_PATH', '')}"
    
    with open(smoke_log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd_smoke, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        smoke_output_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            smoke_output_lines.append(line)
        proc.wait()

    total_smoke_time = time.time() - t_smoke_start
    if proc.returncode != 0:
        raise RuntimeError(f"Smoke training failed with return code {proc.returncode}")

    print(f"\n100-step Smoke Training completed in {total_smoke_time:.2f}s.")

    # -------------------------------------------------------------
    # Step 5: End-to-End Speech Synthesis (10 test prompts)
    # -------------------------------------------------------------
    print("\n=======================================================")
    print("STEP 5: Generating End-to-End Speech for 10 Prompts")
    print("=======================================================")
    best_ckpt = smoke_out_dir / "best_checkpoint.pt"
    if not best_ckpt.is_file():
        # Fallback to checkpoint_000100.pt
        best_ckpt = smoke_out_dir / "checkpoint_000100.pt"

    cmd_gen = [
        PYTHON_EXEC,
        "scripts/generate_smoke_samples.py",
        "--thinker", THINKER_PATH,
        "--checkpoint", str(best_ckpt),
        "--output-dir", str(smoke_out_dir / "audio"),
        "--device", "cuda",
    ]
    run_cmd(cmd_gen, "10-prompt Speech Synthesis")

    # Load generation report
    gen_report_path = smoke_out_dir / "audio" / "generation_report.json"
    with open(gen_report_path, "r", encoding="utf-8") as f:
        gen_reports = json.load(f)

    # -------------------------------------------------------------
    # Final Pipeline Summary
    # -------------------------------------------------------------
    total_pipeline_time = time.time() - t_pipeline_start
    summary_path = OUTPUT_DIR / "preflight_summary.json"
    summary_data = {
        "pipeline_status": "SUCCESS",
        "total_runtime_sec": round(total_pipeline_time, 2),
        "quantizers": 16,
        "codec_summary": codec_summary,
        "verification": verify_res,
        "benchmark": {
            "all_candidates": bench_results,
            "selected_candidate": best_candidate,
        },
        "smoke": {
            "runtime_sec": round(total_smoke_time, 2),
            "checkpoint_used": str(best_ckpt),
        },
        "generation": gen_reports,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("=== [PREFLIGHT PIPELINE COMPLETE] ===")
    print(f"Summary JSON saved to: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
