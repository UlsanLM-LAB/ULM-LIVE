#!/usr/bin/env python3
"""Precompute 16q Mimi codec tokens for ULM-LIVE dataset samples.
Strictly validates token ranges [0, 2047], quantizer count K, and non-empty frames.
Resumable by default.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

# Ensure repo root is in sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import soundfile as sf
import torch
from ulm_live.codec import build_codec


def parse_args():
    p = argparse.ArgumentParser(description="Precompute Mimi codec tokens.")
    p.add_argument("--manifest", required=True, help="Input manifest JSONL.")
    p.add_argument("--output-dir", required=True, help="Directory to save .pt codec files.")
    p.add_argument("--output-manifest", required=True, help="Output manifest with updated codec_path.")
    p.add_argument("--num-quantizers", type=int, default=16, help="Number of RVQ quantizers (default: 16).")
    p.add_argument("--device", default="cuda", help="Device to use for Mimi (default: cuda).")
    return p.parse_args()


def main():
    args = parse_args()
    in_manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_manifest = Path(args.output_manifest)
    base_dir = in_manifest.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Mimi codec (K={args.num_quantizers}, device={args.device})...")
    codec = build_codec(device=args.device)

    with open(in_manifest, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    print(f"Processing {len(rows)} samples from {in_manifest}...")
    t0 = time.time()
    frames_list = []
    skipped = 0
    processed = 0

    for idx, row in enumerate(rows):
        ident = row.get("id")
        pt_path = out_dir / f"{ident}.pt"
        rel_codec_path = f"codec/{ident}.pt"

        if pt_path.is_file():
            # Already exists, verify integrity
            try:
                tokens = torch.load(pt_path, map_location="cpu", weights_only=True)
                if tokens.dim() == 3:
                    tokens = tokens.squeeze(0)
                if tokens.shape[0] == args.num_quantizers and tokens.shape[1] > 0:
                    if 0 <= tokens.min().item() and tokens.max().item() < 2048:
                        row["codec_path"] = rel_codec_path
                        frames_list.append(tokens.shape[1])
                        skipped += 1
                        continue
            except Exception:
                pass  # Recompute if invalid

        # Load audio
        rel_audio = row["audio_path"]
        audio_file = base_dir / rel_audio if not Path(rel_audio).is_absolute() else Path(rel_audio)
        data, sr = sf.read(str(audio_file), dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim == 2:
            wav = wav.T
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Encode on device
        wav = wav.to(args.device)
        with torch.no_grad():
            enc = codec.encode(wav, sr, num_quantizers=args.num_quantizers)
            codes = enc.codes
            if codes.dim() == 3:
                codes = codes.squeeze(0)

        # Strict validation
        k, t = codes.shape
        if k != args.num_quantizers:
            raise ValueError(f"Sample {ident} produced {k} quantizers, expected {args.num_quantizers}")
        if t == 0:
            raise ValueError(f"Sample {ident} produced 0 frames")
        t_min = int(codes.min().item())
        t_max = int(codes.max().item())
        if t_min < 0 or t_max >= 2048:
            raise ValueError(f"Sample {ident} produced out-of-range token: [{t_min}, {t_max}]")

        # Save to file
        torch.save(codes.cpu(), pt_path)
        row["codec_path"] = rel_codec_path
        frames_list.append(t)
        processed += 1

        if (idx + 1) % 500 == 0 or (idx + 1) == len(rows):
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print(f"  [{idx + 1}/{len(rows)}] processed={processed}, skipped={skipped} ({rate:.1f} samples/s)")

    elapsed = time.time() - t0
    # Write output manifest
    with open(out_manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_bytes = sum(f.stat().st_size for f in out_dir.glob("*.pt"))
    total_mb = total_bytes / (1024 * 1024)
    avg_frames = sum(frames_list) / len(frames_list) if frames_list else 0

    print("\n=== Codec Precomputation Summary ===")
    print(f"Total samples:     {len(rows)}")
    print(f"Newly processed:   {processed}")
    print(f"Skipped existing:  {skipped}")
    print(f"Runtime:           {elapsed:.2f}s")
    print(f"Samples/sec:       {len(rows) / elapsed:.1f}")
    print(f"Cache size:        {total_mb:.2f} MB")
    print(f"Mean frames/sample:{avg_frames:.1f}")
    print(f"Manifest saved to: {out_manifest}")


if __name__ == "__main__":
    main()
