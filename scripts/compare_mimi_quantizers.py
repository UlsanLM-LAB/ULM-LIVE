#!/usr/bin/env python3
"""Small held-out Mimi 8/16/32-quantizer reconstruction comparison."""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ulm_live.codec import build_codec  # noqa: E402
from ulm_live.utils.audio import (  # noqa: E402
    get_duration,
    load_wav,
    resample_audio,
    save_wav,
    to_mono,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wavs", nargs="+")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    codec = build_codec(device=a.device)
    report = []
    for source in map(Path, a.wavs):
        wav, sr = load_wav(source)
        stem = out / source.stem
        stem.mkdir(exist_ok=True)
        save_wav(stem / "original.wav", wav, sr)
        for q in (8, 16, 32):
            if codec.num_quantizers < q:
                raise ValueError(
                    f"codec exposes {codec.num_quantizers}, cannot test {q}"
                )
            t = time.perf_counter()
            enc = codec.encode(wav, sr, num_quantizers=q)
            encode_s = time.perf_counter() - t
            t = time.perf_counter()
            recon, recon_sr = codec.decode(enc)
            decode_s = time.perf_counter() - t
            save_wav(stem / f"{q}q.wav", recon.squeeze(0), recon_sr)
            # Waveform MSE is alignment-sensitive but useful as a simple regression metric.
            ref = to_mono(wav)
            if sr != recon_sr:
                ref = resample_audio(ref, sr, recon_sr)
            ref = ref.squeeze()
            pred = recon.squeeze()[: ref.numel()]
            ref = ref[: pred.numel()]
            mse = torch.mean((ref.cpu() - pred.cpu()) ** 2).item()
            report.append(
                {
                    "source": str(source),
                    "quantizers": q,
                    "duration": get_duration(wav, sr),
                    "encode_seconds": encode_s,
                    "decode_seconds": decode_s,
                    "waveform_mse": mse,
                }
            )
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
