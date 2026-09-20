import argparse
from pathlib import Path
import sys
import time

# Ensure repo root is on sys.path for direct script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ulm_live.codec import build_codec
from ulm_live.talker import (
    SpeechSynthesizer,
    TalkerConfig,
    TalkerGenerationConfig,
    ULMTalker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick test for Talker generation and Codec decoding pipeline."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/test_generation.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=25,
        help="Number of codec frames to generate (~2.0s at 12.5 Hz).",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else (args.device if args.device != "auto" else "cpu")
    )

    print("=== Testing Talker Generation & Codec Decode Pipeline ===")
    print(f"Device: {device}")

    # 1. Initialize Codec
    print("Loading Phase 1 Mimi Codec...")
    codec = build_codec(backend="mimi", device=device)

    # 2. Initialize Talker prototype
    cfg = TalkerConfig(
        semantic_dim=2048,
        talker_dim=512,
        num_layers=4,
        num_heads=8,
        num_quantizers=codec.num_quantizers,
        codebook_size=2048,
    )
    talker = ULMTalker(cfg).to(device)

    synthesizer = SpeechSynthesizer(
        thinker=None,
        talker=talker,
        codec=codec,
        device=device,
    )

    # 3. Synthetic semantic states
    B = 1
    S = 8
    synthetic_semantic = torch.randn(B, S, cfg.semantic_dim, device=device)
    gen_cfg = TalkerGenerationConfig(
        max_new_tokens=args.max_tokens, do_sample=False, stop_threshold=2.0
    )

    print(
        f"Generating {args.max_tokens} codec frames from synthetic semantic states..."
    )
    result = synthesizer.synthesize_from_hidden(
        synthetic_semantic,
        speaker_id=0,
        dialect_id=0,
        generation_config=gen_cfg,
    )

    out_path = Path(args.output)
    result.save(out_path)

    print("\nGeneration Result:")
    print(f"  Codec tokens shape:  {tuple(result.codec_tokens.shape)}")
    print(f"  Waveform shape:      {tuple(result.waveform.shape)}")
    print(f"  Sample rate:         {result.sample_rate} Hz")
    print(f"  Audio duration:      {result.duration:.2f} s")
    print(f"  Talker time:         {result.timings['talker_time']:.4f} s")
    print(f"  Decode time:         {result.timings['decode_time']:.4f} s")
    print(f"  Total time:          {result.timings['total_time']:.4f} s")
    print(f"  RTF:                 {result.rtf:.2f}")
    print(f"  Output saved to:     {out_path}")
    print("\nPipeline Verification: SUCCESS")


if __name__ == "__main__":
    main()
