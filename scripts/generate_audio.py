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
    load_talker_checkpoint,
)
from ulm_live.thinker import ULMThinker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end speech generation CLI: Text -> Thinker -> Talker -> Codec -> WAV."
    )
    parser.add_argument(
        "--thinker",
        "-t",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Path or identifier for ULM Thinker model (default: 'Qwen/Qwen3-1.7B').",
    )
    parser.add_argument(
        "--talker",
        "-k",
        type=str,
        default=None,
        help="Path to Talker checkpoint (.pt). If omitted, an untrained model is used with a warning.",
    )
    parser.add_argument(
        "--talker-config",
        type=str,
        default="configs/talker.yaml",
        help="Path to Talker config YAML (used when --talker checkpoint is not provided).",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="오늘 날씨 와 이리 덥노",
        help="Text prompt to synthesize.",
    )
    parser.add_argument(
        "--speaker",
        "-s",
        type=str,
        default="speaker_001",
        help="Speaker identifier (default: 'speaker_001').",
    )
    parser.add_argument(
        "--dialect",
        "-d",
        type=str,
        default="ulsan",
        help="Dialect identifier (default: 'ulsan').",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="outputs/generated.wav",
        help="Output WAV audio file path (default: 'outputs/generated.wav').",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=3.0,
        help="Target maximum audio duration in seconds (default: 3.0s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k filtering for sampling.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus filtering for sampling.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Use stochastic sampling instead of greedy decoding.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")

    # 1. Initialize Codec
    try:
        codec = build_codec(backend="mimi", device=device)
    except Exception as err:
        print(f"Error initializing codec: {err}", file=sys.stderr)
        sys.exit(1)

    # 2. Initialize Talker
    is_trained_talker = False
    speaker2id = {"speaker_001": 0}
    dialect2id = {"ulsan": 0}

    if args.talker is not None and Path(args.talker).is_file():
        try:
            talker, meta = load_talker_checkpoint(args.talker, device=device)
            speaker2id = meta.get("speaker2id") or speaker2id
            dialect2id = meta.get("dialect2id") or dialect2id
            is_trained_talker = True
            checkpoint_desc = str(args.talker)
        except Exception as err:
            print(f"Error loading Talker checkpoint '{args.talker}': {err}", file=sys.stderr)
            sys.exit(1)
    else:
        print("=" * 70, file=sys.stderr)
        print("WARNING:", file=sys.stderr)
        print("No trained Talker checkpoint was provided.", file=sys.stderr)
        print("The Talker is randomly initialized.", file=sys.stderr)
        print("Generated audio is not expected to contain intelligible speech.", file=sys.stderr)
        print("This run only validates the generation pipeline.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)

        cfg_path = Path(args.talker_config)
        cfg = TalkerConfig.from_yaml(cfg_path) if cfg_path.is_file() else TalkerConfig()
        cfg.num_quantizers = codec.num_quantizers
        talker = ULMTalker(cfg).to(device)
        checkpoint_desc = "None (Random initialization)"

    # 3. Initialize Thinker
    try:
        thinker = ULMThinker(
            model_name_or_path=args.thinker,
            device=device,
            freeze=True,
            load_pretrained=True,
        )
    except Exception as err:
        print(f"Error loading Thinker model '{args.thinker}': {err}", file=sys.stderr)
        sys.exit(1)

    # 4. Build Synthesizer
    synthesizer = SpeechSynthesizer(
        thinker=thinker,
        talker=talker,
        codec=codec,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        device=device,
    )

    gen_cfg = TalkerGenerationConfig(
        max_audio_seconds=args.max_seconds,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    # 5. Execute Synthesis
    try:
        result = synthesizer.synthesize(
            text=args.text,
            speaker=args.speaker,
            dialect=args.dialect,
            generation_config=gen_cfg,
        )
    except Exception as err:
        print(f"Error during speech synthesis: {err}", file=sys.stderr)
        sys.exit(1)

    # 6. Save WAV
    out_path = Path(args.output)
    try:
        result.save(out_path)
    except Exception as err:
        print(f"Error saving output audio to '{out_path}': {err}", file=sys.stderr)
        sys.exit(1)

    num_params = sum(p.numel() for p in talker.parameters())
    mode_str = "sampling" if args.do_sample else "greedy"

    # 7. Print Required Report
    print("Input")
    print(f"Text:                  {args.text}")
    print(f"Speaker:               {args.speaker}")
    print(f"Dialect:               {args.dialect}")
    print()
    print("Thinker")
    print(f"Model:                 {args.thinker}")
    print(f"Hidden states:         (1, seq_len, {thinker.hidden_size})")
    print()
    print("Talker")
    print(f"Checkpoint:            {checkpoint_desc}")
    print(f"Parameters:            {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"Generation mode:       {mode_str}")
    print(f"Generated frames:      {result.num_audio_frames}")
    print(f"Codec token shape:     {tuple(result.codec_tokens.shape)}")
    print()
    print("Codec")
    print(f"Backend:               {codec.backend_name}")
    print(f"Codebooks:             {codec.num_quantizers}")
    print(f"Decode time:           {result.timings['decode_time']:.4f}s")
    print()
    print("Audio")
    print(f"Sample rate:           {result.sample_rate} Hz")
    print(f"Duration:              {result.duration:.2f}s")
    print(f"Output path:           {out_path}")
    print()
    if not is_trained_talker:
        print("Audio quality:")
        print("UNTRAINED / PIPELINE TEST ONLY")
        print()
    print("Performance")
    print(f"Thinker time:          {result.timings['thinker_time']:.4f}s")
    print(f"Talker time:           {result.timings['talker_time']:.4f}s")
    print(f"Codec decode time:     {result.timings['decode_time']:.4f}s")
    print(f"Total time:            {result.timings['total_time']:.4f}s")
    print(f"RTF:                   {result.rtf:.2f}")
    print()
    print("Pipeline:")
    print("SUCCESS")


if __name__ == "__main__":
    main()
