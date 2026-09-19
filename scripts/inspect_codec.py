import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path for direct script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.codec import build_codec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect neural audio codec architecture, codebook sizes, and token rates."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/codec.yaml",
        help="Path to codec YAML configuration file.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default=None,
        help="Codec backend name (default: read from config or 'mimi').",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="Device to load model on (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        codec = build_codec(
            backend=args.backend,
            config_path=args.config if Path(args.config).is_file() else None,
            device=args.device,
        )
    except Exception as err:
        print(f"Error loading codec: {err}", file=sys.stderr)
        sys.exit(1)

    print("=== Neural Audio Codec Inspection ===")
    print(f"Backend:            {codec.backend_name}")
    print(f"Device:             {codec.device}")
    print(f"Sample Rate:        {codec.sample_rate} Hz")
    print(f"Frame Rate:         {codec.frame_rate:.2f} Hz")
    print(f"Quantizers (RVQ):   {codec.num_quantizers}")
    print(f"Approx Token Rate:  {codec.approx_token_rate:.1f} tokens/s")

    if hasattr(codec, "model"):
        model = codec.model
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameter Count:    {num_params:,} ({num_params / 1e6:.2f}M)")

        if hasattr(model, "config"):
            cfg = model.config
            codebook_size = getattr(cfg, "codebook_size", "unknown")
            print(f"Codebook Size:      {codebook_size}")
            codebook_dim = getattr(cfg, "codebook_dim", "unknown")
            print(f"Codebook Dim:       {codebook_dim}")
            hidden_size = getattr(cfg, "hidden_size", "unknown")
            print(f"Hidden Size:        {hidden_size}")


if __name__ == "__main__":
    main()
