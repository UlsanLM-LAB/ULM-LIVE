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
from ulm_live.utils import get_duration, load_wav, save_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode audio to discrete codec tokens and reconstruct back to WAV."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input WAV audio file.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output reconstructed WAV file.",
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
        help="Override codec backend (default: read from config or 'mimi').",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=None,
        help="Execution device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--num-quantizers",
        "-q",
        type=int,
        default=None,
        help="Number of RVQ codebooks to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"Error: Input audio file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)

    # 1. Load input waveform
    try:
        waveform, sr = load_wav(input_path)
    except Exception as err:
        print(f"Error loading input audio '{input_path}': {err}", file=sys.stderr)
        sys.exit(1)

    in_duration = get_duration(waveform, sr)
    in_channels = waveform.shape[0]

    # 2. Build Codec
    try:
        build_kwargs = {}
        if args.device is not None:
            build_kwargs["device"] = args.device
        if args.num_quantizers is not None:
            build_kwargs["num_quantizers"] = args.num_quantizers

        codec = build_codec(
            backend=args.backend,
            config_path=args.config if Path(args.config).is_file() else None,
            **build_kwargs,
        )
    except Exception as err:
        print(f"Error initializing codec backend: {err}", file=sys.stderr)
        sys.exit(1)

    # Synchronize CUDA before timing if on CUDA
    if codec.device.type == "cuda":
        torch.cuda.synchronize()

    # 3. Encode
    t0 = time.perf_counter()
    try:
        encoded = codec.encode(waveform, sample_rate=sr)
    except Exception as err:
        print(f"Error during audio encoding: {err}", file=sys.stderr)
        sys.exit(1)

    if codec.device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    encode_time = t1 - t0

    # 4. Decode
    t2 = time.perf_counter()
    try:
        recon_audio, out_sr = codec.decode(encoded)
    except Exception as err:
        print(f"Error during audio decoding: {err}", file=sys.stderr)
        sys.exit(1)

    if codec.device.type == "cuda":
        torch.cuda.synchronize()
    t3 = time.perf_counter()
    decode_time = t3 - t2

    # If decoded audio is 3D (batch, channels, samples), flatten to 2D (channels, samples)
    if recon_audio.ndim == 3:
        recon_audio = recon_audio.squeeze(0)

    # Align duration to original input if desired, or keep as reconstructed
    # Save reconstructed audio
    try:
        save_wav(output_path, recon_audio, out_sr)
    except Exception as err:
        print(f"Error saving output audio '{output_path}': {err}", file=sys.stderr)
        sys.exit(1)

    out_duration = get_duration(recon_audio, out_sr)

    # 5. Output summary
    print("Input")
    print(f"sample rate: {sr}")
    print(f"channels: {in_channels}")
    print(f"duration: {in_duration:.2f}s")
    print()
    print("Codec")
    print(f"backend: {codec.backend_name}")
    print(f"representation shape: {tuple(encoded.codes.shape)}")
    print(f"approx token rate: {codec.approx_token_rate:.1f} tokens/s")
    print(f"encode time: {encode_time:.4f}s")
    print(f"decode time: {decode_time:.4f}s")
    print()
    print("Output")
    print(f"sample rate: {out_sr}")
    print(f"duration: {out_duration:.2f}s")
    print(f"path: {output_path}")


if __name__ == "__main__":
    main()
