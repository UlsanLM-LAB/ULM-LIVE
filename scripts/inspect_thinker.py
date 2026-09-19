import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path for direct script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.thinker import ULMThinker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect ULM Thinker (language model) architecture and hidden state dimensions."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model path or Hugging Face model identifier (e.g. 'Qwen/Qwen3-1.7B', '/path/to/ULM-1.7B').",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        help="Device to load model on ('cpu', 'cuda', 'auto'). Default: cpu.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Torch dtype ('auto', 'bf16', 'fp16', 'fp32'). Default: auto.",
    )
    parser.add_argument(
        "--hidden-layer",
        type=int,
        default=-1,
        help="Selected hidden layer index to extract (default: -1 for last layer).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading Thinker model from: {args.model} ...")
    try:
        thinker = ULMThinker(
            model_name_or_path=args.model,
            device=args.device,
            torch_dtype=args.dtype,
            hidden_layer=args.hidden_layer,
            freeze=True,
            load_pretrained=True,
        )
    except Exception as err:
        print(f"Error: Failed to load Thinker model: {err}", file=sys.stderr)
        sys.exit(1)

    print("\n=== ULM Thinker Inspection ===")
    print(f"Model:                 {args.model}")
    print(f"Hidden size:           {thinker.hidden_size}")
    print(f"Layers:                {thinker.num_layers}")
    print(f"Vocabulary size:       {thinker.vocab_size}")
    print(f"Selected hidden layer: {thinker.hidden_layer}")
    print(f"Device:                {thinker.device}")
    print(f"Dtype:                 {thinker.dtype}")

    # Quick forward test on sample text
    sample_prompt = "울산 사투리 음성 모델 테스트"
    try:
        inputs = thinker.tokenize(sample_prompt)
        hidden = thinker.forward_hidden(inputs["input_ids"], attention_mask=inputs.get("attention_mask"))
        print(f"\nForward Hidden Test:")
        print(f"  Input tokens:        {inputs['input_ids'].shape[1]}")
        print(f"  Hidden state shape:  {tuple(hidden.shape)}")
        print("  Status:              OK")
    except Exception as err:
        print(f"\nWarning: Forward hidden test failed: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
