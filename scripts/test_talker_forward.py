import argparse
from pathlib import Path
import sys

# Ensure repo root is on sys.path for direct script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ulm_live.talker import TalkerConfig, ULMTalker
from ulm_live.thinker import ULMThinker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run forward and backward test for ULMTalker prototype."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/talker.yaml",
        help="Path to talker configuration YAML.",
    )
    parser.add_argument(
        "--allow-synthetic-thinker",
        action="store_true",
        help="Explicitly use random semantics for this architecture-only test.",
    )
    parser.add_argument(
        "--thinker",
        "-t",
        type=str,
        default=None,
        help="Path or identifier for ULM Thinker model.",
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

    # 1. Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 2. Load Talker config and model
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        config = TalkerConfig.from_yaml(cfg_path)
    else:
        config = TalkerConfig()

    talker = ULMTalker(config).to(device)
    talker.train()

    # 3. Obtain semantic hidden states
    batch_size = 2
    num_audio_frames = 25  # ~2 seconds of audio at 12.5 Hz frame rate
    K = config.num_quantizers

    if args.thinker is not None:
        try:
            print(f"Loading Thinker model from: {args.thinker}")
            thinker = ULMThinker(
                model_name_or_path=args.thinker,
                device=str(device),
                freeze=True,
                load_pretrained=True,
            )
            sample_prompts = [
                "오늘 날씨 와 이리 덥노",
                "울산 남구 삼산동 가자",
            ]
            tokens = thinker.tokenize(sample_prompts)
            semantic_hidden = thinker.forward_hidden(
                tokens["input_ids"], attention_mask=tokens.get("attention_mask")
            )
        except Exception as err:
            raise RuntimeError(
                f"Failed to load requested Thinker '{args.thinker}': {err}"
            ) from err
    else:
        if not args.allow_synthetic_thinker:
            raise ValueError(
                "Pass --thinker or explicitly opt in with --allow-synthetic-thinker"
            )
        semantic_hidden = torch.randn(batch_size, 8, config.semantic_dim, device=device)

    # 4. Prepare audio codes, targets, and conditioning IDs
    targets = torch.randint(
        0, config.codebook_size, (batch_size, K, num_audio_frames), device=device
    )
    audio_codes = torch.full_like(targets, config.pad_token_id)
    audio_codes[:, :, 0] = config.bos_token_id
    audio_codes[:, :, 1:] = targets[:, :, :-1]
    speaker_ids = torch.tensor(
        [0, 1 % config.num_speakers], dtype=torch.long, device=device
    )
    dialect_ids = torch.tensor([0, 0], dtype=torch.long, device=device)

    # 5. Forward Pass
    out = talker(
        semantic_hidden_states=semantic_hidden,
        audio_codes=audio_codes,
        speaker_ids=speaker_ids,
        dialect_ids=dialect_ids,
        targets=targets,
        stop_targets=torch.cat(
            (torch.zeros(batch_size, num_audio_frames - 1), torch.ones(batch_size, 1)),
            dim=1,
        ).to(device),
    )

    # 6. Backward Pass
    loss = out.loss
    if loss is None:
        print("Error: Loss is None", file=sys.stderr)
        sys.exit(1)

    loss.backward()

    # 7. Print verification report
    num_params = sum(p.numel() for p in talker.parameters())
    print("Semantic hidden:       " + str(tuple(semantic_hidden.shape)))
    print("Audio codes:           " + str(tuple(audio_codes.shape)))
    print("Speaker ids:           " + str(speaker_ids.tolist()))
    print("Dialect ids:           " + str(dialect_ids.tolist()))
    print()
    print("Talker:")
    print(f"Parameters:            {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"Hidden size:           {config.talker_dim}")
    print(f"Layers:                {config.num_layers}")
    print(f"Heads:                 {config.num_heads}")
    print()
    print("Output:")
    print("Logits shape:          " + str(tuple(out.logits.shape)))
    print(f"Loss:                  {loss.item():.4f}")
    print("Backward:              OK")


if __name__ == "__main__":
    main()
