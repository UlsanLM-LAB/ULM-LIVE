import argparse
from pathlib import Path
import sys
import time

# Ensure repo root is on sys.path for direct script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from torch.utils.data import DataLoader

from ulm_live.talker import (
    TalkerCollator,
    TalkerConfig,
    TalkerDataset,
    ULMTalker,
    build_id_mappings,
)
from ulm_live.thinker import ULMThinker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training entrypoint for ULMTalker prototype."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="configs/talker.yaml",
        help="Path to Talker configuration YAML.",
    )
    parser.add_argument(
        "--manifest",
        "-m",
        type=str,
        required=True,
        help="Path to training manifest JSONL file.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=None,
        help="Path to validation manifest JSONL file (optional).",
    )
    parser.add_argument(
        "--thinker",
        "-t",
        type=str,
        default=None,
        help="Path or identifier for ULM Thinker model. If omitted, uses synthetic semantic states for fast testing.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/talker_checkpoint",
        help="Directory to save Talker checkpoints and mapping files.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=2,
        help="Training batch size (default: 2).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4).",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1).",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        help="Execution device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 mixed precision.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 mixed precision.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 step forward and backward, then exit with verification report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Error: Training manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Device and precision
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    amp_dtype = None
    if args.bf16:
        amp_dtype = torch.bfloat16
    elif args.fp16:
        amp_dtype = torch.float16

    use_amp = (device.type == "cuda" and amp_dtype is not None)

    # 2. Config & ID mappings
    cfg_path = Path(args.config)
    config = TalkerConfig.from_yaml(cfg_path) if cfg_path.is_file() else TalkerConfig()

    manifest_files = [manifest_path]
    if args.val_manifest and Path(args.val_manifest).is_file():
        manifest_files.append(Path(args.val_manifest))

    speaker2id, dialect2id = build_id_mappings(manifest_files, output_dir=output_dir)
    config.num_speakers = max(config.num_speakers, len(speaker2id) + 1)
    config.num_dialects = max(config.num_dialects, len(dialect2id) + 1)

    # 3. Model & Thinker
    thinker: ULMThinker | None = None
    if args.thinker is not None:
        print(f"Loading Thinker model from {args.thinker}...")
        try:
            thinker = ULMThinker(
                model_name_or_path=args.thinker,
                device=str(device),
                torch_dtype="auto",
                freeze=True,
                load_pretrained=True,
            )
            config.semantic_dim = thinker.hidden_size
        except Exception as err:
            print(f"Warning: Failed to load Thinker model: {err}. Using synthetic semantic hidden states.", file=sys.stderr)
            thinker = None

    talker = ULMTalker(config).to(device)
    optimizer = torch.optim.AdamW(
        talker.parameters(),
        lr=args.lr,
        weight_decay=config.to_dict().get("weight_decay", 0.01),
    )

    # 4. Dataset & DataLoader
    tokenizer = thinker.tokenizer if thinker is not None else None
    train_dataset = TalkerDataset(
        manifest_path=manifest_path,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=config.num_quantizers,
    )
    collator = TalkerCollator(
        tokenizer=tokenizer,
        pad_token_id=config.pad_token_id,
        max_audio_len=config.max_seq_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    print(f"\n=== ULMTalker Training Setup ===")
    print(f"Model parameters:    {sum(p.numel() for p in talker.parameters()):,}")
    print(f"Device:              {device} (AMP: {use_amp}, dtype: {amp_dtype})")
    print(f"Dataset samples:     {len(train_dataset)}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Learning rate:       {args.lr}")
    print(f"Dry-run mode:        {args.dry_run}")
    print()

    step = 0
    talker.train()

    for epoch in range(args.epochs):
        for batch in train_loader:
            step += 1
            audio_codes = batch["audio_codes"].to(device)
            targets = batch["targets"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            dialect_ids = batch["dialect_ids"].to(device)

            # Obtain semantic hidden states
            if thinker is not None and "input_ids" in batch:
                input_ids = batch["input_ids"].to(device)
                att_mask = batch.get("attention_mask")
                if att_mask is not None:
                    att_mask = att_mask.to(device)
                semantic_hidden = thinker.forward_hidden(input_ids, attention_mask=att_mask).float()
            else:
                B = audio_codes.shape[0]
                semantic_hidden = torch.randn(B, 8, config.semantic_dim, device=device)

            # Forward with optional AMP
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = talker(
                        semantic_hidden_states=semantic_hidden,
                        audio_codes=audio_codes,
                        speaker_ids=speaker_ids,
                        dialect_ids=dialect_ids,
                        targets=targets,
                    )
                    loss = out.loss / args.grad_accum
            else:
                out = talker(
                    semantic_hidden_states=semantic_hidden,
                    audio_codes=audio_codes,
                    speaker_ids=speaker_ids,
                    dialect_ids=dialect_ids,
                    targets=targets,
                )
                loss = out.loss / args.grad_accum

            loss.backward()

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(talker.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            curr_loss = (loss * args.grad_accum).item()
            print(f"[Epoch {epoch+1} | Step {step}] Loss: {curr_loss:.4f} | Logits shape: {tuple(out.logits.shape)}")

            if args.dry_run:
                print("\n[Dry-run Verification]")
                print("  Forward pass:  OK")
                print(f"  Loss:          {curr_loss:.4f}")
                print("  Backward pass: OK")
                print("Dry-run finished successfully.")
                return

    # Save final checkpoint
    ckpt_path = output_dir / "talker_final.pt"
    torch.save(
        {
            "model_state_dict": talker.state_dict(),
            "config": config.to_dict(),
            "speaker2id": speaker2id,
            "dialect2id": dialect2id,
        },
        ckpt_path,
    )
    print(f"\nTraining completed. Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
