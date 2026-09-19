import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

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
        description="Training entrypoint for ULMTalker prototype with checkpointing and validation."
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
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides epochs if reached).",
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
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (default: 1).",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="Save periodic checkpoint every N optimizer steps (default: 100).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate on validation set every N optimizer steps (default: 100).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt to resume training from.",
    )
    parser.add_argument(
        "--strict-codec",
        action="store_true",
        default=True,
        help="Fail fast if any dataset item has missing or invalid codec tokens (default: True).",
    )
    parser.add_argument(
        "--no-strict-codec",
        action="store_false",
        dest="strict_codec",
        help="Allow fallback zero tokens for missing codecs.",
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


def evaluate(
    talker: ULMTalker,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: Any,
    thinker: ULMThinker | None,
    config: TalkerConfig,
    max_eval_batches: int = 50,
) -> float:
    """Evaluate mean validation loss."""
    talker.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx >= max_eval_batches:
                break
            audio_codes = batch["audio_codes"].to(device)
            targets = batch["targets"].to(device)
            speaker_ids = batch["speaker_ids"].to(device)
            dialect_ids = batch["dialect_ids"].to(device)

            if thinker is not None and "input_ids" in batch:
                input_ids = batch["input_ids"].to(device)
                att_mask = batch.get("attention_mask")
                if att_mask is not None:
                    att_mask = att_mask.to(device)
                semantic_hidden = thinker.forward_hidden(input_ids, attention_mask=att_mask).float()
            else:
                B = audio_codes.shape[0]
                semantic_hidden = torch.randn(B, 8, config.semantic_dim, device=device)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = talker(
                        semantic_hidden_states=semantic_hidden,
                        audio_codes=audio_codes,
                        speaker_ids=speaker_ids,
                        dialect_ids=dialect_ids,
                        targets=targets,
                    )
            else:
                out = talker(
                    semantic_hidden_states=semantic_hidden,
                    audio_codes=audio_codes,
                    speaker_ids=speaker_ids,
                    dialect_ids=dialect_ids,
                    targets=targets,
                )

            total_loss += out.loss.item()
            num_batches += 1

    talker.train()
    return total_loss / num_batches if num_batches > 0 else float("inf")


def save_checkpoint(
    path: Path,
    talker: ULMTalker,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    epoch: int,
    val_loss: float | None,
    config: TalkerConfig,
    speaker2id: dict[str, int],
    dialect2id: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": talker.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config.to_dict(),
        "speaker2id": speaker2id,
        "dialect2id": dialect2id,
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"Error: Training manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_steps is not None:
        # Allow training loop to continue across epochs until max_steps is reached
        args.epochs = 999999

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
    val_manifest_path = Path(args.val_manifest) if args.val_manifest else None
    if val_manifest_path and val_manifest_path.is_file():
        manifest_files.append(val_manifest_path)

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
            raise RuntimeError(f"Critical: Failed to load Thinker model from '{args.thinker}': {err}")

    talker = ULMTalker(config).to(device)
    optimizer = torch.optim.AdamW(
        talker.parameters(),
        lr=args.lr,
        weight_decay=config.to_dict().get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.max_steps or 1000, 100))

    start_step = 0
    start_epoch = 0
    best_val_loss = float("inf")

    # Resume from checkpoint if specified
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_file():
            print(f"Resuming training from {resume_path}...")
            ckpt = torch.load(resume_path, map_location=device)
            talker.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"]:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"Resumed at step {start_step}, epoch {start_epoch}, best_val_loss: {best_val_loss}")
        else:
            raise FileNotFoundError(f"Checkpoint for resume not found: {resume_path}")

    # 4. Dataset & DataLoader
    tokenizer = thinker.tokenizer if thinker is not None else None
    train_dataset = TalkerDataset(
        manifest_path=manifest_path,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=config.num_quantizers,
        strict_codec=args.strict_codec,
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

    val_loader = None
    if val_manifest_path and val_manifest_path.is_file():
        val_dataset = TalkerDataset(
            manifest_path=val_manifest_path,
            speaker2id=speaker2id,
            dialect2id=dialect2id,
            num_quantizers=config.num_quantizers,
            strict_codec=args.strict_codec,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    print(f"\n=== ULMTalker Training Setup ===")
    print(f"Model parameters:    {sum(p.numel() for p in talker.parameters()):,}")
    print(f"Device:              {device} (AMP: {use_amp}, dtype: {amp_dtype})")
    print(f"Dataset samples:     {len(train_dataset)}")
    if val_loader:
        print(f"Val samples:         {len(val_loader.dataset)}")
    print(f"Batch size:          {args.batch_size}")
    print(f"Effective batch:     {args.batch_size * args.grad_accum}")
    print(f"Learning rate:       {args.lr}")
    print(f"Max steps:           {args.max_steps}")
    print(f"Strict codec:        {args.strict_codec}")
    print(f"Dry-run mode:        {args.dry_run}")
    print()

    global_step = start_step
    batch_step = 0
    talker.train()
    start_train_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            batch_step += 1
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

            curr_loss = (loss * args.grad_accum).item()
            if math.isnan(curr_loss) or math.isinf(curr_loss):
                raise FloatingPointError(f"Loss is NaN or Inf at batch {batch_step}: {curr_loss}")

            is_opt_step = (batch_step % args.grad_accum == 0)
            if is_opt_step:
                torch.nn.utils.clip_grad_norm_(talker.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 5 == 0 or global_step == 1 or args.dry_run:
                    print(f"[Epoch {epoch+1} | Step {global_step}] Loss: {curr_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6e}")

                if args.dry_run:
                    print("\n[Dry-run Verification]")
                    print("  Forward pass:  OK")
                    print(f"  Loss:          {curr_loss:.4f}")
                    print("  Backward pass: OK")
                    print("Dry-run finished successfully.")
                    return

                # Validation evaluation
                if val_loader is not None and global_step % args.eval_steps == 0:
                    val_loss = evaluate(talker, val_loader, device, use_amp, amp_dtype, thinker, config)
                    print(f"--> [Validation Step {global_step}] Val Loss: {val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_path = output_dir / "best.pt"
                        save_checkpoint(best_path, talker, optimizer, scheduler, global_step, epoch, best_val_loss, config, speaker2id, dialect2id)
                        print(f"    New best validation loss! Saved to {best_path}")

                # Periodic checkpoint saving
                if global_step % args.save_steps == 0:
                    ckpt_p = output_dir / f"checkpoint-{global_step}.pt"
                    save_checkpoint(ckpt_p, talker, optimizer, scheduler, global_step, epoch, best_val_loss, config, speaker2id, dialect2id)
                    print(f"    Saved periodic checkpoint to {ckpt_p}")

                # Max steps check (e.g. for smoke testing)
                if args.max_steps is not None and global_step >= args.max_steps:
                    print(f"\nReached max_steps ({args.max_steps} optimizer steps). Finishing training loop.")
                    break

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    total_time = time.time() - start_train_time
    # Save final checkpoints
    final_path = output_dir / "talker_final.pt"
    save_checkpoint(final_path, talker, optimizer, scheduler, global_step, epoch, best_val_loss, config, speaker2id, dialect2id)
    # Also save as final.pt
    save_checkpoint(output_dir / "final.pt", talker, optimizer, scheduler, global_step, epoch, best_val_loss, config, speaker2id, dialect2id)

    # Save training state summary json
    trainer_state = {
        "final_step": global_step,
        "epochs_completed": epoch + 1,
        "runtime_seconds": total_time,
        "steps_per_second": global_step / total_time if total_time > 0 else 0,
        "seconds_per_step": total_time / global_step if global_step > 0 else 0,
        "best_val_loss": best_val_loss if not math.isinf(best_val_loss) else None,
        "last_train_loss": curr_loss,
    }
    with open(output_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)

    print(f"\nTraining completed in {total_time:.2f}s ({global_step} steps). Saved checkpoints to: {output_dir}")


if __name__ == "__main__":
    main()
