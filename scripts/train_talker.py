#!/usr/bin/env python3
"""Fail-fast Talker trainer. No production synthetic-semantic fallback."""

from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from ulm_live.talker import (  # noqa: E402
    TalkerCollator,
    TalkerConfig,
    TalkerDataset,
    ULMTalker,
    build_id_mappings,
    save_training_checkpoint,
    load_training_checkpoint,
)
from ulm_live.thinker import ULMThinker  # noqa: E402
from ulm_live.talker.semantic_cache import thinker_fingerprint  # noqa: E402


def optimizer_step_counts(
    micro_batches: int, grad_accum: int, epochs: int, max_steps: int | None = None
):
    if micro_batches <= 0 or grad_accum <= 0 or epochs <= 0:
        raise ValueError("steps, accumulation, and epochs must be positive")
    per_epoch = math.ceil(micro_batches / grad_accum)
    return per_epoch, (max_steps if max_steps is not None else per_epoch * epochs)


def build_scheduler(optimizer, total_steps, warmup_steps=0, warmup_ratio=0.0):
    warmup = warmup_steps or round(total_steps * warmup_ratio)
    if not 0 <= warmup < total_steps:
        raise ValueError("warmup must be smaller than total steps")

    def factor(step):
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def manifest_has_complete_cache(path):
    rows = [
        json.loads(x)
        for x in Path(path).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    flags = [bool(x.get("semantic_path")) for x in rows]
    if any(flags) and not all(flags):
        raise ValueError(
            "manifest has a partial semantic cache; every row or no row must define semantic_path"
        )
    return bool(rows) and all(flags)


def validate_train_val_split(train_path, val_path):
    def read(path):
        return [
            json.loads(x)
            for x in Path(path).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]

    tr, va = read(train_path), read(val_path)
    train_speakers = {str(x.get("speaker_id")) for x in tr}
    unseen = {str(x.get("speaker_id")) for x in va} - train_speakers

    def keys(rows):
        ids = [str(x.get("utterance_id") or x.get("id")) for x in rows]
        groups = [
            str(
                x.get("session_id")
                or x.get("source_audio_id")
                or x.get("utterance_id")
                or x.get("id")
            )
            for x in rows
        ]
        return ids, groups

    ti, tg = keys(tr)
    vi, vg = keys(va)
    if len(ti) != len(set(ti)) or len(vi) != len(set(vi)):
        raise ValueError("duplicate utterance identity within a split")
    if set(ti) & set(vi) or set(tg) & set(vg):
        raise ValueError(
            "utterance/session/source-clip leakage between train and validation"
        )
    if unseen:
        raise ValueError(f"validation speakers absent from train: {sorted(unseen)}")


def validate_semantic_source(manifest, thinker, allow_synthetic, dry_run):
    cached = manifest_has_complete_cache(manifest)
    if allow_synthetic and not dry_run:
        raise ValueError("--allow-synthetic-thinker is restricted to --dry-run")
    if not thinker and not cached and not allow_synthetic:
        raise ValueError(
            "training requires --thinker or a complete semantic_path cache"
        )
    return cached


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/talker.yaml")
    p.add_argument("--manifest", "-m", required=True)
    p.add_argument("--val-manifest")
    p.add_argument("--thinker", "-t")
    p.add_argument("--semantic-cache-thinker")
    p.add_argument("--semantic-cache-hidden-layer", type=int, default=-1)
    p.add_argument("--output-dir", "-o", default="outputs/talker_checkpoint")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--batch-size", "-b", type=int, default=2)
    p.add_argument("--lr", type=float)
    p.add_argument("--grad-accum", "--gradient-accumulation-steps", type=int, default=1)
    p.add_argument(
        "--strict-codec",
        action="store_true",
        help="Compatibility flag; strict codec validation is always enabled",
    )
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--resume")
    p.add_argument("--device", default="auto")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-synthetic-thinker", action="store_true")
    p.add_argument("--max-eval-batches", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=None
    )
    p.add_argument("--prefetch-factor", type=int)
    p.add_argument("--warmup-steps", type=int, default=0)
    p.add_argument("--warmup-ratio", type=float)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def semantic_batch(batch, thinker, config, device, allow_synthetic=False):
    if "semantic_hidden_states" in batch:
        return batch["semantic_hidden_states"].to(device), batch[
            "semantic_attention_mask"
        ].to(device)
    if thinker is not None:
        mask = batch.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        return thinker.forward_hidden(batch["input_ids"].to(device), mask), mask
    if allow_synthetic:
        b = batch["audio_codes"].shape[0]
        return torch.randn(b, 8, config.semantic_dim, device=device), torch.ones(
            b, 8, dtype=torch.long, device=device
        )
    raise RuntimeError("no real Thinker or validated semantic cache")


def evaluate(
    talker, loader, device, thinker=None, max_eval_batches=None, allow_synthetic=False
):
    was = talker.training
    talker.eval()
    sums = {
        k: 0.0
        for k in (
            "total_loss",
            "codec_loss",
            "stop_loss",
            "stop_accuracy",
            "length_error_frames",
            "codebook_0_ce",
            "mean_residual_ce",
        )
    }
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_eval_batches is not None and i >= max_eval_batches:
                break
            sem, sm = semantic_batch(
                batch, thinker, talker.config, device, allow_synthetic
            )
            kw = {
                k: batch[k].to(device)
                for k in (
                    "audio_codes",
                    "targets",
                    "speaker_ids",
                    "dialect_ids",
                    "audio_attention_mask",
                    "stop_targets",
                )
            }
            out = talker(sem, semantic_attention_mask=sm, **kw)
            valid = kw["stop_targets"] >= 0
            pred = out.stop_logits >= 0
            target = kw["stop_targets"].bool()
            sums["total_loss"] += out.loss.item()
            sums["codec_loss"] += out.codec_loss.item()
            sums["stop_loss"] += out.stop_loss.item()
            sums["stop_accuracy"] += (
                (pred[valid] == target[valid]).float().mean().item()
            )
            sums["codebook_0_ce"] += out.codebook_losses[0].item()
            sums["mean_residual_ce"] += (
                torch.stack(out.codebook_losses[1:]).mean().item()
                if len(out.codebook_losses) > 1
                else 0.0
            )
            n += 1
            target_len = valid.sum(1)
            eligible = (
                torch.arange(out.stop_logits.shape[1], device=device)[None]
                >= talker.config.min_audio_frames - 1
            )
            detected = (
                torch.sigmoid(out.stop_logits) >= talker.config.stop_threshold
            ) & eligible
            first = (
                torch.where(
                    detected,
                    torch.arange(1, out.stop_logits.shape[1] + 1, device=device)[None],
                    out.stop_logits.shape[1] + 1,
                )
                .min(1)
                .values
            )
            first = torch.minimum(
                first, target_len.new_full(first.shape, out.stop_logits.shape[1])
            )
            sums["length_error_frames"] += (
                (first - target_len).abs().float().mean().item()
            )
    talker.train(was)
    if not n:
        raise ValueError("validation loader produced no batches")
    return {k: v / n for k, v in sums.items()} | {"batches": n}


def loader_for(dataset, collator, args, epoch, shuffle):
    workers = args.num_workers
    if workers is None:
        workers = 4 if torch.cuda.is_available() else 0
    gen = torch.Generator().manual_seed(args.seed + epoch)
    kw = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=args.pin_memory,
        generator=gen,
    )
    if workers:
        kw.update(
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
        )
    return DataLoader(**kw)


def main(argv=None):
    args = parse_args(argv)
    manifest = Path(args.manifest)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    cache_mode = validate_semantic_source(
        manifest, args.thinker, args.allow_synthetic_thinker, args.dry_run
    )
    val_cache_mode = (
        manifest_has_complete_cache(args.val_manifest) if args.val_manifest else False
    )
    if (
        args.val_manifest
        and not args.thinker
        and not val_cache_mode
        and not args.allow_synthetic_thinker
    ):
        raise ValueError("validation requires a Thinker or complete semantic cache")
    expected_cache = None
    if cache_mode and not args.thinker:
        if not args.semantic_cache_thinker:
            raise ValueError(
                "cache-only training requires --semantic-cache-thinker to bind cache identity"
            )
        expected_cache = {
            "thinker_id": args.semantic_cache_thinker,
            "thinker_fingerprint": thinker_fingerprint(args.semantic_cache_thinker),
            "hidden_layer": args.semantic_cache_hidden_layer,
        }
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    torch.manual_seed(args.seed)
    if args.val_manifest:
        validate_train_val_split(manifest, args.val_manifest)
    config = TalkerConfig.from_yaml(args.config)
    args.num_workers = (
        config.num_workers if args.num_workers is None else args.num_workers
    )
    args.pin_memory = (
        (config.pin_memory and device.type == "cuda")
        if args.pin_memory is None
        else args.pin_memory
    )
    args.persistent_workers = (
        config.persistent_workers
        if args.persistent_workers is None
        else args.persistent_workers
    )
    args.prefetch_factor = (
        config.prefetch_factor if args.prefetch_factor is None else args.prefetch_factor
    )
    args.lr = config.learning_rate if args.lr is None else args.lr
    args.warmup_ratio = (
        config.warmup_ratio if args.warmup_ratio is None else args.warmup_ratio
    )
    args.max_eval_batches = (
        config.max_eval_batches
        if args.max_eval_batches is None
        else args.max_eval_batches
    )
    if config.best_metric not in {"total_loss", "codec_loss", "stop_loss"}:
        raise ValueError(f"unsupported best checkpoint metric: {config.best_metric}")
    paths = [manifest] + ([Path(args.val_manifest)] if args.val_manifest else [])
    speaker2id, dialect2id = build_id_mappings(paths)
    config.num_speakers = max(config.num_speakers, len(speaker2id))
    config.num_dialects = max(config.num_dialects, len(dialect2id))
    thinker = None
    if args.thinker:
        try:
            thinker = ULMThinker(
                args.thinker,
                device=str(device),
                torch_dtype="bf16" if args.bf16 else "auto",
                freeze=True,
            )
            config.semantic_dim = thinker.hidden_size
        except Exception as e:
            raise RuntimeError(
                f"failed to load requested Thinker '{args.thinker}': {e}"
            ) from e
        if cache_mode or val_cache_mode:
            expected_cache = {
                "thinker_id": args.thinker,
                "thinker_fingerprint": thinker_fingerprint(args.thinker),
                "hidden_layer": thinker.hidden_layer,
            }
    tokenizer = thinker.tokenizer if thinker else None
    train = TalkerDataset(
        manifest,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=config.num_quantizers,
        cache_only=cache_mode and thinker is None,
        semantic_cache_metadata=expected_cache,
    )
    if cache_mode:
        cached_dim = train[0]["semantic_hidden_states"].shape[-1]
        if thinker is not None and cached_dim != thinker.hidden_size:
            raise ValueError(
                f"semantic cache dim {cached_dim} does not match Thinker dim {thinker.hidden_size}"
            )
        config.semantic_dim = cached_dim
    collator = TalkerCollator(
        tokenizer,
        config.pad_token_id,
        config.bos_token_id,
        config.max_seq_len,
        acoustic_delay_frames=config.acoustic_delay_frames,
    )
    probe = loader_for(train, collator, args, 0, True)
    per_epoch, total_steps = optimizer_step_counts(
        len(probe), args.grad_accum, args.epochs, args.max_steps
    )
    run_epochs = (
        math.ceil(total_steps / per_epoch)
        if args.max_steps is not None
        else args.epochs
    )
    model = ULMTalker(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, total_steps, args.warmup_steps, args.warmup_ratio
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.fp16)
    val = None
    if args.val_manifest:
        val = TalkerDataset(
            args.val_manifest,
            speaker2id=speaker2id,
            dialect2id=dialect2id,
            num_quantizers=config.num_quantizers,
            cache_only=val_cache_mode and thinker is None,
            semantic_cache_metadata=expected_cache if val_cache_mode else None,
        )
        if (
            val_cache_mode
            and val[0]["semantic_hidden_states"].shape[-1] != config.semantic_dim
        ):
            raise ValueError(
                "validation semantic cache dimension does not match training"
            )
    start_epoch = start_pos = global_step = 0
    best = {"metric": config.best_metric, "value": float("inf"), "step": None}
    if args.resume:
        state = load_training_checkpoint(
            args.resume, model, optimizer, scheduler, scaler
        )
        global_step = state["global_step"]
        start_epoch = state["epoch"]
        start_pos = state["micro_batch_position"]
        best = state["best_validation"]
        if state["speaker2id"] != speaker2id or state["dialect2id"] != dialect2id:
            raise ValueError("checkpoint ID mappings do not match manifests")
        previous_args = state.get("training_args", {})
        for key in ("batch_size", "grad_accum", "seed", "epochs", "max_steps"):
            if key in previous_args and previous_args[key] != getattr(args, key):
                raise ValueError(f"resume argument mismatch for {key}")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad()
    started = time.time()
    last_loss = None
    if global_step >= total_steps:
        print(f"checkpoint already reached configured total steps ({total_steps})")
        return
    for epoch in range(start_epoch, run_epochs):
        loader = loader_for(train, collator, args, epoch, True)
        accumulation = 0
        for pos, batch in enumerate(loader):
            if epoch == start_epoch and pos < start_pos:
                continue
            sem, sm = semantic_batch(
                batch, thinker, config, device, args.allow_synthetic_thinker
            )
            kw = {
                k: batch[k].to(device)
                for k in (
                    "audio_codes",
                    "targets",
                    "speaker_ids",
                    "dialect_ids",
                    "audio_attention_mask",
                    "stop_targets",
                )
            }
            dtype = torch.bfloat16 if args.bf16 else torch.float16
            with torch.autocast(
                device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and (args.bf16 or args.fp16),
            ):
                result = model(sem, semantic_attention_mask=sm, **kw)
                loss = result.loss / args.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch}, batch={pos}"
                )
            scaler.scale(loss).backward()
            accumulation += 1
            last_loss = result.loss.item()
            boundary = accumulation == args.grad_accum or pos + 1 == len(loader)
            if not boundary:
                continue
            scaler.unscale_(optimizer)
            if accumulation < args.grad_accum:
                correction = args.grad_accum / accumulation
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            accumulation = 0
            next_epoch, next_pos = (
                (epoch + 1, 0) if pos + 1 == len(loader) else (epoch, pos + 1)
            )
            if args.dry_run:
                print(f"dry-run OK: loss={last_loss:.4f}")
                return
            if val is not None and global_step % args.eval_steps == 0:
                metrics = evaluate(
                    model,
                    loader_for(val, collator, args, epoch, False),
                    device,
                    thinker,
                    args.max_eval_batches,
                )
                print(json.dumps({"step": global_step, "validation": metrics}))
                if metrics[best["metric"]] < best["value"]:
                    best = {
                        "metric": config.best_metric,
                        "value": metrics[config.best_metric],
                        "step": global_step,
                    }
                    save_training_checkpoint(
                        outdir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        global_step=global_step,
                        epoch=next_epoch,
                        micro_batch_position=next_pos,
                        accumulation_step=0,
                        speaker2id=speaker2id,
                        dialect2id=dialect2id,
                        training_args=vars(args),
                        best_validation=best,
                    )
            if global_step % args.save_steps == 0:
                save_training_checkpoint(
                    outdir / f"checkpoint-{global_step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    global_step=global_step,
                    epoch=next_epoch,
                    micro_batch_position=next_pos,
                    accumulation_step=0,
                    speaker2id=speaker2id,
                    dialect2id=dialect2id,
                    training_args=vars(args),
                    best_validation=best,
                )
            if global_step >= total_steps:
                break
        start_pos = 0
        if global_step >= total_steps:
            break
    save_training_checkpoint(
        outdir / "final.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        global_step=global_step,
        epoch=run_epochs,
        micro_batch_position=0,
        accumulation_step=0,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        training_args=vars(args),
        best_validation=best,
    )
    print(
        json.dumps(
            {
                "steps": global_step,
                "last_loss": last_loss,
                "seconds": time.time() - started,
                "parameters": model.parameter_breakdown(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
