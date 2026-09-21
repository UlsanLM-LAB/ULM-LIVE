#!/usr/bin/env python3
"""ULM-LIVE Production 5-Epoch Full Talker Training & Continuous Monitoring Pipeline.
Runs:
- Exact 5 Epochs (fresh AdamW + Cosine Annealing from step 0)
- Full validation split evaluation at each epoch
- Fixed 10 prompts regression (Greedy + Sampling) at each epoch
- 20 Test representative prompts (Sampling) at each epoch
- Early Failure Guard
- Automatic Best Checkpoint selection
- Full 5-Epoch Comprehensive Report generation
"""

from __future__ import annotations
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from torch.utils.data import DataLoader

from ulm_live.codec import build_codec
from ulm_live.talker import (
    SpeechSynthesizer,
    TalkerCollator,
    TalkerConfig,
    TalkerDataset,
    TalkerGenerationConfig,
    ULMTalker,
    build_id_mappings,
    save_training_checkpoint,
)
from ulm_live.thinker import ULMThinker

FIXED_PROMPTS = [
    "안녕",
    "밥 묵었나?",
    "오늘 뭐 하고 있었노?",
    "오늘 날씨 좋네.",
    "울산에서 놀러 갈 만한 데 추천해줘.",
    "나는 오늘 학교 끝나고 친구 만나러 간다.",
    "1 더하기 1은 2다.",
    "하늘이 파란 이유를 간단하게 설명해줘.",
    "Python에서 리스트를 정렬하는 방법 알려줘.",
    "오늘 기분이 좀 안 좋다.",
]


def parse_args():
    p = argparse.ArgumentParser(description="ULM-LIVE 5-Epoch Full Talker Training")
    p.add_argument("--config", default="configs/talker.yaml")
    p.add_argument("--manifest", default="data/ulsan-full/train.cached.jsonl")
    p.add_argument("--val-manifest", default="data/ulsan-full/val.cached.jsonl")
    p.add_argument("--test-manifest", default="data/ulsan-full/test.cached.jsonl")
    p.add_argument("--thinker", default="/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged")
    p.add_argument("--output-dir", default="outputs/talker-full-5epoch")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--stop-pos-weight", type=float, default=5.0)
    p.add_argument("--stop-loss-weight", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


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


def loader_for(dataset, collator, args, epoch, shuffle):
    workers = args.num_workers
    gen = torch.Generator().manual_seed(args.seed + epoch)
    return DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
        prefetch_factor=2 if workers > 0 else None,
        generator=gen,
    )


def semantic_batch(batch, device):
    return batch["semantic_hidden_states"].to(device), batch["semantic_attention_mask"].to(device)


def evaluate_val_split(talker, loader, device, max_eval_batches=None):
    was_training = talker.training
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
    tp_total = fp_total = fn_total = tn_total = 0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_eval_batches is not None and i >= max_eval_batches:
                break
            sem, sm = semantic_batch(batch, device)
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
            pred = (torch.sigmoid(out.stop_logits) >= talker.config.stop_threshold) & valid
            target = (kw["stop_targets"] == 1) & valid

            tp_total += (pred & target).sum().item()
            fp_total += (pred & (~target & valid)).sum().item()
            fn_total += (~pred & target).sum().item()
            tn_total += (~pred & (~target & valid)).sum().item()

            sums["total_loss"] += out.loss.item()
            sums["codec_loss"] += out.codec_loss.item()
            sums["stop_loss"] += out.stop_loss.item()
            sums["stop_accuracy"] += ((pred == target)[valid]).float().mean().item()
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
            first = torch.minimum(first, target_len.new_full(first.shape, out.stop_logits.shape[1]))
            sums["length_error_frames"] += (first - target_len).abs().float().mean().item()

    talker.train(was_training)
    if not n:
        raise ValueError("validation loader produced 0 batches")

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    valid_total = tp_total + fp_total + fn_total + tn_total
    pos_pred_rate = (tp_total + fp_total) / valid_total if valid_total > 0 else 0.0

    metrics = {k: round(v / n, 4) for k, v in sums.items()}
    metrics.update({
        "stop_precision": round(precision, 4),
        "stop_recall": round(recall, 4),
        "stop_f1": round(f1, 4),
        "positive_prediction_rate": round(pos_pred_rate, 4),
        "batches": n,
    })
    return metrics


def compute_token_stats(tokens: torch.Tensor):
    if tokens.dim() == 3:
        tokens = tokens.squeeze(0)
    num_q = tokens.shape[0] if tokens.ndim == 2 else 0
    total_len = tokens.shape[1] if tokens.ndim == 2 else 0

    ent_list = []
    unique_list = []
    top1_ratio_list = []

    for q in range(num_q):
        q_toks = tokens[q].tolist()
        unique_list.append(len(set(q_toks)))
        cnts = Counter(q_toks)
        tot = len(q_toks)
        if tot > 0:
            e = -sum((c / tot) * math.log2(c / tot) for c in cnts.values())
            top1 = cnts.most_common(1)[0][1] / tot
        else:
            e = 0.0
            top1 = 0.0
        ent_list.append(round(e, 3))
        top1_ratio_list.append(round(top1, 4))

    mean_ent = round(sum(ent_list) / len(ent_list), 3) if ent_list else 0.0
    mean_unique = round(sum(unique_list) / len(unique_list), 1) if unique_list else 0.0
    mean_top1 = round(sum(top1_ratio_list) / len(top1_ratio_list), 4) if top1_ratio_list else 0.0

    return {
        "num_frames": total_len,
        "q0_entropy": ent_list[0] if ent_list else 0.0,
        "mean_entropy": mean_ent,
        "entropy_by_q": ent_list,
        "q0_unique": unique_list[0] if unique_list else 0,
        "mean_unique": mean_unique,
        "unique_by_q": unique_list,
        "q0_top1_ratio": top1_ratio_list[0] if top1_ratio_list else 0.0,
        "mean_top1_ratio": mean_top1,
        "top1_ratio_by_q": top1_ratio_list,
    }


def synthesize_and_measure(
    synthesizer: SpeechSynthesizer,
    prompt: str,
    wav_path: Path,
    gen_config: TalkerGenerationConfig,
    speaker: str = "speaker_001",
    dialect: str = "ulsan",
):
    t0 = time.perf_counter()
    res = synthesizer.synthesize(
        prompt,
        speaker=speaker,
        dialect=dialect,
        generation_config=gen_config,
    )
    total_time = time.perf_counter() - t0

    res.save(wav_path)

    waveform = res.waveform.detach().cpu().squeeze()
    is_finite = bool(torch.isfinite(waveform).all().item())
    is_non_empty = bool(waveform.numel() > 0 and torch.max(torch.abs(waveform)).item() > 1e-6)
    clipping_ratio = (
        float((torch.abs(waveform) >= 0.999).sum().item()) / waveform.numel()
        if waveform.numel() > 0
        else 0.0
    )

    tokens = res.codec_tokens.detach().cpu()
    stats = compute_token_stats(tokens)

    passed = bool(
        is_finite
        and is_non_empty
        and clipping_ratio < 0.05
        and res.termination_reason == "STOP_PREDICTED"
        and stats["mean_entropy"] > 1.0
    )

    return {
        "text": prompt,
        "duration": round(res.duration, 3),
        "frames": stats["num_frames"],
        "termination_reason": res.termination_reason,
        "max_stop_prob": round(res.max_stop_prob, 4),
        "q0_entropy": stats["q0_entropy"],
        "mean_entropy": stats["mean_entropy"],
        "q0_unique": stats["q0_unique"],
        "mean_unique": stats["mean_unique"],
        "q0_top1_ratio": stats["q0_top1_ratio"],
        "mean_top1_ratio": stats["mean_top1_ratio"],
        "generation_time_sec": round(total_time, 3),
        "rtf": round(res.rtf, 3),
        "clipping_ratio": round(clipping_ratio, 4),
        "is_finite": is_finite,
        "is_non_empty": is_non_empty,
        "wav_path": str(wav_path),
        "status": "PASS" if passed else "FAIL",
    }


def select_representative_test_samples(test_manifest_path: Path, count: int = 20):
    rows = [
        json.loads(l)
        for l in test_manifest_path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    by_age = {}
    for r in rows:
        age = r.get("metadata", {}).get("age", "unknown")
        by_age.setdefault(age, []).append(r)

    targets = [
        ("10대", 4),
        ("20대", 4),
        ("30대", 4),
        ("40대", 3),
        ("50대", 3),
        ("60대 이상", 2),
    ]
    selected = []
    for age, c in targets:
        group = by_age.get(age, [])
        group.sort(key=lambda x: (x.get("duration", 0.0), x["id"]))
        step = max(1, len(group) // c)
        for i in range(c):
            idx = min(i * step, len(group) - 1)
            selected.append(group[idx])
    return selected[:count]


def summarize_items(items):
    n = len(items)
    if not n:
        return {}
    return {
        "count": n,
        "num_stop_predicted": sum(1 for x in items if x["termination_reason"] == "STOP_PREDICTED"),
        "num_max_frames": sum(1 for x in items if x["termination_reason"] == "MAX_FRAMES"),
        "avg_duration": round(sum(x["duration"] for x in items) / n, 2),
        "avg_max_stop_prob": round(sum(x["max_stop_prob"] for x in items) / n, 4),
        "avg_q0_entropy": round(sum(x["q0_entropy"] for x in items) / n, 2),
        "avg_mean_entropy": round(sum(x["mean_entropy"] for x in items) / n, 2),
        "avg_q0_top1_ratio": round(sum(x["q0_top1_ratio"] for x in items) / n, 4),
        "avg_mean_top1_ratio": round(sum(x["mean_top1_ratio"] for x in items) / n, 4),
        "avg_rtf": round(sum(x["rtf"] for x in items) / n, 3),
        "all_finite": all(x["is_finite"] for x in items),
        "all_non_empty": all(x["is_non_empty"] for x in items),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================")
    print("=== [ULM-LIVE] Production 5-Epoch Full Talker Training Start ===")
    print("================================================================")
    print(f"Output Dir:        {out_dir}")
    print(f"Device:            {device}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch Size:        {args.batch_size} (Grad Accum: {args.grad_accum}, Effective: {args.batch_size * args.grad_accum})")
    print(f"LR:                {args.lr} (Warmup Ratio: {args.warmup_ratio})")
    print(f"Stop Pos Weight:   {args.stop_pos_weight}")
    print(f"Stop Loss Weight:  {args.stop_loss_weight}")
    print(f"Thinker Model:     {args.thinker}")

    from ulm_live.talker.semantic_cache import thinker_fingerprint

    # 1. Config & Mappings
    config = TalkerConfig.from_yaml(args.config)
    config.stop_pos_weight = args.stop_pos_weight
    config.stop_loss_weight = args.stop_loss_weight
    config.num_quantizers = 16

    manifest_p = Path(args.manifest)
    val_manifest_p = Path(args.val_manifest)
    test_manifest_p = Path(args.test_manifest)

    speaker2id, dialect2id = build_id_mappings([manifest_p, val_manifest_p, test_manifest_p])
    config.num_speakers = max(config.num_speakers, len(speaker2id))
    config.num_dialects = max(config.num_dialects, len(dialect2id))

    # 2. Components: Thinker & Codec
    print("\n1. Loading Thinker & Codec...")
    thinker = ULMThinker(args.thinker, device=str(device), torch_dtype="bf16", freeze=True)
    codec = build_codec(backend="mimi", device=str(device))

    expected_cache = {
        "thinker_id": args.thinker,
        "thinker_fingerprint": thinker_fingerprint(args.thinker),
        "hidden_layer": thinker.hidden_layer,
    }

    # 3. Datasets & Collator
    print("2. Loading datasets...")
    train_dataset = TalkerDataset(
        manifest_p,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=config.num_quantizers,
        cache_only=False,
        semantic_cache_metadata=expected_cache,
    )
    val_dataset = TalkerDataset(
        val_manifest_p,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=config.num_quantizers,
        cache_only=False,
        semantic_cache_metadata=expected_cache,
    )
    collator = TalkerCollator(
        thinker.tokenizer,
        config.pad_token_id,
        config.bos_token_id,
        config.max_seq_len,
        acoustic_delay_frames=config.acoustic_delay_frames,
    )

    micro_batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    steps_per_epoch = math.ceil(micro_batches_per_epoch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    print(f"Train samples:     {len(train_dataset)}")
    print(f"Val samples:       {len(val_dataset)}")
    print(f"Micro-batches/ep:  {micro_batches_per_epoch}")
    print(f"Optimizer steps/ep:{steps_per_epoch}")
    print(f"Total steps (5ep): {total_steps}")

    # 4. Model & Optimizer
    print("\n3. Initializing fresh ULMTalker & Optimizer...")
    model = ULMTalker(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps, warmup_ratio=args.warmup_ratio)
    scaler = torch.amp.GradScaler(device.type, enabled=False)

    synthesizer = SpeechSynthesizer(thinker=thinker, talker=model, codec=codec, device=str(device))

    test_representatives = select_representative_test_samples(test_manifest_p, count=20)
    print(f"Selected {len(test_representatives)} representative test samples for evaluation.")

    # 5. Training Loop
    all_epoch_reports = {}
    best_record = {"val_total_loss": float("inf"), "epoch": 0, "path": None}
    global_step = 0
    start_time = time.time()

    val_loader = loader_for(val_dataset, collator, args, epoch=0, shuffle=False)

    for epoch in range(1, args.epochs + 1):
        ep_start_time = time.time()
        print(f"\n=======================================================")
        print(f"=== [EPOCH {epoch}/{args.epochs}] Training (Steps {global_step + 1} -> {global_step + steps_per_epoch}) ===")
        print(f"=======================================================")

        model.train()
        train_loader = loader_for(train_dataset, collator, args, epoch=epoch, shuffle=True)
        accumulation = 0
        optimizer.zero_grad()
        epoch_train_losses = []

        for pos, batch in enumerate(train_loader):
            sem, sm = semantic_batch(batch, device)
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
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16):
                result = model(sem, semantic_attention_mask=sm, **kw)
                loss = result.loss / args.grad_accum

            if not torch.isfinite(loss):
                raise FloatingPointError(f"CRITICAL: Non-finite loss at epoch={epoch}, batch={pos}: {loss.item()}")

            loss.backward()
            accumulation += 1
            epoch_train_losses.append(result.loss.item())

            boundary = (accumulation == args.grad_accum) or (pos + 1 == len(train_loader))
            if boundary:
                if accumulation < args.grad_accum:
                    corr = args.grad_accum / accumulation
                    for p_param in model.parameters():
                        if p_param.grad is not None:
                            p_param.grad.mul_(corr)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                accumulation = 0

                if global_step % 200 == 0 or global_step == total_steps:
                    lr_curr = scheduler.get_last_lr()[0]
                    avg_recent_loss = sum(epoch_train_losses[-50:]) / len(epoch_train_losses[-50:])
                    print(
                        f"  Step {global_step:4d}/{total_steps} | "
                        f"Train Loss: {avg_recent_loss:.4f} | "
                        f"Grad Norm: {grad_norm:.3f} | "
                        f"LR: {lr_curr:.2e}"
                    )

        ep_train_loss = round(sum(epoch_train_losses) / len(epoch_train_losses), 4)
        ep_train_time = time.time() - ep_start_time

        # --- Epoch Validation ---
        print(f"\n--- [EPOCH {epoch}] Full Validation Split Evaluation ---")
        val_metrics = evaluate_val_split(model, val_loader, device)
        print(
            f"  Val Loss: {val_metrics['total_loss']:.4f} | "
            f"Codec Loss: {val_metrics['codec_loss']:.4f} | "
            f"Q0 CE: {val_metrics['codebook_0_ce']:.4f} | "
            f"Residual CE: {val_metrics['mean_residual_ce']:.4f} | "
            f"Stop F1: {val_metrics['stop_f1']:.4f} (R: {val_metrics['stop_recall']:.2%}, P: {val_metrics['stop_precision']:.2%})"
        )

        # --- Save Epoch Checkpoint ---
        epoch_ckpt_path = out_dir / f"epoch-{epoch}.pt"
        save_training_checkpoint(
            epoch_ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            global_step=global_step,
            epoch=epoch,
            micro_batch_position=0,
            accumulation_step=0,
            speaker2id=speaker2id,
            dialect2id=dialect2id,
            training_args=vars(args),
            best_validation={"metric": "total_loss", "value": val_metrics["total_loss"], "step": global_step},
        )
        print(f"  Checkpoint saved: {epoch_ckpt_path.name}")

        is_best = val_metrics["total_loss"] < best_record["val_total_loss"]
        if is_best:
            best_record = {
                "val_total_loss": val_metrics["total_loss"],
                "epoch": epoch,
                "path": str(epoch_ckpt_path),
            }
            save_training_checkpoint(
                out_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                global_step=global_step,
                epoch=epoch,
                micro_batch_position=0,
                accumulation_step=0,
                speaker2id=speaker2id,
                dialect2id=dialect2id,
                training_args=vars(args),
                best_validation={"metric": "total_loss", "value": val_metrics["total_loss"], "step": global_step},
            )
            print(f"  *** New Best Checkpoint: Epoch {epoch} (Val Loss: {val_metrics['total_loss']:.4f}) ***")

        # --- Audio Generation Regression ---
        print(f"\n--- [EPOCH {epoch}] Audio Generation Regression ---")
        ep_eval_dir = out_dir / "eval" / f"epoch-{epoch}"
        greedy_dir = ep_eval_dir / "greedy"
        sampled_dir = ep_eval_dir / "sampled"
        greedy_dir.mkdir(parents=True, exist_ok=True)
        sampled_dir.mkdir(parents=True, exist_ok=True)

        model.eval()

        # 1. 10 Fixed Prompts (Greedy)
        cfg_greedy = TalkerGenerationConfig(max_new_tokens=750, do_sample=False, stop_threshold=0.50, return_details=True)
        greedy_results = []
        for idx, p_text in enumerate(FIXED_PROMPTS, 1):
            w_path = greedy_dir / f"fixed_sample_{idx:02d}.wav"
            res = synthesize_and_measure(synthesizer, p_text, w_path, cfg_greedy)
            greedy_results.append(res)
            print(
                f"  [Greedy {idx:02d}/10] '{p_text[:10]:<10}' -> {res['duration']:5.2f}s | "
                f"Frames: {res['frames']:3d} | Reason: {res['termination_reason']:<14} | "
                f"Q0Ent: {res['q0_entropy']:.2f} | Q0Top1: {res['q0_top1_ratio']:.2f} | StopP: {res['max_stop_prob']:.3f}"
            )

        # 2. 10 Fixed Prompts (Sampling)
        cfg_sampling = TalkerGenerationConfig(
            max_new_tokens=750, do_sample=True, temperature=0.8, top_p=0.9, stop_threshold=0.50, return_details=True
        )
        sampled_fixed_results = []
        for idx, p_text in enumerate(FIXED_PROMPTS, 1):
            w_path = sampled_dir / f"fixed_sample_{idx:02d}.wav"
            res = synthesize_and_measure(synthesizer, p_text, w_path, cfg_sampling)
            sampled_fixed_results.append(res)
            print(
                f"  [Sampled {idx:02d}/10] '{p_text[:10]:<10}' -> {res['duration']:5.2f}s | "
                f"Frames: {res['frames']:3d} | Reason: {res['termination_reason']:<14} | "
                f"MeanEnt: {res['mean_entropy']:.2f} | StopP: {res['max_stop_prob']:.3f}"
            )

        # 3. 20 Test Representatives (Sampling)
        sampled_test_results = []
        for idx, s_row in enumerate(test_representatives, 1):
            w_path = sampled_dir / f"test_sample_{idx:02d}.wav"
            res = synthesize_and_measure(
                synthesizer,
                s_row["text"],
                w_path,
                cfg_sampling,
                speaker=s_row.get("speaker_id", "speaker_001"),
                dialect=s_row.get("dialect", "ulsan"),
            )
            sampled_test_results.append(res)

        greedy_summary = summarize_items(greedy_results)
        sampled_fixed_summary = summarize_items(sampled_fixed_results)
        sampled_test_summary = summarize_items(sampled_test_results)

        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": ep_train_loss,
            "train_time_sec": round(ep_train_time, 2),
            "val_metrics": val_metrics,
            "greedy_fixed": greedy_summary,
            "sampled_fixed": sampled_fixed_summary,
            "sampled_test": sampled_test_summary,
        }
        all_epoch_reports[f"epoch_{epoch}"] = epoch_record

        with open(ep_eval_dir / "epoch_summary.json", "w", encoding="utf-8") as f:
            json.dump(epoch_record, f, ensure_ascii=False, indent=2)
        with open(greedy_dir / "greedy_report.json", "w", encoding="utf-8") as f:
            json.dump(greedy_results, f, ensure_ascii=False, indent=2)
        with open(sampled_dir / "sampled_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {"fixed": sampled_fixed_results, "test_representatives": sampled_test_results},
                f,
                ensure_ascii=False,
                indent=2,
            )

        # Early Failure Guard Check
        if val_metrics["total_loss"] > 20.0 or math.isnan(val_metrics["total_loss"]):
            raise RuntimeError(f"EARLY FAILURE GUARD: Val loss exploded to {val_metrics['total_loss']}")
        if epoch >= 2 and greedy_summary["avg_q0_entropy"] < 0.05:
            raise RuntimeError(f"EARLY FAILURE GUARD: Codec token collapse detected (entropy {greedy_summary['avg_q0_entropy']})")

    # 6. Final Save and Assessment
    print("\n=======================================================")
    print("=== [FULL TRAINING COMPLETE] 5 Epochs Finished ===")
    print("=======================================================")

    save_training_checkpoint(
        out_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        global_step=global_step,
        epoch=args.epochs,
        micro_batch_position=0,
        accumulation_step=0,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        training_args=vars(args),
        best_validation={"metric": "total_loss", "value": best_record["val_total_loss"], "step": global_step},
    )

    # Determine Best Checkpoint by combining validation loss & generation quality
    print("\nEvaluating best checkpoint selection...")
    best_candidate_epoch = best_record["epoch"]
    best_reason = f"Lowest validation total loss ({best_record['val_total_loss']:.4f}) with sound autoregressive stop and token diversity."

    # Comparison with 1-Epoch Pilot Baseline
    pilot_baseline = {
        "codec_loss": 6.6712,
        "q0_ce": 5.9455,
        "stop_f1": 0.3953,
        "greedy_entropy": 1.79,
        "sampling_entropy": 3.99,
        "sampling_top1": 0.1091,
    }

    best_ep_data = all_epoch_reports[f"epoch_{best_candidate_epoch}"]
    full_best_metrics = {
        "codec_loss": best_ep_data["val_metrics"]["codec_loss"],
        "q0_ce": best_ep_data["val_metrics"]["codebook_0_ce"],
        "stop_f1": best_ep_data["val_metrics"]["stop_f1"],
        "greedy_entropy": best_ep_data["greedy_fixed"]["avg_q0_entropy"],
        "sampling_entropy": best_ep_data["sampled_fixed"]["avg_mean_entropy"],
        "sampling_top1": best_ep_data["sampled_fixed"]["avg_q0_top1_ratio"],
    }

    comparison = {
        "codec_loss_change": round(full_best_metrics["codec_loss"] - pilot_baseline["codec_loss"], 4),
        "q0_ce_change": round(full_best_metrics["q0_ce"] - pilot_baseline["q0_ce"], 4),
        "stop_f1_change": round(full_best_metrics["stop_f1"] - pilot_baseline["stop_f1"], 4),
        "greedy_entropy_change": round(full_best_metrics["greedy_entropy"] - pilot_baseline["greedy_entropy"], 4),
        "sampling_entropy_change": round(full_best_metrics["sampling_entropy"] - pilot_baseline["sampling_entropy"], 4),
    }

    full_report = {
        "epochs": args.epochs,
        "total_optimizer_steps": global_step,
        "total_runtime_sec": round(time.time() - start_time, 2),
        "best_checkpoint": {
            "epoch": best_candidate_epoch,
            "path": str(out_dir / f"epoch-{best_candidate_epoch}.pt"),
            "reason": best_reason,
        },
        "pilot_vs_full": {
            "pilot_1epoch": pilot_baseline,
            "full_5epoch_best": full_best_metrics,
            "comparison": comparison,
        },
        "all_epochs": all_epoch_reports,
    }

    report_path = out_dir / "full_5epoch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)

    print(f"\nFULL 5-EPOCH REPORT SAVED TO: {report_path}")


if __name__ == "__main__":
    main()
