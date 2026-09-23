#!/usr/bin/env python3
# ruff: noqa: E402
"""Fresh production Talker retraining with token-accuracy and listening controls."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.codec import EncodedAudio, build_codec
from ulm_live.talker import (
    SpeechSynthesizer,
    TalkerCollator,
    TalkerConfig,
    TalkerDataset,
    TalkerGenerationConfig,
    ULMTalker,
    build_id_mappings,
)
from ulm_live.talker.checkpoint import capture_rng_state, restore_rng_state
from ulm_live.talker.semantic_cache import thinker_fingerprint
from ulm_live.thinker import ULMThinker
from ulm_live.utils.audio import load_wav, save_wav


SEED = 20260922
EVALUATION_MAX_FRAMES = 250  # 20 s at 12.5 Hz; model capacity remains 750 frames.
OLD_BASELINE = {
    "mean_token_accuracy": 0.05874,
    "q0_accuracy": 0.15932,
    "codec_ce": 6.028972,
}
FIXED_PROMPTS = [
    {"category": "short", "text": "안녕."},
    {"category": "short", "text": "밥 묵었나?"},
    {"category": "short", "text": "오늘 날씨 좋네."},
    {"category": "short", "text": "지금 어디고?"},
    {"category": "short", "text": "조심해서 가라."},
    {"category": "normal", "text": "오늘 뭐 하고 있었어?"},
    {"category": "normal", "text": "나는 오늘 학교 끝나고 친구를 만나러 간다."},
    {"category": "normal", "text": "주말에 시간 되면 같이 영화 보러 갈래?"},
    {"category": "normal", "text": "점심은 먹었어? 아직이면 내가 맛있는 곳을 알려줄게."},
    {"category": "normal", "text": "오늘 기분이 조금 안 좋아서 일찍 쉬려고 해."},
    {"category": "ulsan_dialect", "text": "니 오늘 와 이리 늦었노?"},
    {"category": "ulsan_dialect", "text": "그거 내가 아까 말했제."},
    {"category": "ulsan_dialect", "text": "날씨가 추우이 단디 입고 가라."},
    {"category": "ulsan_dialect", "text": "마, 그 정도면 진짜 잘했다 아이가."},
    {"category": "ulsan_dialect", "text": "아무리 바빠도 밥은 챙겨 묵어야 된다."},
    {"category": "long", "text": "하늘이 파란 이유를 처음 듣는 사람도 이해할 수 있게 간단히 설명해줘."},
    {"category": "long", "text": "울산의 태화강 국가정원이 시민들에게 어떤 의미가 있는지 차근차근 설명해줘."},
    {"category": "long", "text": "비가 많이 오는 날에는 길이 미끄러우니 평소보다 천천히 운전하고 안전거리를 충분히 확보해야 한다."},
    {"category": "long", "text": "친구와 의견이 다를 때는 상대방의 말을 끝까지 듣고 차분하게 내 생각을 이야기하는 것이 좋다."},
    {"category": "long", "text": "아침에 물을 한 잔 마시고 가볍게 몸을 풀면 하루를 조금 더 편안하게 시작할 수 있다."},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/talker.yaml"))
    parser.add_argument("--train-manifest", type=Path, default=Path("data/ulsan-full/train.cached.jsonl"))
    parser.add_argument("--val-manifest", type=Path, default=Path("data/ulsan-full/val.cached.jsonl"))
    parser.add_argument("--test-manifest", type=Path, default=Path("data/ulsan-full/test.cached.jsonl"))
    parser.add_argument("--thinker", default="/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/talker-full-retrain-v2"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--stop-pos-weight", type=float, default=5.0)
    parser.add_argument("--stop-loss-weight", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def hardlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def loader_for(dataset, collator, args, epoch: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED + epoch)
    workers = args.num_workers
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        generator=generator,
    )


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = round(total_steps * warmup_ratio)

    def factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor), warmup_steps


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate_validation(
    model: ULMTalker,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    q_loss_sum = np.zeros(model.config.num_quantizers, dtype=np.float64)
    q_correct = np.zeros(model.config.num_quantizers, dtype=np.int64)
    q_count = np.zeros(model.config.num_quantizers, dtype=np.int64)
    total_loss_sum = codec_loss_sum = stop_loss_sum = 0.0
    batches = 0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    batch["semantic_hidden_states"],
                    batch["audio_codes"],
                    batch["speaker_ids"],
                    batch["dialect_ids"],
                    targets=batch["targets"],
                    semantic_attention_mask=batch["semantic_attention_mask"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    stop_targets=batch["stop_targets"],
                )
            total_loss_sum += float(output.loss)
            codec_loss_sum += float(output.codec_loss)
            stop_loss_sum += float(output.stop_loss)
            batches += 1
            prediction = output.logits.argmax(-1)
            for q in range(model.config.num_quantizers):
                valid = batch["targets"][:, q] != model.config.pad_token_id
                count = int(valid.sum())
                q_count[q] += count
                q_correct[q] += int((prediction[:, q][valid] == batch["targets"][:, q][valid]).sum())
                q_loss_sum[q] += float(output.codebook_losses[q]) * count
            valid_stop = batch["stop_targets"] >= 0
            target_stop = (batch["stop_targets"] == 1) & valid_stop
            predicted_stop = (torch.sigmoid(output.stop_logits) >= 0.5) & valid_stop
            tp += int((predicted_stop & target_stop).sum())
            fp += int((predicted_stop & (~target_stop & valid_stop)).sum())
            fn += int(((~predicted_stop) & target_stop).sum())
            tn += int(((~predicted_stop) & (~target_stop & valid_stop)).sum())
    if not batches or np.any(q_count == 0):
        raise RuntimeError("validation produced no valid codec tokens")
    per_q_ce = q_loss_sum / q_count
    per_q_accuracy = q_correct / q_count
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "batches": batches,
        "total_loss": round(total_loss_sum / batches, 6),
        "codec_loss": round(codec_loss_sum / batches, 6),
        "stop_loss": round(stop_loss_sum / batches, 6),
        "q0_ce": round(float(per_q_ce[0]), 6),
        "mean_residual_ce": round(float(per_q_ce[1:].mean()), 6),
        "per_codebook_ce": [round(float(value), 6) for value in per_q_ce],
        "mean_token_accuracy": round(float(q_correct.sum() / q_count.sum()), 6),
        "q0_accuracy": round(float(per_q_accuracy[0]), 6),
        "mean_residual_accuracy": round(float(q_correct[1:].sum() / q_count[1:].sum()), 6),
        "per_codebook_accuracy": [round(float(value), 6) for value in per_q_accuracy],
        "stop_precision": round(precision, 6),
        "stop_recall": round(recall, 6),
        "stop_f1": round(f1, 6),
        "stop_tp": tp,
        "stop_fp": fp,
        "stop_fn": fn,
        "stop_tn": tn,
    }


def token_distribution(tokens: torch.Tensor) -> dict[str, Any]:
    if tokens.ndim == 3:
        tokens = tokens[0]
    entropy: list[float] = []
    top1: list[float] = []
    unique: list[int] = []
    for q in range(tokens.shape[0]):
        values = tokens[q].detach().cpu().tolist()
        counts = Counter(values)
        total = len(values)
        entropy.append(
            -sum((count / total) * math.log2(count / total) for count in counts.values())
            if total
            else 0.0
        )
        top1.append(counts.most_common(1)[0][1] / total if total else 0.0)
        unique.append(len(counts))
    return {
        "q0_entropy": round(entropy[0], 4),
        "q0_unique": unique[0],
        "q0_top1_ratio": round(top1[0], 6),
        "entropy_by_q": [round(value, 4) for value in entropy],
        "top1_ratio_by_q": [round(value, 6) for value in top1],
        "unique_by_q": unique,
    }


def validate_wav(path: Path) -> None:
    waveform, sample_rate = load_wav(path)
    if sample_rate != 24000 or waveform.shape[0] != 1 or waveform.shape[-1] <= 0:
        raise ValueError(f"invalid WAV: {path}")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite WAV: {path}")


def fixed_prompt_evaluation(
    synthesizer: SpeechSynthesizer,
    epoch_dir: Path,
    epoch: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"sampling": [], "greedy": []}
    jobs = (
        (
            "sampling",
            FIXED_PROMPTS,
            TalkerGenerationConfig(
                max_new_tokens=EVALUATION_MAX_FRAMES,
                max_audio_frames=EVALUATION_MAX_FRAMES,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                stop_threshold=0.50,
                return_details=True,
            ),
        ),
        (
            "greedy",
            FIXED_PROMPTS[:10],
            TalkerGenerationConfig(
                max_new_tokens=EVALUATION_MAX_FRAMES,
                max_audio_frames=EVALUATION_MAX_FRAMES,
                do_sample=False,
                stop_threshold=0.50,
                return_details=True,
            ),
        ),
    )
    for mode, prompts, config in jobs:
        output_dir = epoch_dir / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        hashes: set[str] = set()
        for index, prompt in enumerate(prompts, 1):
            if mode == "sampling":
                seed_everything(1000 + index)
            result = synthesizer.synthesize(
                prompt["text"],
                speaker="DKSR20000953_1",
                dialect="ulsan",
                generation_config=config,
            )
            wav_path = output_dir / f"sample_{index:03d}.wav"
            result.save(wav_path)
            validate_wav(wav_path)
            stats = token_distribution(result.codec_tokens)
            digest = hashlib.sha256(result.codec_tokens.numpy().tobytes()).hexdigest()
            hashes.add(digest)
            item = {
                "prompt_id": f"prompt_{index:03d}",
                "category": prompt["category"],
                "text": prompt["text"],
                "speaker": "DKSR20000953_1",
                "dialect": "ulsan",
                "mode": mode,
                "seed": 1000 + index if mode == "sampling" else None,
                "temperature": 0.8 if mode == "sampling" else None,
                "top_p": 0.9 if mode == "sampling" else None,
                "stop_threshold": 0.50,
                "duration": result.duration,
                "frames": result.num_audio_frames,
                "termination_reason": result.termination_reason,
                "max_stop_probability": result.max_stop_prob,
                "wav_path": str(wav_path.relative_to(epoch_dir)),
                **stats,
            }
            report[mode].append(item)
        report[f"{mode}_summary"] = {
            "count": len(prompts),
            "unique_outputs": len(hashes),
            "collapse_warning": any(
                item["q0_top1_ratio"] > 0.90 or item["q0_entropy"] < 0.1
                for item in report[mode]
            )
            or len(hashes) != len(prompts),
            "stop_predicted": sum(
                item["termination_reason"] == "STOP_PREDICTED" for item in report[mode]
            ),
            "max_frames": sum(
                item["termination_reason"] == "MAX_FRAMES" for item in report[mode]
            ),
        }
    write_json(epoch_dir / "generation_report.json", report)
    return report


def clean_test_row(row: dict[str, Any]) -> bool:
    text = str(row.get("text", "")).strip()
    duration = float(row.get("duration", 0.0))
    return (
        1.0 <= duration <= 8.0
        and len(text) >= 3
        and not re.search(r"[{}&]|\(\(\)\)", text)
        and bool(row.get("codec_path"))
        and bool(row.get("semantic_path"))
    )


def select_test_rows(path: Path, count: int = 20) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eligible = sorted((row for row in rows if clean_test_row(row)), key=lambda row: row["id"])
    indices = [round(index * (len(eligible) - 1) / (count - 1)) for index in range(count)]
    return [eligible[index] for index in indices]


def decode_codes(codec, codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    encoded = EncodedAudio(
        codes=codes,
        sample_rate=codec.sample_rate,
        frame_rate=codec.frame_rate,
        metadata={"backend": "mimi", "num_quantizers": codes.shape[-2]},
    )
    waveform, sample_rate = codec.decode(encoded)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] != 1 or waveform.shape[-1] == 0 or not torch.isfinite(waveform).all():
        raise ValueError("invalid Mimi decoded waveform")
    return waveform.detach().cpu(), int(sample_rate)


def teacher_forced(
    model: ULMTalker,
    sample: dict[str, Any],
    collator: TalkerCollator,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    batch = to_device(collator([sample]), device)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        output = model(
            batch["semantic_hidden_states"],
            batch["audio_codes"],
            batch["speaker_ids"],
            batch["dialect_ids"],
            targets=batch["targets"],
            semantic_attention_mask=batch["semantic_attention_mask"],
            audio_attention_mask=batch["audio_attention_mask"],
        )
    prediction = output.logits.argmax(-1)[0].detach().cpu()
    target = sample["audio_codes"]
    correct = prediction == target
    per_q = correct.float().mean(1)
    return prediction, {
        "codec_ce": round(float(output.codec_loss), 6),
        "mean_token_accuracy": round(float(correct.float().mean()), 6),
        "q0_accuracy": round(float(per_q[0]), 6),
        "mean_residual_accuracy": round(float(per_q[1:].mean()), 6),
        "per_codebook_accuracy": [round(float(value), 6) for value in per_q],
    }


def autoregressive_reconstruction(
    model: ULMTalker,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    config = TalkerGenerationConfig(
        max_new_tokens=EVALUATION_MAX_FRAMES,
        max_audio_frames=EVALUATION_MAX_FRAMES,
        do_sample=False,
        stop_threshold=0.50,
        return_details=True,
    )
    model.eval()
    with torch.no_grad(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        codes, lengths, reasons, stop_probs = model.generate(
            sample["semantic_hidden_states"].unsqueeze(0).to(device),
            torch.tensor([sample["speaker_id"]], device=device),
            torch.tensor([sample["dialect_id"]], device=device),
            generation_config=config,
            frame_rate=12.5,
            semantic_attention_mask=sample["semantic_attention_mask"].unsqueeze(0).to(device),
        )
    generated = codes[0, :, : int(lengths[0])].detach().cpu()
    target = sample["audio_codes"]
    overlap = min(generated.shape[-1], target.shape[-1])
    correct = generated[:, :overlap] == target[:, :overlap]
    per_q = correct.float().mean(1)
    return generated, {
        "generated_frames": int(generated.shape[-1]),
        "ground_truth_frames": int(target.shape[-1]),
        "length_ratio": round(generated.shape[-1] / target.shape[-1], 6),
        "overlap_frames": overlap,
        "mean_token_agreement": round(float(correct.float().mean()), 6),
        "q0_token_agreement": round(float(per_q[0]), 6),
        "per_codebook_agreement": [round(float(value), 6) for value in per_q],
        "termination_reason": reasons[0],
        "max_stop_probability": round(float(stop_probs[0]), 6),
        **token_distribution(generated),
    }


def reconstruction_evaluation(
    model: ULMTalker,
    codec,
    test_dataset: TalkerDataset,
    selected_rows: list[dict[str, Any]],
    collator: TalkerCollator,
    epoch_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    index_by_id = {row["id"]: index for index, row in enumerate(test_dataset.items)}
    reports: list[dict[str, Any]] = []
    for index, row in enumerate(selected_rows, 1):
        sample_id = f"sample_{index:03d}"
        destination = epoch_dir / "reconstruction" / sample_id
        destination.mkdir(parents=True, exist_ok=True)
        sample = test_dataset[index_by_id[row["id"]]]
        original_path = test_dataset._path(row["audio_path"])
        hardlink(original_path, destination / "original.wav")
        target = sample["audio_codes"]
        ground_truth_waveform, sample_rate = decode_codes(codec, target.to(device))
        save_wav(destination / "mimi_gt.wav", ground_truth_waveform, sample_rate)
        teacher_codes, teacher_metrics = teacher_forced(model, sample, collator, device)
        teacher_waveform, sample_rate = decode_codes(codec, teacher_codes.to(device))
        save_wav(destination / "teacher_forced.wav", teacher_waveform, sample_rate)
        generated, autoregressive_metrics = autoregressive_reconstruction(model, sample, device)
        generated_waveform, sample_rate = decode_codes(codec, generated.to(device))
        save_wav(destination / "autoregressive.wav", generated_waveform, sample_rate)
        for wav_name in ("original.wav", "mimi_gt.wav", "teacher_forced.wav", "autoregressive.wav"):
            validate_wav(destination / wav_name)
        (destination / "text.txt").write_text(str(row["text"]) + "\n", encoding="utf-8")
        metadata = {
            "sample_id": sample_id,
            "dataset_id": row["id"],
            "text": row["text"],
            "speaker": row["speaker_id"],
            "dialect": row["dialect"],
            "teacher_forced": teacher_metrics,
            "autoregressive": autoregressive_metrics,
        }
        write_json(destination / "metadata.json", metadata)
        reports.append(metadata)
    summary = {
        "sample_count": len(reports),
        "mean_teacher_forced_accuracy": round(
            float(np.mean([item["teacher_forced"]["mean_token_accuracy"] for item in reports])), 6
        ),
        "mean_teacher_forced_q0_accuracy": round(
            float(np.mean([item["teacher_forced"]["q0_accuracy"] for item in reports])), 6
        ),
        "mean_autoregressive_token_agreement": round(
            float(np.mean([item["autoregressive"]["mean_token_agreement"] for item in reports])), 6
        ),
        "mean_autoregressive_q0_agreement": round(
            float(np.mean([item["autoregressive"]["q0_token_agreement"] for item in reports])), 6
        ),
        "mean_length_ratio": round(
            float(np.mean([item["autoregressive"]["length_ratio"] for item in reports])), 6
        ),
        "samples": reports,
    }
    write_json(epoch_dir / "reconstruction_report.json", summary)
    return summary


def save_model_checkpoint(
    path: Path,
    model: ULMTalker,
    epoch: int,
    global_step: int,
    speaker2id: dict[str, int],
    dialect2id: dict[str, int],
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 3,
            "checkpoint_kind": "model_only",
            "fresh_training_run": True,
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "speaker2id": speaker2id,
            "dialect2id": dialect2id,
            "epoch": epoch,
            "global_step": global_step,
            "step": global_step,
            "metrics": metrics,
            "training_args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def save_resume_state(
    path: Path,
    model: ULMTalker,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    reports: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "run": "talker-full-retrain-v2",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "reports": reports,
            "rng_state": capture_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def write_epoch_table(reports: dict[str, Any], path: Path) -> None:
    lines = [
        "| Epoch | Train Codec | Val Codec | Q0 CE | Residual CE | Mean Token Accuracy | Q0 Accuracy | Stop F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(reports, key=lambda value: int(value.split("_")[-1])):
        item = reports[key]
        val = item["validation"]
        lines.append(
            f"| {item['epoch']} | {item['train']['codec_loss']:.4f} | {val['codec_loss']:.4f} | "
            f"{val['q0_ce']:.4f} | {val['mean_residual_ce']:.4f} | "
            f"{val['mean_token_accuracy']:.3%} | {val['q0_accuracy']:.3%} | {val['stop_f1']:.3%} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs > 20:
        raise ValueError("20 epochs is the hard maximum")
    if args.stop_loss_weight != 0.05:
        raise ValueError("production retrain v2 requires stop_loss_weight=0.05")
    args.config = args.config.resolve()
    args.train_manifest = args.train_manifest.resolve()
    args.val_manifest = args.val_manifest.resolve()
    args.test_manifest = args.test_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    device = torch.device(args.device)
    seed_everything()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(output_dir.glob("epoch-*.pt")):
        raise RuntimeError("fresh run output already contains epoch checkpoints")

    speaker2id, dialect2id = build_id_mappings(
        [args.train_manifest, args.val_manifest, args.test_manifest]
    )
    config = TalkerConfig.from_yaml(args.config)
    config.num_quantizers = 16
    config.stop_pos_weight = args.stop_pos_weight
    config.stop_loss_weight = args.stop_loss_weight
    config.learning_rate = args.learning_rate
    config.warmup_ratio = args.warmup_ratio
    config.num_speakers = max(config.num_speakers, len(speaker2id))
    config.num_dialects = max(config.num_dialects, len(dialect2id))
    print("Loading production Thinker and Mimi codec...")
    thinker = ULMThinker(args.thinker, device=args.device, torch_dtype="bf16", freeze=True)
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=16)
    expected_cache = {
        "thinker_id": args.thinker,
        "thinker_fingerprint": thinker_fingerprint(args.thinker),
        "hidden_layer": thinker.hidden_layer,
    }
    train_dataset = TalkerDataset(
        args.train_manifest,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=16,
        cache_only=True,
        semantic_cache_metadata=expected_cache,
    )
    val_dataset = TalkerDataset(
        args.val_manifest,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=16,
        cache_only=True,
        semantic_cache_metadata=expected_cache,
    )
    test_dataset = TalkerDataset(
        args.test_manifest,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        num_quantizers=16,
        cache_only=True,
        semantic_cache_metadata=expected_cache,
    )
    if (len(train_dataset), len(val_dataset), len(test_dataset)) != (22150, 2769, 2768):
        raise ValueError(
            f"canonical split mismatch: {len(train_dataset)}, {len(val_dataset)}, {len(test_dataset)}"
        )
    collator = TalkerCollator(
        thinker.tokenizer,
        config.pad_token_id,
        config.bos_token_id,
        config.max_seq_len,
        acoustic_delay_frames=config.acoustic_delay_frames,
    )
    micro_batches_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.grad_accum)
    total_steps = optimizer_steps_per_epoch * args.epochs
    model = ULMTalker(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=config.weight_decay
    )
    scheduler, warmup_steps = build_scheduler(optimizer, total_steps, args.warmup_ratio)
    reports: dict[str, Any] = {}
    global_step = 0
    start_epoch = 1
    resume_path = output_dir / "resume-latest.pt"
    if args.resume:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        reports = payload["reports"]
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"]) + 1
        restore_rng_state(payload["rng_state"])
        print(f"Resuming this v2 run from epoch {start_epoch}; no legacy checkpoint used")

    selected_test_rows = select_test_rows(args.test_manifest, 20)
    write_json(output_dir / "fixed_prompts.json", FIXED_PROMPTS)
    write_json(output_dir / "reconstruction_selection.json", selected_test_rows)
    run_config = {
        "fresh_initialization": not args.resume,
        "legacy_checkpoint_resume": False,
        "tiny_checkpoint_resume": False,
        "train_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "dtype": "bfloat16",
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "scheduler": "cosine",
        "stop_pos_weight": args.stop_pos_weight,
        "stop_loss_weight": args.stop_loss_weight,
        "total_optimizer_steps_planned": total_steps,
        "full_generation_epochs": list(range(2, args.epochs + 1, 2)),
        "evaluation_max_frames": EVALUATION_MAX_FRAMES,
        "evaluation_max_seconds": EVALUATION_MAX_FRAMES / 12.5,
    }
    write_json(output_dir / "training_config.json", run_config)
    synthesizer = SpeechSynthesizer(
        thinker=thinker,
        talker=model,
        codec=codec,
        speaker2id=speaker2id,
        dialect2id=dialect2id,
        device=device,
    )
    run_start = time.time()
    best_codec = (float("inf"), None)
    best_accuracy = (-1.0, None)
    for key, report in reports.items():
        epoch = int(key.split("_")[-1])
        val = report["validation"]
        if val["codec_loss"] < best_codec[0]:
            best_codec = (val["codec_loss"], epoch)
        if val["mean_token_accuracy"] > best_accuracy[0]:
            best_accuracy = (val["mean_token_accuracy"], epoch)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loader = loader_for(train_dataset, collator, args, epoch, shuffle=True)
        optimizer.zero_grad(set_to_none=True)
        accumulation = 0
        train_total_sum = train_codec_sum = train_stop_sum = 0.0
        train_micro_batches = 0
        gradient_norms: list[float] = []
        for position, raw_batch in enumerate(train_loader):
            batch = to_device(raw_batch, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(
                    batch["semantic_hidden_states"],
                    batch["audio_codes"],
                    batch["speaker_ids"],
                    batch["dialect_ids"],
                    targets=batch["targets"],
                    semantic_attention_mask=batch["semantic_attention_mask"],
                    audio_attention_mask=batch["audio_attention_mask"],
                    stop_targets=batch["stop_targets"],
                )
                scaled_loss = output.loss / args.grad_accum
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f"non-finite loss epoch={epoch} batch={position}")
            scaled_loss.backward()
            accumulation += 1
            train_micro_batches += 1
            train_total_sum += float(output.loss.detach())
            train_codec_sum += float(output.codec_loss.detach())
            train_stop_sum += float(output.stop_loss.detach())
            boundary = accumulation == args.grad_accum or position + 1 == len(train_loader)
            if boundary:
                if accumulation < args.grad_accum:
                    correction = args.grad_accum / accumulation
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(correction)
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                gradient_norms.append(gradient_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                accumulation = 0
        train_metrics = {
            "total_loss": round(train_total_sum / train_micro_batches, 6),
            "codec_loss": round(train_codec_sum / train_micro_batches, 6),
            "stop_loss": round(train_stop_sum / train_micro_batches, 6),
            "mean_gradient_norm": round(float(np.mean(gradient_norms)), 6),
            "max_gradient_norm": round(float(np.max(gradient_norms)), 6),
            "learning_rate": scheduler.get_last_lr()[0],
            "micro_batches": train_micro_batches,
        }
        val_loader = loader_for(val_dataset, collator, args, epoch=0, shuffle=False)
        validation = evaluate_validation(model, val_loader, device)
        epoch_report: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "validation": validation,
            "epoch_runtime_seconds_before_generation": round(time.time() - epoch_start, 3),
        }
        reports[f"epoch_{epoch}"] = epoch_report
        checkpoint_path = output_dir / f"epoch-{epoch:02d}.pt"
        save_model_checkpoint(
            checkpoint_path,
            model,
            epoch,
            global_step,
            speaker2id,
            dialect2id,
            epoch_report,
            args,
        )
        if validation["codec_loss"] < best_codec[0]:
            best_codec = (validation["codec_loss"], epoch)
            hardlink(checkpoint_path, output_dir / "best_codec.pt")
        if validation["mean_token_accuracy"] > best_accuracy[0]:
            best_accuracy = (validation["mean_token_accuracy"], epoch)
            hardlink(checkpoint_path, output_dir / "best_accuracy.pt")
        print(
            f"EPOCH {epoch:02d} train_codec={train_metrics['codec_loss']:.4f} "
            f"val_codec={validation['codec_loss']:.4f} q0_CE={validation['q0_ce']:.4f} "
            f"accuracy={validation['mean_token_accuracy']:.3%} q0={validation['q0_accuracy']:.3%} "
            f"stop_F1={validation['stop_f1']:.3%}"
        )
        if epoch % 2 == 0:
            eval_dir = output_dir / "eval" / f"epoch-{epoch:02d}"
            print(f"Running full listening evaluation for epoch {epoch:02d}...")
            generation = fixed_prompt_evaluation(synthesizer, eval_dir, epoch)
            reconstruction = reconstruction_evaluation(
                model,
                codec,
                test_dataset,
                selected_test_rows,
                collator,
                eval_dir,
                device,
            )
            epoch_report["generation"] = {
                "sampling": generation["sampling_summary"],
                "greedy": generation["greedy_summary"],
            }
            epoch_report["reconstruction"] = {
                key: value for key, value in reconstruction.items() if key != "samples"
            }
            write_json(output_dir / "best_listening_candidate.json", {
                "status": "HUMAN_UNRATED",
                "human_listening_required": True,
                "evaluated_epochs": [
                    report["epoch"]
                    for report in reports.values()
                    if report.get("generation") is not None
                ],
                "provisional_candidates": sorted(
                    [
                        {
                            "epoch": report["epoch"],
                            "val_codec_loss": report["validation"]["codec_loss"],
                            "mean_token_accuracy": report["validation"]["mean_token_accuracy"],
                            "q0_accuracy": report["validation"]["q0_accuracy"],
                            "path": f"eval/epoch-{report['epoch']:02d}",
                        }
                        for report in reports.values()
                        if report.get("generation") is not None
                    ],
                    key=lambda item: (-item["mean_token_accuracy"], item["val_codec_loss"]),
                )[:3],
            })
        epoch_report["epoch_runtime_seconds"] = round(time.time() - epoch_start, 3)
        write_json(output_dir / "reports" / f"epoch-{epoch:02d}.json", epoch_report)
        write_json(output_dir / "reports" / "all_epochs.json", reports)
        write_epoch_table(reports, output_dir / "reports" / "epoch_table.md")
        if epoch == 10:
            latest = validation
            mid = {
                "epochs": list(range(1, 11)),
                "epoch_table": "reports/epoch_table.md",
                "generation_epochs": [2, 4, 6, 8, 10],
                "human_questions": {
                    "speech_like_structure": "HUMAN_LISTENING_REQUIRED",
                    "recognizable_korean_syllables": "HUMAN_LISTENING_REQUIRED",
                },
                "teacher_forced_accuracy": latest["mean_token_accuracy"],
                "old_epoch_5_accuracy": OLD_BASELINE["mean_token_accuracy"],
                "absolute_accuracy_gain": round(
                    latest["mean_token_accuracy"] - OLD_BASELINE["mean_token_accuracy"], 6
                ),
                "continue_to_epoch_20": True,
                "reason": "No catastrophic failure; human quality is unscored and objective codec learning continues.",
            }
            write_json(output_dir / "reports" / "mid_training_epoch_10.json", mid)
        save_resume_state(
            resume_path, model, optimizer, scheduler, epoch, global_step, reports
        )

    final_epoch = args.epochs
    hardlink(output_dir / f"epoch-{final_epoch:02d}.pt", output_dir / "final.pt")
    candidates = json.loads((output_dir / "best_listening_candidate.json").read_text(encoding="utf-8"))
    best_codec_epoch = int(best_codec[1])
    best_accuracy_epoch = int(best_accuracy[1])
    best_accuracy_metrics = reports[f"epoch_{best_accuracy_epoch}"]["validation"]
    final_generation = reports[f"epoch_{final_epoch}"].get("generation", {})
    final_collapse = bool(
        final_generation.get("sampling", {}).get("collapse_warning", False)
        or final_generation.get("greedy", {}).get("collapse_warning", False)
    )
    objective_resolved = bool(
        best_accuracy_metrics["mean_token_accuracy"] > OLD_BASELINE["mean_token_accuracy"]
        and best_accuracy_metrics["codec_loss"] < OLD_BASELINE["codec_ce"]
    )
    final_report = {
        "title": "ULM-LIVE FULL RETRAIN V2 REPORT",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip(),
        "training": {
            "epochs_completed": final_epoch,
            "optimizer_steps": global_step,
            "runtime_seconds": round(time.time() - run_start, 3),
        },
        "config": run_config,
        "epochs": reports,
        "best_codec_checkpoint": {
            "epoch": best_codec_epoch,
            "path": "best_codec.pt",
            "val_codec_loss": best_codec[0],
        },
        "best_token_accuracy_checkpoint": {
            "epoch": best_accuracy_epoch,
            "path": "best_accuracy.pt",
            "mean_token_accuracy": best_accuracy[0],
        },
        "recommended_human_listening_candidates": candidates["provisional_candidates"],
        "human_listening_status": "HUMAN LISTENING REQUIRED",
        "final_token_collapse": final_collapse,
        "final_stop_stability": reports[f"epoch_{final_epoch}"]["validation"]["stop_f1"],
        "old_5epoch_reference": OLD_BASELINE,
        "new_best": {
            "mean_token_accuracy": best_accuracy_metrics["mean_token_accuracy"],
            "q0_accuracy": best_accuracy_metrics["q0_accuracy"],
            "codec_ce": best_accuracy_metrics["codec_loss"],
        },
        "root_result": "UNDERTRAINING RESOLVED" if objective_resolved else "STILL UNDERTRAINED",
        "production_talker_candidate": "best_accuracy.pt" if objective_resolved else None,
        "production_ready": False,
        "full_training_complete": True,
    }
    write_json(output_dir / "full_retrain_v2_report.json", final_report)
    print("TRAINING COMPLETE")
    print("ROOT RESULT:", final_report["root_result"])
    print("HUMAN LISTENING REQUIRED")


if __name__ == "__main__":
    main()
