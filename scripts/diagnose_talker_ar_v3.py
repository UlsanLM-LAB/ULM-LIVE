#!/usr/bin/env python3
# ruff: noqa: E402
"""Measure Talker conditioning, free-running errors, and cache alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ulm_live.codec import EncodedAudio, build_codec
from ulm_live.talker import TalkerCollator, TalkerDataset, TalkerGenerationConfig, load_talker_checkpoint
from ulm_live.talker.data import _load_cache
from ulm_live.utils.audio import save_wav


CHECKPOINT_ROOT = Path("outputs/talker-full-retrain-v2")
OUTPUT_ROOT = Path("outputs/talker-ar-diagnostics-v3")
DATA_ROOT = Path("data/ulsan-full")
SEED = 20260923


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("ablation", "rollout", "alignment", "text", "hybrid", "forced_audio", "embedding", "position_probe"))
    p.add_argument("--epochs", type=int, nargs="+", default=[6, 8, 10])
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--same-speaker", action="store_true", help="use one speaker for paired semantic ablation")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_ROOT)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def selected_indices(dataset: TalkerDataset, count: int) -> list[int]:
    eligible = sorted(
        (
            i for i, row in enumerate(dataset.items)
            if 1 <= float(row.get("duration", 0)) <= 8
            and row.get("semantic_path") and row.get("codec_path")
            and len(str(row.get("text", ""))) >= 3),
        key=lambda i: dataset.items[i]["id"],
    )
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible test rows")
    return [eligible[round(i * (len(eligible) - 1) / (count - 1))] for i in range(count)]


def load_model_and_data(args: argparse.Namespace, epoch: int):
    model, metadata = load_talker_checkpoint(
        args.checkpoint_dir / f"epoch-{epoch:02d}.pt", args.device
    )
    dataset = TalkerDataset(
        args.data_dir / "test.cached.jsonl",
        speaker2id=metadata["speaker2id"],
        dialect2id=metadata["dialect2id"],
        num_quantizers=16,
        cache_only=True,
    )
    collator = TalkerCollator(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        max_audio_len=model.config.max_seq_len,
        acoustic_delay_frames=model.config.acoustic_delay_frames,
    )
    return model, dataset, collator


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def teacher_prediction(model, collator, sample, device):
    batch = to_device(collator([sample]), device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(
            batch["semantic_hidden_states"], batch["audio_codes"],
            batch["speaker_ids"], batch["dialect_ids"],
            targets=batch["targets"],
            semantic_attention_mask=batch["semantic_attention_mask"],
            audio_attention_mask=batch["audio_attention_mask"],
        )
    return output, batch


def per_q_accuracy(pred: torch.Tensor, target: torch.Tensor) -> list[float]:
    return (pred == target).float().mean(-1).tolist()


def first_divergence(pred: torch.Tensor, target: torch.Tensor) -> int | None:
    """First frame where any quantizer differs, using [Q,T] tensors."""
    matches = (pred == target).all(0)
    positions = (~matches).nonzero(as_tuple=False)
    return int(positions[0, 0]) if positions.numel() else None


def first_q0_divergence(pred: torch.Tensor, target: torch.Tensor) -> int | None:
    positions = (pred[0] != target[0]).nonzero(as_tuple=False)
    return int(positions[0, 0]) if positions.numel() else None


def distribution_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction = logits.argmax(-1)
    correct = prediction == target
    losses = [
        float(F.cross_entropy(logits[:, q].float().reshape(-1, logits.shape[-1]), target[:, q].reshape(-1)))
        for q in range(target.shape[1])
    ]
    return {
        "codec_ce": float(np.mean(losses)),
        "mean_accuracy": float(correct.float().mean()),
        "q0_accuracy": float(correct[:, 0].float().mean()),
        "residual_accuracy": float(correct[:, 1:].float().mean()),
        "q0_ce": losses[0],
        "residual_ce": float(np.mean(losses[1:])),
    }


def semantic_variant(hidden: torch.Tensor, other: torch.Tensor, mode: str, generator: torch.Generator) -> torch.Tensor:
    if mode == "correct" or mode == "same":
        return hidden
    if mode == "zero":
        return torch.zeros_like(hidden)
    if mode == "random":
        return torch.randn(hidden.shape, generator=generator, dtype=torch.float32).to(hidden.dtype) * hidden.float().std() + hidden.float().mean()
    if mode == "shuffled":
        if other.shape[0] >= hidden.shape[0]:
            return other[:hidden.shape[0]].to(hidden.dtype)
        repeats = math.ceil(hidden.shape[0] / other.shape[0])
        return other.repeat(repeats, 1)[:hidden.shape[0]].to(hidden.dtype)
    raise ValueError(mode)


def run_ablation(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    generator = torch.Generator().manual_seed(SEED)
    for epoch in args.epochs:
        model, dataset, collator = load_model_and_data(args, epoch)
        indices = selected_indices(dataset, args.samples)
        if args.same_speaker:
            eligible = [i for i, row in enumerate(dataset.items)
                        if 1 <= float(row.get("duration", 0)) <= 8
                        and row.get("semantic_path") and row.get("codec_path")]
            by_speaker: dict[str, list[int]] = {}
            for i in eligible:
                by_speaker.setdefault(str(dataset.items[i]["speaker_id"]), []).append(i)
            speaker, pool = max(by_speaker.items(), key=lambda pair: len(pair[1]))
            if len(pool) < args.samples:
                raise ValueError(f"speaker {speaker} has only {len(pool)} samples")
            pool.sort(key=lambda i: dataset.items[i]["id"])
            indices = [pool[round(i * (len(pool) - 1) / (args.samples - 1))] for i in range(args.samples)]
        samples = [dataset[i] for i in indices]
        rows = []
        for n, sample in enumerate(samples):
            other = samples[(n + 1) % len(samples)]
            baseline_logits = None
            for mode in ("correct", "same", "zero", "random", "shuffled"):
                variant = dict(sample)
                variant["semantic_hidden_states"] = semantic_variant(
                    sample["semantic_hidden_states"], other["semantic_hidden_states"], mode, generator
                )
                output, batch = teacher_prediction(model, collator, variant, device)
                logits = output.logits.float()
                metrics = distribution_metrics(logits, batch["targets"])
                if baseline_logits is None:
                    baseline_logits = logits
                    metrics["kl_from_correct"] = 0.0
                    metrics["mean_abs_logit_diff"] = 0.0
                else:
                    base_log_p = F.log_softmax(baseline_logits, -1)
                    base_p = base_log_p.exp()
                    metrics["kl_from_correct"] = float(
                        F.kl_div(F.log_softmax(logits, -1), base_p, reduction="batchmean")
                        / (logits.shape[1] * logits.shape[2])
                    )
                    metrics["mean_abs_logit_diff"] = float((logits - baseline_logits).abs().mean())
                rows.append({"epoch": epoch, "sample_id": sample["id"], "mode": mode, **metrics})
            if (n + 1) % 10 == 0:
                print(f"ablation epoch={epoch} samples={n+1}/{len(samples)}", flush=True)
        suffix = "-same-speaker" if args.same_speaker else ""
        out = args.output_dir / "ablation" / f"epoch-{epoch:02d}{suffix}.csv"
        write_csv(out, rows)
        summary = {}
        for mode in ("correct", "same", "zero", "random", "shuffled"):
            selected = [row for row in rows if row["mode"] == mode]
            summary[mode] = {k: float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ("epoch", "sample_id", "mode")}
        write_json(out.with_suffix(".json"), summary)
        print(f"ablation epoch={epoch} {summary}", flush=True)
        del model
        torch.cuda.empty_cache()


def rollout(model, sample, switch: int, limit: int, device: torch.device):
    """Use GT previous frames before switch, then feed model predictions."""
    target = sample["audio_codes"].to(device)
    sem = sample["semantic_hidden_states"].unsqueeze(0).to(device)
    mask = sample["semantic_attention_mask"].unsqueeze(0).to(device)
    speaker = torch.tensor([sample["speaker_id"]], device=device)
    dialect = torch.tensor([sample["dialect_id"]], device=device)
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        state = model.init_generation_state(sem, speaker, dialect, mask)
        previous = torch.full((1, model.config.num_quantizers), model.config.bos_token_id, device=device, dtype=torch.long)
        predicted = []
        stop_probs = []
        for frame in range(limit):
            if 0 < frame <= switch and frame <= target.shape[-1]:
                previous = target[:, frame - 1].unsqueeze(0)
            tokens, stop = model.step(state, previous)
            predicted.append(tokens[0].cpu())
            stop_probs.append(float(stop[0]))
            previous = tokens
    return torch.stack(predicted, -1), stop_probs


def stop_frame(stop_probs: list[float], minimum: int, threshold: float = 0.5) -> int:
    return next((i + 1 for i, p in enumerate(stop_probs) if i + 1 >= minimum and p >= threshold), len(stop_probs))


def run_rollout(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    for epoch in args.epochs:
        model, dataset, collator = load_model_and_data(args, epoch)
        indices = selected_indices(dataset, args.samples)
        summary_rows = []
        frame_rows = []
        for n, idx in enumerate(indices, 1):
            sample = dataset[idx]
            target = sample["audio_codes"]
            length = target.shape[-1]
            output, batch = teacher_prediction(model, collator, sample, device)
            teacher = output.logits.argmax(-1)[0].cpu()
            teacher_q = per_q_accuracy(teacher, target)
            teacher_matches = (teacher == target).float()
            switches = sorted(set([0, length // 4, length // 2, max(0, length - 10)]))
            for switch in switches:
                limit = max(length, 250) if switch == 0 else length
                limit = min(limit, model.config.max_seq_len - sample["semantic_hidden_states"].shape[0])
                pred, stop_probs = rollout(model, sample, switch, limit, device)
                if switch == 0:
                    normal_len = stop_frame(stop_probs, model.config.min_audio_frames)
                    normal = pred[:, :normal_len]
                    normal_overlap = min(normal_len, length)
                    normal_q = per_q_accuracy(normal[:, :normal_overlap], target[:, :normal_overlap])
                    summary_rows.append({
                        "epoch": epoch, "sample_id": sample["id"], "mode": "normal_ar",
                        "switch_frame": 0, "target_frames": length, "generated_frames": normal_len,
                        "length_ratio": normal_len / length, "mean_accuracy": float(np.mean(normal_q)),
                        "q0_accuracy": normal_q[0], "residual_accuracy": float(np.mean(normal_q[1:])),
                        "per_q_accuracy": json.dumps(normal_q),
                        "first_incorrect_frame": first_divergence(normal[:, :normal_overlap], target[:, :normal_overlap]),
                        "first_q0_incorrect_frame": first_q0_divergence(normal[:, :normal_overlap], target[:, :normal_overlap]),
                    })
                fixed = pred[:, :length]
                overlap = min(fixed.shape[-1], length)
                q_acc = per_q_accuracy(fixed[:, :overlap], target[:, :overlap])
                mode = "forced_gt_length" if switch == 0 else "switch_rollout"
                summary_rows.append({
                    "epoch": epoch, "sample_id": sample["id"], "mode": mode,
                    "switch_frame": switch, "target_frames": length, "generated_frames": overlap,
                    "length_ratio": overlap / length, "mean_accuracy": float(np.mean(q_acc)),
                    "q0_accuracy": q_acc[0], "residual_accuracy": float(np.mean(q_acc[1:])),
                    "per_q_accuracy": json.dumps(q_acc),
                    "first_incorrect_frame": first_divergence(fixed[:, :overlap], target[:, :overlap]),
                    "first_q0_incorrect_frame": first_q0_divergence(fixed[:, :overlap], target[:, :overlap]),
                })
                matches = (fixed[:, :overlap] == target[:, :overlap]).float()
                for frame in range(overlap):
                    frame_rows.append({
                        "epoch": epoch, "sample_id": sample["id"], "switch_frame": switch,
                        "frame": frame, "relative_to_switch": frame - switch,
                        "frame_token_accuracy": float(matches[:, frame].mean()),
                        "q0_correct": float(matches[0, frame]),
                        "residual_accuracy": float(matches[1:, frame].mean()),
                        "cumulative_token_accuracy": float(matches[:, :frame+1].mean()),
                        "stop_probability": stop_probs[frame],
                        "teacher_frame_token_accuracy": float(teacher_matches[:, frame].mean()),
                        "teacher_q0_correct": float(teacher_matches[0, frame]),
                    })
            summary_rows.append({
                "epoch": epoch, "sample_id": sample["id"], "mode": "teacher_forced",
                "switch_frame": -1, "target_frames": length, "generated_frames": length,
                "length_ratio": 1.0, "mean_accuracy": float(np.mean(teacher_q)),
                "q0_accuracy": teacher_q[0], "residual_accuracy": float(np.mean(teacher_q[1:])),
                "per_q_accuracy": json.dumps(teacher_q),
                "first_incorrect_frame": first_divergence(teacher, target),
                "first_q0_incorrect_frame": first_q0_divergence(teacher, target),
            })
            if n % 10 == 0:
                print(f"rollout epoch={epoch} samples={n}/{len(indices)}", flush=True)
        write_csv(args.output_dir / "rollout" / f"epoch-{epoch:02d}.csv", summary_rows)
        write_csv(args.output_dir / "rollout" / f"epoch-{epoch:02d}-frames.csv", frame_rows)
        report = {}
        for mode in ("teacher_forced", "normal_ar", "forced_gt_length", "switch_rollout"):
            subset = [row for row in summary_rows if row["mode"] == mode]
            report[mode] = {key: float(np.mean([row[key] for row in subset])) for key in ("mean_accuracy", "q0_accuracy", "residual_accuracy", "length_ratio")}
            report[mode]["sample_count"] = len(subset)
            report[mode]["median_first_incorrect_frame"] = float(np.median([row["first_incorrect_frame"] for row in subset if row["first_incorrect_frame"] is not None]))
            q0_first = [row["first_q0_incorrect_frame"] for row in subset if row["first_q0_incorrect_frame"] is not None]
            report[mode]["median_first_q0_incorrect_frame"] = float(np.median(q0_first)) if q0_first else None
        write_json(args.output_dir / "rollout" / f"epoch-{epoch:02d}.json", report)
        print(f"rollout epoch={epoch} {report}", flush=True)
        del model
        torch.cuda.empty_cache()


def run_alignment(args: argparse.Namespace) -> None:
    rows = []
    for split in ("train", "val", "test"):
        manifest = args.data_dir / f"{split}.cached.jsonl"
        items = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for n, item in enumerate(items, 1):
            path = Path(item["semantic_path"])
            if not path.is_absolute():
                path = args.data_dir / path
            hidden, mask, _ = _load_cache(path, None)
            sem_len = int(mask.sum())
            duration = float(item["duration"])
            codec_path = Path(item["codec_path"])
            if not codec_path.is_absolute():
                codec_path = args.data_dir / codec_path
            cached_codes = torch.load(codec_path, map_location="cpu", weights_only=True)
            audio_frames = int(cached_codes.shape[-1])
            rows.append({
                "split": split, "sample_id": item["id"], "speaker": item["speaker_id"],
                "text_chars": len(str(item["text"])), "semantic_tokens": sem_len,
                "audio_frames": audio_frames, "duration": duration,
                "semantic_audio_ratio": sem_len / max(1, audio_frames),
                "sequence_length": sem_len + audio_frames,
            })
            if n % 5000 == 0:
                print(f"alignment {split} {n}/{len(items)}", flush=True)
    out = args.output_dir / "alignment" / "lengths.csv"
    write_csv(out, rows)
    summary = {
        "samples": len(rows),
        "semantic_length_quantiles": np.quantile([r["semantic_tokens"] for r in rows], [0, .25, .5, .75, .95, .99, 1]).tolist(),
        "audio_frame_quantiles": np.quantile([r["audio_frames"] for r in rows], [0, .25, .5, .75, .95, .99, 1]).tolist(),
        "combined_over_2048": sum(r["sequence_length"] > 2048 for r in rows),
        "text_semantic_correlation": float(np.corrcoef([r["text_chars"] for r in rows], [r["semantic_tokens"] for r in rows])[0, 1]),
        "text_audio_correlation": float(np.corrcoef([r["text_chars"] for r in rows], [r["audio_frames"] for r in rows])[0, 1]),
        "semantic_audio_correlation": float(np.corrcoef([r["semantic_tokens"] for r in rows], [r["audio_frames"] for r in rows])[0, 1]),
    }
    write_json(args.output_dir / "alignment" / "summary.json", summary)
    print(summary, flush=True)


def decode_save(codec, codes: torch.Tensor, path: Path) -> None:
    if codes.numel() == 0 or int(codes.min()) < 0 or int(codes.max()) >= 2048:
        raise ValueError(f"invalid codes for {path}")
    waveform, sr = codec.decode(EncodedAudio(
        codes=codes.to(codec.device), sample_rate=24000, frame_rate=12.5,
        metadata={"backend": "mimi", "num_quantizers": 16},
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_wav(path, waveform.squeeze(0).cpu(), sr)


def run_text(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=16)
    for epoch in args.epochs:
        model, dataset, _ = load_model_and_data(args, epoch)
        speaker_counts: dict[str, list[int]] = {}
        for idx, row in enumerate(dataset.items):
            if 1 <= float(row.get("duration", 0)) <= 8:
                speaker_counts.setdefault(str(row["speaker_id"]), []).append(idx)
        speaker, indices = max(speaker_counts.items(), key=lambda pair: len(pair[1]))
        indices = sorted(indices, key=lambda i: dataset.items[i]["id"])
        if len(indices) < 20:
            raise ValueError("no speaker has 20 test utterances")
        indices = [indices[round(i * (len(indices) - 1) / 19)] for i in range(20)]
        outputs = []
        rows = []
        for n, idx in enumerate(indices, 1):
            sample = dataset[idx]
            sem = sample["semantic_hidden_states"].unsqueeze(0).to(device)
            mask = sample["semantic_attention_mask"].unsqueeze(0).to(device)
            speaker_id = torch.tensor([sample["speaker_id"]], device=device)
            dialect_id = torch.tensor([sample["dialect_id"]], device=device)
            cfg = TalkerGenerationConfig(max_new_tokens=100, max_audio_frames=100,
                                         do_sample=False, stop_threshold=0.5, return_details=True)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                codes, lengths, reasons, _ = model.generate(
                    sem, speaker_id, dialect_id, cfg, semantic_attention_mask=mask
                )
            codes = codes[0, :, :int(lengths[0])].cpu()
            outputs.append(codes)
            dest = args.output_dir / "text" / f"epoch-{epoch:02d}" / f"sample_{n:03d}.wav"
            decode_save(codec, codes, dest)
            rows.append({"epoch": epoch, "index": n, "sample_id": sample["id"],
                         "speaker": speaker, "text": sample["text"],
                         "frames": codes.shape[-1], "termination": reasons[0],
                         "wav_path": str(dest.relative_to(args.output_dir))})
            print(f"text epoch={epoch} sample={n}/20 frames={codes.shape[-1]}", flush=True)
        pairs = []
        for i in range(20):
            for j in range(i + 1, 20):
                overlap = min(outputs[i].shape[-1], outputs[j].shape[-1])
                matches = outputs[i][:, :overlap] == outputs[j][:, :overlap]
                pairs.append({"epoch": epoch, "left": i + 1, "right": j + 1,
                              "overlap_frames": overlap,
                              "mean_token_agreement": float(matches.float().mean()),
                              "q0_agreement": float(matches[0].float().mean()),
                              "residual_agreement": float(matches[1:].float().mean()),
                              "exact_sequence_same": bool(torch.equal(outputs[i], outputs[j]))})
        dest = args.output_dir / "text" / f"epoch-{epoch:02d}"
        write_csv(dest / "samples.csv", rows)
        write_csv(dest / "pairs.csv", pairs)
        summary = {
            "speaker": speaker, "sample_count": 20, "pairs": len(pairs),
            "exact_same_pairs": sum(pair["exact_sequence_same"] for pair in pairs),
            "mean_pairwise_token_agreement": float(np.mean([pair["mean_token_agreement"] for pair in pairs])),
            "mean_pairwise_q0_agreement": float(np.mean([pair["q0_agreement"] for pair in pairs])),
            "median_generated_frames": float(np.median([row["frames"] for row in rows])),
        }
        write_json(dest / "summary.json", summary)
        print(f"text epoch={epoch} {summary}", flush=True)
        del model
        torch.cuda.empty_cache()


def run_hybrid(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=16)
    for epoch in args.epochs:
        model, dataset, collator = load_model_and_data(args, epoch)
        indices = selected_indices(dataset, args.samples)
        rows = []
        for n, idx in enumerate(indices, 1):
            sample = dataset[idx]
            output, _ = teacher_prediction(model, collator, sample, device)
            prediction = output.logits.argmax(-1)[0].cpu()
            target = sample["audio_codes"]
            variants = {"gt_all": target, "pred_all": prediction}
            for count in (1, 2, 4, 8):
                a = target.clone()
                a[:count] = prediction[:count]
                variants[f"pred_q0_to_{count-1}_gt_rest"] = a
                b = prediction.clone()
                b[:count] = target[:count]
                variants[f"gt_q0_to_{count-1}_pred_rest"] = b
            for name, codes in variants.items():
                dest = args.output_dir / "hybrid" / f"epoch-{epoch:02d}" / f"sample_{n:03d}" / f"{name}.wav"
                decode_save(codec, codes, dest)
                rows.append({"epoch": epoch, "sample_id": sample["id"], "index": n,
                             "text": sample["text"], "variant": name,
                             "token_accuracy": float((codes == target).float().mean()),
                             "q0_accuracy": float((codes[0] == target[0]).float().mean()),
                             "wav_path": str(dest.relative_to(args.output_dir))})
            (args.output_dir / "hybrid" / f"epoch-{epoch:02d}" / f"sample_{n:03d}" / "text.txt").write_text(sample["text"] + "\n", encoding="utf-8")
            if n % 5 == 0:
                print(f"hybrid epoch={epoch} samples={n}/{len(indices)}", flush=True)
        write_csv(args.output_dir / "hybrid" / f"epoch-{epoch:02d}" / "manifest.csv", rows)
        del model
        torch.cuda.empty_cache()


def run_forced_audio(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=16)
    for epoch in args.epochs:
        model, dataset, _ = load_model_and_data(args, epoch)
        indices = selected_indices(dataset, args.samples)
        rows = []
        for n, idx in enumerate(indices, 1):
            sample = dataset[idx]
            target = sample["audio_codes"]
            length = target.shape[-1]
            limit = min(max(250, length), model.config.max_seq_len - sample["semantic_hidden_states"].shape[0])
            generated, stop_probs = rollout(model, sample, 0, limit, device)
            normal_length = stop_frame(stop_probs, model.config.min_audio_frames)
            variants = {
                "ground_truth": target,
                "normal_ar": generated[:, :normal_length],
                "forced_gt_length_ar": generated[:, :length],
            }
            dest = args.output_dir / "forced_audio" / f"epoch-{epoch:02d}" / f"sample_{n:03d}"
            for name, codes in variants.items():
                decode_save(codec, codes, dest / f"{name}.wav")
            (dest / "text.txt").write_text(sample["text"] + "\n", encoding="utf-8")
            rows.append({"epoch": epoch, "sample_id": sample["id"], "text": sample["text"],
                         "target_frames": length, "normal_frames": normal_length,
                         "normal_length_ratio": normal_length / length,
                         "forced_frames": length,
                         "normal_wav": str((dest / "normal_ar.wav").relative_to(args.output_dir)),
                         "forced_wav": str((dest / "forced_gt_length_ar.wav").relative_to(args.output_dir))})
            if n % 5 == 0:
                print(f"forced audio epoch={epoch} samples={n}/{len(indices)}", flush=True)
        write_csv(args.output_dir / "forced_audio" / f"epoch-{epoch:02d}" / "manifest.csv", rows)
        del model
        torch.cuda.empty_cache()


def run_embedding(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    codec = build_codec(backend="mimi", device="cpu", num_quantizers=16)
    quantizer = codec.model.quantizer
    layers = [quantizer.semantic_residual_vector_quantizer.layers[0]]
    layers += list(quantizer.acoustic_residual_vector_quantizer.layers[:15])
    embeddings = [layer.codebook.embed.float() for layer in layers]
    generator = torch.Generator().manual_seed(SEED)
    random_reference = []
    for embed in embeddings:
        a = torch.randint(0, embed.shape[0], (10000,), generator=generator)
        b = torch.randint(0, embed.shape[0], (10000,), generator=generator)
        random_reference.append(float(torch.linalg.vector_norm(embed[a] - embed[b], dim=-1).mean()))
    for epoch in args.epochs:
        model, dataset, collator = load_model_and_data(args, epoch)
        indices = selected_indices(dataset, args.samples)
        rows = []
        for idx in indices:
            sample = dataset[idx]
            output, _ = teacher_prediction(model, collator, sample, device)
            predicted = output.logits.argmax(-1)[0].cpu()
            target = sample["audio_codes"]
            for q, embed in enumerate(embeddings):
                distances = torch.linalg.vector_norm(embed[predicted[q]] - embed[target[q]], dim=-1)
                wrong = predicted[q] != target[q]
                rows.append({"epoch": epoch, "sample_id": sample["id"], "q": q,
                             "frames": target.shape[-1], "exact_accuracy": float((~wrong).float().mean()),
                             "mean_embedding_distance": float(distances.mean()),
                             "wrong_only_embedding_distance": float(distances[wrong].mean()) if wrong.any() else 0.0,
                             "random_pair_distance": random_reference[q]})
        out = args.output_dir / "embedding" / f"epoch-{epoch:02d}.csv"
        write_csv(out, rows)
        summary = []
        for q in range(16):
            selected = [row for row in rows if row["q"] == q]
            weights = [row["frames"] for row in selected]
            summary.append({"q": q, "exact_accuracy": float(np.average([row["exact_accuracy"] for row in selected], weights=weights)),
                            "mean_distance": float(np.average([row["mean_embedding_distance"] for row in selected], weights=weights)),
                            "random_distance": random_reference[q]})
        write_json(out.with_suffix(".json"), {"samples": len(indices), "per_q": summary})
        print(f"embedding epoch={epoch} q0={summary[0]} residual_mean_ratio={np.mean([item['mean_distance']/item['random_distance'] for item in summary[1:]]):.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


def run_position_probe(args: argparse.Namespace) -> None:
    """Probe the batch-padding change to audio absolute positions without changing valid semantics."""
    device = torch.device(args.device)
    for epoch in args.epochs:
        model, dataset, collator = load_model_and_data(args, epoch)
        indices = selected_indices(dataset, args.samples)
        rows = []
        ar_count = 0
        for idx in indices:
            sample = dataset[idx]
            original_length = sample["semantic_hidden_states"].shape[0]
            for padded_length in sorted(set([original_length, max(original_length, 24), max(original_length, 36), max(original_length, 48)])):
                variant = dict(sample)
                if padded_length > original_length:
                    extra = padded_length - original_length
                    variant["semantic_hidden_states"] = F.pad(sample["semantic_hidden_states"], (0, 0, 0, extra))
                    variant["semantic_attention_mask"] = F.pad(sample["semantic_attention_mask"], (0, extra))
                output, batch = teacher_prediction(model, collator, variant, device)
                metrics = distribution_metrics(output.logits.float(), batch["targets"])
                row = {"epoch": epoch, "sample_id": sample["id"], "semantic_length": original_length,
                       "padded_length": padded_length, "position_shift": padded_length - original_length,
                       "codec_ce": metrics["codec_ce"], "mean_accuracy": metrics["mean_accuracy"],
                       "q0_accuracy": metrics["q0_accuracy"], "residual_accuracy": metrics["residual_accuracy"],
                       "ar_mean_accuracy": "", "ar_q0_accuracy": ""}
                if ar_count < 20:
                    target = sample["audio_codes"]
                    predicted, _ = rollout(model, variant, 0, target.shape[-1], device)
                    matches = (predicted == target).float()
                    row["ar_mean_accuracy"] = float(matches.mean())
                    row["ar_q0_accuracy"] = float(matches[0].mean())
                rows.append(row)
            if ar_count < 20:
                ar_count += 1
            if len(rows) % 40 == 0:
                print(f"position epoch={epoch} rows={len(rows)}", flush=True)
        out = args.output_dir / "position_probe" / f"epoch-{epoch:02d}.csv"
        write_csv(out, rows)
        summary = {}
        for shift_name, chosen in (("unshifted", [r for r in rows if r["position_shift"] == 0]),
                                   ("shifted", [r for r in rows if r["position_shift"] > 0])):
            summary[shift_name] = {key: float(np.mean([float(r[key]) for r in chosen]))
                                   for key in ("codec_ce", "mean_accuracy", "q0_accuracy")}
            ar = [r for r in chosen if r["ar_mean_accuracy"] != ""]
            summary[shift_name]["ar_mean_accuracy"] = float(np.mean([r["ar_mean_accuracy"] for r in ar])) if ar else None
            summary[shift_name]["ar_q0_accuracy"] = float(np.mean([r["ar_q0_accuracy"] for r in ar])) if ar else None
            summary[shift_name]["ar_sample_count"] = len(ar)
        write_json(out.with_suffix(".json"), summary)
        print(f"position epoch={epoch} {summary}", flush=True)
        del model
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    if args.stage == "ablation":
        run_ablation(args)
    elif args.stage == "rollout":
        run_rollout(args)
    elif args.stage == "alignment":
        run_alignment(args)
    elif args.stage == "text":
        run_text(args)
    elif args.stage == "hybrid":
        run_hybrid(args)
    elif args.stage == "forced_audio":
        run_forced_audio(args)
    elif args.stage == "embedding":
        run_embedding(args)
    else:
        run_position_probe(args)


if __name__ == "__main__":
    main()
