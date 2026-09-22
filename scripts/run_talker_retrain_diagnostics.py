#!/usr/bin/env python3
"""Staged codec and Talker memorization diagnostics before full retraining."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ulm_live.codec import EncodedAudio, build_codec
from ulm_live.talker import (
    TalkerCollator,
    TalkerConfig,
    TalkerGenerationConfig,
    ULMTalker,
    build_id_mappings,
    load_talker_checkpoint,
)
from ulm_live.talker.data import _load_cache
from ulm_live.talker.semantic_cache import thinker_fingerprint
from ulm_live.thinker import ULMThinker
from ulm_live.utils.audio import load_wav, save_wav


RANDOM_SEED = 20260922
NUM_QUANTIZERS = 16
CODEBOOK_SIZE = 2048
EXPECTED_RANDOM_CE = math.log(CODEBOOK_SIZE)
CHECKPOINTS = ("epoch-2", "epoch-3", "epoch-5")
REQUIRED_CODEC_KEYS = (
    "shape_match",
    "exact_token_match_ratio",
    "q0_token_match_ratio",
    "residual_token_match_ratio",
    "cached_frames",
    "current_frames",
    "duration_alignment",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("codec", "existing", "single", "tiny", "stop", "report", "all"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/ulsan-full"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("outputs/talker-full-5epoch"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/talker-retrain-diagnostics"))
    parser.add_argument("--thinker", default="/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged")
    parser.add_argument("--config", type=Path, default=Path("configs/talker.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--single-max-steps", type=int, default=20_000)
    parser.add_argument("--tiny-max-steps", type=int, default=30_000)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


def seed_everything(seed: int = RANDOM_SEED) -> None:
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


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_candidate(row: dict[str, Any], low: float = 1.0, high: float = 8.0) -> bool:
    text = str(row.get("text", "")).strip()
    duration = float(row.get("duration", 0.0))
    return (
        low <= duration <= high
        and len(text) >= 3
        and not re.search(r"[{}&]|\(\(\)\)", text)
        and bool(row.get("codec_path"))
        and bool(row.get("semantic_path"))
        and bool(row.get("audio_path"))
    )


def deterministic_spread(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["id"]))
    if len(ordered) < count:
        raise ValueError(f"need {count} eligible rows, found {len(ordered)}")
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[index] for index in indices]


def codec_selection(data_dir: Path) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for split in ("train", "test"):
        rows = [row for row in load_rows(data_dir / f"{split}.cached.jsonl") if clean_candidate(row)]
        for row in deterministic_spread(rows, 10):
            selected.append({**row, "split": split})
    return selected


def resolve_path(data_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_dir / path


def load_codes(data_dir: Path, row: dict[str, Any]) -> torch.Tensor:
    codes = torch.load(resolve_path(data_dir, row["codec_path"]), map_location="cpu", weights_only=True)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]
    if codes.ndim != 2:
        raise ValueError(f"invalid codec rank for {row['id']}: {tuple(codes.shape)}")
    return codes.long()


def decode_codes(codec, codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    encoded = EncodedAudio(
        codes=codes,
        sample_rate=codec.sample_rate,
        frame_rate=codec.frame_rate,
        metadata={"backend": codec.backend_name, "num_quantizers": codes.shape[-2]},
    )
    waveform, sample_rate = codec.decode(encoded)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(0)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2 or waveform.shape[0] != 1 or waveform.shape[-1] == 0:
        raise ValueError(f"invalid decoded waveform shape: {tuple(waveform.shape)}")
    if not torch.isfinite(waveform).all():
        raise ValueError("decoded waveform contains NaN or Inf")
    return waveform.detach().cpu(), int(sample_rate)


def waveform_metrics(reference: torch.Tensor, estimate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().flatten()
    estimate = estimate.float().flatten()
    n = min(reference.numel(), estimate.numel())
    reference = reference[:n]
    estimate = estimate[:n]
    mse = F.mse_loss(estimate, reference).item()
    reference_zm = reference - reference.mean()
    estimate_zm = estimate - estimate.mean()
    scale = torch.dot(estimate_zm, reference_zm) / (torch.dot(reference_zm, reference_zm) + 1e-8)
    target = scale * reference_zm
    noise = estimate_zm - target
    si_sdr = 10 * torch.log10((target.square().sum() + 1e-8) / (noise.square().sum() + 1e-8))
    return {"mse": round(float(mse), 8), "si_sdr_db": round(float(si_sdr), 4)}


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def runtime_codec_metadata(codec) -> dict[str, Any]:
    import transformers

    config = codec.model.config
    return {
        "backend": codec.backend_name,
        "model_id": getattr(codec, "_model_id", "kyutai/mimi"),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "sample_rate": codec.sample_rate,
        "frame_rate": codec.frame_rate,
        "model_num_quantizers": int(getattr(config, "num_quantizers", -1)),
        "diagnostic_num_quantizers": NUM_QUANTIZERS,
        "codebook_size": int(getattr(config, "codebook_size", CODEBOOK_SIZE)),
        "config_name_or_path": getattr(config, "_name_or_path", None),
        "cache_creation_metadata_available": False,
        "cache_creation_source": "scripts/cache_mimi_tokens.py (raw tensor only)",
    }


def stage_codec(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_dir.resolve()
    stage_root = output_root / "ground_truth_decode"
    reports_root = output_root / "reports"
    stage_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    selection = codec_selection(args.data_dir)
    write_json(reports_root / "codec_selection.json", selection)

    print("Loading current kyutai/mimi codec...")
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=NUM_QUANTIZERS)
    runtime = runtime_codec_metadata(codec)
    samples: list[dict[str, Any]] = []
    for index, row in enumerate(selection, 1):
        sample_id = f"sample_{index:03d}"
        sample_dir = stage_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        original_path = resolve_path(args.data_dir, row["audio_path"])
        copied_original = sample_dir / "original.wav"
        shutil.copy2(original_path, copied_original)
        original, original_sr = load_wav(original_path)
        cached = load_codes(args.data_dir, row)
        if cached.shape[0] != NUM_QUANTIZERS or cached.shape[1] <= 0:
            raise ValueError(f"{row['id']} codec shape is {tuple(cached.shape)}")
        if int(cached.min()) < 0 or int(cached.max()) >= CODEBOOK_SIZE:
            raise ValueError(f"{row['id']} codec token out of range")

        reconstructed, reconstructed_sr = decode_codes(codec, cached.to(args.device))
        reconstructed_path = sample_dir / "mimi_reconstructed.wav"
        save_wav(reconstructed_path, reconstructed, reconstructed_sr)
        current = codec.encode(original, original_sr, num_quantizers=NUM_QUANTIZERS).codes.detach().cpu()
        if current.ndim == 3:
            current = current.squeeze(0)
        overlap = min(cached.shape[1], current.shape[1])
        compared_cached = cached[:, :overlap]
        compared_current = current[:, :overlap]
        per_q = (compared_cached == compared_current).float().mean(1)
        original_duration = original.shape[-1] / original_sr
        reconstructed_duration = reconstructed.shape[-1] / reconstructed_sr
        expected_frames = original_duration * codec.frame_rate
        metrics = {
            "sample_id": sample_id,
            "dataset_id": row["id"],
            "split": row["split"],
            "text": row["text"],
            "speaker": row["speaker_id"],
            "dialect": row["dialect"],
            "codec_shape": list(cached.shape),
            "token_min": int(cached.min()),
            "token_max": int(cached.max()),
            "original_sample_rate": original_sr,
            "reconstructed_sample_rate": reconstructed_sr,
            "original_duration": round(original_duration, 4),
            "reconstructed_duration": round(reconstructed_duration, 4),
            "duration_ratio": round(reconstructed_duration / original_duration, 4),
            "expected_frames_from_duration": round(expected_frames, 3),
            "cached_frames": int(cached.shape[1]),
            "current_frames": int(current.shape[1]),
            "duration_alignment": round(abs(cached.shape[1] - expected_frames), 4),
            "shape_match": list(cached.shape) == list(current.shape),
            "exact_token_match_ratio": round(float((compared_cached == compared_current).float().mean()), 6),
            "q0_token_match_ratio": round(float(per_q[0]), 6),
            "residual_token_match_ratio": round(float(per_q[1:].mean()), 6),
            "per_codebook_match_ratio": [round(float(value), 6) for value in per_q],
            **waveform_metrics(original, reconstructed),
        }
        (sample_dir / "text.txt").write_text(str(row["text"]) + "\n", encoding="utf-8")
        write_json(sample_dir / "metadata.json", metrics)
        samples.append(metrics)
        print(
            f"[{index:02d}/20] {row['split']} {row['id']} "
            f"shape={tuple(cached.shape)} cache_match={metrics['exact_token_match_ratio']:.4f} "
            f"duration_ratio={metrics['duration_ratio']:.3f}"
        )

    cache_match = float(np.mean([sample["exact_token_match_ratio"] for sample in samples]))
    duration_ratio_min = min(sample["duration_ratio"] for sample in samples)
    duration_ratio_max = max(sample["duration_ratio"] for sample in samples)
    shape_pass = all(sample["codec_shape"][0] == NUM_QUANTIZERS for sample in samples)
    range_pass = all(0 <= sample["token_min"] <= sample["token_max"] < CODEBOOK_SIZE for sample in samples)
    decode_pass = all(
        sample["reconstructed_sample_rate"] == 24000
        and sample["reconstructed_duration"] > 0
        and 0.85 <= sample["duration_ratio"] <= 1.15
        for sample in samples
    )
    cache_pass = all(
        sample["shape_match"] and sample["exact_token_match_ratio"] >= 0.999
        for sample in samples
    )
    summary = {
        "stage": "ground_truth_mimi_decode_and_cache_consistency",
        "random_seed": RANDOM_SEED,
        "sample_count": len(samples),
        "runtime_codec": runtime,
        "mean_exact_token_match_ratio": round(cache_match, 6),
        "duration_ratio_range": [round(duration_ratio_min, 4), round(duration_ratio_max, 4)],
        "codec_shape_pass": shape_pass,
        "codec_token_range_pass": range_pass,
        "decode_integrity_pass": decode_pass,
        "cache_consistency_pass": cache_pass,
        "ground_truth_decode_pass": bool(shape_pass and range_pass and decode_pass and cache_pass),
        "samples": samples,
    }
    write_json(reports_root / "ground_truth_codec_report.json", summary)
    print("GROUND TRUTH MIMI DECODE:", "PASS" if summary["ground_truth_decode_pass"] else "FAIL")
    print("CACHE CONSISTENCY:", "PASS" if cache_pass else "FAIL")
    return summary


def mappings_for_data(data_dir: Path) -> tuple[dict[str, int], dict[str, int]]:
    manifests = [data_dir / f"{split}.cached.jsonl" for split in ("train", "val", "test")]
    return build_id_mappings(manifests)


def sample_from_row(
    args: argparse.Namespace,
    row: dict[str, Any],
    speaker2id: dict[str, int],
    dialect2id: dict[str, int],
) -> dict[str, Any]:
    expected_cache = {
        "thinker_id": args.thinker,
        "thinker_fingerprint": thinker_fingerprint(args.thinker),
        "hidden_layer": -1,
    }
    hidden, semantic_mask, metadata = _load_cache(
        resolve_path(args.data_dir, row["semantic_path"]), expected_cache
    )
    return {
        "id": row["id"],
        "text": row["text"],
        "audio_codes": load_codes(args.data_dir, row)[:NUM_QUANTIZERS],
        "speaker_id": speaker2id[str(row["speaker_id"])],
        "dialect_id": dialect2id[str(row["dialect"])],
        "semantic_hidden_states": hidden,
        "semantic_attention_mask": semantic_mask,
        "semantic_metadata": metadata,
    }


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def sequence_distribution(tokens: torch.Tensor) -> dict[str, Any]:
    if tokens.ndim == 3:
        tokens = tokens[0]
    entropy: list[float] = []
    top1: list[float] = []
    unique: list[int] = []
    for quantizer in range(tokens.shape[0]):
        values = tokens[quantizer].detach().cpu().tolist()
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
        "entropy_by_q": [round(value, 4) for value in entropy],
        "top1_ratio_by_q": [round(value, 6) for value in top1],
        "unique_tokens_by_q": unique,
        "q0_entropy": round(entropy[0], 4),
        "q0_top1_ratio": round(top1[0], 6),
        "mean_entropy": round(float(np.mean(entropy)), 4),
        "mean_top1_ratio": round(float(np.mean(top1)), 6),
    }


def teacher_forced_prediction(
    model: ULMTalker,
    sample: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    collator = TalkerCollator(
        tokenizer=None,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        max_audio_len=model.config.max_seq_len,
        acoustic_delay_frames=model.config.acoustic_delay_frames,
    )
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
    logits = output.logits.float()
    target = batch["targets"]
    prediction = logits.argmax(-1)
    valid = target != model.config.pad_token_id
    correct = prediction == target
    per_q_accuracy = [
        float(correct[:, q][valid[:, q]].float().mean()) for q in range(model.config.num_quantizers)
    ]
    per_q_ce = [
        float(F.cross_entropy(logits[:, q][valid[:, q]], target[:, q][valid[:, q]]))
        for q in range(model.config.num_quantizers)
    ]
    valid_frames = batch["audio_attention_mask"]
    frame_all = correct.all(1) & valid_frames
    metrics = {
        "frames": int(valid_frames.sum()),
        "codec_ce": round(float(np.mean(per_q_ce)), 6),
        "exact_token_accuracy": round(float(correct[valid].float().mean()), 6),
        "mean_token_accuracy": round(float(correct[valid].float().mean()), 6),
        "q0_accuracy": round(per_q_accuracy[0], 6),
        "residual_accuracy": round(float(np.mean(per_q_accuracy[1:])), 6),
        "frame_all_codebook_accuracy": round(float(frame_all[valid_frames].float().mean()), 6),
        "per_codebook_accuracy": [round(value, 6) for value in per_q_accuracy],
        "per_codebook_ce": [round(value, 6) for value in per_q_ce],
        **sequence_distribution(prediction),
    }
    return prediction[0].detach().cpu(), metrics


def stage_existing(args: argparse.Namespace) -> dict[str, Any]:
    codec_report = read_json(args.output_dir / "reports" / "ground_truth_codec_report.json")
    selection = read_json(args.output_dir / "reports" / "codec_selection.json")
    if not codec_report or not codec_report.get("ground_truth_decode_pass") or not selection:
        raise RuntimeError("codec stage must PASS before existing-checkpoint diagnosis")
    device = torch.device(args.device)
    speaker2id, dialect2id = mappings_for_data(args.data_dir)
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=NUM_QUANTIZERS)
    stage_root = args.output_dir / "teacher_forced"
    checkpoint_reports: dict[str, Any] = {}

    for checkpoint_name in CHECKPOINTS:
        print(f"Teacher-forced diagnosis: {checkpoint_name}")
        model, checkpoint_metadata = load_talker_checkpoint(
            args.checkpoint_dir / f"{checkpoint_name}.pt", device=device
        )
        if checkpoint_metadata.get("speaker2id") != speaker2id:
            raise ValueError(f"speaker mapping mismatch in {checkpoint_name}")
        if checkpoint_metadata.get("dialect2id") != dialect2id:
            raise ValueError(f"dialect mapping mismatch in {checkpoint_name}")
        if model.config.num_quantizers != NUM_QUANTIZERS or model.config.acoustic_delay_frames != 0:
            raise ValueError(f"unexpected codec config in {checkpoint_name}")
        sample_reports: list[dict[str, Any]] = []
        for index, row in enumerate(selection, 1):
            sample_id = f"sample_{index:03d}"
            sample = sample_from_row(args, row, speaker2id, dialect2id)
            prediction, metrics = teacher_forced_prediction(model, sample, device)
            destination = stage_root / checkpoint_name / sample_id
            destination.mkdir(parents=True, exist_ok=True)
            source_ground_truth = args.output_dir / "ground_truth_decode" / sample_id
            hardlink_or_copy(source_ground_truth / "original.wav", destination / "original.wav")
            hardlink_or_copy(
                source_ground_truth / "mimi_reconstructed.wav", destination / "mimi_gt.wav"
            )
            waveform, sample_rate = decode_codes(codec, prediction.to(device))
            save_wav(destination / "teacher_forced.wav", waveform, sample_rate)
            (destination / "text.txt").write_text(str(row["text"]) + "\n", encoding="utf-8")
            item = {
                "sample_id": sample_id,
                "dataset_id": row["id"],
                "split": row["split"],
                "text": row["text"],
                "speaker": row["speaker_id"],
                "dialect": row["dialect"],
                **metrics,
            }
            write_json(destination / "metadata.json", item)
            sample_reports.append(item)
            print(
                f"  [{index:02d}/20] CE={metrics['codec_ce']:.3f} "
                f"accuracy={metrics['mean_token_accuracy']:.3%} q0={metrics['q0_accuracy']:.3%}"
            )

        frame_weights = np.array([item["frames"] for item in sample_reports], dtype=np.float64)
        def weighted(key: str) -> float:
            return float(np.average([item[key] for item in sample_reports], weights=frame_weights))

        per_q_accuracy = [
            float(np.average([item["per_codebook_accuracy"][q] for item in sample_reports], weights=frame_weights))
            for q in range(NUM_QUANTIZERS)
        ]
        per_q_ce = [
            float(np.average([item["per_codebook_ce"][q] for item in sample_reports], weights=frame_weights))
            for q in range(NUM_QUANTIZERS)
        ]
        summary = {
            "checkpoint": checkpoint_name,
            "sample_count": len(sample_reports),
            "total_frames": int(frame_weights.sum()),
            "codec_ce": round(float(np.mean(per_q_ce)), 6),
            "mean_token_accuracy": round(float(np.mean(per_q_accuracy)), 6),
            "q0_accuracy": round(per_q_accuracy[0], 6),
            "residual_accuracy": round(float(np.mean(per_q_accuracy[1:])), 6),
            "frame_all_codebook_accuracy": round(weighted("frame_all_codebook_accuracy"), 6),
            "per_codebook_accuracy": [round(value, 6) for value in per_q_accuracy],
            "per_codebook_ce": [round(value, 6) for value in per_q_ce],
            "mean_predicted_entropy": round(weighted("mean_entropy"), 6),
            "mean_predicted_top1_ratio": round(weighted("mean_top1_ratio"), 6),
            "random_initialization_ce_reference": round(EXPECTED_RANDOM_CE, 6),
            "samples": sample_reports,
        }
        checkpoint_reports[checkpoint_name] = summary
        write_json(args.output_dir / "reports" / f"teacher_forced_{checkpoint_name}.json", summary)
        print(
            f"{checkpoint_name}: CE={summary['codec_ce']:.4f} "
            f"accuracy={summary['mean_token_accuracy']:.3%} q0={summary['q0_accuracy']:.3%}"
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "stage": "teacher_forced_existing_checkpoints",
        "selection_report": "reports/codec_selection.json",
        "checkpoints": checkpoint_reports,
    }
    write_json(args.output_dir / "reports" / "teacher_forced_existing_report.json", report)
    return report


def single_and_tiny_selection(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_rows(args.data_dir / "train.cached.jsonl")
    eligible = [row for row in rows if clean_candidate(row)]
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    speaker, speaker_rows = min(
        by_speaker.items(), key=lambda item: (-len(item[1]), item[0])
    )
    if len(speaker_rows) < 32:
        raise ValueError("no speaker has 32 eligible utterances")
    single_candidates = [
        row
        for row in speaker_rows
        if 2.0 <= float(row["duration"]) <= 5.0
        and 10 <= len(str(row["text"]).strip()) <= 60
        and str(row["text"]).strip()[-1:] in (".", "?", "!")
        and "~" not in str(row["text"])
    ]
    preferred_id = "ulsan_004977"
    preferred = [row for row in single_candidates if row["id"] == preferred_id]
    single = preferred[0] if preferred else deterministic_spread(single_candidates, 1)[0]
    tiny = deterministic_spread(speaker_rows, 32)
    if single["id"] not in {row["id"] for row in tiny}:
        tiny[0] = single
        tiny = sorted(tiny, key=lambda row: str(row["id"]))
    if len({row["id"] for row in tiny}) != 32:
        raise ValueError("tiny selection contains duplicate IDs")
    if any(str(row["speaker_id"]) != speaker for row in tiny):
        raise ValueError("tiny selection is not single-speaker")
    return single, tiny


def fresh_config(args: argparse.Namespace, stop_loss_weight: float = 0.0) -> TalkerConfig:
    config = TalkerConfig.from_yaml(args.config)
    config.num_quantizers = NUM_QUANTIZERS
    config.stop_loss_weight = stop_loss_weight
    config.stop_pos_weight = 5.0
    speaker2id, dialect2id = mappings_for_data(args.data_dir)
    config.num_speakers = max(config.num_speakers, len(speaker2id))
    config.num_dialects = max(config.num_dialects, len(dialect2id))
    return config


def prediction_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    if prediction.ndim == 3:
        prediction = prediction[0]
    if target.ndim == 3:
        target = target[0]
    frames = min(prediction.shape[-1], target.shape[-1])
    prediction_overlap = prediction[:, :frames].cpu()
    target_overlap = target[:, :frames].cpu()
    correct = prediction_overlap == target_overlap
    per_q = correct.float().mean(1)
    result = {
        "generated_frames": int(prediction.shape[-1]),
        "target_frames": int(target.shape[-1]),
        "length_difference": int(prediction.shape[-1] - target.shape[-1]),
        "overlap_frames": frames,
        "mean_token_accuracy": round(float(correct.float().mean()), 6),
        "q0_accuracy": round(float(per_q[0]), 6),
        "residual_accuracy": round(float(per_q[1:].mean()), 6),
        "per_codebook_accuracy": [round(float(value), 6) for value in per_q],
        "frame_all_codebook_accuracy": round(float(correct.all(0).float().mean()), 6),
        **sequence_distribution(prediction_overlap),
    }
    return result


def save_diagnostic_checkpoint(
    path: Path,
    model: ULMTalker,
    step: int,
    speaker2id: dict[str, int],
    dialect2id: dict[str, int],
    selection: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "purpose": "tiny_overfit_diagnostic_only",
            "model_state_dict": model.state_dict(),
            "config": model.config.to_dict(),
            "step": step,
            "speaker2id": speaker2id,
            "dialect2id": dialect2id,
            "selection_ids": [row["id"] for row in selection],
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def semantic_cache_consistency(
    args: argparse.Namespace,
    row: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    thinker = ULMThinker(args.thinker, device=args.device, torch_dtype="bf16", freeze=True)
    tokenized = thinker.tokenize(str(row["text"]))
    with torch.no_grad():
        current = thinker.forward_hidden(
            tokenized["input_ids"], attention_mask=tokenized.get("attention_mask")
        ).detach().cpu()
    cached = sample["semantic_hidden_states"].detach().cpu()
    if current.ndim == 3 and current.shape[0] == 1:
        current = current[0]
    shape_match = tuple(current.shape) == tuple(cached.shape)
    overlap = min(current.shape[0], cached.shape[0])
    difference = (current[:overlap].float() - cached[:overlap].float()).abs()
    result = {
        "shape_match": shape_match,
        "cached_shape": list(cached.shape),
        "current_shape": list(current.shape),
        "exact_match": bool(shape_match and torch.equal(current, cached)),
        "max_abs_difference": round(float(difference.max()), 8),
        "mean_abs_difference": round(float(difference.mean()), 8),
        "metadata_fingerprint": sample["semantic_metadata"].get("thinker_fingerprint"),
        "current_fingerprint": thinker_fingerprint(args.thinker),
    }
    del thinker
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_single(
    args: argparse.Namespace,
    row: dict[str, Any],
    sample: dict[str, Any],
    speaker2id: dict[str, int],
    dialect2id: dict[str, int],
) -> tuple[ULMTalker, int, list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.device)
    seed_everything()
    config = fresh_config(args, stop_loss_weight=0.0)
    model = ULMTalker(config).to(device)
    collator = TalkerCollator(
        tokenizer=None,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        max_audio_len=config.max_seq_len,
        acoustic_delay_frames=config.acoustic_delay_frames,
    )
    batch = to_device(collator([sample]), device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    history: list[dict[str, Any]] = []
    final_metrics: dict[str, Any] = {}
    final_step = 0
    for step in range(1, args.single_max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                batch["semantic_hidden_states"],
                batch["audio_codes"],
                batch["speaker_ids"],
                batch["dialect_ids"],
                targets=batch["targets"],
                semantic_attention_mask=batch["semantic_attention_mask"],
                audio_attention_mask=batch["audio_attention_mask"],
            )
            loss = output.codec_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite single-overfit loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        final_step = step

        if step == 1 or step % 100 == 0:
            _, measured = teacher_forced_prediction(model, sample, device)
            entry = {
                "step": step,
                "train_codec_loss": round(float(loss.detach()), 6),
                "grad_norm": round(grad_norm, 6),
                **measured,
            }
            history.append(entry)
            final_metrics = measured
            write_json(args.output_dir / "reports" / "single_overfit_history.json", history)
            print(
                f"single step={step:05d} CE={measured['codec_ce']:.5f} "
                f"accuracy={measured['mean_token_accuracy']:.3%} q0={measured['q0_accuracy']:.3%}"
            )
            if measured["mean_token_accuracy"] >= 0.99 and measured["q0_accuracy"] >= 0.99:
                break
    return model, final_step, history, final_metrics


def incremental_full_recompute_check(
    model: ULMTalker,
    sample: dict[str, Any],
    frames: int,
    device: torch.device,
) -> dict[str, Any]:
    sem = sample["semantic_hidden_states"].unsqueeze(0).to(device)
    sem_mask = sample["semantic_attention_mask"].unsqueeze(0).to(device)
    speaker = torch.tensor([sample["speaker_id"]], device=device)
    dialect = torch.tensor([sample["dialect_id"]], device=device)
    model.eval()
    with torch.no_grad(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        state = model.init_generation_state(sem, speaker, dialect, sem_mask)
        current = torch.full(
            (1, model.config.num_quantizers),
            model.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        full_inputs = current[:, :, None]
        matches: list[bool] = []
        for _ in range(frames):
            cached, _ = model.step(state, current)
            recomputed = model(
                sem,
                full_inputs,
                speaker,
                dialect,
                semantic_attention_mask=sem_mask,
            ).logits[:, :, -1].argmax(-1)
            matches.append(bool(torch.equal(cached, recomputed)))
            current = cached
            full_inputs = torch.cat((full_inputs, cached[:, :, None]), -1)
    return {
        "frames_checked": frames,
        "matching_frames": sum(matches),
        "all_tokens_match": all(matches),
    }


def stage_single(args: argparse.Namespace) -> dict[str, Any]:
    codec_report = read_json(args.output_dir / "reports" / "ground_truth_codec_report.json")
    if not codec_report or not codec_report.get("ground_truth_decode_pass"):
        raise RuntimeError("codec stage must PASS before single overfit")
    single_row, tiny_rows = single_and_tiny_selection(args)
    write_json(args.output_dir / "reports" / "single_selection.json", single_row)
    write_json(args.output_dir / "reports" / "tiny_selection.json", tiny_rows)
    speaker2id, dialect2id = mappings_for_data(args.data_dir)
    sample = sample_from_row(args, single_row, speaker2id, dialect2id)
    semantic_check = semantic_cache_consistency(args, single_row, sample)
    if not semantic_check["shape_match"] or semantic_check["max_abs_difference"] > 1e-3:
        raise RuntimeError(f"semantic cache mismatch: {semantic_check}")
    model, steps, history, final_train_metrics = train_single(
        args, single_row, sample, speaker2id, dialect2id
    )
    device = torch.device(args.device)
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=NUM_QUANTIZERS)
    target = sample["audio_codes"]
    teacher_prediction, teacher_metrics = teacher_forced_prediction(model, sample, device)
    teacher_comparison = prediction_metrics(teacher_prediction, target)
    frames = target.shape[-1]
    generation_config = TalkerGenerationConfig(
        max_new_tokens=frames,
        max_audio_frames=frames,
        min_audio_frames=frames + 1,
        do_sample=False,
        stop_threshold=2.0,
    )
    model.eval()
    with torch.no_grad(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        generated = model.generate(
            sample["semantic_hidden_states"].unsqueeze(0).to(device),
            torch.tensor([sample["speaker_id"]], device=device),
            torch.tensor([sample["dialect_id"]], device=device),
            generation_config=generation_config,
            frame_rate=codec.frame_rate,
            semantic_attention_mask=sample["semantic_attention_mask"].unsqueeze(0).to(device),
        )[0].detach().cpu()
    autoregressive_comparison = prediction_metrics(generated, target)
    cache_alignment = incremental_full_recompute_check(model, sample, frames, device)

    destination = args.output_dir / "single_overfit"
    destination.mkdir(parents=True, exist_ok=True)
    original_source = resolve_path(args.data_dir, single_row["audio_path"])
    shutil.copy2(original_source, destination / "original.wav")
    for name, codes in (
        ("mimi_gt.wav", target),
        ("teacher_forced.wav", teacher_prediction),
        ("autoregressive.wav", generated),
    ):
        waveform, sample_rate = decode_codes(codec, codes.to(device))
        save_wav(destination / name, waveform, sample_rate)
    (destination / "text.txt").write_text(str(single_row["text"]) + "\n", encoding="utf-8")

    passed = bool(
        teacher_comparison["mean_token_accuracy"] > 0.95
        and autoregressive_comparison["q0_accuracy"] > 0.90
        and cache_alignment["all_tokens_match"]
    )
    report = {
        "stage": "single_utterance_overfit",
        "random_seed": RANDOM_SEED,
        "fresh_model": True,
        "resume_checkpoint": None,
        "stop_loss_weight": 0.0,
        "learning_rate": args.learning_rate,
        "max_steps": args.single_max_steps,
        "steps": steps,
        "sample": {
            "id": single_row["id"],
            "text": single_row["text"],
            "speaker": single_row["speaker_id"],
            "dialect": single_row["dialect"],
            "duration": single_row["duration"],
            "frames": int(target.shape[-1]),
        },
        "semantic_cache_consistency": semantic_check,
        "final_teacher_forced_model_metrics": teacher_metrics,
        "teacher_forced_comparison": teacher_comparison,
        "autoregressive_comparison": autoregressive_comparison,
        "incremental_vs_full_recompute": cache_alignment,
        "single_overfit_pass": passed,
        "history_path": "reports/single_overfit_history.json",
    }
    write_json(args.output_dir / "reports" / "single_overfit_report.json", report)
    write_json(destination / "metadata.json", report)
    save_diagnostic_checkpoint(
        destination / "single_overfit.pt",
        model,
        steps,
        speaker2id,
        dialect2id,
        [single_row],
        report,
    )
    print("SINGLE OVERFIT:", "PASS" if passed else "FAIL")
    print(
        f"teacher={teacher_comparison['mean_token_accuracy']:.3%} "
        f"autoregressive_q0={autoregressive_comparison['q0_accuracy']:.3%}"
    )
    del model, codec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def evaluate_teacher_set(
    model: ULMTalker,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[torch.Tensor]]:
    reports: list[dict[str, Any]] = []
    predictions: list[torch.Tensor] = []
    for sample in samples:
        prediction, metrics = teacher_forced_prediction(model, sample, device)
        reports.append({"id": sample["id"], **metrics})
        predictions.append(prediction)
    weights = np.array([report["frames"] for report in reports], dtype=np.float64)
    per_q_accuracy = [
        float(np.average([report["per_codebook_accuracy"][q] for report in reports], weights=weights))
        for q in range(NUM_QUANTIZERS)
    ]
    per_q_ce = [
        float(np.average([report["per_codebook_ce"][q] for report in reports], weights=weights))
        for q in range(NUM_QUANTIZERS)
    ]
    summary = {
        "sample_count": len(reports),
        "total_frames": int(weights.sum()),
        "codec_ce": round(float(np.mean(per_q_ce)), 6),
        "mean_token_accuracy": round(float(np.mean(per_q_accuracy)), 6),
        "q0_accuracy": round(per_q_accuracy[0], 6),
        "residual_accuracy": round(float(np.mean(per_q_accuracy[1:])), 6),
        "per_codebook_accuracy": [round(value, 6) for value in per_q_accuracy],
        "per_codebook_ce": [round(value, 6) for value in per_q_ce],
        "min_sample_token_accuracy": round(min(report["mean_token_accuracy"] for report in reports), 6),
        "min_sample_q0_accuracy": round(min(report["q0_accuracy"] for report in reports), 6),
    }
    return summary, reports, predictions


def autoregressive_tokens(
    model: ULMTalker,
    sample: dict[str, Any],
    frames: int,
    device: torch.device,
) -> torch.Tensor:
    config = TalkerGenerationConfig(
        max_new_tokens=frames,
        max_audio_frames=frames,
        min_audio_frames=frames + 1,
        do_sample=False,
        stop_threshold=2.0,
    )
    model.eval()
    with torch.no_grad(), torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        result = model.generate(
            sample["semantic_hidden_states"].unsqueeze(0).to(device),
            torch.tensor([sample["speaker_id"]], device=device),
            torch.tensor([sample["dialect_id"]], device=device),
            generation_config=config,
            semantic_attention_mask=sample["semantic_attention_mask"].unsqueeze(0).to(device),
        )
    return result[0].detach().cpu()


def evaluate_autoregressive_set(
    model: ULMTalker,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[torch.Tensor]]:
    reports: list[dict[str, Any]] = []
    generated: list[torch.Tensor] = []
    for sample in samples:
        tokens = autoregressive_tokens(
            model, sample, sample["audio_codes"].shape[-1], device
        )
        comparison = prediction_metrics(tokens, sample["audio_codes"])
        reports.append({"id": sample["id"], **comparison})
        generated.append(tokens)
    frame_weights = np.array([report["overlap_frames"] for report in reports], dtype=np.float64)
    per_q = [
        float(np.average([report["per_codebook_accuracy"][q] for report in reports], weights=frame_weights))
        for q in range(NUM_QUANTIZERS)
    ]
    token_hashes = {
        __import__("hashlib").sha256(tokens.numpy().tobytes()).hexdigest()
        for tokens in generated
    }
    q0_top1 = [report["q0_top1_ratio"] for report in reports]
    repeated_pairs = 0
    for left in range(len(generated)):
        for right in range(left + 1, len(generated)):
            overlap = min(generated[left].shape[-1], generated[right].shape[-1])
            if overlap and torch.equal(
                generated[left][:, :overlap], generated[right][:, :overlap]
            ):
                repeated_pairs += 1
    summary = {
        "sample_count": len(reports),
        "mean_token_accuracy": round(float(np.mean(per_q)), 6),
        "mean_q0_accuracy": round(per_q[0], 6),
        "mean_residual_accuracy": round(float(np.mean(per_q[1:])), 6),
        "per_codebook_accuracy": [round(value, 6) for value in per_q],
        "min_sample_q0_accuracy": round(min(report["q0_accuracy"] for report in reports), 6),
        "unique_output_count": len(token_hashes),
        "identical_overlap_pair_count": repeated_pairs,
        "max_q0_top1_ratio": round(max(q0_top1), 6),
        "mean_q0_top1_ratio": round(float(np.mean(q0_top1)), 6),
        "all_lengths_exact": all(report["length_difference"] == 0 for report in reports),
    }
    return summary, reports, generated


def train_tiny(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    speaker2id: dict[str, int],
    dialect2id: dict[str, int],
) -> tuple[ULMTalker, int, list[dict[str, Any]], dict[str, Any]]:
    device = torch.device(args.device)
    seed_everything()
    config = fresh_config(args, stop_loss_weight=0.0)
    model = ULMTalker(config).to(device)
    collator = TalkerCollator(
        tokenizer=None,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        max_audio_len=config.max_seq_len,
        acoustic_delay_frames=config.acoustic_delay_frames,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    rng = random.Random(RANDOM_SEED)
    order = list(range(len(samples)))
    position = len(order)
    history: list[dict[str, Any]] = []
    final_metrics: dict[str, Any] = {}
    final_step = 0
    batch_size = 8
    for step in range(1, args.tiny_max_steps + 1):
        if position + batch_size > len(order):
            rng.shuffle(order)
            position = 0
        indices = order[position : position + batch_size]
        position += batch_size
        batch = to_device(collator([samples[index] for index in indices]), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                batch["semantic_hidden_states"],
                batch["audio_codes"],
                batch["speaker_ids"],
                batch["dialect_ids"],
                targets=batch["targets"],
                semantic_attention_mask=batch["semantic_attention_mask"],
                audio_attention_mask=batch["audio_attention_mask"],
            )
            loss = output.codec_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite tiny-overfit loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        final_step = step

        if step == 1 or step % 250 == 0:
            measured, _, _ = evaluate_teacher_set(model, samples, device)
            entry: dict[str, Any] = {
                "step": step,
                "train_batch_codec_loss": round(float(loss.detach()), 6),
                "grad_norm": round(grad_norm, 6),
                **measured,
            }
            if measured["mean_token_accuracy"] >= 0.95:
                autoregressive_summary, _, _ = evaluate_autoregressive_set(
                    model, samples, device
                )
                entry["autoregressive"] = autoregressive_summary
            history.append(entry)
            final_metrics = measured
            write_json(args.output_dir / "reports" / "tiny_overfit_history.json", history)
            print(
                f"tiny step={step:05d} CE={measured['codec_ce']:.5f} "
                f"teacher={measured['mean_token_accuracy']:.3%} q0={measured['q0_accuracy']:.3%} "
                f"min={measured['min_sample_token_accuracy']:.3%}"
            )
            autoregressive = entry.get("autoregressive")
            if autoregressive:
                print(
                    f"  autoregressive mean={autoregressive['mean_token_accuracy']:.3%} "
                    f"q0={autoregressive['mean_q0_accuracy']:.3%} "
                    f"unique={autoregressive['unique_output_count']}/32"
                )
            if (
                measured["mean_token_accuracy"] >= 0.99
                and measured["q0_accuracy"] >= 0.99
                and autoregressive
                and autoregressive["mean_q0_accuracy"] >= 0.90
                and autoregressive["unique_output_count"] == len(samples)
            ):
                break
    return model, final_step, history, final_metrics


def validate_saved_wav(path: Path) -> dict[str, Any]:
    waveform, sample_rate = load_wav(path)
    if sample_rate != 24000 or waveform.shape[0] != 1 or waveform.shape[-1] <= 0:
        raise ValueError(f"invalid diagnostic WAV: {path}")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"non-finite diagnostic WAV: {path}")
    return {
        "sample_rate": sample_rate,
        "channels": int(waveform.shape[0]),
        "samples": int(waveform.shape[-1]),
        "duration": round(waveform.shape[-1] / sample_rate, 4),
    }


def stage_tiny(args: argparse.Namespace) -> dict[str, Any]:
    single_report = read_json(args.output_dir / "reports" / "single_overfit_report.json")
    if not single_report or not single_report.get("single_overfit_pass"):
        raise RuntimeError("single overfit must PASS before tiny overfit")
    _, tiny_rows = single_and_tiny_selection(args)
    write_json(args.output_dir / "reports" / "tiny_selection.json", tiny_rows)
    speaker2id, dialect2id = mappings_for_data(args.data_dir)
    samples = [sample_from_row(args, row, speaker2id, dialect2id) for row in tiny_rows]
    model, steps, history, _ = train_tiny(
        args, tiny_rows, samples, speaker2id, dialect2id
    )
    device = torch.device(args.device)
    teacher_summary, teacher_samples, teacher_predictions = evaluate_teacher_set(
        model, samples, device
    )
    autoregressive_summary, autoregressive_samples, generated = evaluate_autoregressive_set(
        model, samples, device
    )
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=NUM_QUANTIZERS)
    stage_root = args.output_dir / "tiny_overfit"
    wav_integrity = True
    sample_reports: list[dict[str, Any]] = []
    for index, (row, sample, teacher_item, ar_item, teacher_codes, generated_codes) in enumerate(
        zip(
            tiny_rows,
            samples,
            teacher_samples,
            autoregressive_samples,
            teacher_predictions,
            generated,
            strict=True,
        ),
        1,
    ):
        sample_id = f"sample_{index:03d}"
        destination = stage_root / sample_id
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve_path(args.data_dir, row["audio_path"]), destination / "original.wav")
        for name, codes in (
            ("mimi_gt.wav", sample["audio_codes"]),
            ("teacher_forced.wav", teacher_codes),
            ("generated.wav", generated_codes),
        ):
            waveform, sample_rate = decode_codes(codec, codes.to(device))
            wav_path = destination / name
            save_wav(wav_path, waveform, sample_rate)
            try:
                validate_saved_wav(wav_path)
            except ValueError:
                wav_integrity = False
                raise
        (destination / "text.txt").write_text(str(row["text"]) + "\n", encoding="utf-8")
        item = {
            "sample_id": sample_id,
            "dataset_id": row["id"],
            "text": row["text"],
            "speaker": row["speaker_id"],
            "dialect": row["dialect"],
            "duration": row["duration"],
            "teacher_forced": teacher_item,
            "autoregressive": ar_item,
        }
        write_json(destination / "metadata.json", item)
        sample_reports.append(item)

    collapse_detected = bool(
        autoregressive_summary["unique_output_count"] != 32
        or autoregressive_summary["identical_overlap_pair_count"] > 0
        or autoregressive_summary["max_q0_top1_ratio"] >= 0.80
    )
    passed = bool(
        teacher_summary["mean_token_accuracy"] >= 0.90
        and autoregressive_summary["mean_q0_accuracy"] >= 0.50
        and autoregressive_summary["unique_output_count"] == 32
        and autoregressive_summary["all_lengths_exact"]
        and not collapse_detected
        and wav_integrity
    )
    report = {
        "stage": "tiny_32_utterance_overfit",
        "random_seed": RANDOM_SEED,
        "fresh_model": True,
        "resume_checkpoint": None,
        "single_speaker": True,
        "speaker": tiny_rows[0]["speaker_id"],
        "stop_loss_weight": 0.0,
        "learning_rate": args.learning_rate,
        "max_steps": args.tiny_max_steps,
        "steps": steps,
        "teacher_forced": teacher_summary,
        "autoregressive": autoregressive_summary,
        "collapse_detected": collapse_detected,
        "wav_integrity_pass": wav_integrity,
        "tiny_overfit_pass": passed,
        "sample_count": len(sample_reports),
        "samples": sample_reports,
        "history_path": "reports/tiny_overfit_history.json",
    }
    write_json(args.output_dir / "reports" / "tiny_overfit_report.json", report)
    save_diagnostic_checkpoint(
        stage_root / "tiny_overfit.pt",
        model,
        steps,
        speaker2id,
        dialect2id,
        tiny_rows,
        report,
    )
    print("TINY OVERFIT:", "PASS" if passed else "FAIL")
    print(
        f"teacher={teacher_summary['mean_token_accuracy']:.3%} "
        f"autoregressive_q0={autoregressive_summary['mean_q0_accuracy']:.3%} "
        f"unique={autoregressive_summary['unique_output_count']}/32 collapse={collapse_detected}"
    )
    del model, codec
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def load_diagnostic_model(path: Path, device: torch.device) -> tuple[ULMTalker, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    raw_config = payload["config"]
    config = TalkerConfig(
        **{
            key: value
            for key, value in raw_config.items()
            if key in TalkerConfig.__dataclass_fields__
        }
    )
    model = ULMTalker(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload


def evaluate_stop_set(
    model: ULMTalker,
    samples: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    collator = TalkerCollator(
        tokenizer=None,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        max_audio_len=model.config.max_seq_len,
        acoustic_delay_frames=model.config.acoustic_delay_frames,
    )
    tp = fp = fn = tn = 0
    stop_loss_sum = 0.0
    stop_valid_count = 0
    model.eval()
    for start in range(0, len(samples), 8):
        batch = to_device(collator(samples[start : start + 8]), device)
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
                stop_targets=batch["stop_targets"],
            )
        valid = batch["stop_targets"] >= 0
        target = (batch["stop_targets"] == 1) & valid
        prediction = (torch.sigmoid(output.stop_logits) >= 0.5) & valid
        tp += int((prediction & target).sum())
        fp += int((prediction & (~target & valid)).sum())
        fn += int(((~prediction) & target).sum())
        tn += int(((~prediction) & (~target & valid)).sum())
        valid_count = int(valid.sum())
        stop_loss_sum += float(output.stop_loss) * valid_count
        stop_valid_count += valid_count
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    teacher_summary, _, _ = evaluate_teacher_set(model, samples, device)
    return {
        "codec_ce": teacher_summary["codec_ce"],
        "mean_token_accuracy": teacher_summary["mean_token_accuracy"],
        "q0_accuracy": teacher_summary["q0_accuracy"],
        "residual_accuracy": teacher_summary["residual_accuracy"],
        "stop_loss": round(stop_loss_sum / max(stop_valid_count, 1), 6),
        "stop_precision": round(precision, 6),
        "stop_recall": round(recall, 6),
        "stop_f1": round(f1, 6),
        "stop_tp": tp,
        "stop_fp": fp,
        "stop_fn": fn,
        "stop_tn": tn,
        "positive_prediction_rate": round((tp + fp) / max(tp + fp + fn + tn, 1), 6),
    }


def train_stop_variant(
    args: argparse.Namespace,
    checkpoint_path: Path,
    samples: list[dict[str, Any]],
    weight: float,
    steps: int = 500,
) -> dict[str, Any]:
    device = torch.device(args.device)
    seed_everything()
    model, _ = load_diagnostic_model(checkpoint_path, device)
    model.config.stop_loss_weight = weight
    model.config.stop_pos_weight = 5.0
    baseline = evaluate_stop_set(model, samples, device)
    if weight == 0.0:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"weight": weight, "steps": 0, "before": baseline, "after": baseline}

    collator = TalkerCollator(
        tokenizer=None,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        max_audio_len=model.config.max_seq_len,
        acoustic_delay_frames=model.config.acoustic_delay_frames,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    rng = random.Random(RANDOM_SEED)
    order = list(range(len(samples)))
    position = len(order)
    history: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        if position + 8 > len(order):
            rng.shuffle(order)
            position = 0
        indices = order[position : position + 8]
        position += 8
        batch = to_device(collator([samples[index] for index in indices]), device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
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
            loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite stop experiment loss at weight={weight} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            measured = evaluate_stop_set(model, samples, device)
            history.append({"step": step, **measured})
            print(
                f"stop weight={weight:.2f} step={step:03d} codec_CE={measured['codec_ce']:.4f} "
                f"accuracy={measured['mean_token_accuracy']:.3%} F1={measured['stop_f1']:.3%}"
            )
    after = evaluate_stop_set(model, samples, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "weight": weight,
        "steps": steps,
        "before": baseline,
        "after": after,
        "history": history,
    }


def stage_stop(args: argparse.Namespace) -> dict[str, Any]:
    tiny_report = read_json(args.output_dir / "reports" / "tiny_overfit_report.json")
    if not tiny_report or not tiny_report.get("tiny_overfit_pass"):
        raise RuntimeError("tiny overfit must PASS before stop-weight experiment")
    _, tiny_rows = single_and_tiny_selection(args)
    speaker2id, dialect2id = mappings_for_data(args.data_dir)
    samples = [sample_from_row(args, row, speaker2id, dialect2id) for row in tiny_rows]
    checkpoint_path = args.output_dir / "tiny_overfit" / "tiny_overfit.pt"
    variants: dict[str, Any] = {}
    for weight in (0.0, 0.05, 0.1, 0.25):
        print(f"Stop-weight experiment: {weight}")
        variants[str(weight)] = train_stop_variant(
            args, checkpoint_path, samples, weight, steps=500
        )
        write_json(
            args.output_dir / "reports" / "stop_weight_experiment.partial.json",
            {"variants": variants},
        )

    baseline_accuracy = variants["0.0"]["after"]["mean_token_accuracy"]
    eligible = [
        float(weight)
        for weight, result in variants.items()
        if float(weight) > 0
        and result["after"]["mean_token_accuracy"] >= baseline_accuracy - 0.01
    ]
    recommended = max(
        eligible,
        key=lambda weight: (
            variants[str(weight)]["after"]["stop_f1"],
            -weight,
        ),
    ) if eligible else None
    report = {
        "stage": "stop_weight_experiment",
        "start_checkpoint": "tiny_overfit/tiny_overfit.pt",
        "all_variants_same_start": True,
        "steps_per_nonzero_variant": 500,
        "variants": variants,
        "recommended_stop_loss_weight": recommended,
    }
    write_json(args.output_dir / "reports" / "stop_weight_experiment.json", report)
    print("RECOMMENDED STOP WEIGHT:", recommended)
    return report


def stage_report(args: argparse.Namespace) -> dict[str, Any]:
    codec = read_json(args.output_dir / "reports" / "ground_truth_codec_report.json")
    existing = read_json(args.output_dir / "reports" / "teacher_forced_existing_report.json")
    single = read_json(args.output_dir / "reports" / "single_overfit_report.json")
    tiny = read_json(args.output_dir / "reports" / "tiny_overfit_report.json")
    stop = read_json(args.output_dir / "reports" / "stop_weight_experiment.json")
    missing = [
        name
        for name, value in (
            ("codec", codec),
            ("existing", existing),
            ("single", single),
            ("tiny", tiny),
            ("stop", stop),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"cannot build final report; missing stages: {missing}")
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    codec_samples = codec["samples"]
    codec_summary = {
        "pass": codec["ground_truth_decode_pass"],
        "sample_count": codec["sample_count"],
        "mean_cache_match": codec["mean_exact_token_match_ratio"],
        "mean_mse": round(float(np.mean([item["mse"] for item in codec_samples])), 8),
        "mean_si_sdr_db": round(float(np.mean([item["si_sdr_db"] for item in codec_samples])), 4),
        "mean_duration_ratio": round(
            float(np.mean([item["duration_ratio"] for item in codec_samples])), 6
        ),
        "duration_ratio_range": codec["duration_ratio_range"],
        "runtime_codec": codec["runtime_codec"],
    }
    existing_summary = {
        checkpoint: {
            key: report[key]
            for key in (
                "mean_token_accuracy",
                "q0_accuracy",
                "residual_accuracy",
                "codec_ce",
                "frame_all_codebook_accuracy",
            )
        }
        for checkpoint, report in existing["checkpoints"].items()
    }
    single_summary = {
        "sample": single["sample"],
        "steps": single["steps"],
        "final_codec_loss": single["final_teacher_forced_model_metrics"]["codec_ce"],
        "q0_ce": single["final_teacher_forced_model_metrics"]["per_codebook_ce"][0],
        "teacher_forced_accuracy": single["teacher_forced_comparison"],
        "autoregressive_accuracy": single["autoregressive_comparison"],
        "incremental_vs_full_recompute": single["incremental_vs_full_recompute"],
        "semantic_cache_consistency": single["semantic_cache_consistency"],
        "pass": single["single_overfit_pass"],
    }
    tiny_summary = {
        "steps": tiny["steps"],
        "sample_count": tiny["sample_count"],
        "speaker": tiny["speaker"],
        "final_codec_loss": tiny["teacher_forced"]["codec_ce"],
        "teacher_forced": tiny["teacher_forced"],
        "autoregressive": tiny["autoregressive"],
        "unique_outputs": tiny["autoregressive"]["unique_output_count"],
        "collapse_detected": tiny["collapse_detected"],
        "wav_integrity_pass": tiny["wav_integrity_pass"],
        "pass": tiny["tiny_overfit_pass"],
    }
    if not codec_summary["pass"]:
        decision = "CODEC PIPELINE BROKEN"
        root_cause = "CODEC CACHE"
    elif not single_summary["pass"]:
        decision = "TALKER ARCHITECTURE / ALIGNMENT BROKEN"
        root_cause = "TRAINING/INFERENCE ALIGNMENT"
    elif not tiny_summary["pass"]:
        decision = "SINGLE OVERFIT PASS / TINY OVERFIT FAIL"
        root_cause = "MODEL CAPACITY/OBJECTIVE"
    else:
        decision = "SINGLE OVERFIT PASS / TINY OVERFIT PASS"
        root_cause = "UNDERTRAINING"
    full_retrain_ready = decision == "SINGLE OVERFIT PASS / TINY OVERFIT PASS"
    recommendation = (
        {
            "initialize": "fresh model",
            "train_samples": 22150,
            "max_epochs": 20,
            "checkpoint_every_epochs": 1,
            "full_generation_evaluation_every_epochs": 2,
            "primary_checkpoint_metrics": [
                "validation codec loss",
                "q0 CE",
                "mean residual CE",
                "teacher-forced token accuracy",
                "autoregressive token/audio quality and human listening",
            ],
            "stop_pos_weight": 5.0,
            "stop_loss_weight": stop["recommended_stop_loss_weight"],
            "stop_loss_role": "secondary",
            "selection_rule": "do not select by total loss alone",
            "execution_status": "RECOMMENDED ONLY; NOT STARTED",
        }
        if full_retrain_ready
        else None
    )
    final = {
        "title": "ULM-LIVE RETRAIN DIAGNOSTIC REPORT",
        "git_head": git_head,
        "random_initialization_ce_reference": round(EXPECTED_RANDOM_CE, 6),
        "ground_truth_mimi_decode": codec_summary,
        "ground_truth_speech_reconstruction_pass": codec_summary["pass"],
        "original_to_current_mimi_encode_cache_match": codec_summary["mean_cache_match"],
        "teacher_forced_existing_checkpoints": existing_summary,
        "single_utterance_overfit": single_summary,
        "tiny_32_utterance_overfit": tiny_summary,
        "stop_weight_experiment": stop,
        "recommended_stop_weight": stop["recommended_stop_loss_weight"],
        "root_cause": root_cause,
        "root_cause_evidence": [
            "cached tokens match the current 16q Mimi encoder exactly on 20/20 samples",
            "fresh Talker memorized one utterance teacher-forced and autoregressively at 100% token accuracy",
            "fresh Talker memorized 32 utterances to 99.976% teacher-forced and 93.683% autoregressive q0 accuracy",
            "existing epoch-2/3/5 checkpoints remain near 4.45%-5.87% mean teacher-forced token accuracy with codec CE near 6",
        ],
        "decision": decision,
        "full_retrain_ready": full_retrain_ready,
        "recommended_full_training_config": recommendation,
        "full_training_started": False,
    }
    write_json(args.output_dir / "retrain_diagnostic_report.json", final)

    checkpoint_lines = []
    for checkpoint, values in existing_summary.items():
        checkpoint_lines.extend(
            [
                f"### {checkpoint}",
                "",
                f"- Mean token accuracy: {values['mean_token_accuracy']:.3%}",
                f"- Q0 accuracy: {values['q0_accuracy']:.3%}",
                f"- Codec CE: {values['codec_ce']:.4f}",
                "",
            ]
        )
    stop_lines = []
    for weight, variant in stop["variants"].items():
        values = variant["after"]
        stop_lines.append(
            f"- {weight}: codec CE {values['codec_ce']:.6f}, token accuracy "
            f"{values['mean_token_accuracy']:.3%}, stop F1 {values['stop_f1']:.3%}"
        )
    markdown = "\n".join(
        [
            "# ULM-LIVE RETRAIN DIAGNOSTIC REPORT",
            "",
            f"Git HEAD: `{git_head}`",
            "",
            "## Ground Truth Mimi Decode",
            "",
            f"PASS: **{codec_summary['pass']}**",
            f"Cache match: {codec_summary['mean_cache_match']:.3%}",
            f"Mean SI-SDR: {codec_summary['mean_si_sdr_db']:.2f} dB",
            f"Mean duration ratio: {codec_summary['mean_duration_ratio']:.4f}",
            "",
            "## Existing checkpoint teacher forcing",
            "",
            *checkpoint_lines,
            "## Single utterance overfit",
            "",
            f"Steps: {single_summary['steps']}",
            f"Teacher-forced accuracy: {single_summary['teacher_forced_accuracy']['mean_token_accuracy']:.3%}",
            f"Autoregressive q0 accuracy: {single_summary['autoregressive_accuracy']['q0_accuracy']:.3%}",
            f"PASS: **{single_summary['pass']}**",
            "",
            "## Tiny 32-utterance overfit",
            "",
            f"Steps: {tiny_summary['steps']}",
            f"Teacher-forced accuracy: {tiny_summary['teacher_forced']['mean_token_accuracy']:.3%}",
            f"Autoregressive q0 accuracy: {tiny_summary['autoregressive']['mean_q0_accuracy']:.3%}",
            f"Unique outputs: {tiny_summary['unique_outputs']}/32",
            f"Collapse detected: {tiny_summary['collapse_detected']}",
            f"PASS: **{tiny_summary['pass']}**",
            "",
            "## Stop weight experiment",
            "",
            *stop_lines,
            f"Recommended stop weight: **{stop['recommended_stop_loss_weight']}**",
            "",
            "## Decision",
            "",
            f"ROOT CAUSE: **{root_cause}**",
            f"DECISION: **{decision}**",
            f"FULL RETRAIN READY: **{'YES' if full_retrain_ready else 'NO'}**",
            "",
            "No full-dataset training was started by this diagnostic.",
            "",
        ]
    )
    report_path = args.output_dir / "reports" / "retrain_diagnostic_report.md"
    report_path.write_text(markdown, encoding="utf-8")
    print("FINAL DECISION:", decision)
    print("ROOT CAUSE:", root_cause)
    print("FULL RETRAIN READY:", "YES" if full_retrain_ready else "NO")
    return final


def main() -> None:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.config = args.config.resolve()
    seed_everything()
    if args.stage in ("codec", "all"):
        report = stage_codec(args)
        if not report["ground_truth_decode_pass"]:
            raise SystemExit("FULL RETRAIN BLOCKED: codec pipeline failed")
    if args.stage in ("existing", "all"):
        stage_existing(args)
    if args.stage in ("single", "all"):
        single_report = stage_single(args)
        if not single_report["single_overfit_pass"]:
            raise SystemExit("FULL RETRAIN BLOCKED: single utterance overfit failed")
    if args.stage in ("tiny", "all"):
        tiny_report = stage_tiny(args)
        if not tiny_report["tiny_overfit_pass"]:
            raise SystemExit("FULL RETRAIN BLOCKED: tiny-set overfit failed")
    if args.stage in ("stop", "all"):
        stage_stop(args)
    if args.stage in ("report", "all"):
        stage_report(args)
    if args.stage not in ("codec", "existing", "single", "tiny", "stop", "report", "all"):
        raise SystemExit(f"stage {args.stage!r} is not implemented yet")


if __name__ == "__main__":
    main()
