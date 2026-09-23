#!/usr/bin/env python3
# ruff: noqa: E402
"""Controlled fresh Talker subset pilots for AR collapse diagnosis."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
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
from ulm_live.talker import TalkerCollator, TalkerConfig, TalkerDataset, TalkerGenerationConfig, ULMTalker, build_id_mappings
from ulm_live.utils.audio import save_wav


SEED = 20260923
DATA_ROOT = Path("data/ulsan-full")
OUTPUT_ROOT = Path("outputs/talker-ar-diagnostics-v3/pilots")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, choices=(1, 32, 128, 512, 2048), required=True)
    p.add_argument("--speakers", choices=("single", "concentrated", "multi"), required=True)
    p.add_argument("--variant", choices=("baseline", "larger", "q0x2", "q0x4", "coarse", "codec_only", "staged_stop", "scheduled", "continuation", "semantic_margin"), default="baseline")
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--init-checkpoint", type=Path)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def select_subset(items: list[dict[str, Any]], size: int, mode: str) -> list[int]:
    eligible = [i for i, row in enumerate(items)
                if 1 <= float(row.get("duration", 0)) <= 8 and row.get("semantic_path")
                and row.get("codec_path") and len(str(row.get("text", ""))) >= 3]
    by_speaker: dict[str, list[int]] = {}
    for i in eligible:
        by_speaker.setdefault(str(items[i]["speaker_id"]), []).append(i)
    for indices in by_speaker.values():
        indices.sort(key=lambda i: items[i]["id"])
    ranked = sorted(by_speaker, key=lambda speaker: (-len(by_speaker[speaker]), speaker))
    if mode == "single":
        if len(by_speaker[ranked[0]]) < size:
            raise ValueError(f"largest single speaker has {len(by_speaker[ranked[0]])} eligible utterances; cannot select {size}")
        pool = by_speaker[ranked[0]]
        if size == 1:
            return [pool[len(pool) // 2]]
        return [pool[round(j * (len(pool) - 1) / (size - 1))] for j in range(size)]
    if mode == "concentrated":
        pool = []
        for speaker in ranked:
            pool.extend(by_speaker[speaker])
            if len(pool) >= size:
                break
        pool.sort(key=lambda i: items[i]["id"])
        return [pool[round(j * (len(pool) - 1) / (size - 1))] for j in range(size)]
    if mode == "multi":
        rng = random.Random(SEED)
        speakers = sorted(by_speaker)
        rng.shuffle(speakers)
        picked = []
        depth = 0
        while len(picked) < size:
            for speaker in speakers:
                indices = by_speaker[speaker]
                if depth < len(indices):
                    picked.append(indices[depth])
                    if len(picked) == size:
                        break
            depth += 1
        return picked
    raise ValueError(mode)


def loss_weights(variant: str, device: torch.device) -> torch.Tensor:
    values = [1.0] * 16
    if variant == "q0x2":
        values[0] = 2.0
    elif variant == "q0x4":
        values[0] = 4.0
    elif variant == "coarse":
        values[:4] = [4.0, 2.0, 1.5, 1.5]
    return torch.tensor(values, device=device)


def feedback_probability(variant: str, step: int, total_steps: int) -> float:
    if variant != "scheduled":
        return 0.0
    progress = (step - 1) / total_steps
    return 0.0 if progress < .1 else .1 if progress < .3 else .25 if progress < .6 else .5


def prime_with_teacher_inputs(model: ULMTalker, batch: dict[str, torch.Tensor], start: int):
    """Fill the temporal KV cache with a teacher-forced prefix in one causal pass."""
    state = model.init_generation_state(
        batch["semantic_hidden_states"], batch["speaker_ids"],
        batch["dialect_ids"], batch["semantic_attention_mask"],
    )
    if start == 0:
        return state
    sem_len = batch["semantic_hidden_states"].shape[1]
    positions = state.next_position[:, None] + torch.arange(start, device=batch["audio_codes"].device)
    if torch.any(positions >= model.config.max_seq_len):
        raise ValueError("generation exceeds max_seq_len")
    x = model.audio_embed(batch["audio_codes"][:, :, :start], state.condition)
    x = x + model.pos_embedding(positions)
    causal = torch.cat((
        torch.ones(start, sem_len, dtype=torch.bool, device=x.device),
        torch.ones(start, start, dtype=torch.bool, device=x.device).tril(),
    ), dim=1)
    padding = torch.cat((state.key_padding_mask, ~batch["audio_attention_mask"][:, :start].bool()), dim=1)
    caches = []
    for block, past in zip(model.temporal_transformer, state.layer_kv, strict=True):
        x, kv = block(x, mask=causal, padding=padding, past=past, cache=True)
        caches.append(kv)
    state.layer_kv = caches
    state.key_padding_mask = padding
    state.next_position = state.next_position + start
    return state


def mixed_previous_frames(model: ULMTalker, batch: dict[str, torch.Tensor], probability: float) -> torch.Tensor:
    """Inject detached feedback following the selected causal path in an eight-frame window."""
    inputs = batch["audio_codes"].clone()
    if probability <= 0 or inputs.shape[-1] <= 1:
        return inputs
    count = min(8, inputs.shape[-1] - 1)
    start = int(torch.randint(0, inputs.shape[-1] - count, (1,)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=inputs.device.type == "cuda"):
            state = prime_with_teacher_inputs(model, batch, start)
            previous = inputs[:, :, start]
            for offset in range(1, count + 1):
                tokens, _ = model.step(state, previous)
                frame = start + offset
                take = (torch.rand(inputs.shape[0], device=inputs.device) < probability) & batch["audio_attention_mask"][:, frame]
                inputs[:, :, frame] = torch.where(take[:, None], tokens.detach(), inputs[:, :, frame])
                previous = inputs[:, :, frame]
    finally:
        model.train(was_training)
    return inputs


def forward(model, batch, inputs=None):
    return model(
        batch["semantic_hidden_states"], inputs if inputs is not None else batch["audio_codes"],
        batch["speaker_ids"], batch["dialect_ids"], targets=batch["targets"],
        semantic_attention_mask=batch["semantic_attention_mask"],
        audio_attention_mask=batch["audio_attention_mask"], stop_targets=batch["stop_targets"],
    )


def measure_teacher(model, samples, collator, device):
    model.eval()
    correct = np.zeros(16, dtype=np.float64)
    count = np.zeros(16, dtype=np.float64)
    ce = np.zeros(16, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(samples), 8):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in collator(samples[start:start+8]).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = forward(model, batch)
            pred = output.logits.argmax(-1)
            for q in range(16):
                valid = batch["targets"][:, q] != model.config.pad_token_id
                n = int(valid.sum())
                correct[q] += int((pred[:, q][valid] == batch["targets"][:, q][valid]).sum())
                count[q] += n
                ce[q] += float(output.codebook_losses[q]) * n
    per_q = correct / count
    return {"codec_ce": float(np.mean(ce / count)), "mean_accuracy": float(correct.sum() / count.sum()),
            "q0_accuracy": float(per_q[0]), "residual_accuracy": float(correct[1:].sum() / count[1:].sum()),
            "per_q_accuracy": per_q.tolist(), "per_q_ce": (ce / count).tolist()}


def measure_ar(model, samples, device, normal: bool):
    model.eval()
    results = []
    for sample in samples:
        target = sample["audio_codes"]
        length = target.shape[-1]
        max_frames = min(150, model.config.max_seq_len - sample["semantic_hidden_states"].shape[0]) if normal else length
        cfg = TalkerGenerationConfig(
            max_new_tokens=max_frames, max_audio_frames=max_frames,
            do_sample=False, stop_threshold=0.5 if normal else 2.0,
            min_audio_frames=model.config.min_audio_frames if normal else length + 1,
            return_details=True,
        )
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            codes, lengths, reasons, _ = model.generate(
                sample["semantic_hidden_states"].unsqueeze(0).to(device),
                torch.tensor([sample["speaker_id"]], device=device),
                torch.tensor([sample["dialect_id"]], device=device), cfg,
                semantic_attention_mask=sample["semantic_attention_mask"].unsqueeze(0).to(device),
            )
        codes = codes[0, :, :int(lengths[0])].cpu()
        overlap = min(codes.shape[-1], length)
        matches = (codes[:, :overlap] == target[:, :overlap]).float()
        results.append({"id": sample["id"], "mean_accuracy": float(matches.mean()),
                        "q0_accuracy": float(matches[0].mean()),
                        "residual_accuracy": float(matches[1:].mean()),
                        "length_ratio": codes.shape[-1] / length,
                        "frames": int(codes.shape[-1]), "reason": reasons[0], "codes": codes})
    hashes = {hashlib.sha256(r["codes"].numpy().tobytes()).hexdigest() for r in results}
    return {"mean_accuracy": float(np.mean([r["mean_accuracy"] for r in results])),
            "q0_accuracy": float(np.mean([r["q0_accuracy"] for r in results])),
            "residual_accuracy": float(np.mean([r["residual_accuracy"] for r in results])),
            "length_ratio": float(np.mean([r["length_ratio"] for r in results])),
            "unique_outputs": len(hashes), "sample_count": len(results),
            "stop_predicted": sum(r["reason"] == "STOP_PREDICTED" for r in results)}, results


def save_wavs(model, codec, samples, device, dest):
    _, ar_items = measure_ar(model, samples, device, normal=False)
    for i, (sample, item) in enumerate(zip(samples, ar_items, strict=True), 1):
        batch = TalkerCollator(
            pad_token_id=model.config.pad_token_id, bos_token_id=model.config.bos_token_id,
            max_audio_len=model.config.max_seq_len, acoustic_delay_frames=model.config.acoustic_delay_frames,
        )([sample])
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            teacher = forward(model, batch).logits.argmax(-1)[0].cpu()
        sample_dir = dest / f"sample_{i:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for name, codes in (("ground_truth", sample["audio_codes"]),
                            ("teacher_forced", teacher), ("autoregressive", item["codes"])):
            audio, sr = codec.decode(EncodedAudio(
                codes=codes.to(device), sample_rate=24000, frame_rate=12.5,
                metadata={"backend": "mimi", "num_quantizers": 16},
            ))
            save_wav(sample_dir / f"{name}.wav", audio.squeeze(0).cpu(), sr)
        (sample_dir / "text.txt").write_text(sample["text"] + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.speakers == "single" and args.size > 128:
        raise ValueError("canonical training data has no single speaker with 512 utterances")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device(args.device)
    train_manifest = args.data_dir / "train.cached.jsonl"
    speaker2id, dialect2id = build_id_mappings([
        train_manifest, args.data_dir / "val.cached.jsonl", args.data_dir / "test.cached.jsonl"
    ])
    dataset = TalkerDataset(train_manifest, speaker2id=speaker2id, dialect2id=dialect2id,
                            num_quantizers=16, cache_only=True)
    indices = select_subset(dataset.items, args.size, args.speakers)
    samples = [dataset[i] for i in indices]
    output = args.output_dir / f"{args.size}-{args.speakers}-{args.variant}"
    if (output / "result.json").exists():
        raise FileExistsError(output / "result.json")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "selection.json", [{"id": dataset.items[i]["id"],
                "speaker": dataset.items[i]["speaker_id"], "duration": dataset.items[i]["duration"]} for i in indices])
    config = TalkerConfig.from_yaml(ROOT / "configs/talker.yaml")
    config.num_quantizers = 16
    config.stop_loss_weight = 0.05
    config.stop_pos_weight = 5.0
    config.num_speakers = max(config.num_speakers, len(speaker2id))
    config.num_dialects = max(config.num_dialects, len(dialect2id))
    if args.variant == "larger":
        config.talker_dim = 768
        config.num_heads = 12
        config.dim_feedforward = 3072
    model = ULMTalker(config).to(device)
    if args.init_checkpoint:
        payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if payload["config"] != config.to_dict() or payload["selection_ids"] != [s["id"] for s in samples]:
            raise ValueError("initial checkpoint config/selection mismatch")
        model.load_state_dict(payload["model_state_dict"])
    collator = TalkerCollator(pad_token_id=config.pad_token_id, bos_token_id=config.bos_token_id,
                             max_audio_len=config.max_seq_len, acoustic_delay_frames=config.acoustic_delay_frames)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.0)
    weights = loss_weights(args.variant, device)
    rng = random.Random(SEED)
    order = list(range(len(samples)))
    position = len(order)
    history = []
    ar_examples = [samples[i] for i in [round(j * (len(samples) - 1) / 11) for j in range(12)]]
    for step in range(1, args.steps + 1):
        if position + 8 > len(order):
            rng.shuffle(order)
            position = 0
        batch_indices = order[position:position+8]
        position += 8
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in collator([samples[i] for i in batch_indices]).items()}
        model.train()
        inputs = mixed_previous_frames(model, batch, feedback_probability(args.variant, step, args.steps))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = forward(model, batch, inputs)
            codec_loss = (torch.stack(prediction.codebook_losses) * weights).sum() / weights.sum()
            stop_weight = 0.0 if args.variant == "codec_only" else (0.0 if args.variant == "staged_stop" and step <= args.steps // 2 else 0.05)
            loss = codec_loss + stop_weight * prediction.stop_loss
            if args.variant == "semantic_margin":
                wrong = dict(batch)
                wrong["semantic_hidden_states"] = batch["semantic_hidden_states"].roll(1, 0)
                wrong["semantic_attention_mask"] = batch["semantic_attention_mask"].roll(1, 0)
                wrong_prediction = forward(model, wrong, inputs)
                loss = loss + F.relu(0.3 + prediction.codec_loss - wrong_prediction.codec_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite pilot loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optimizer.step()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            teacher = measure_teacher(model, samples, collator, device)
            ar, _ = measure_ar(model, ar_examples[:4], device, normal=False)
            normal_ar = None
            if args.variant == "scheduled":
                normal_ar, _ = measure_ar(model, ar_examples[:4], device, normal=True)
            item = {"step": step, "train_codec_loss": float(codec_loss.detach()),
                    "gradient_norm": grad_norm, "teacher": teacher, "ar_probe": ar,
                    "normal_ar_probe": normal_ar,
                    "feedback_probability": feedback_probability(args.variant, step, args.steps)}
            history.append(item)
            write_json(output / "history.json", history)
            print(f"{args.size}-{args.speakers}-{args.variant} step={step} codec={teacher['codec_ce']:.4f} teacher={teacher['mean_accuracy']:.3%} q0={teacher['q0_accuracy']:.3%} AR={ar['mean_accuracy']:.3%} ARq0={ar['q0_accuracy']:.3%}", flush=True)
            if teacher["mean_accuracy"] >= .99 and ar["mean_accuracy"] >= .99 and ar["q0_accuracy"] >= .99:
                break
    final_teacher = measure_teacher(model, samples, collator, device)
    ar_forced, _ = measure_ar(model, ar_examples, device, normal=False)
    ar_normal, _ = measure_ar(model, ar_examples, device, normal=True)
    result = {"size": args.size, "speaker_mode": args.speakers, "variant": args.variant,
              "selected_speakers": len(Counter(s["speaker_id"] for s in samples)),
              "seed": SEED, "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
              "steps": step, "planned_steps": args.steps, "learning_rate": 2e-4,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "teacher": final_teacher, "ar_forced": ar_forced, "ar_normal": ar_normal}
    write_json(output / "result.json", result)
    if args.size >= 512:
        torch.save({"model_state_dict": model.state_dict(), "config": config.to_dict(),
                    "selection_ids": [s["id"] for s in samples], "step": step}, output / "final.pt")
    codec = build_codec(backend="mimi", device=args.device, num_quantizers=16)
    save_wavs(model, codec, ar_examples[:2], device, output / "audio")
    print(f"PILOT COMPLETE {output} {result}", flush=True)


if __name__ == "__main__":
    main()
