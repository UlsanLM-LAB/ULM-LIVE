#!/usr/bin/env python3
"""Matched teacher and learner-history continuation from the post-fix 2048 pilot."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.pilot_talker_ar_v3 import (  # noqa: E402
    SEED, forward, measure_ar, measure_teacher, prime_with_teacher_inputs, write_json,
)
from ulm_live.talker import (  # noqa: E402
    TalkerCollator, TalkerConfig, TalkerDataset, ULMTalker, build_id_mappings,
)

BASE = Path("outputs/talker-ar-diagnostics-v3/pilots-postfix/2048-multi-baseline")
OUT = Path("outputs/talker-decisive-v4")
DATA = Path("data/ulsan-full")
STEPS = 8000


def target_ratio(step: int) -> float:
    return .10 if step <= 800 else .25 if step <= 2800 else .40 if step <= 5600 else .50


def rollout_starts(mask: torch.Tensor, ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = mask.long().sum(1)
    eligible = (lengths - 1).clamp_min(0)
    counts = torch.round(eligible.float() * ratio).long()
    return lengths - counts, counts


def feedback_metrics(original: torch.Tensor, mixed: torch.Tensor, mask: torch.Tensor,
                     starts: torch.Tensor) -> dict:
    b, _, t = original.shape
    lengths = mask.long().sum(1)
    eligible = int((lengths - 1).clamp_min(0).sum())
    positions = torch.arange(t, device=mask.device)[None]
    selected = mask.bool() & (positions > 0) & (positions >= starts[:, None])
    changed = selected & (original != mixed).any(1)
    lengths_rollout = selected.sum(1).tolist()
    generated = int(selected.sum())
    return {
        "eligible_valid_frames": eligible,
        "generated_feedback_frames": generated,
        "changed_feedback_frames": int(changed.sum()),
        "effective_feedback_ratio": generated / eligible if eligible else 0.,
        "perturbed_feedback_ratio": int(changed.sum()) / eligible if eligible else 0.,
        "rollout_length_sum": int(sum(lengths_rollout)),
        "sample_count": b,
        "exposed_samples": sum(x > 0 for x in lengths_rollout),
        "rollout_length_histogram": dict(Counter(map(str, lengths_rollout))),
    }


def merge_feedback(items: list[dict]) -> dict:
    additive = ("eligible_valid_frames", "generated_feedback_frames", "changed_feedback_frames",
                "rollout_length_sum", "sample_count", "exposed_samples")
    total = {key: sum(x[key] for x in items) for key in additive}
    hist = Counter()
    for x in items:
        hist.update(x["rollout_length_histogram"])
    e = total["eligible_valid_frames"]
    n = total["sample_count"]
    total.update(effective_feedback_ratio=total["generated_feedback_frames"] / e if e else 0.,
                 perturbed_feedback_ratio=total["changed_feedback_frames"] / e if e else 0.,
                 mean_contiguous_rollout_length=total["rollout_length_sum"] / n if n else 0.,
                 exposed_sample_fraction=total["exposed_samples"] / n if n else 0.,
                 rollout_length_histogram=dict(sorted(hist.items(), key=lambda x: int(x[0]))))
    return total


def learner_history(model: ULMTalker, batch: dict, ratio: float) -> tuple[torch.Tensor, dict]:
    original = batch["audio_codes"]
    mixed = original.clone()
    starts, counts = rollout_starts(batch["audio_attention_mask"], ratio)
    if not counts.any():
        return mixed, feedback_metrics(original, mixed, batch["audio_attention_mask"], starts)
    first = int(starts[counts > 0].min())
    last = int(batch["audio_attention_mask"].long().sum(1).max())
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast(original.device.type, dtype=torch.bfloat16,
                                             enabled=original.device.type == "cuda"):
            state = prime_with_teacher_inputs(model, batch, first - 1)
            for pos in range(first - 1, last - 1):
                generated, _ = model.step(state, mixed[:, :, pos])
                next_pos = pos + 1
                take = (counts > 0) & (next_pos >= starts) & batch["audio_attention_mask"][:, next_pos]
                mixed[:, :, next_pos] = torch.where(take[:, None], generated.detach(), mixed[:, :, next_pos])
    finally:
        model.train(was_training)
    return mixed.detach(), feedback_metrics(original, mixed, batch["audio_attention_mask"], starts)


def select_validation(train_ids: list[str], val_rows: list[dict]) -> tuple[list[str], list[str]]:
    pool = sorted(row["id"] for row in val_rows if row.get("semantic_path") and row.get("codec_path")
                  and 1 <= float(row.get("duration", 0)) <= 8 and len(str(row.get("text", ""))) >= 3)
    if len(pool) < 100:
        raise ValueError("fewer than 100 eligible validation utterances")
    rng = random.Random(SEED)
    chosen = rng.sample(pool, 100)
    if set(chosen) & set(train_ids):
        raise ValueError("validation overlaps training subset")
    return chosen, chosen[:20]


def survival(items: list[dict], samples: list[dict]) -> dict:
    offsets = (0, 1, 2, 4, 8, 16, 32)
    sums = {str(k): [0., 0] for k in offsets}
    no_error = Counter()
    first_errors = []
    for item, sample in zip(items, samples, strict=True):
        target = sample["audio_codes"]
        codes = item["codes"]
        overlap = min(codes.shape[-1], target.shape[-1])
        frame_match = (codes[:, :overlap] == target[:, :overlap]).float().mean(0)
        wrong = (frame_match < 1).nonzero().flatten()
        first = int(wrong[0]) if len(wrong) else overlap
        first_errors.append(first)
        for k in offsets:
            pos = first + k
            if pos < overlap:
                sums[str(k)][0] += float(frame_match[pos])
                sums[str(k)][1] += 1
        for t in (0, 1, 2, 4, 8, 16, 32):
            if first > t:
                no_error[t] += 1
    return {"first_error_frames": first_errors,
            "accuracy_after_first_error": {k: {"mean": v[0] / v[1] if v[1] else None, "n": v[1]}
                                           for k, v in sums.items()},
            "survival": {str(t): no_error[t] / len(items) for t in (0, 1, 2, 4, 8, 16, 32)}}


def load_data():
    paths = [DATA / f"{split}.cached.jsonl" for split in ("train", "val", "test")]
    speaker2id, dialect2id = build_id_mappings(paths)
    train = TalkerDataset(paths[0], speaker2id=speaker2id, dialect2id=dialect2id,
                          num_quantizers=16, cache_only=True)
    val = TalkerDataset(paths[1], speaker2id=speaker2id, dialect2id=dialect2id,
                        num_quantizers=16, cache_only=True)
    ids = [x["id"] for x in json.loads((BASE / "selection.json").read_text())]
    train_by_id = {r["id"]: i for i, r in enumerate(train.items)}
    val_by_id = {r["id"]: i for i, r in enumerate(val.items)}
    selected = [train[train_by_id[x]] for x in ids]
    val_ids, ar_ids = select_validation(ids, val.items)
    return ids, selected, [val[val_by_id[x]] for x in val_ids], [val[val_by_id[x]] for x in ar_ids]


def evaluate(model, train_probe, val_tf, val_ar, collator, device):
    teacher = measure_teacher(model, val_tf, collator, device)
    train_teacher = measure_teacher(model, train_probe, collator, device)
    forced, forced_items = measure_ar(model, val_ar, device, normal=False)
    normal, _ = measure_ar(model, val_ar, device, normal=True)
    return {"train_tf": train_teacher, "val_tf": teacher, "forced_ar": forced,
            "normal_ar": normal, "first_error": survival(forced_items, val_ar)}


def smoke32(payload, samples, device):
    """New learner-history smoke on 32 distinct utterances before the large rollout."""
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = ULMTalker(TalkerConfig(**payload["config"])).to(device)
    collator = TalkerCollator(pad_token_id=model.config.pad_token_id,
                             bos_token_id=model.config.bos_token_id,
                             max_audio_len=model.config.max_seq_len,
                             acoustic_delay_frames=model.config.acoustic_delay_frames)
    selected = samples[:32]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.)
    rng = random.Random(SEED)
    order = list(range(32))
    position = 32
    feedback = []
    output = OUT / "smoke32"
    output.mkdir(parents=True, exist_ok=True)
    for step in range(1, 2501):
        if position + 8 > 32:
            rng.shuffle(order)
            position = 0
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in collator([selected[i] for i in order[position:position + 8]]).items()}
        position += 8
        model.train()
        ratio = 0. if step <= 1250 else .1 if step <= 1450 else .25 if step <= 1650 else .5
        inputs, metrics = learner_history(model, batch, ratio)
        feedback.append(metrics)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            result = forward(model, batch, inputs)
            loss = result.codec_loss + .05 * result.stop_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 250 == 0:
            teacher = measure_teacher(model, selected, collator, device)
            forced, _ = measure_ar(model, selected[:12], device, normal=False)
            normal, _ = measure_ar(model, selected[:12], device, normal=True)
            item = {"step": step, "teacher": teacher, "forced_ar": forced,
                    "normal_ar": normal, "feedback": merge_feedback(feedback),
                    "selection_ids": [x["id"] for x in selected]}
            write_json(output / "result.json", item)
            print(f"smoke32 {step} TF={teacher['mean_accuracy']:.3%} "
                  f"AR={forced['mean_accuracy']:.3%}", flush=True)
            feedback = []
            if step >= 1750 and teacher["mean_accuracy"] >= .99 and forced["mean_accuracy"] >= .99:
                return
    raise RuntimeError("32-sample learner-history smoke did not reach 99% TF and AR")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--branch", choices=("tf", "rollout", "base", "smoke32"), required=True)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    ids, samples, val_tf, val_ar = load_data()
    payload = torch.load(BASE / "final.pt", map_location="cpu", weights_only=False)
    if payload["step"] != 8000 or payload["selection_ids"] != ids:
        raise ValueError("incorrect base checkpoint")
    if args.branch == "smoke32":
        smoke32(payload, samples, device)
        return
    model = ULMTalker(TalkerConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["model_state_dict"])
    collator = TalkerCollator(pad_token_id=model.config.pad_token_id,
                             bos_token_id=model.config.bos_token_id,
                             max_audio_len=model.config.max_seq_len,
                             acoustic_delay_frames=model.config.acoustic_delay_frames)
    output = OUT / f"2048-cont-{args.branch}"
    output.mkdir(parents=True, exist_ok=True)
    selection = {"train_ids": ids, "val_tf_ids": [x["id"] for x in val_tf],
                 "val_ar_ids": [x["id"] for x in val_ar]}
    if (OUT / "selection.json").exists() and json.loads((OUT / "selection.json").read_text()) != selection:
        raise ValueError("validation selection changed")
    write_json(OUT / "selection.json", selection)
    if args.branch == "base":
        write_json(output / "final.json", {"step": 8000, **evaluate(model, samples[:100], val_tf, val_ar,
                                                                      collator, device)})
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    rng = random.Random(SEED)
    order = list(range(len(samples)))
    position = len(order)
    history = []
    start_step = 1
    latest = output / "latest.pt"
    if args.resume and latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        rng.setstate(state["batch_rng"])
        order, position = state["order"], state["position"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        history = json.loads((output / "history.json").read_text())
        start_step = state["step"] + 1
    feedback_window = []
    for step in range(start_step, args.steps + 1):
        if position + 8 > len(order):
            rng.shuffle(order)
            position = 0
        indices = order[position:position + 8]
        position += 8
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in collator([samples[i] for i in indices]).items()}
        model.train()
        if args.branch == "rollout":
            inputs, metrics = learner_history(model, batch, target_ratio(step))
            feedback_window.append(metrics)
        else:
            inputs = batch["audio_codes"]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            result = forward(model, batch, inputs)
            loss = result.codec_loss + .05 * result.stop_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            item = {"step": 8000 + step, "continuation_step": step,
                    "train_batch_loss": float(loss.detach()),
                    "target_feedback": target_ratio(step) if args.branch == "rollout" else 0.,
                    "feedback": merge_feedback(feedback_window) if feedback_window else None,
                    **evaluate(model, samples[:100], val_tf, val_ar, collator, device)}
            history.append(item)
            write_json(output / "history.json", history)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "batch_rng": rng.getstate(), "order": order, "position": position,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "step": step}, latest)
            print(f"{args.branch} +{step} valTF={item['val_tf']['mean_accuracy']:.3%} "
                  f"AR={item['forced_ar']['mean_accuracy']:.3%} "
                  f"feedback={item['feedback']['effective_feedback_ratio']:.3%}"
                  if item["feedback"] else f"{args.branch} +{step} valTF={item['val_tf']['mean_accuracy']:.3%} "
                  f"AR={item['forced_ar']['mean_accuracy']:.3%}", flush=True)
            feedback_window = []
    write_json(output / "final.json", history[-1])


if __name__ == "__main__":
    main()
