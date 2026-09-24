#!/usr/bin/env python3
"""Conditional 512-sample pilot for token-aligned explicit linguistic identity."""

from __future__ import annotations

from pathlib import Path
import json
import random
import sys

import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decisive_ar_v4 import DATA, OUT  # noqa: E402
from scripts.pilot_talker_ar_v3 import (  # noqa: E402
    SEED, forward, measure_ar, measure_teacher, write_json,
)
from ulm_live.talker import (  # noqa: E402
    TalkerCollator, TalkerConfig, TalkerDataset, ULMTalker, build_id_mappings,
)

BASE512 = Path("outputs/talker-ar-diagnostics-v3/pilots-postfix/512-multi-baseline/final.pt")
RESULT = OUT / "linguistic-512"


class AlignedTextFusion(nn.Module):
    """Modulate each cached Thinker token state using its aligned Qwen token ID."""

    def __init__(self, vocabulary: int, semantic_dim: int, token_dim: int = 64):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary, token_dim)
        self.modulation = nn.Linear(token_dim, semantic_dim * 2)
        nn.init.normal_(self.token_embedding.weight, 0., .02)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        if hidden.shape[:2] != token_ids.shape:
            raise ValueError("Qwen token IDs must align exactly with cached semantic states")
        scale, shift = self.modulation(self.token_embedding(token_ids)).chunk(2, -1)
        return hidden.float() * (1 + .1 * torch.tanh(scale)) + .1 * shift


def token_ids(tokenizer, samples):
    result = {}
    for sample in samples:
        ids = tokenizer(sample["text"], return_tensors="pt")["input_ids"][0]
        if ids.numel() != sample["semantic_hidden_states"].shape[0]:
            raise ValueError(f"semantic/token alignment mismatch: {sample['id']}")
        result[sample["id"]] = ids
    return result


def fuse_batch(fusion, batch, ids):
    max_length = batch["semantic_hidden_states"].shape[1]
    tokens = torch.zeros(len(ids), max_length, dtype=torch.long,
                         device=batch["semantic_hidden_states"].device)
    for i, sequence in enumerate(ids):
        tokens[i, :sequence.numel()] = sequence.to(tokens.device)
    return {**batch, "semantic_hidden_states": fusion(batch["semantic_hidden_states"], tokens)}


def fused_samples(fusion, samples, ids_by_sample, device):
    result = []
    fusion.eval()
    with torch.no_grad():
        for sample in samples:
            hidden = sample["semantic_hidden_states"].unsqueeze(0).to(device)
            ids = ids_by_sample[sample["id"]].unsqueeze(0).to(device)
            fused = fusion(hidden, ids)[0].cpu()
            result.append({**sample, "semantic_hidden_states": fused})
    return result


def evaluate(talker, fusion, train, val_tf, val_ar, text_ids, collator, device):
    train_fused = fused_samples(fusion, train[:100], text_ids, device)
    val_fused = fused_samples(fusion, val_tf, text_ids, device)
    ar_fused = val_fused[:20]
    return {"train_tf": measure_teacher(talker, train_fused, collator, device),
            "val_tf": measure_teacher(talker, val_fused, collator, device),
            "forced_ar": measure_ar(talker, ar_fused, device, normal=False)[0],
            "normal_ar": measure_ar(talker, ar_fused, device, normal=True)[0]}


def same_speaker_wrong_ids(tokenizer, val_ar, val_data, train_data):
    """Find equal-length wrong utterances so token alignment remains valid."""
    wrong = {}
    for sample in val_ar:
        speaker = sample["speaker_id"]
        length = sample["semantic_hidden_states"].shape[0]
        for dataset in (val_data, train_data):
            for row in dataset.items:
                if (row["id"] == sample["id"] or str(row["text"]).strip() == sample["text"]
                        or dataset.speaker2id[str(row["speaker_id"])] != speaker):
                    continue
                ids = tokenizer(str(row["text"]).strip(), return_tensors="pt")["input_ids"][0]
                if ids.numel() == length:
                    wrong[sample["id"]] = ids
                    break
            if sample["id"] in wrong:
                break
    return wrong


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    snapshot_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots"
    snapshots = sorted(snapshot_root.glob("*"))
    if not snapshots:
        raise FileNotFoundError("Qwen tokenizer is not already cached on EC2")
    tokenizer = AutoTokenizer.from_pretrained(snapshots[-1], local_files_only=True)
    selection = json.loads((OUT / "selection.json").read_text())
    ids512 = [x["id"] for x in json.loads((BASE512.parent / "selection.json").read_text())]
    if ids512 != selection["train_ids"][:512]:
        raise ValueError("512 pilot selection differs from existing baseline")
    paths = [DATA / f"{split}.cached.jsonl" for split in ("train", "val", "test")]
    speaker2id, dialect2id = build_id_mappings(paths)
    train_data = TalkerDataset(paths[0], speaker2id=speaker2id, dialect2id=dialect2id,
                               num_quantizers=16, cache_only=True)
    val_data = TalkerDataset(paths[1], speaker2id=speaker2id, dialect2id=dialect2id,
                             num_quantizers=16, cache_only=True)
    train_index = {row["id"]: i for i, row in enumerate(train_data.items)}
    val_index = {row["id"]: i for i, row in enumerate(val_data.items)}
    samples = [train_data[train_index[x]] for x in ids512]
    val_tf = [val_data[val_index[x]] for x in selection["val_tf_ids"]]
    val_ar = val_tf[:20]
    text_ids = token_ids(tokenizer, samples + val_tf)
    old = torch.load(BASE512, map_location="cpu", weights_only=False)
    config = TalkerConfig(**old["config"])
    collator = TalkerCollator(pad_token_id=config.pad_token_id,
                             bos_token_id=config.bos_token_id, max_audio_len=config.max_seq_len,
                             acoustic_delay_frames=config.acoustic_delay_frames)
    RESULT.mkdir(parents=True, exist_ok=True)
    baseline = ULMTalker(config).to(device).eval()
    baseline.load_state_dict(old["model_state_dict"])
    baseline_result = {"val_tf": measure_teacher(baseline, val_tf, collator, device),
                       "forced_ar": measure_ar(baseline, val_ar, device, normal=False)[0],
                       "normal_ar": measure_ar(baseline, val_ar, device, normal=True)[0]}
    write_json(RESULT / "matched_512_baseline.json", baseline_result)
    del baseline
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    talker = ULMTalker(config).to(device)
    fusion = AlignedTextFusion(len(tokenizer), config.semantic_dim).to(device)
    optimizer = torch.optim.AdamW(list(talker.parameters()) + list(fusion.parameters()),
                                  lr=2e-4, weight_decay=0.)
    rng = random.Random(SEED)
    order = list(range(512))
    position = 512
    history = []
    for step in range(1, 4001):
        if position + 8 > 512:
            rng.shuffle(order)
            position = 0
        indices = order[position:position + 8]
        position += 8
        current = [samples[i] for i in indices]
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in collator(current).items()}
        talker.train()
        fusion.train()
        batch = fuse_batch(fusion, batch, [text_ids[x["id"]] for x in current])
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            output = forward(talker, batch)
            loss = output.codec_loss + .05 * output.stop_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(talker.parameters()) + list(fusion.parameters()), 1.)
        optimizer.step()
        if step % 1000 == 0:
            result = {"step": step, "train_batch_loss": float(loss.detach()),
                      **evaluate(talker, fusion, samples, val_tf, val_ar, text_ids,
                                 collator, device)}
            history.append(result)
            write_json(RESULT / "history.json", history)
            print(f"linguistic512 {step} TF={result['val_tf']['mean_accuracy']:.3%} "
                  f"AR={result['forced_ar']['mean_accuracy']:.3%}", flush=True)
    torch.save({"model": talker.state_dict(), "fusion": fusion.state_dict(),
                "config": config.to_dict(), "selection_ids": ids512,
                "step": step}, RESULT / "final.pt")
    wrong_ids = same_speaker_wrong_ids(tokenizer, val_ar, val_data, train_data)
    comparable = [x for x in val_ar if x["id"] in wrong_ids]
    if comparable:
        correct = fused_samples(fusion, comparable, text_ids, device)
        incorrect = fused_samples(fusion, comparable, wrong_ids, device)
        write_json(RESULT / "linguistic_identity.json", {
            "sample_count": len(comparable), "correct": measure_teacher(talker, correct, collator, device),
            "same_speaker_wrong_text_ids": measure_teacher(talker, incorrect, collator, device),
            "sample_ids": [x["id"] for x in comparable],
        })


if __name__ == "__main__":
    main()
