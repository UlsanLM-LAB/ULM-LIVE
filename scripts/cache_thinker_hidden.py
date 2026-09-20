#!/usr/bin/env python3
"""Precompute frozen Thinker hidden states and write a derived manifest."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ulm_live.thinker import ULMThinker  # noqa: E402
from ulm_live.talker.semantic_cache import cache_metadata  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--thinker", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-manifest", required=True)
    p.add_argument("--hidden-layer", type=int, default=-1)
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main():
    a = parse_args()
    source = Path(a.manifest)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    thinker = ULMThinker(
        a.thinker,
        device=a.device,
        torch_dtype=a.dtype,
        hidden_layer=a.hidden_layer,
        freeze=True,
    )
    meta = cache_metadata(
        a.thinker,
        a.hidden_layer,
        getattr(thinker.tokenizer, "name_or_path", a.thinker),
        thinker.hidden_size,
    )
    rows = []
    for i, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"empty text at row {i}")
        tok = thinker.tokenize(text)
        with torch.no_grad():
            hidden = thinker.forward_hidden(
                tok["input_ids"], tok.get("attention_mask"), a.hidden_layer
            )
        hidden = hidden.to(torch.bfloat16 if a.dtype == "bf16" else torch.float16).cpu()
        mask = tok.get(
            "attention_mask", torch.ones(hidden.shape[:2], dtype=torch.long)
        ).cpu()
        ident = str(row.get("id", i))
        path = out / f"{ident}.pt"
        torch.save(
            {
                "hidden_states": hidden,
                "attention_mask": mask.squeeze(0),
                "sequence_length": hidden.shape[1],
                "dtype": str(hidden.dtype),
                "metadata": meta,
            },
            path,
        )
        row["semantic_path"] = str(path.resolve())
        rows.append(row)
    dest = Path(a.output_manifest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    print(f"cached {len(rows)} samples -> {dest}")


if __name__ == "__main__":
    main()
