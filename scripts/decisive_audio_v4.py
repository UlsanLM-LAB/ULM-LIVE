#!/usr/bin/env python3
"""Build the fixed held-out listening pack on the training host only."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.decisive_ar_v4 import BASE, DATA, OUT  # noqa: E402
from scripts.pilot_talker_ar_v3 import measure_ar, write_json  # noqa: E402
from ulm_live.codec import EncodedAudio, build_codec  # noqa: E402
from ulm_live.talker import TalkerConfig, TalkerDataset, ULMTalker, build_id_mappings  # noqa: E402
from ulm_live.utils.audio import save_wav  # noqa: E402


def decode(codec, codes, path):
    audio, sr = codec.decode(EncodedAudio(
        codes=codes, sample_rate=24000, frame_rate=12.5,
        metadata={"backend": "mimi", "num_quantizers": 16},
    ))
    save_wav(path, audio.squeeze(0).cpu(), sr)


def main():
    selection = json.loads((OUT / "selection.json").read_text())
    paths = [DATA / f"{split}.cached.jsonl" for split in ("train", "val", "test")]
    speaker2id, dialect2id = build_id_mappings(paths)
    dataset = TalkerDataset(paths[1], speaker2id=speaker2id, dialect2id=dialect2id,
                            num_quantizers=16, cache_only=True)
    by_id = {row["id"]: i for i, row in enumerate(dataset.items)}
    samples = [dataset[by_id[ident]] for ident in selection["val_ar_ids"]]
    output = OUT / "listening"
    output.mkdir(parents=True, exist_ok=True)
    codec = build_codec(backend="mimi", device="cuda", num_quantizers=16)
    for i, sample in enumerate(samples, 1):
        dest = output / f"sample_{i:03}"
        dest.mkdir(parents=True, exist_ok=True)
        row = dataset.items[by_id[sample["id"]]]
        (dest / "text.txt").write_text(sample["text"] + "\n", encoding="utf-8")
        if not (dest / "original.wav").exists():
            shutil.copy2(dataset._path(row["audio_path"]), dest / "original.wav")
        if not (dest / "mimi_gt.wav").exists():
            decode(codec, sample["audio_codes"].to("cuda"), dest / "mimi_gt.wav")
    paths = {"base": BASE / "final.pt",
             "tf_cont": OUT / "2048-cont-tf" / "latest.pt",
             "rollout": OUT / "2048-cont-rollout" / "latest.pt"}
    base = torch.load(paths["base"], map_location="cpu", weights_only=False)
    for label, path in paths.items():
        payload = base if label == "base" else torch.load(path, map_location="cpu", weights_only=False)
        if label != "base" and payload["step"] != 8000:
            raise ValueError(f"{label} continuation incomplete")
        model = ULMTalker(TalkerConfig(**base["config"])).to("cuda").eval()
        model.load_state_dict(payload["model_state_dict"] if label == "base" else payload["model"])
        for i, sample in enumerate(samples, 1):
            dest = output / f"sample_{i:03}"
            target = dest / f"{label}_ar.wav"
            if target.exists():
                continue
            _, result = measure_ar(model, [sample], torch.device("cuda"), normal=True)
            decode(codec, result[0]["codes"].to("cuda"), target)
            print(f"audio {label} {i}/20", flush=True)
        del model
        torch.cuda.empty_cache()
    files = [f for f in output.rglob("*") if f.is_file()]
    summary = {"path": str(output.resolve()), "file_count": len(files),
               "size_bytes": sum(f.stat().st_size for f in files),
               "sample_ids": selection["val_ar_ids"]}
    write_json(OUT / "listening_summary.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
