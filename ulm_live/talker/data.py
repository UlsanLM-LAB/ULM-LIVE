"""Strict Talker dataset, BOS alignment, semantic cache loading and acoustic delay."""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset
from ulm_live.codec import AudioCodec
from ulm_live.talker.semantic_cache import validate_cache_metadata


def build_id_mappings(paths, output_dir=None):
    paths = (
        [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    )
    speakers = set()
    dialects = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                speakers.add(str(x["speaker_id"]))
                dialects.add(str(x["dialect"]))
    sm = {v: i for i, v in enumerate(sorted(speakers))}
    dm = {v: i for i, v in enumerate(sorted(dialects))}
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "speaker2id.json").write_text(
            json.dumps(sm, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out / "dialect2id.json").write_text(
            json.dumps(dm, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return sm, dm


def apply_acoustic_delay(
    codes: torch.Tensor, delay: int, pad_token_id: int = 2048, bos_token_id: int = 2049
) -> torch.Tensor:
    if delay not in (0, 1, 2):
        raise ValueError("delay must be 0, 1, or 2")
    if delay == 0:
        return codes.clone()
    k, t = codes.shape[-2:]
    out = codes.new_full((*codes.shape[:-2], k, t + (k - 1) * delay), pad_token_id)
    for q in range(k):
        out[..., q, q * delay : q * delay + t] = codes[..., q, :]
    return out


def remove_acoustic_delay(
    codes: torch.Tensor, delay: int, original_frames: int | None = None
) -> torch.Tensor:
    if delay not in (0, 1, 2):
        raise ValueError("delay must be 0, 1, or 2")
    if delay == 0:
        return codes.clone()
    k = codes.shape[-2]
    length = (
        original_frames
        if original_frames is not None
        else codes.shape[-1] - (k - 1) * delay
    )
    return torch.stack(
        [codes[..., q, q * delay : q * delay + length] for q in range(k)], -2
    )


def _load_cache(path: Path, expected: dict[str, Any] | None):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("hidden_states"), torch.Tensor
    ):
        raise ValueError(f"invalid semantic cache: {path}")
    meta = payload.get("metadata", {})
    try:
        validate_cache_metadata(meta, expected or {})
    except ValueError as error:
        raise ValueError(f"{error}: {path}") from error
    h = payload["hidden_states"]
    mask = payload.get("attention_mask", torch.ones(h.shape[-2], dtype=torch.long))
    if h.ndim == 3 and h.shape[0] == 1:
        h = h[0]
    if h.ndim != 2 or mask.ndim != 1 or mask.numel() != h.shape[0]:
        raise ValueError(f"invalid semantic cache shapes: {path}")
    return h, mask, meta


class TalkerDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        base_dir=None,
        speaker2id=None,
        dialect2id=None,
        codec: AudioCodec | None = None,
        num_quantizers=32,
        strict_codec=True,
        cache_only=False,
        semantic_cache_metadata=None,
        allow_test_placeholders=False,
    ):
        self.manifest_path = Path(manifest_path)
        self.base_dir = Path(base_dir) if base_dir else self.manifest_path.parent
        self.codec = codec
        self.num_quantizers = num_quantizers
        self.strict_codec = strict_codec
        self.cache_only = cache_only
        self.expected_cache = semantic_cache_metadata
        self.allow_test_placeholders = allow_test_placeholders
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        self.items = [
            json.loads(x)
            for x in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        sm, dm = build_id_mappings(self.manifest_path)
        self.speaker2id = speaker2id or sm
        self.dialect2id = dialect2id or dm

    def __len__(self):
        return len(self.items)

    def _path(self, value):
        p = Path(value)
        return p if p.is_absolute() else self.base_dir / p

    def __getitem__(self, idx):
        x = self.items[idx]
        ident = x.get("id", f"sample_{idx}")
        text = str(x.get("text", "")).strip()
        if not text:
            raise ValueError(f"empty text: {ident}")
        if x.get("split") is not None and x["split"] not in ("train", "val", "test"):
            raise ValueError(f"invalid split for {ident}: {x['split']}")
        if "duration" in x and (
            not math.isfinite(float(x["duration"])) or float(x["duration"]) <= 0
        ):
            raise ValueError(f"invalid audio duration for {ident}")
        if "sample_rate" in x and int(x["sample_rate"]) <= 0:
            raise ValueError(f"invalid sample rate for {ident}")
        spk = str(x.get("speaker_id", ""))
        dia = str(x.get("dialect", ""))
        if spk not in self.speaker2id:
            raise ValueError(f"invalid speaker ID: {spk}")
        if dia not in self.dialect2id:
            raise ValueError(f"invalid dialect ID: {dia}")
        codes = None
        cp = x.get("codec_path")
        if cp:
            path = self._path(cp)
            if path.is_file():
                try:
                    codes = torch.load(path, map_location="cpu", weights_only=True)
                except Exception as e:
                    raise ValueError(f"corrupt codec tensor for {ident}: {e}") from e
        if codes is None and self.codec is not None and x.get("audio_path"):
            from ulm_live.utils.audio import load_wav

            wav, sr = load_wav(self._path(x["audio_path"]))
            codes = self.codec.encode(
                wav, sr, num_quantizers=self.num_quantizers
            ).codes.cpu()
        if codes is None:
            if not self.allow_test_placeholders:
                raise FileNotFoundError(f"missing codec tensor for {ident}: {cp}")
            codes = torch.zeros(self.num_quantizers, 1, dtype=torch.long)
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        if codes.ndim != 2 or codes.shape[1] == 0:
            raise ValueError(f"invalid codec tensor rank/length for {ident}")
        if codes.shape[0] < self.num_quantizers:
            raise ValueError(
                f"requested {self.num_quantizers} quantizers but {ident} has {codes.shape[0]}"
            )
        codes = codes[: self.num_quantizers].long()
        if torch.any((codes < 0) | (codes >= 2048)):
            raise ValueError(f"codec token outside [0, 2047] for {ident}")
        out = {
            "id": ident,
            "text": text,
            "audio_codes": codes,
            "speaker_id": self.speaker2id[spk],
            "dialect_id": self.dialect2id[dia],
        }
        semantic = x.get("semantic_path")
        if semantic:
            path = self._path(semantic)
            if not path.is_file():
                raise FileNotFoundError(f"missing semantic cache for {ident}: {path}")
            (
                out["semantic_hidden_states"],
                out["semantic_attention_mask"],
                out["semantic_metadata"],
            ) = _load_cache(path, self.expected_cache)
        elif self.cache_only:
            raise FileNotFoundError(
                f"semantic_path missing in cache-only mode: {ident}"
            )
        return out


class TalkerCollator:
    def __init__(
        self,
        tokenizer=None,
        pad_token_id=2048,
        bos_token_id=2049,
        max_audio_len=1024,
        max_text_len=256,
        acoustic_delay_frames=0,
    ):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len
        self.delay = acoustic_delay_frames

    def __call__(self, batch):
        delayed = [
            apply_acoustic_delay(
                x["audio_codes"][:, : self.max_audio_len],
                self.delay,
                self.pad_token_id,
                self.bos_token_id,
            )
            for x in batch
        ]
        b = len(batch)
        k = delayed[0].shape[0]
        t = max(x.shape[1] for x in delayed)
        inputs = torch.full((b, k, t), self.pad_token_id, dtype=torch.long)
        target = torch.full_like(inputs, self.pad_token_id)
        audio_mask = torch.zeros(b, t, dtype=torch.bool)
        stop = torch.full((b, t), -1, dtype=torch.long)
        for i, c in enumerate(delayed):
            n = c.shape[1]
            target[i, :, :n] = c
            inputs[i, :, 0] = self.bos_token_id
            if n > 1:
                inputs[i, :, 1:n] = c[:, :-1]
            # A delayed frame remains valid while any codebook carries a real code.
            valid = (c != self.pad_token_id).any(0)
            audio_mask[i, :n] = valid
            stop[i, :n][valid] = 0
            stop[i, valid.nonzero()[-1]] = 1
        out = {
            "ids": [x["id"] for x in batch],
            "audio_codes": inputs,
            "targets": target,
            "audio_attention_mask": audio_mask,
            "stop_targets": stop,
            "speaker_ids": torch.tensor([x["speaker_id"] for x in batch]),
            "dialect_ids": torch.tensor([x["dialect_id"] for x in batch]),
        }
        if "semantic_hidden_states" in batch[0]:
            s = max(x["semantic_hidden_states"].shape[0] for x in batch)
            d = batch[0]["semantic_hidden_states"].shape[1]
            dtype = batch[0]["semantic_hidden_states"].dtype
            h = torch.zeros(b, s, d, dtype=dtype)
            m = torch.zeros(b, s, dtype=torch.long)
            for i, x in enumerate(batch):
                n = x["semantic_hidden_states"].shape[0]
                h[i, :n] = x["semantic_hidden_states"]
                m[i, :n] = x["semantic_attention_mask"]
            out["semantic_hidden_states"] = h
            out["semantic_attention_mask"] = m
        elif self.tokenizer is not None:
            tok = self.tokenizer(
                [x["text"] for x in batch],
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            out["input_ids"] = tok["input_ids"]
            out["attention_mask"] = tok.get("attention_mask")
        return out
