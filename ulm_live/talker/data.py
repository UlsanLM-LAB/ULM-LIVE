import json
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import Dataset

from ulm_live.codec import AudioCodec


def build_id_mappings(
    manifest_paths: list[str | Path] | str | Path,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build deterministic, sorted string-to-integer mappings for speaker and dialect IDs."""
    if isinstance(manifest_paths, (str, Path)):
        paths = [Path(manifest_paths)]
    else:
        paths = [Path(p) for p in manifest_paths]

    speakers = set()
    dialects = set()

    for p in paths:
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if "speaker_id" in item:
                    speakers.add(str(item["speaker_id"]))
                if "dialect" in item:
                    dialects.add(str(item["dialect"]))

    # Sort for reproducibility
    speaker2id = {spk: idx for idx, spk in enumerate(sorted(list(speakers)))}
    dialect2id = {dia: idx for idx, dia in enumerate(sorted(list(dialects)))}

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "speaker2id.json", "w", encoding="utf-8") as f:
            json.dump(speaker2id, f, ensure_ascii=False, indent=2)
        with open(out / "dialect2id.json", "w", encoding="utf-8") as f:
            json.dump(dialect2id, f, ensure_ascii=False, indent=2)

    return speaker2id, dialect2id


class TalkerDataset(Dataset):
    """Dataset for training Talker model from ULM manifest JSONL files."""

    def __init__(
        self,
        manifest_path: str | Path,
        base_dir: str | Path | None = None,
        speaker2id: dict[str, int] | None = None,
        dialect2id: dict[str, int] | None = None,
        codec: AudioCodec | None = None,
        num_quantizers: int = 32,
        strict_codec: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.base_dir = Path(base_dir) if base_dir is not None else self.manifest_path.parent
        self.codec = codec
        self.num_quantizers = num_quantizers
        self.strict_codec = strict_codec

        self.items: list[dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.items.append(json.loads(line))

        # Build mappings if not provided
        if speaker2id is None or dialect2id is None:
            s2id, d2id = build_id_mappings(self.manifest_path)
            self.speaker2id = speaker2id or s2id
            self.dialect2id = dialect2id or d2id
        else:
            self.speaker2id = speaker2id
            self.dialect2id = dialect2id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        item_id = item.get("id", f"sample_{idx}")
        text = item.get("text", "")
        spk_str = str(item.get("speaker_id", "unknown"))
        dia_str = str(item.get("dialect", "ulsan"))

        spk_id = self.speaker2id.get(spk_str, 0)
        dia_id = self.dialect2id.get(dia_str, 0)

        # 1. Load codec tokens
        audio_codes = None
        codec_rel = item.get("codec_path")
        if codec_rel:
            codec_path = self.base_dir / codec_rel
            if codec_path.is_file():
                try:
                    # Stored tensor shape: (1, K, T) or (K, T)
                    loaded = torch.load(codec_path, weights_only=True)
                    if loaded.ndim == 3:
                        loaded = loaded.squeeze(0)
                    audio_codes = loaded[: self.num_quantizers, :]
                except Exception:
                    audio_codes = None

        # 2. Fallback to runtime audio encoding if needed and codec provided
        if audio_codes is None and self.codec is not None:
            audio_rel = item.get("audio_path")
            if audio_rel:
                audio_path = self.base_dir / audio_rel
                if audio_path.is_file():
                    from ulm_live.utils.audio import load_wav
                    wav, sr = load_wav(audio_path)
                    enc = self.codec.encode(wav, sr, num_quantizers=self.num_quantizers)
                    audio_codes = enc.codes.squeeze(0).cpu()

        # If still None, raise error in strict mode or create dummy placeholder
        if audio_codes is None:
            if self.strict_codec:
                raise RuntimeError(
                    f"Missing or invalid codec tokens for sample '{item_id}' (codec_path: '{codec_rel}')"
                )
            audio_codes = torch.zeros((self.num_quantizers, 1), dtype=torch.long)

        return {
            "id": item_id,
            "text": text,
            "audio_codes": audio_codes, # (K, T)
            "speaker_id": spk_id,
            "dialect_id": dia_id,
        }


class TalkerCollator:
    """Collates variable-length audio tokens and text prompts into padded batches."""

    def __init__(
        self,
        tokenizer: Any = None,
        pad_token_id: int = 2048,
        max_audio_len: int = 1024,
        max_text_len: int = 256,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_audio_len = max_audio_len
        self.max_text_len = max_text_len

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        B = len(batch)
        K = batch[0]["audio_codes"].shape[0]

        # 1. Audio token padding & shift-for-targets
        max_T = max(item["audio_codes"].shape[1] for item in batch)
        max_T = min(max_T, self.max_audio_len)

        # In autoregressive training:
        # Input tokens: codes[:, :, :-1]
        # Target tokens: codes[:, :, 1:]
        # If length is 1, keep input as is
        shifted_len = max(1, max_T - 1)

        input_codes = torch.full((B, K, shifted_len), self.pad_token_id, dtype=torch.long)
        targets = torch.full((B, K, shifted_len), self.pad_token_id, dtype=torch.long)

        for b, item in enumerate(batch):
            codes = item["audio_codes"][:, :max_T] # (K, T_cur)
            T_cur = codes.shape[1]
            if T_cur > 1:
                input_codes[b, :, : T_cur - 1] = codes[:, :-1]
                targets[b, :, : T_cur - 1] = codes[:, 1:]
            else:
                input_codes[b, :, :T_cur] = codes
                targets[b, :, :T_cur] = codes

        speaker_ids = torch.tensor([item["speaker_id"] for item in batch], dtype=torch.long)
        dialect_ids = torch.tensor([item["dialect_id"] for item in batch], dtype=torch.long)

        result: dict[str, Any] = {
            "ids": [item["id"] for item in batch],
            "audio_codes": input_codes,
            "targets": targets,
            "speaker_ids": speaker_ids,
            "dialect_ids": dialect_ids,
        }

        # 2. Tokenize text if tokenizer provided
        if self.tokenizer is not None:
            texts = [item["text"] for item in batch]
            tok = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            result["input_ids"] = tok["input_ids"]
            result["attention_mask"] = tok.get("attention_mask")

        return result
