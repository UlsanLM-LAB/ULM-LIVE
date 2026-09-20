"""Versioned semantic-cache metadata shared by precompute and training."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_CACHE_KEYS = (
    "thinker_id",
    "thinker_fingerprint",
    "hidden_layer",
    "tokenizer_id",
    "semantic_dim",
)


def thinker_fingerprint(
    identifier: str | Path, config: dict[str, Any] | None = None
) -> str:
    """Stable identity hash; hashes local model config/tokenizer metadata, never weights."""
    p = Path(identifier)
    h = hashlib.sha256()
    h.update(str(identifier).encode())
    if config is not None:
        h.update(json.dumps(config, sort_keys=True, default=str).encode())
    if p.is_dir():
        for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
            f = p / name
            if f.is_file():
                h.update(name.encode())
                h.update(f.read_bytes())
    return h.hexdigest()


def cache_metadata(
    thinker_id: str,
    hidden_layer: int,
    tokenizer_id: str,
    semantic_dim: int,
    config=None,
):
    return {
        "format_version": 1,
        "thinker_id": thinker_id,
        "thinker_fingerprint": thinker_fingerprint(thinker_id, config),
        "hidden_layer": hidden_layer,
        "tokenizer_id": tokenizer_id,
        "semantic_dim": semantic_dim,
    }


def validate_cache_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_CACHE_KEYS if k not in actual]
    mismatch = [k for k, v in expected.items() if actual.get(k) != v]
    if missing or mismatch:
        raise ValueError(
            f"semantic cache metadata invalid; missing={missing}, mismatch={mismatch}"
        )
