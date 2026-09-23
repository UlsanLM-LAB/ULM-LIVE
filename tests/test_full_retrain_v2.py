import importlib.util
import json
from pathlib import Path
import sys

import torch


SPEC = importlib.util.spec_from_file_location(
    "train_full_retrain_v2",
    Path(__file__).parents[1] / "scripts" / "train_full_retrain_v2.py",
)
train = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train
SPEC.loader.exec_module(train)


def test_fixed_prompt_composition() -> None:
    categories = [prompt["category"] for prompt in train.FIXED_PROMPTS]
    assert len(categories) == 20
    assert categories.count("short") == 5
    assert categories.count("normal") == 5
    assert categories.count("ulsan_dialect") == 5
    assert categories.count("long") == 5
    assert train.EVALUATION_MAX_FRAMES == 250


def test_token_distribution_exposes_single_token_loop() -> None:
    tokens = torch.zeros(16, 12, dtype=torch.long)
    stats = train.token_distribution(tokens)
    assert stats["q0_entropy"] == 0.0
    assert stats["q0_unique"] == 1
    assert stats["q0_top1_ratio"] == 1.0
    assert stats["top1_ratio_by_q"] == [1.0] * 16


def test_reconstruction_selection_is_deterministic(tmp_path: Path) -> None:
    manifest = tmp_path / "test.cached.jsonl"
    rows = [
        {
            "id": f"sample_{index:03d}",
            "text": f"정상 문장 {index}",
            "duration": 2.0,
            "codec_path": f"codec/{index}.pt",
            "semantic_path": f"semantic/{index}.pt",
        }
        for index in range(30)
    ]
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    first = train.select_test_rows(manifest, 20)
    second = train.select_test_rows(manifest, 20)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert len({row["id"] for row in first}) == 20
