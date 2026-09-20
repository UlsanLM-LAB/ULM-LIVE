import json
from pathlib import Path
import pytest
import torch

from ulm_live.talker.data import (
    TalkerCollator,
    TalkerDataset,
    build_id_mappings,
)


def test_build_id_mappings_deterministic(tmp_path: Path) -> None:
    manifest_data = [
        {"id": "1", "speaker_id": "spk_B", "dialect": "ulsan"},
        {"id": "2", "speaker_id": "spk_A", "dialect": "busan"},
        {"id": "3", "speaker_id": "spk_C", "dialect": "ulsan"},
    ]
    m_file = tmp_path / "test_m.jsonl"
    with open(m_file, "w", encoding="utf-8") as f:
        for it in manifest_data:
            f.write(json.dumps(it) + "\n")

    s2id_1, d2id_1 = build_id_mappings(m_file)
    s2id_2, d2id_2 = build_id_mappings(m_file)

    # Deterministic alphabetical assignment
    assert s2id_1 == s2id_2
    assert s2id_1["spk_A"] == 0
    assert s2id_1["spk_B"] == 1
    assert s2id_1["spk_C"] == 2
    assert d2id_1["busan"] == 0
    assert d2id_1["ulsan"] == 1


def test_talker_dataset_and_collator(tmp_path: Path) -> None:
    # Prepare fixtures
    codec_dir = tmp_path / "codec"
    codec_dir.mkdir()

    # Save 2 mock token files of shape (1, 8, 15) and (1, 8, 25)
    torch.save(torch.randint(0, 2048, (1, 8, 15)), codec_dir / "item1.pt")
    torch.save(torch.randint(0, 2048, (1, 8, 25)), codec_dir / "item2.pt")

    manifest = [
        {
            "id": "item1",
            "text": "첫번째 문장",
            "codec_path": "codec/item1.pt",
            "speaker_id": "spk_01",
            "dialect": "ulsan",
        },
        {
            "id": "item2",
            "text": "두번째 긴 문장입니다",
            "codec_path": "codec/item2.pt",
            "speaker_id": "spk_02",
            "dialect": "ulsan",
        },
    ]
    m_path = tmp_path / "manifest.jsonl"
    with open(m_path, "w", encoding="utf-8") as f:
        for it in manifest:
            f.write(json.dumps(it) + "\n")

    dataset = TalkerDataset(m_path, base_dir=tmp_path, num_quantizers=8)
    assert len(dataset) == 2

    item0 = dataset[0]
    assert item0["id"] == "item1"
    assert item0["audio_codes"].shape == (8, 15)

    collator = TalkerCollator(pad_token_id=2048, max_audio_len=100)
    batch = collator([dataset[0], dataset[1]])

    # Verify batch collation
    # BOS alignment preserves all 25 real targets.
    assert batch["audio_codes"].shape == (2, 8, 25)
    assert batch["targets"].shape == (2, 8, 25)
    assert batch["speaker_ids"].shape == (2,)
    assert batch["dialect_ids"].shape == (2,)

    assert torch.all(batch["audio_codes"][:, :, 0] == 2049)
    assert torch.equal(batch["targets"][0, :, :15], dataset[0]["audio_codes"])
    assert torch.all(batch["audio_codes"][0, :, 15:] == 2048)
    assert torch.all(batch["targets"][0, :, 15:] == 2048)


def test_talker_dataset_strict_codec_error(tmp_path: Path) -> None:
    manifest = [
        {
            "id": "missing_codec",
            "text": "테스트",
            "codec_path": "codec/nonexistent.pt",
            "speaker_id": "spk_01",
            "dialect": "ulsan",
        },
    ]
    m_path = tmp_path / "manifest.jsonl"
    with open(m_path, "w", encoding="utf-8") as f:
        for it in manifest:
            f.write(json.dumps(it) + "\n")

    # Production never fabricates codec tensors, even if the legacy flag is false.
    dataset_lenient = TalkerDataset(
        m_path, base_dir=tmp_path, num_quantizers=8, strict_codec=False
    )
    with pytest.raises(FileNotFoundError):
        _ = dataset_lenient[0]
    dataset_debug = TalkerDataset(
        m_path, base_dir=tmp_path, num_quantizers=8, allow_test_placeholders=True
    )
    item = dataset_debug[0]
    assert item["audio_codes"].shape == (8, 1)

    # Strict mode should raise RuntimeError
    dataset_strict = TalkerDataset(
        m_path, base_dir=tmp_path, num_quantizers=8, strict_codec=True
    )
    with pytest.raises(FileNotFoundError, match="missing codec tensor"):
        _ = dataset_strict[0]
