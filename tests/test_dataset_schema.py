import json
from pathlib import Path
import pytest

from ulm_live.data.aihub import (
    AIHubParser,
    FieldMatcher,
    RegionClassifier,
    inspect_aihub_dataset,
)
from ulm_live.data.schema import DatasetItem, SpeakerInfo, UtteranceInfo


def test_dataset_item_serialization() -> None:
    item = DatasetItem(
        id="ulsan_000001",
        audio_path="audio/ulsan_000001.wav",
        text="와 이래 덥노 오늘",
        speaker_id="speaker_001",
        dialect="ulsan",
        sample_rate=24000,
        duration=3.821,
        source="aihub",
        codec_path="codec/ulsan_000001.pt",
    )
    d = item.to_dict()
    assert d["id"] == "ulsan_000001"
    assert d["duration"] == 3.821
    assert d["dialect"] == "ulsan"
    assert d["codec_path"] == "codec/ulsan_000001.pt"


def test_field_matcher() -> None:
    mappings = {
        "text_fields": ["dialect_form", "transcription", "text"],
        "speaker_id_fields": ["spk_id", "speaker_id"],
    }
    matcher = FieldMatcher(mappings)
    sample_dict = {"dialect_form": "밥 묵었나", "spk_id": "SPK01"}
    assert matcher.get_value(sample_dict, "text_fields") == "밥 묵었나"
    assert matcher.get_value(sample_dict, "speaker_id_fields") == "SPK01"
    assert matcher.get_value(sample_dict, "unknown_type", default="fallback") == "fallback"


def test_region_classifier() -> None:
    classifier = RegionClassifier(
        priority=["principal_residence", "birthplace", "current_residence"],
        aliases={
            "ulsan": ["울산", "울산광역시", "ulsan"],
            "busan": ["부산", "부산광역시"],
        },
    )

    # Priority check: principal_residence wins over birthplace
    spk1 = SpeakerInfo(
        speaker_id="spk_01",
        birthplace="부산광역시",
        principal_residence="울산광역시 남구",
        current_residence="서울시",
    )
    assert classifier.determine_region(spk1) == "ulsan"
    assert classifier.is_target_region(spk1, "ulsan") is True
    assert classifier.is_target_region(spk1, "busan") is False

    # Fallback to birthplace
    spk2 = SpeakerInfo(
        speaker_id="spk_02",
        birthplace="부산시 해운대구",
        principal_residence=None,
    )
    assert classifier.determine_region(spk2) == "busan"
    assert classifier.is_target_region(spk2, "busan") is True
    assert classifier.is_target_region(spk2, "ulsan") is False


def test_aihub_parser_multi_speaker(tmp_path: Path) -> None:
    sample_json = {
        "metadata": {"title": "경상도 대화", "audio_file": "dialogue_01.wav"},
        "speaker": [
            {
                "id": "SPK_ULSAN",
                "birthplace": "울산광역시",
                "principal_residence": "울산광역시 중구",
            },
            {
                "id": "SPK_BUSAN",
                "birthplace": "부산광역시",
                "principal_residence": "부산광역시 동래구",
            },
        ],
        "utterance": [
            {
                "id": "utt_01",
                "speaker_id": "SPK_ULSAN",
                "start": 1.0,
                "end": 4.5,
                "dialect_form": "와 이라노 진짜",
            },
            {
                "id": "utt_02",
                "speaker_id": "SPK_BUSAN",
                "start": 5.0,
                "end": 8.2,
                "dialect_form": "맞나 니 말이 맞다",
            },
        ],
    }

    json_path = tmp_path / "dialogue_01.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_json, f)

    parser = AIHubParser()
    speakers, utterances, audio_hint = parser.parse_file(json_path)

    assert len(speakers) == 2
    assert "SPK_ULSAN" in speakers
    assert "SPK_BUSAN" in speakers
    assert len(utterances) == 2
    assert utterances[0].text == "와 이라노 진짜"
    assert utterances[0].duration == 3.5
    assert audio_hint == "dialogue_01.wav"


def test_inspect_aihub_dataset_synthetic(tmp_path: Path) -> None:
    # Create synthetic directory
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()

    sample_json = {
        "speaker": [{"id": "SPK1", "principal_residence": "울산"}],
        "utterance": [{"id": "1", "speaker_id": "SPK1", "dialect_form": "테스트 발화"}],
    }
    with open(data_dir / "sample1.json", "w", encoding="utf-8") as f:
        json.dump(sample_json, f)
    # Dummy audio file
    (data_dir / "sample1.wav").write_bytes(b"RIFFdummy")

    results = inspect_aihub_dataset(data_dir)
    assert results["num_json_files"] == 1
    assert results["num_audio_files"] == 1
    assert "ulsan" in results["regions_found"]
    assert results["total_speakers_scanned"] == 1
