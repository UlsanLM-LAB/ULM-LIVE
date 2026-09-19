import json
from pathlib import Path
import pytest
import torch

from ulm_live.data.manifest import (
    compute_dataset_summary,
    split_dataset,
    write_jsonl,
)
from ulm_live.data.schema import DatasetItem, UtteranceInfo
from ulm_live.data.segment import AudioSegmenter, match_audio_file, slice_waveform
from ulm_live.utils.audio import load_wav, save_wav


def test_slice_waveform() -> None:
    sr = 16000
    # 5 seconds waveform
    waveform = torch.randn(1, 5 * sr)

    # Slice 1.0s to 3.0s -> 2.0s (32000 samples)
    sliced = slice_waveform(waveform, sr, start=1.0, end=3.0)
    assert sliced.shape == (1, 2 * sr)

    # Slice without start/end -> full
    full = slice_waveform(waveform, sr)
    assert full.shape == waveform.shape

    # Clamp beyond bounds
    clamped = slice_waveform(waveform, sr, start=4.0, end=10.0)
    assert clamped.shape == (1, 1 * sr)


def test_match_audio_file() -> None:
    audio_by_stem = {"audio_01": Path("/data/audio_01.wav")}
    audio_by_name = {"audio_01.wav": Path("/data/audio_01.wav")}

    json_path = Path("/data/audio_01.json")
    assert match_audio_file(None, json_path, audio_by_stem, audio_by_name) == Path("/data/audio_01.wav")
    assert match_audio_file("audio_01.wav", Path("/data/other.json"), audio_by_stem, audio_by_name) == Path("/data/audio_01.wav")
    assert match_audio_file("non_existent.wav", Path("/data/other.json"), audio_by_stem, audio_by_name) is None


def test_manifest_and_summary(tmp_path: Path) -> None:
    items = [
        DatasetItem(
            id=f"ulsan_{i:04d}",
            audio_path=f"audio/ulsan_{i:04d}.wav",
            text="테스트 발화입니다",
            speaker_id="spk_A" if i < 3 else "spk_B",
            dialect="ulsan",
            sample_rate=24000,
            duration=2.5,
        )
        for i in range(5)
    ]

    out_file = tmp_path / "manifest.jsonl"
    write_jsonl(items, out_file)
    assert out_file.is_file()

    with open(out_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 5
    assert lines[0]["id"] == "ulsan_0000"

    summary = compute_dataset_summary(items)
    assert summary.num_samples == 5
    assert summary.num_speakers == 2
    assert summary.mean_duration == 2.5
    assert summary.total_hours == round((5 * 2.5) / 3600.0, 3)
    assert "spk_A" in summary.speaker_stats
    assert summary.speaker_stats["spk_A"]["samples"] == 3


def test_split_dataset_speaker_leakage() -> None:
    # 6 speakers, 2 samples each = 12 samples
    items = []
    for spk_idx in range(6):
        spk_id = f"spk_{spk_idx:02d}"
        for s in range(2):
            items.append(
                DatasetItem(
                    id=f"utt_{spk_id}_{s}",
                    audio_path=f"audio/test_{spk_id}_{s}.wav",
                    text="방언 발화 테스트",
                    speaker_id=spk_id,
                    dialect="ulsan",
                    sample_rate=24000,
                    duration=3.0,
                )
            )

    train, val, test = split_dataset(items, strategy="speaker", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    assert len(train) + len(val) + len(test) == len(items)

    train_spks = {it.speaker_id for it in train}
    val_spks = {it.speaker_id for it in val}
    test_spks = {it.speaker_id for it in test}

    # Verify zero speaker leakage across splits
    assert len(train_spks.intersection(val_spks)) == 0
    assert len(train_spks.intersection(test_spks)) == 0
    assert len(val_spks.intersection(test_spks)) == 0


def test_audio_segmenter_process_utterance(tmp_path: Path) -> None:
    # Create a 3-second source 48kHz audio file
    sr = 48000
    t = torch.linspace(0, 3.0, sr * 3)
    tone = 0.5 * torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)
    source_wav_path = tmp_path / "source.wav"
    save_wav(source_wav_path, tone, sr)

    out_dataset_dir = tmp_path / "processed_dataset"
    segmenter = AudioSegmenter(output_dir=out_dataset_dir, target_sample_rate=24000)

    utt = UtteranceInfo(
        utterance_id="utt_001",
        speaker_id="spk_ulsan_01",
        text="와 이라노 오늘 날씨 와 이래 덥노",
        start=0.5,
        end=2.5,
    )

    item, reason = segmenter.process_utterance(
        item_id="ulsan_000001",
        utterance=utt,
        source_audio_path=source_wav_path,
        dialect="ulsan",
    )

    assert reason is None
    assert item is not None
    assert item.id == "ulsan_000001"
    assert item.sample_rate == 24000
    assert pytest.approx(item.duration, rel=1e-2) == 2.0
    assert Path(out_dataset_dir / item.audio_path).is_file()

    # Verify saved audio sample rate and duration
    loaded_wav, loaded_sr = load_wav(out_dataset_dir / item.audio_path)
    assert loaded_sr == 24000
    assert loaded_wav.shape[0] == 1 # mono
    assert abs(loaded_wav.shape[-1] - 48000) < 1000
