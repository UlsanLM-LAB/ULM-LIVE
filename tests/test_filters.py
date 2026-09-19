from pathlib import Path
import pytest
import torch

from ulm_live.data.filters import (
    QualityFilter,
    clean_transcript,
    compute_silence_ratio,
)


def test_clean_transcript() -> None:
    raw = "((기침소리)) 그래가꼬 [웃음] 와 그라노 o/ b/ 저기요"
    cleaned = clean_transcript(raw)
    assert "기침소리" not in cleaned
    assert "웃음" not in cleaned
    assert "o/" not in cleaned
    assert "b/" not in cleaned
    assert "그래가꼬" in cleaned
    assert "와 그라노" in cleaned
    assert "저기요" in cleaned


def test_quality_filter_text() -> None:
    qf = QualityFilter(min_text_len=2, max_text_len=10)
    # Too short
    valid, _, reason = qf.validate_text("a")
    assert not valid
    assert "text_too_short" in reason

    # Too long
    valid, _, reason = qf.validate_text("이것은너무길어서탈락해야하는문장입니다")
    assert not valid
    assert "text_too_long" in reason

    # Valid
    valid, cleaned, reason = qf.validate_text("밥 묵자")
    assert valid
    assert reason is None
    assert cleaned == "밥 묵자"


def test_compute_silence_ratio() -> None:
    sr = 16000
    # 1 second pure silence
    silence = torch.zeros(1, sr)
    ratio_silent = compute_silence_ratio(silence, sr)
    assert ratio_silent == 1.0

    # 1 second loud sine tone
    t = torch.linspace(0, 1.0, sr)
    tone = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)
    ratio_tone = compute_silence_ratio(tone, sr)
    assert ratio_tone < 0.1


def test_quality_filter_audio() -> None:
    qf = QualityFilter(min_duration=1.0, max_duration=5.0, min_sample_rate=16000, max_silence_ratio=0.5)

    sr = 16000
    # 0.5s audio -> duration too short
    short_wav = torch.randn(1, int(sr * 0.5))
    valid, dur, sil, reason = qf.validate_audio(short_wav, sr)
    assert not valid
    assert "duration_too_short" in reason

    # 6.0s audio -> duration too long
    long_wav = torch.randn(1, int(sr * 6.0))
    valid, dur, sil, reason = qf.validate_audio(long_wav, sr)
    assert not valid
    assert "duration_too_long" in reason

    # Low sample rate (8000Hz)
    valid, _, _, reason = qf.validate_audio(torch.randn(1, 16000), 8000)
    assert not valid
    assert "sample_rate_too_low" in reason

    # 2.0s valid sine tone
    t = torch.linspace(0, 2.0, sr * 2)
    tone = torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)
    valid, dur, sil, reason = qf.validate_audio(tone, sr)
    assert valid
    assert reason is None
    assert pytest.approx(dur, rel=1e-2) == 2.0
    assert sil < 0.1
