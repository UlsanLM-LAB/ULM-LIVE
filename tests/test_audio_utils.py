from pathlib import Path
import pytest
import torch

from ulm_live.utils import (
    get_duration,
    load_wav,
    peak_normalize,
    resample_audio,
    save_wav,
    to_mono,
)


def test_save_and_load_wav_roundtrip(tmp_path: Path) -> None:
    test_file = tmp_path / "roundtrip.wav"
    sample_rate = 24000
    # Generate 0.5s of 440Hz sine wave
    t = torch.linspace(0, 0.5, int(sample_rate * 0.5))
    expected_waveform = 0.8 * torch.sin(2 * torch.pi * 440.0 * t).unsqueeze(0)

    save_wav(test_file, expected_waveform, sample_rate, encoding="pcm_16")
    loaded_waveform, loaded_sr = load_wav(test_file)

    assert loaded_sr == sample_rate
    assert loaded_waveform.shape == expected_waveform.shape
    # Check that 16-bit PCM quantization error is small
    assert torch.max(torch.abs(loaded_waveform - expected_waveform)).item() < 1e-3


def test_save_and_load_float32(tmp_path: Path) -> None:
    test_file = tmp_path / "float32.wav"
    sample_rate = 16000
    waveform = torch.randn(1, 8000)
    save_wav(test_file, waveform, sample_rate, encoding="float32")
    loaded, sr = load_wav(test_file)
    assert sr == sample_rate
    assert torch.allclose(loaded, waveform, atol=1e-5)


def test_to_mono() -> None:
    # 1D
    mono_1d = torch.randn(16000)
    out_1d = to_mono(mono_1d)
    assert out_1d.shape == (1, 16000)

    # 2D stereo
    left = torch.ones(1, 1000)
    right = torch.full((1, 1000), 3.0)
    stereo = torch.cat([left, right], dim=0) # (2, 1000)
    out_2d = to_mono(stereo)
    assert out_2d.shape == (1, 1000)
    assert torch.allclose(out_2d, torch.full((1, 1000), 2.0))

    # 3D batch stereo
    batch_stereo = torch.stack([stereo, stereo], dim=0) # (2, 2, 1000)
    out_3d = to_mono(batch_stereo)
    assert out_3d.shape == (2, 1, 1000)
    assert torch.allclose(out_3d, torch.full((2, 1, 1000), 2.0))


def test_resample_audio() -> None:
    orig_sr = 48000
    target_sr = 24000
    waveform = torch.randn(1, orig_sr)
    resampled = resample_audio(waveform, orig_sr, target_sr)
    assert resampled.shape == (1, target_sr)

    # Same sample rate should return immediately
    same = resample_audio(waveform, orig_sr, orig_sr)
    assert same is waveform

    # 44.1kHz to 24kHz
    audio_44k = torch.randn(2, 44100)
    resampled_24k = resample_audio(audio_44k, 44100, 24000)
    assert resampled_24k.shape[0] == 2
    assert resampled_24k.shape[1] == 24000


def test_peak_normalize() -> None:
    waveform = torch.tensor([[0.2, -0.4, 0.1]])
    normalized = peak_normalize(waveform, target_peak=0.8)
    assert pytest.approx(torch.max(torch.abs(normalized)).item(), abs=1e-5) == 0.8

    # All zeros
    zeros = torch.zeros(1, 100)
    assert torch.equal(peak_normalize(zeros), zeros)

    # Invalid target peak
    with pytest.raises(ValueError):
        peak_normalize(waveform, target_peak=1.5)


def test_get_duration() -> None:
    waveform = torch.zeros(1, 48000)
    assert get_duration(waveform, 24000) == 2.0

    with pytest.raises(ValueError):
        get_duration(waveform, 0)


def test_nonexistent_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_wav("non_existent_audio_path_xyz.wav")
