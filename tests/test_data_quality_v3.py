"""Audio-quality measurements used to inspect canonical Talker data."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import wave

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_talker_data_v3.py"
SPEC = importlib.util.spec_from_file_location("inspect_talker_data_v3", SCRIPT)
assert SPEC and SPEC.loader
inspect = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect)


def test_spread_indices_cover_endpoints() -> None:
    assert inspect.spread_indices(10, 3) == [0, 4, 9]
    assert inspect.spread_indices(10, 1) == [5]


def test_audio_stats_decode_pcm_and_measure_silence(tmp_path: Path) -> None:
    path = tmp_path / "sample.wav"
    samples = np.zeros(24000, dtype="<i2")
    samples[12000:] = 16384
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(samples.tobytes())
    result = inspect.audio_stats(path)
    assert result["duration"] == 1
    assert result["silence_fraction"] == .5
    assert result["rms"] == pytest.approx(2**-.5 / 2)
