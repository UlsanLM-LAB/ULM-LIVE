from pathlib import Path
from typing import Any
import pytest
import torch

from ulm_live.codec import AudioCodec, EncodedAudio, build_codec
from ulm_live.talker import (
    SpeechGenerationResult,
    SpeechSynthesizer,
    TalkerConfig,
    TalkerGenerationConfig,
    ULMTalker,
    load_talker_checkpoint,
    save_talker_checkpoint,
)
from ulm_live.thinker import ULMThinker
from ulm_live.utils.audio import load_wav


class MockCodec(AudioCodec):
    """Fast mock audio codec for unit tests without downloading weights."""

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def sample_rate(self) -> int:
        return 24000

    @property
    def frame_rate(self) -> float:
        return 12.5

    @property
    def num_quantizers(self) -> int:
        return 8

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def to(self, device: torch.device | str) -> "MockCodec":
        return self

    def encode(
        self, waveform: torch.Tensor, sample_rate: int, **kwargs: Any
    ) -> EncodedAudio:
        T = max(1, waveform.shape[-1] // 1920)
        codes = torch.zeros((1, 8, T), dtype=torch.long)
        return EncodedAudio(codes=codes, sample_rate=24000, frame_rate=12.5)

    def decode(self, encoded: EncodedAudio, **kwargs: Any) -> tuple[torch.Tensor, int]:
        num_samples = encoded.codes.shape[-1] * 1920
        waveform = torch.zeros((1, 1, num_samples), dtype=torch.float32)
        return waveform, 24000


def test_checkpoint_save_and_restore(tmp_path: Path) -> None:
    cfg = TalkerConfig(
        semantic_dim=128,
        talker_dim=64,
        num_layers=1,
        num_heads=2,
        num_quantizers=8,
    )
    talker = ULMTalker(cfg)

    ckpt_file = tmp_path / "talker_ckpt.pt"
    save_talker_checkpoint(
        ckpt_file,
        model=talker,
        speaker2id={"spk1": 0, "spk2": 1},
        dialect2id={"ulsan": 0},
        codec_metadata={"backend": "mock"},
        step=42,
    )
    assert ckpt_file.is_file()

    loaded_talker, meta = load_talker_checkpoint(ckpt_file)
    assert meta["speaker2id"] == {"spk1": 0, "spk2": 1}
    assert meta["dialect2id"] == {"ulsan": 0}
    assert meta["step"] == 42
    assert loaded_talker.config.semantic_dim == 128
    assert loaded_talker.config.num_quantizers == 8


def test_synthesizer_from_hidden_with_mock_codec(tmp_path: Path) -> None:
    cfg = TalkerConfig(
        semantic_dim=128,
        talker_dim=64,
        num_layers=1,
        num_heads=2,
        num_quantizers=8,
    )
    talker = ULMTalker(cfg)
    codec = MockCodec()

    synthesizer = SpeechSynthesizer(
        thinker=None,
        talker=talker,
        codec=codec,
        device="cpu",
    )

    sem = torch.randn(1, 4, 128)
    gen_cfg = TalkerGenerationConfig(max_new_tokens=10, stop_threshold=2.0)
    result = synthesizer.synthesize_from_hidden(sem, generation_config=gen_cfg)

    assert isinstance(result, SpeechGenerationResult)
    assert result.codec_tokens.shape == (8, 10)
    assert result.sample_rate == 24000
    assert result.waveform.ndim == 2
    assert result.waveform.shape[0] == 1
    assert result.waveform.shape[1] == 10 * 1920
    assert result.duration == pytest.approx(10 * 1920 / 24000.0, rel=1e-2)
    assert result.rtf > 0.0

    # Test WAV save and reload
    out_wav = tmp_path / "mock_output.wav"
    result.save(out_wav)
    assert out_wav.is_file()

    loaded_wav, loaded_sr = load_wav(out_wav)
    assert loaded_sr == 24000
    assert loaded_wav.shape == result.waveform.shape


# ---------------------------------------------------------
# Integration Tests (executed with: pytest -m integration)
# ---------------------------------------------------------


@pytest.mark.integration
def test_real_speech_synthesizer_with_mimi(tmp_path: Path) -> None:
    """Validate Talker generate + real Mimi codec decode integration."""
    codec = build_codec(backend="mimi", device="cpu")
    cfg = TalkerConfig(
        semantic_dim=256,
        talker_dim=128,
        num_layers=2,
        num_heads=4,
        num_quantizers=codec.num_quantizers,  # 32
        codebook_size=2048,
    )
    talker = ULMTalker(cfg)

    synthesizer = SpeechSynthesizer(
        thinker=None,
        talker=talker,
        codec=codec,
        device="cpu",
    )

    sem = torch.randn(1, 4, 256)
    gen_cfg = TalkerGenerationConfig(max_new_tokens=13)  # ~1.04s
    result = synthesizer.synthesize_from_hidden(sem, generation_config=gen_cfg)

    assert result.sample_rate == 24000
    assert result.codec_tokens.shape[0] == 32
    assert result.codec_tokens.shape[1] == 13
    assert not torch.isnan(result.waveform).any()
    assert result.duration > 0.5

    out_wav = tmp_path / "real_mimi.wav"
    result.save(out_wav)
    assert out_wav.is_file()
