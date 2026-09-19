from typing import Any
import pytest
import torch

from ulm_live.codec import AudioCodec, EncodedAudio, MimiCodec, build_codec


class DummyCodec(AudioCodec):
    """Minimal dummy codec for fast unit tests without network or heavy weights."""

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)

    @property
    def backend_name(self) -> str:
        return "dummy"

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
        return self._device

    def to(self, device: torch.device | str) -> "DummyCodec":
        self._device = torch.device(device)
        return self

    def encode(self, waveform: torch.Tensor, sample_rate: int, **kwargs: Any) -> EncodedAudio:
        # 1 frame per 1920 samples (24000 / 12.5 = 1920)
        num_frames = max(1, waveform.shape[-1] // 1920)
        codes = torch.zeros((1, self.num_quantizers, num_frames), dtype=torch.long, device=self._device)
        return EncodedAudio(
            codes=codes,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            metadata={"backend": "dummy"},
        )

    def decode(self, encoded: EncodedAudio, **kwargs: Any) -> tuple[torch.Tensor, int]:
        num_samples = encoded.num_frames * 1920
        waveform = torch.zeros((1, 1, num_samples), dtype=torch.float32, device=self._device)
        return waveform, self.sample_rate


def test_encoded_audio_dataclass() -> None:
    codes = torch.randint(0, 1024, (2, 8, 25))
    enc = EncodedAudio(codes=codes, sample_rate=24000, frame_rate=12.5)
    assert enc.num_quantizers == 8
    assert enc.num_frames == 25
    assert enc.sample_rate == 24000
    assert enc.frame_rate == 12.5


def test_dummy_codec_contract() -> None:
    codec = DummyCodec()
    assert codec.backend_name == "dummy"
    assert codec.sample_rate == 24000
    assert codec.frame_rate == 12.5
    assert codec.num_quantizers == 8
    assert codec.approx_token_rate == 100.0

    wav = torch.randn(1, 24000)
    encoded = codec.encode(wav, 24000)
    assert isinstance(encoded, EncodedAudio)
    assert encoded.codes.shape == (1, 8, 12)

    decoded_wav, sr = codec.decode(encoded)
    assert sr == 24000
    assert decoded_wav.shape == (1, 1, 12 * 1920)


def test_build_codec_invalid_backend() -> None:
    with pytest.raises(ValueError, match="Unknown codec backend"):
        build_codec(backend="non_existent_codec")


def test_build_codec_lazy_load() -> None:
    # Test initialization without loading weights
    codec = build_codec(backend="mimi", load_pretrained=False)
    assert isinstance(codec, MimiCodec)
    assert codec.backend_name == "mimi"


# ---------------------------------------------------------
# Integration Tests (executed with: pytest -m integration)
# ---------------------------------------------------------


@pytest.mark.integration
def test_real_mimi_encode_decode() -> None:
    """Validate end-to-end encoding and decoding using real Mimi model weights."""
    codec = build_codec(backend="mimi", device="cpu")
    assert codec.backend_name == "mimi"
    assert codec.sample_rate == 24000
    assert codec.frame_rate == 12.5
    assert codec.num_quantizers == 32

    # 1 second of audio at 24kHz
    audio_1s = torch.sin(2 * torch.pi * 440.0 * torch.linspace(0, 1.0, 24000)).unsqueeze(0)

    encoded = codec.encode(audio_1s, sample_rate=24000)
    assert isinstance(encoded, EncodedAudio)
    assert encoded.codes.ndim == 3
    assert encoded.codes.shape[0] == 1
    assert encoded.codes.shape[1] == 32
    # At 12.5 Hz frame rate, 1 second audio yields ~13 frames
    assert 12 <= encoded.codes.shape[2] <= 14

    reconstructed, sr = codec.decode(encoded)
    assert sr == 24000
    assert reconstructed.ndim == 3
    assert reconstructed.shape[0] == 1 # batch
    assert reconstructed.shape[1] == 1 # mono
    # Check length is approximately 1 second
    assert abs(reconstructed.shape[-1] - 24000) < 2000


@pytest.mark.integration
def test_real_mimi_resampling_stereo() -> None:
    """Test that input audio with stereo channels and 48kHz is automatically converted."""
    codec = build_codec(backend="mimi", device="cpu")

    # 0.5s stereo audio at 48kHz
    stereo_48k = torch.randn(2, 24000)
    encoded = codec.encode(stereo_48k, sample_rate=48000)
    assert encoded.sample_rate == 24000
    assert encoded.codes.shape[1] == 32
    # 0.5s at 12.5Hz -> ~7 frames
    assert 6 <= encoded.codes.shape[2] <= 8

    recon, sr = codec.decode(encoded)
    assert sr == 24000
