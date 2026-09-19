from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class EncodedAudio:
    """Container for discrete audio codec representations and metadata."""

    codes: torch.Tensor
    sample_rate: int
    bandwidth: float | None = None
    frame_rate: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_quantizers(self) -> int:
        """Number of RVQ codebooks (quantizers)."""
        if self.codes.ndim >= 2:
            return self.codes.shape[-2]
        return 1

    @property
    def num_frames(self) -> int:
        """Number of acoustic token frames."""
        return self.codes.shape[-1]


class AudioCodec(ABC):
    """Abstract base class for neural audio codecs."""

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Name of the codec backend."""
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Target audio sample rate in Hz."""
        ...

    @property
    @abstractmethod
    def frame_rate(self) -> float:
        """Codec frame rate in Hz (frames per second)."""
        ...

    @property
    @abstractmethod
    def num_quantizers(self) -> int:
        """Number of quantizer codebooks."""
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device where codec model weights reside."""
        ...

    @property
    def approx_token_rate(self) -> float:
        """Approximate number of discrete tokens per second."""
        return self.frame_rate * float(self.num_quantizers)

    @abstractmethod
    def encode(self, waveform: torch.Tensor, sample_rate: int, **kwargs: Any) -> EncodedAudio:
        """Encode waveform into discrete audio tokens.

        Args:
            waveform: Audio tensor of shape (channels, samples) or (batch, channels, samples).
            sample_rate: Sample rate of the input waveform in Hz.

        Returns:
            EncodedAudio instance containing discrete codes and metadata.
        """
        ...

    @abstractmethod
    def decode(self, encoded: EncodedAudio, **kwargs: Any) -> tuple[torch.Tensor, int]:
        """Decode discrete audio tokens back into waveform.

        Args:
            encoded: EncodedAudio instance.

        Returns:
            Tuple of (reconstructed_waveform, sample_rate).
            reconstructed_waveform has shape (channels, samples) or (batch, channels, samples).
        """
        ...

    @abstractmethod
    def to(self, device: torch.device | str) -> "AudioCodec":
        """Move codec model weights to the target device."""
        ...
