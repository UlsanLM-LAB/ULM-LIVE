from pathlib import Path
from typing import Any
import torch
import yaml

from ulm_live.codec.base import AudioCodec, EncodedAudio
from ulm_live.utils.audio import resample_audio, to_mono


def _resolve_device(device: str | torch.device) -> torch.device:
    """Resolve device string to torch.device with CUDA fallback."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return target


class MimiCodec(AudioCodec):
    """Neural audio codec backed by Kyutai's Mimi model.
    
    Provides 12.5 Hz frame rate representation optimized for Spoken Language Models.
    """

    def __init__(
        self,
        model_id: str = "kyutai/mimi",
        device: str | torch.device = "auto",
        num_quantizers: int | None = None,
        load_pretrained: bool = True,
    ) -> None:
        self._model_id = model_id
        self._target_device = _resolve_device(device)
        self._num_quantizers_override = num_quantizers
        self._model: Any = None

        if load_pretrained:
            self._load_model()

    def _load_model(self) -> None:
        """Load pretrained Mimi model from Hugging Face."""
        try:
            from transformers import MimiModel
        except ImportError as err:
            raise ImportError(
                "transformers is required for MimiCodec. Please install via 'pip install transformers'."
            ) from err

        try:
            self._model = MimiModel.from_pretrained(self._model_id)
            self._model.to(self._target_device)
            self._model.eval()
        except Exception as err:
            raise RuntimeError(
                f"Failed to load Mimi model from '{self._model_id}': {err}"
            ) from err

    @property
    def model(self) -> Any:
        """Underlying Hugging Face MimiModel."""
        if self._model is None:
            self._load_model()
        return self._model

    @property
    def backend_name(self) -> str:
        return "mimi"

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.model.config, "sampling_rate", 24000))

    @property
    def frame_rate(self) -> float:
        return float(getattr(self.model.config, "frame_rate", 12.5))

    @property
    def num_quantizers(self) -> int:
        if self._num_quantizers_override is not None:
            return self._num_quantizers_override
        return int(getattr(self.model.config, "num_quantizers", 32))

    @property
    def device(self) -> torch.device:
        return self._target_device

    def to(self, device: torch.device | str) -> "MimiCodec":
        self._target_device = _resolve_device(device)
        if self._model is not None:
            self._model.to(self._target_device)
        return self

    def encode(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        num_quantizers: int | None = None,
        **kwargs: Any,
    ) -> EncodedAudio:
        """Encode waveform into discrete Mimi audio tokens.

        Args:
            waveform: Tensor of shape (channels, samples) or (batch, channels, samples).
            sample_rate: Input sample rate in Hz.
            num_quantizers: Optional number of quantizers to use (default: self.num_quantizers).
        """
        # Ensure 3D tensor: (batch_size, channels, sequence_length)
        if waveform.ndim == 1:
            wav = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            wav = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            wav = waveform
        else:
            raise ValueError(f"Expected waveform rank 1, 2, or 3, got rank {waveform.ndim}")

        # Resample to 24kHz if needed
        if sample_rate != self.sample_rate:
            wav = resample_audio(wav, sample_rate, self.sample_rate)

        # Convert to mono if channels > 1
        if wav.shape[1] > 1:
            wav = to_mono(wav)

        wav = wav.to(device=self._target_device, dtype=torch.float32)

        nq = num_quantizers if num_quantizers is not None else self.num_quantizers

        with torch.no_grad():
            encoder_output = self.model.encode(wav, num_quantizers=nq, **kwargs)
            codes = encoder_output.audio_codes

        return EncodedAudio(
            codes=codes,
            sample_rate=self.sample_rate,
            frame_rate=self.frame_rate,
            metadata={
                "backend": self.backend_name,
                "model_id": self._model_id,
                "num_quantizers": nq,
            },
        )

    def decode(
        self,
        encoded: EncodedAudio,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, int]:
        """Decode discrete Mimi audio tokens back into waveform.

        Args:
            encoded: EncodedAudio instance.

        Returns:
            Tuple of (waveform, sample_rate). waveform has shape (batch, channels, samples).
        """
        if hasattr(encoded, "codes"):
            codes = encoded.codes.to(device=self._target_device)
        elif isinstance(encoded, torch.Tensor):
            codes = encoded.to(device=self._target_device)
        else:
            raise TypeError(f"Expected EncodedAudio or torch.Tensor, got {type(encoded)}")

        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        elif codes.ndim != 3:
            raise ValueError(
                f"Expected codes of shape (batch, quantizers, frames) or (quantizers, frames), got {tuple(codes.shape)}"
            )

        with torch.no_grad():
            decoder_output = self.model.decode(codes, **kwargs)
            audio_values = decoder_output.audio_values

        return audio_values, self.sample_rate


def build_codec(
    backend: str | None = None,
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> AudioCodec:
    """Factory function to instantiate an audio codec by backend name or config file."""
    config: dict[str, Any] = {}
    if config_path is not None:
        cfg_file = Path(config_path)
        if not cfg_file.is_file():
            raise FileNotFoundError(f"Codec configuration file not found: {cfg_file}")
        with open(cfg_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if raw and "codec" in raw:
                config = raw["codec"]
            elif raw:
                config = raw

    target_backend = backend or config.get("backend", "mimi")

    if target_backend == "mimi":
        model_id = kwargs.get("model_id", config.get("model_id", "kyutai/mimi"))
        device = kwargs.get("device", config.get("device", "auto"))
        num_quantizers = kwargs.get("num_quantizers", config.get("num_quantizers", None))
        load_pretrained = kwargs.get("load_pretrained", True)
        return MimiCodec(
            model_id=model_id,
            device=device,
            num_quantizers=num_quantizers,
            load_pretrained=load_pretrained,
        )

    raise ValueError(f"Unknown codec backend: '{target_backend}'. Available backends: ['mimi']")
