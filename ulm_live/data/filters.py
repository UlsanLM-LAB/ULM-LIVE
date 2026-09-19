import math
import re
from typing import Any
import torch

from ulm_live.data.schema import UtteranceInfo

# Common noise / non-speech markup patterns in AI Hub dialect scripts
AIHUB_TAG_PATTERN = re.compile(r"\([^\)]*\)|\[[^\]]*\]|\{[^\}]*\}|<[^>]*>|[nbuslo]/\??")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")


def clean_transcript(text: str) -> str:
    """Clean AI Hub annotation artifacts, non-speech markers, and collapse whitespace."""
    if not text:
        return ""
    # Remove markup like (()), [[]], o/, b/, l/
    cleaned = AIHUB_TAG_PATTERN.sub(" ", text)
    # Remove common punctuation artifacts but keep Korean letters, numbers, and basic punctuation
    cleaned = re.sub(r"[^\w\s가-힣0-9\.\,\?\!\'\"]", " ", cleaned)
    cleaned = MULTIPLE_SPACES_PATTERN.sub(" ", cleaned).strip()
    return cleaned


def compute_silence_ratio(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_length_ms: float = 25.0,
    frame_shift_ms: float = 10.0,
    threshold_db: float = -40.0,
) -> float:
    """Compute the ratio of non-active (silence) frames in the audio waveform."""
    if waveform.numel() == 0:
        return 1.0

    wav = waveform.detach().cpu().float()
    if wav.ndim > 1:
        wav = wav.mean(dim=0) # convert to 1D for energy calculation

    frame_len = int(sample_rate * (frame_length_ms / 1000.0))
    frame_shift = int(sample_rate * (frame_shift_ms / 1000.0))

    if frame_len <= 0 or frame_shift <= 0 or wav.shape[0] < frame_len:
        return 0.0

    # Unfold into frames
    frames = wav.unfold(0, frame_len, frame_shift)
    # Root Mean Square per frame
    rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-12)
    max_rms = torch.max(rms).item()

    if max_rms < 1e-7:
        return 1.0 # Completely silent

    # Relative decibels relative to peak frame RMS
    db = 20.0 * torch.log10(rms / (max_rms + 1e-12))
    silent_frames = torch.sum(db < threshold_db).item()
    total_frames = frames.shape[0]

    return float(silent_frames) / float(total_frames)


class QualityFilter:
    """Configurable quality gate for speech utterances and audio waveforms."""

    def __init__(
        self,
        min_duration: float = 1.0,
        max_duration: float = 20.0,
        min_sample_rate: int = 16000,
        max_silence_ratio: float = 0.6,
        min_text_len: int = 2,
        max_text_len: int = 200,
        clean_text_enabled: bool = True,
    ) -> None:
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_sample_rate = min_sample_rate
        self.max_silence_ratio = max_silence_ratio
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.clean_text_enabled = clean_text_enabled

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None) -> "QualityFilter":
        cfg = config.get("filters", {}) if config else {}
        return cls(
            min_duration=float(cfg.get("min_duration", 1.0)),
            max_duration=float(cfg.get("max_duration", 20.0)),
            min_sample_rate=int(cfg.get("min_sample_rate", 16000)),
            max_silence_ratio=float(cfg.get("max_silence_ratio", 0.6)),
            min_text_len=int(cfg.get("min_text_len", 2)),
            max_text_len=int(cfg.get("max_text_len", 200)),
            clean_text_enabled=bool(cfg.get("clean_text", True)),
        )

    def validate_text(self, raw_text: str) -> tuple[bool, str, str | None]:
        """Validate and clean transcript. Returns (is_valid, cleaned_text, rejection_reason)."""
        cleaned = clean_transcript(raw_text) if self.clean_text_enabled else raw_text.strip()
        if len(cleaned) < self.min_text_len:
            return False, cleaned, f"text_too_short (<{self.min_text_len})"
        if len(cleaned) > self.max_text_len:
            return False, cleaned, f"text_too_long (>{self.max_text_len})"
        return True, cleaned, None

    def validate_duration(self, duration: float) -> tuple[bool, str | None]:
        """Validate duration. Returns (is_valid, rejection_reason)."""
        if duration < self.min_duration:
            return False, f"duration_too_short ({duration:.2f}s < {self.min_duration}s)"
        if duration > self.max_duration:
            return False, f"duration_too_long ({duration:.2f}s > {self.max_duration}s)"
        return True, None

    def validate_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> tuple[bool, float, float, str | None]:
        """Validate waveform properties.

        Returns:
            (is_valid, duration, silence_ratio, rejection_reason)
        """
        if sample_rate < self.min_sample_rate:
            return False, 0.0, 1.0, f"sample_rate_too_low ({sample_rate} < {self.min_sample_rate})"

        num_samples = waveform.shape[-1]
        duration = float(num_samples) / float(sample_rate)

        dur_valid, dur_reason = self.validate_duration(duration)
        if not dur_valid:
            return False, duration, 0.0, dur_reason

        silence_ratio = compute_silence_ratio(waveform, sample_rate)
        if silence_ratio > self.max_silence_ratio:
            return (
                False,
                duration,
                silence_ratio,
                f"too_much_silence ({silence_ratio:.2f} > {self.max_silence_ratio})",
            )

        return True, duration, silence_ratio, None
