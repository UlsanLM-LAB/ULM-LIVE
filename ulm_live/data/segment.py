from pathlib import Path
import torch

from ulm_live.codec import AudioCodec
from ulm_live.data.filters import QualityFilter
from ulm_live.data.schema import DatasetItem, UtteranceInfo
from ulm_live.utils.audio import (
    get_duration,
    load_wav,
    peak_normalize,
    resample_audio,
    save_wav,
    to_mono,
)


def match_audio_file(
    hint_name: str | None,
    json_path: Path,
    audio_files_by_stem: dict[str, Path],
    audio_files_by_name: dict[str, Path],
) -> Path | None:
    """Find matching audio file using hint or json stem."""
    if hint_name:
        p_hint = Path(hint_name)
        if hint_name in audio_files_by_name:
            return audio_files_by_name[hint_name]
        if p_hint.name in audio_files_by_name:
            return audio_files_by_name[p_hint.name]
        if p_hint.stem in audio_files_by_stem:
            return audio_files_by_stem[p_hint.stem]

    # Match by JSON stem
    if json_path.stem in audio_files_by_stem:
        return audio_files_by_stem[json_path.stem]

    return None


def slice_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    start: float | None = None,
    end: float | None = None,
) -> torch.Tensor:
    """Slice waveform tensor along sample dimension based on start/end timestamps."""
    total_samples = waveform.shape[-1]
    if start is None and end is None:
        return waveform

    start_sec = max(0.0, start) if start is not None else 0.0
    end_sec = end if end is not None else float(total_samples) / float(sample_rate)

    start_idx = int(start_sec * sample_rate)
    end_idx = int(end_sec * sample_rate)

    start_idx = max(0, min(start_idx, total_samples))
    end_idx = max(start_idx, min(end_idx, total_samples))

    if end_idx <= start_idx:
        return waveform[:, start_idx:start_idx]

    return waveform[:, start_idx:end_idx]


class AudioSegmenter:
    """Segments, converts, normalizes audio for dataset generation, with optional codec encoding."""

    def __init__(
        self,
        output_dir: str | Path,
        target_sample_rate: int = 24000,
        quality_filter: QualityFilter | None = None,
        codec: AudioCodec | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.codec_dir = self.output_dir / "codec"
        self.target_sample_rate = target_sample_rate
        self.quality_filter = quality_filter or QualityFilter()
        self.codec = codec

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        if self.codec is not None:
            self.codec_dir.mkdir(parents=True, exist_ok=True)

    def process_utterance(
        self,
        item_id: str,
        utterance: UtteranceInfo,
        source_audio_path: Path,
        dialect: str = "ulsan",
    ) -> tuple[DatasetItem | None, str | None]:
        """Process a single utterance: slice, filter, resample, save, and optionally encode.

        Returns:
            (DatasetItem, None) on success, or (None, rejection_reason) on failure/skip.
        """
        # 1. Text validation & cleaning
        text_valid, cleaned_text, text_reason = self.quality_filter.validate_text(
            utterance.text
        )
        if not text_valid:
            return None, text_reason

        # 2. Load audio
        try:
            full_waveform, orig_sr = load_wav(source_audio_path)
        except Exception as err:
            return None, f"load_failed: {err}"

        # 3. Slice audio segment
        segment = slice_waveform(full_waveform, orig_sr, utterance.start, utterance.end)
        if segment.shape[-1] == 0:
            return None, "empty_audio_segment"

        # 4. Validate audio segment quality before heavy processing
        audio_valid, orig_dur, silence_ratio, audio_reason = (
            self.quality_filter.validate_audio(segment, orig_sr)
        )
        if not audio_valid:
            return None, audio_reason

        # 5. Convert to mono, resample to target sample rate, peak normalize
        mono_segment = to_mono(segment)
        resampled_segment = resample_audio(
            mono_segment, orig_sr, self.target_sample_rate
        )
        normalized_segment = peak_normalize(resampled_segment, target_peak=0.95)

        # 6. Save processed WAV
        out_wav_filename = f"{item_id}.wav"
        out_wav_path = self.audio_dir / out_wav_filename
        rel_wav_path = f"audio/{out_wav_filename}"

        try:
            save_wav(
                out_wav_path,
                normalized_segment,
                self.target_sample_rate,
                encoding="pcm_16",
            )
        except Exception as err:
            return None, f"save_failed: {err}"

        final_duration = get_duration(normalized_segment, self.target_sample_rate)

        # 7. Optional codec encoding
        rel_codec_path: str | None = None
        if self.codec is not None:
            out_codec_filename = f"{item_id}.pt"
            out_codec_path = self.codec_dir / out_codec_filename
            try:
                encoded = self.codec.encode(
                    normalized_segment, sample_rate=self.target_sample_rate
                )
                # Store discrete RVQ tokens tensor: shape (batch, quantizers, frames)
                torch.save(encoded.codes.detach().cpu(), out_codec_path)
                rel_codec_path = f"codec/{out_codec_filename}"
            except Exception as err:
                return None, f"codec_encode_failed: {err}"

        item = DatasetItem(
            id=item_id,
            audio_path=rel_wav_path,
            text=cleaned_text,
            speaker_id=utterance.speaker_id,
            dialect=dialect,
            sample_rate=self.target_sample_rate,
            duration=final_duration,
            source="aihub",
            standard_text=utterance.standard_text,
            codec_path=rel_codec_path,
            utterance_id=utterance.utterance_id,
            session_id=source_audio_path.stem,
            source_audio_id=source_audio_path.stem,
            metadata={
                "orig_sample_rate": orig_sr,
                "orig_duration": round(orig_dur, 3),
                "silence_ratio": round(silence_ratio, 3),
            },
        )
        return item, None
