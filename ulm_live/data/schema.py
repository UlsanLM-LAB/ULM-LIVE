from dataclasses import dataclass, field
from typing import Any


@dataclass
class SpeakerInfo:
    """Speaker demographic and geographic metadata."""

    speaker_id: str
    birthplace: str | None = None
    principal_residence: str | None = None
    current_residence: str | None = None
    gender: str | None = None
    age: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UtteranceInfo:
    """Raw utterance timing, script, and speaker binding."""

    utterance_id: str
    speaker_id: str
    text: str
    standard_text: str | None = None
    start: float | None = None
    end: float | None = None
    audio_file: str | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float | None:
        if self.start is not None and self.end is not None:
            return max(0.0, self.end - self.start)
        return None


@dataclass
class DatasetItem:
    """Cleaned sample item for ULM-Live Spoken Language Model training."""

    id: str
    audio_path: str
    text: str
    speaker_id: str
    dialect: str
    sample_rate: int
    duration: float
    source: str = "aihub"
    standard_text: str | None = None
    codec_path: str | None = None
    utterance_id: str | None = None
    session_id: str | None = None
    source_audio_id: str | None = None
    semantic_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary suitable for JSONL manifest serialization."""
        data = {
            "id": self.id,
            "audio_path": self.audio_path,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "dialect": self.dialect,
            "sample_rate": self.sample_rate,
            "duration": round(self.duration, 3),
            "source": self.source,
        }
        if self.standard_text is not None:
            data["standard_text"] = self.standard_text
        if self.codec_path is not None:
            data["codec_path"] = self.codec_path
        for name in ("utterance_id", "session_id", "source_audio_id", "semantic_path"):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass
class DatasetSummary:
    """Overall dataset statistics."""

    num_samples: int
    num_speakers: int
    total_hours: float
    mean_duration: float
    speaker_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_samples": self.num_samples,
            "num_speakers": self.num_speakers,
            "total_hours": round(self.total_hours, 3),
            "mean_duration": round(self.mean_duration, 3),
            "speaker_stats": self.speaker_stats,
        }
