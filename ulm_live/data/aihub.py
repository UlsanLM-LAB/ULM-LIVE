import json
from pathlib import Path
from typing import Any
import yaml

from ulm_live.data.schema import SpeakerInfo, UtteranceInfo

AUDIO_EXTENSIONS = {".wav", ".flac", ".pcm", ".mp3"}


def load_dataset_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load dataset configuration from YAML file or return defaults."""
    default_config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "dataset.yaml"
    target_path = Path(config_path) if config_path is not None else default_config_path

    if target_path.is_file():
        with open(target_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if raw and "dataset" in raw:
                return raw["dataset"]
            return raw or {}
    return {}


def scan_aihub_directory(root_dir: str | Path) -> tuple[list[Path], list[Path]]:
    """Scan directory recursively for JSON metadata files and audio files."""
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    json_files = sorted(list(root.rglob("*.json")))
    audio_files = sorted(
        [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS]
    )
    return json_files, audio_files


class FieldMatcher:
    """Helper to flexibly extract values from dictionary using candidate field names."""

    def __init__(self, mappings: dict[str, list[str]] | None = None) -> None:
        self.mappings = mappings or {}

    def get_value(self, data: dict[str, Any], field_type: str, default: Any = None) -> Any:
        candidates = self.mappings.get(field_type, [])
        for key in candidates:
            if key in data and data[key] is not None:
                return data[key]
        return default

    def find_all_keys(self, data: dict[str, Any]) -> set[str]:
        keys = set(data.keys())
        for v in data.values():
            if isinstance(v, dict):
                keys.update(self.find_all_keys(v))
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                for item in v:
                    if isinstance(item, dict):
                        keys.update(self.find_all_keys(item))
        return keys


class RegionClassifier:
    """Classifies speaker region based on configurable priority and aliases."""

    def __init__(
        self,
        priority: list[str] | None = None,
        aliases: dict[str, list[str]] | None = None,
    ) -> None:
        self.priority = priority or [
            "principal_residence",
            "birthplace",
            "current_residence",
            "region",
        ]
        self.aliases = aliases or {
            "ulsan": ["울산", "울산광역시", "울산시", "ulsan"],
            "busan": ["부산", "부산광역시", "부산시", "busan"],
            "daegu": ["대구", "대구광역시", "대구시", "daegu"],
            "gyeongnam": ["경남", "경상남도", "gyeongnam"],
            "gyeongbuk": ["경북", "경상북도", "gyeongbuk"],
        }

    def match_alias(self, text: str) -> str | None:
        cleaned = text.strip().lower()
        for canonical, alias_list in self.aliases.items():
            for alias in alias_list:
                if alias.lower() in cleaned:
                    return canonical
        return None

    def determine_region(self, speaker: SpeakerInfo) -> str | None:
        # Check in prioritized order
        for field_name in self.priority:
            val = getattr(speaker, field_name, None)
            if not val and field_name in speaker.raw_metadata:
                val = speaker.raw_metadata[field_name]
            if val and isinstance(val, str) and val.strip():
                matched = self.match_alias(val)
                if matched:
                    return matched
                # Return raw string if not recognized in aliases
                return val.strip()
        return None

    def is_target_region(self, speaker: SpeakerInfo, target: str) -> bool:
        matched = self.determine_region(speaker)
        if matched is None:
            return False
        if matched.lower() == target.lower():
            return True
        # Check target aliases
        target_aliases = self.aliases.get(target.lower(), [target.lower()])
        return any(a.lower() in matched.lower() for a in target_aliases)


class AIHubParser:
    """Flexible parser for AI Hub dialect speech metadata."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_dataset_config()
        self.field_mappings = cfg.get("field_mappings", {})
        self.matcher = FieldMatcher(self.field_mappings)
        self.region_classifier = RegionClassifier(
            priority=cfg.get("region_priority"),
            aliases=cfg.get("region_aliases"),
        )

    def parse_file(
        self, json_path: Path
    ) -> tuple[dict[str, SpeakerInfo], list[UtteranceInfo], str | None]:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        speakers: dict[str, SpeakerInfo] = {}
        utterances: list[UtteranceInfo] = []

        # Find speakers container
        speaker_container = self.matcher.get_value(data, "speaker_container_fields")
        if speaker_container is not None:
            if isinstance(speaker_container, list):
                for s in speaker_container:
                    spk = self._parse_speaker_dict(s)
                    if spk:
                        speakers[spk.speaker_id] = spk
            elif isinstance(speaker_container, dict):
                spk = self._parse_speaker_dict(speaker_container)
                if spk:
                    speakers[spk.speaker_id] = spk

        # Find utterances container
        utterance_container = self.matcher.get_value(data, "utterance_container_fields")
        if utterance_container is not None and isinstance(utterance_container, list):
            for i, u in enumerate(utterance_container):
                utt = self._parse_utterance_dict(u, fallback_id=f"{json_path.stem}_{i:04d}")
                if utt:
                    utterances.append(utt)
        else:
            # Check if this JSON is a single-utterance document
            single_utt = self._parse_utterance_dict(data, fallback_id=json_path.stem)
            if single_utt and single_utt.text:
                utterances.append(single_utt)
                if not speakers and single_utt.speaker_id:
                    # Try to extract speaker info from top-level
                    spk = self._parse_speaker_dict(data)
                    if spk:
                        speakers[spk.speaker_id] = spk

        # Check audio file name in metadata
        audio_file_hint = self.matcher.get_value(data, "audio_file_fields")
        if audio_file_hint is None and "metadata" in data and isinstance(data["metadata"], dict):
            audio_file_hint = self.matcher.get_value(data["metadata"], "audio_file_fields")

        return speakers, utterances, audio_file_hint

    def _parse_speaker_dict(self, d: dict[str, Any]) -> SpeakerInfo | None:
        spk_id = self.matcher.get_value(d, "speaker_id_fields")
        if not spk_id:
            # Fallback check
            for k in ["id", "speaker_id", "speaker", "name"]:
                if k in d:
                    spk_id = str(d[k])
                    break
        if not spk_id:
            return None

        return SpeakerInfo(
            speaker_id=str(spk_id),
            birthplace=d.get("birthplace") or d.get("birth_place"),
            principal_residence=d.get("principal_residence") or d.get("residence"),
            current_residence=d.get("current_residence"),
            gender=d.get("gender") or d.get("sex"),
            age=str(d.get("age")) if d.get("age") is not None else None,
            raw_metadata=d,
        )

    def _parse_utterance_dict(self, d: dict[str, Any], fallback_id: str) -> UtteranceInfo | None:
        text = self.matcher.get_value(d, "text_fields")
        if not text:
            return None

        utt_id = d.get("id") or d.get("utterance_id") or fallback_id
        spk_id = self.matcher.get_value(d, "speaker_id_fields", default="unknown_speaker")
        standard_text = self.matcher.get_value(d, "standard_text_fields")

        start_val = self.matcher.get_value(d, "start_fields")
        end_val = self.matcher.get_value(d, "end_fields")

        start = float(start_val) if start_val is not None else None
        end = float(end_val) if end_val is not None else None
        audio_file = self.matcher.get_value(d, "audio_file_fields")

        return UtteranceInfo(
            utterance_id=str(utt_id),
            speaker_id=str(spk_id),
            text=str(text).strip(),
            standard_text=str(standard_text).strip() if standard_text else None,
            start=start,
            end=end,
            audio_file=str(audio_file) if audio_file else None,
            raw_metadata=d,
        )


def inspect_aihub_dataset(
    root_dir: str | Path,
    config: dict[str, Any] | None = None,
    max_scan_files: int = 200,
) -> dict[str, Any]:
    """Inspect AI Hub dataset non-destructively and return statistical summary."""
    json_files, audio_files = scan_aihub_directory(root_dir)
    parser = AIHubParser(config)

    detected_speaker_fields: set[str] = set()
    detected_region_fields: set[str] = set()
    detected_text_fields: set[str] = set()
    regions_found: dict[str, int] = {}
    total_speakers: set[str] = set()
    total_utterances = 0

    scan_targets = json_files[:max_scan_files] if max_scan_files else json_files

    for jf in scan_targets:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                raw = json.load(f)
            keys = parser.matcher.find_all_keys(raw)
            for k in keys:
                k_lower = k.lower()
                if "speaker" in k_lower or "spk" in k_lower:
                    detected_speaker_fields.add(k)
                if any(r in k_lower for r in ["residence", "birth", "region", "place", "location"]):
                    detected_region_fields.add(k)
                if any(t in k_lower for t in ["text", "form", "transcription", "script"]):
                    detected_text_fields.add(k)

            speakers, utterances, _ = parser.parse_file(jf)
            for spk in speakers.values():
                total_speakers.add(spk.speaker_id)
                reg = parser.region_classifier.determine_region(spk)
                if reg:
                    regions_found[reg] = regions_found.get(reg, 0) + 1
            total_utterances += len(utterances)
        except Exception:
            continue

    return {
        "num_json_files": len(json_files),
        "num_audio_files": len(audio_files),
        "scanned_json_files": len(scan_targets),
        "detected_speaker_fields": sorted(list(detected_speaker_fields)),
        "detected_region_fields": sorted(list(detected_region_fields)),
        "detected_text_fields": sorted(list(detected_text_fields)),
        "regions_found": regions_found,
        "total_speakers_scanned": len(total_speakers),
        "total_utterances_scanned": total_utterances,
    }
