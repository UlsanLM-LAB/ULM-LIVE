from pathlib import Path
from typing import Any
import torch

from ulm_live.talker.model import TalkerConfig, ULMTalker


def save_talker_checkpoint(
    output_path: str | Path,
    model: ULMTalker,
    speaker2id: dict[str, int] | None = None,
    dialect2id: dict[str, int] | None = None,
    codec_metadata: dict[str, Any] | None = None,
    step: int | None = None,
) -> Path:
    """Save Talker checkpoint including model weights, config, and vocabulary mappings."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "speaker2id": speaker2id or {},
        "dialect2id": dialect2id or {},
        "codec_metadata": codec_metadata or {},
        "step": step,
    }
    torch.save(checkpoint, path)
    return path


def load_talker_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ULMTalker, dict[str, Any]]:
    """Load Talker model and associated metadata from checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    dev = torch.device(device) if isinstance(device, str) else device
    checkpoint = torch.load(path, map_location=dev, weights_only=False)

    raw_config = checkpoint.get("config", {})
    config = TalkerConfig(**{k: v for k, v in raw_config.items() if k in TalkerConfig.__dataclass_fields__})

    model = ULMTalker(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(dev)
    model.eval()

    metadata = {
        "speaker2id": checkpoint.get("speaker2id", {}),
        "dialect2id": checkpoint.get("dialect2id", {}),
        "codec_metadata": checkpoint.get("codec_metadata", {}),
        "step": checkpoint.get("step"),
    }
    return model, metadata
