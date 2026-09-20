from pathlib import Path
from typing import Any
import random
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
    state_dict = checkpoint.get("model_state_dict", {})
    if any(key.startswith(("output_heads.", "transformer.")) for key in state_dict):
        raise RuntimeError(
            "Legacy independent-head Talker checkpoints are incompatible with "
            "the Temporal+Depth architecture and must not be resumed."
        )

    raw_config = checkpoint.get("config", {})
    config = TalkerConfig(
        **{
            k: v
            for k, v in raw_config.items()
            if k in TalkerConfig.__dataclass_fields__
        }
    )

    model = ULMTalker(config)
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()

    metadata = {
        "speaker2id": checkpoint.get("speaker2id", {}),
        "dialect2id": checkpoint.get("dialect2id", {}),
        "codec_metadata": checkpoint.get("codec_metadata", {}),
        "step": checkpoint.get("step"),
    }
    return model, metadata


def capture_rng_state() -> dict[str, Any]:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    state["cuda"] = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str | Path,
    model: ULMTalker,
    optimizer,
    scheduler,
    scaler,
    *,
    global_step: int,
    epoch: int,
    micro_batch_position: int,
    accumulation_step: int,
    speaker2id: dict[str, int],
    dialect2id: dict[str, int],
    training_args: dict[str, Any],
    best_validation: dict[str, Any],
) -> Path:
    if accumulation_step != 0:
        raise ValueError(
            "checkpoints may only be saved at a gradient accumulation boundary"
        )
    payload = {
        "format_version": 2,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "global_step": global_step,
        "step": global_step,
        "epoch": epoch,
        "micro_batch_position": micro_batch_position,
        "accumulation_step": 0,
        "rng_state": capture_rng_state(),
        "speaker2id": speaker2id,
        "dialect2id": dialect2id,
        "config": model.config.to_dict(),
        "training_args": training_args,
        "best_validation": best_validation,
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(payload, temp)
    temp.replace(dest)
    return dest


def load_training_checkpoint(
    path: str | Path,
    model: ULMTalker,
    optimizer=None,
    scheduler=None,
    scaler=None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("accumulation_step", 0) != 0:
        raise ValueError("cannot resume unresolved gradient accumulation")
    if payload.get("config") != model.config.to_dict():
        raise ValueError("checkpoint Talker config does not match")
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if scheduler is not None and payload.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload
