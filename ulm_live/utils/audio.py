from pathlib import Path
import math
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch


def load_wav(path: str | Path) -> tuple[torch.Tensor, int]:
    """Load a WAV file and return a float32 tensor of shape (channels, samples) and sample rate."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    sample_rate, data = wavfile.read(str(file_path))

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    elif np.issubdtype(data.dtype, np.floating):
        audio = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported audio dtype: {data.dtype}")

    # Ensure shape is (channels, samples)
    if audio.ndim == 1:
        waveform = torch.from_numpy(audio).unsqueeze(0)
    elif audio.ndim == 2:
        waveform = torch.from_numpy(audio.T)
    else:
        raise ValueError(f"Expected 1D or 2D audio array, got {audio.ndim}D")

    return waveform, sample_rate


def save_wav(
    path: str | Path,
    waveform: torch.Tensor,
    sample_rate: int,
    encoding: str = "pcm_16",
) -> None:
    """Save a waveform tensor of shape (channels, samples) or (samples,) as a WAV file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    tensor = waveform.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(waveform.shape)}")

    # scipy expects (samples, channels)
    data = tensor.numpy().T

    if encoding == "pcm_16":
        clipped = np.clip(data, -1.0, 1.0)
        out_data = (clipped * 32767.0).astype(np.int16)
    elif encoding == "float32":
        out_data = data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}. Choose 'pcm_16' or 'float32'")

    wavfile.write(str(file_path), sample_rate, out_data)


def to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Convert multi-channel waveform to mono by averaging channels along the channel dimension.
    
    Supports shapes (channels, samples) or (batch, channels, samples).
    """
    if waveform.ndim == 1:
        return waveform.unsqueeze(0)
    if waveform.ndim == 2:
        # (channels, samples)
        if waveform.shape[0] == 1:
            return waveform
        return waveform.mean(dim=0, keepdim=True)
    if waveform.ndim == 3:
        # (batch, channels, samples)
        if waveform.shape[1] == 1:
            return waveform
        return waveform.mean(dim=1, keepdim=True)
    raise ValueError(f"Expected tensor of rank 1, 2, or 3, got rank {waveform.ndim}")


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resample waveform using polyphase filtering."""
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError(f"Sample rates must be positive, got orig_sr={orig_sr}, target_sr={target_sr}")
    if orig_sr == target_sr:
        return waveform

    device = waveform.device
    dtype = waveform.dtype

    gcd = math.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd

    arr = waveform.detach().cpu().numpy()
    resampled = signal.resample_poly(arr, up=up, down=down, axis=-1)
    resampled_tensor = torch.from_numpy(np.ascontiguousarray(resampled)).to(device=device, dtype=dtype)
    return resampled_tensor


def peak_normalize(waveform: torch.Tensor, target_peak: float = 0.95) -> torch.Tensor:
    """Scale waveform so that its maximum absolute amplitude equals target_peak."""
    if not (0.0 < target_peak <= 1.0):
        raise ValueError(f"target_peak must be in range (0.0, 1.0], got {target_peak}")

    max_val = torch.max(torch.abs(waveform))
    if max_val < 1e-7:
        return waveform
    return waveform * (target_peak / max_val)


def get_duration(waveform: torch.Tensor, sample_rate: int) -> float:
    """Calculate the duration of an audio tensor in seconds."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    num_samples = waveform.shape[-1]
    return float(num_samples) / float(sample_rate)
