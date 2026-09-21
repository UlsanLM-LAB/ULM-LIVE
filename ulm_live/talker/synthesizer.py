from dataclasses import dataclass
from pathlib import Path
import time
import torch

from ulm_live.codec import AudioCodec, EncodedAudio
from ulm_live.talker.generator import TalkerGenerationConfig
from ulm_live.talker.model import ULMTalker
from ulm_live.talker.data import remove_acoustic_delay
from ulm_live.thinker import ULMThinker
from ulm_live.utils.audio import get_duration, save_wav


@dataclass
class SpeechGenerationResult:
    """Container holding synthesized speech waveform, codec tokens, and performance metrics."""

    codec_tokens: torch.Tensor
    waveform: torch.Tensor
    sample_rate: int
    num_audio_frames: int
    duration: float
    timings: dict[str, float]
    rtf: float
    termination_reason: str = "MAX_FRAMES"
    max_stop_prob: float = 0.0

    def save(self, path: str | Path) -> None:
        """Save synthesized waveform to WAV file."""
        save_wav(path, self.waveform, self.sample_rate)


class SpeechSynthesizer:
    """End-to-end speech synthesis pipeline connecting Thinker, Talker, and Codec decoder."""

    def __init__(
        self,
        thinker: ULMThinker | None,
        talker: ULMTalker,
        codec: AudioCodec,
        speaker2id: dict[str, int] | None = None,
        dialect2id: dict[str, int] | None = None,
        device: str | torch.device = "auto",
    ) -> None:
        self.thinker = thinker
        self.talker = talker
        self.codec = codec

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.talker.to(self.device)
        self.talker.eval()

        self.codec.to(self.device)

        self.speaker2id = speaker2id or {"speaker_001": 0}
        self.dialect2id = dialect2id or {"ulsan": 0}

    def synthesize_from_hidden(
        self,
        semantic_hidden: torch.Tensor,
        speaker_id: int = 0,
        dialect_id: int = 0,
        generation_config: TalkerGenerationConfig | None = None,
    ) -> SpeechGenerationResult:
        """Synthesize speech directly from semantic hidden states (bypassing Thinker text tokenization)."""
        gen_cfg = generation_config or TalkerGenerationConfig()

        t_start = time.perf_counter()

        # 1. Talker Generation
        t_talker_0 = time.perf_counter()
        spk_tensor = torch.tensor([speaker_id], dtype=torch.long, device=self.device)
        dia_tensor = torch.tensor([dialect_id], dtype=torch.long, device=self.device)

        term_reason = "MAX_FRAMES"
        max_sp = 0.0
        with torch.no_grad():
            gen_out = self.talker.generate(
                semantic_hidden_states=semantic_hidden.to(self.device),
                speaker_ids=spk_tensor,
                dialect_ids=dia_tensor,
                generation_config=gen_cfg,
                frame_rate=self.codec.frame_rate,
            )
            if gen_cfg.return_details:
                codes, lengths, reasons, max_stop_probs = gen_out
                term_reason = reasons[0]
                max_sp = float(max_stop_probs[0].item())
            elif gen_cfg.return_lengths:
                codes, lengths = gen_out
            else:
                codes = gen_out
        t_talker = time.perf_counter() - t_talker_0

        # 2. Codec Decoding
        t_decode_0 = time.perf_counter()
        decode_codes = remove_acoustic_delay(
            codes, self.talker.config.acoustic_delay_frames
        )
        if torch.any(
            (decode_codes < 0) | (decode_codes >= self.talker.config.codebook_size)
        ):
            raise ValueError("BOS/PAD cannot be passed to Mimi decode")
        encoded = EncodedAudio(
            codes=decode_codes,
            sample_rate=self.codec.sample_rate,
            frame_rate=self.codec.frame_rate,
            metadata={"backend": self.codec.backend_name},
        )
        waveform, sr = self.codec.decode(encoded)
        t_decode = time.perf_counter() - t_decode_0

        # Ensure 2D (1, samples)
        if waveform.ndim == 3:
            waveform = waveform.squeeze(0)

        total_time = time.perf_counter() - t_start
        duration = get_duration(waveform, sr)
        rtf = total_time / max(duration, 1e-4)

        # Sanity validation
        if torch.isnan(waveform).any() or torch.isinf(waveform).any():
            raise RuntimeError("Decoded waveform contains NaN or Inf values.")
        if waveform.shape[-1] == 0:
            raise RuntimeError("Decoded waveform is empty.")

        # Squeeze batch dimension for result tokens -> (K, T)
        token_output = codes.squeeze(0).cpu()

        return SpeechGenerationResult(
            codec_tokens=token_output,
            waveform=waveform.detach().cpu(),
            sample_rate=sr,
            num_audio_frames=int(codes.shape[-1]),
            duration=round(duration, 3),
            timings={
                "thinker_time": 0.0,
                "talker_time": round(t_talker, 4),
                "decode_time": round(t_decode, 4),
                "total_time": round(total_time, 4),
            },
            rtf=round(rtf, 4),
            termination_reason=term_reason,
            max_stop_prob=round(max_sp, 4),
        )

    def synthesize(
        self,
        text: str,
        speaker: str = "speaker_001",
        dialect: str = "ulsan",
        generation_config: TalkerGenerationConfig | None = None,
    ) -> SpeechGenerationResult:
        """End-to-end synthesis from text prompt to speech waveform."""
        if self.thinker is None:
            raise RuntimeError("Thinker model is not loaded. Cannot tokenize text.")

        t_start = time.perf_counter()

        # 1. Thinker Semantic State Extraction
        t_thinker_0 = time.perf_counter()
        inputs = self.thinker.tokenize(text)
        with torch.no_grad():
            hidden = self.thinker.forward_hidden(
                inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )
        t_thinker = time.perf_counter() - t_thinker_0

        # 2. Lookup IDs
        spk_id = self.speaker2id.get(speaker, 0)
        dia_id = self.dialect2id.get(dialect, 0)

        # 3. Generate from hidden
        result = self.synthesize_from_hidden(
            semantic_hidden=hidden,
            speaker_id=spk_id,
            dialect_id=dia_id,
            generation_config=generation_config,
        )

        total_time = time.perf_counter() - t_start
        result.timings["thinker_time"] = round(t_thinker, 4)
        result.timings["total_time"] = round(total_time, 4)
        result.rtf = round(total_time / max(result.duration, 1e-4), 3)

        return result
