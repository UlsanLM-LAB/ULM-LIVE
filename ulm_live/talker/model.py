from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class TalkerConfig:
    """Hyperparameters for ULMTalker model."""

    semantic_dim: int = 2048
    talker_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048

    num_quantizers: int = 32
    codebook_size: int = 2048
    pad_token_id: int = 2048

    speaker_embedding_dim: int = 128
    dialect_embedding_dim: int = 64
    num_speakers: int = 256
    num_dialects: int = 16
    conditioning_mode: str = "additive"

    freeze_thinker: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TalkerConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = raw.get("talker", raw) if raw else {}
        valid_keys = {k for k in cls.__dataclass_fields__}
        filtered = {k: v for k, v in cfg.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TalkerOutput:
    """Output container for ULMTalker forward pass."""

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class ULMTalker(nn.Module):
    """Autoregressive neural acoustic model predicting codec tokens conditioned on Thinker hidden states, speaker, and dialect."""

    def __init__(self, config: TalkerConfig | None = None) -> None:
        super().__init__()
        self.config = config or TalkerConfig()

        # 1. Semantic Projection
        self.semantic_proj = nn.Linear(self.config.semantic_dim, self.config.talker_dim)

        # 2. Speaker Conditioning
        self.speaker_embedding = nn.Embedding(
            self.config.num_speakers, self.config.speaker_embedding_dim
        )
        self.speaker_proj = nn.Linear(
            self.config.speaker_embedding_dim, self.config.talker_dim
        )

        # 3. Dialect Conditioning
        self.dialect_embedding = nn.Embedding(
            self.config.num_dialects, self.config.dialect_embedding_dim
        )
        self.dialect_proj = nn.Linear(
            self.config.dialect_embedding_dim, self.config.talker_dim
        )

        # 4. Audio Codec Token Embeddings (one embedding table per codebook quantizer)
        # Size is codebook_size + 1 to accommodate pad_token_id
        vocab_size = self.config.codebook_size + 1
        self.codebook_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_size, self.config.talker_dim, padding_idx=self.config.pad_token_id)
                for _ in range(self.config.num_quantizers)
            ]
        )

        # Positional Embedding
        self.pos_embedding = nn.Embedding(self.config.max_seq_len, self.config.talker_dim)

        # 5. Causal Transformer Backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.talker_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.num_layers,
            enable_nested_tensor=False,
        )
        self.ln_f = nn.LayerNorm(self.config.talker_dim)

        # 6. Multi-codebook Independent Output Heads
        self.output_heads = nn.ModuleList(
            [
                nn.Linear(self.config.talker_dim, self.config.codebook_size)
                for _ in range(self.config.num_quantizers)
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _build_causal_mask(
        self,
        num_semantic: int,
        num_audio: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create prefix-causal attention mask:
        
        - Semantic tokens (0..S-1) can attend bidirectionally to each other.
        - Audio tokens (S..S+T-1) can attend to all semantic tokens and past audio tokens.
        """
        total_len = num_semantic + num_audio
        mask = torch.full((total_len, total_len), float("-inf"), device=device)

        if num_semantic > 0:
            # Semantic prefix: bidirectional
            mask[:num_semantic, :num_semantic] = 0.0
            # Audio tokens attend to all semantic tokens
            mask[num_semantic:, :num_semantic] = 0.0

        if num_audio > 0:
            # Audio tokens attend causally to previous audio tokens
            audio_causal = torch.triu(
                torch.full((num_audio, num_audio), float("-inf"), device=device),
                diagonal=1,
            )
            mask[num_semantic:, num_semantic:] = audio_causal

        return mask

    def forward(
        self,
        semantic_hidden_states: torch.Tensor,
        audio_codes: torch.Tensor,
        speaker_ids: torch.Tensor,
        dialect_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> TalkerOutput:
        """Forward pass for ULMTalker.

        Args:
            semantic_hidden_states: (batch_size, num_semantic_tokens, semantic_dim)
            audio_codes: (batch_size, num_quantizers, num_audio_frames)
            speaker_ids: (batch_size,) integer speaker ids
            dialect_ids: (batch_size,) integer dialect ids
            targets: Optional (batch_size, num_quantizers, num_audio_frames) for loss calculation

        Returns:
            TalkerOutput containing logits and optional scalar loss.
        """
        device = audio_codes.device
        B = audio_codes.shape[0]
        K = audio_codes.shape[1]
        T = audio_codes.shape[2]
        S = semantic_hidden_states.shape[1] if semantic_hidden_states.numel() > 0 else 0

        # 1. Validate inputs
        if K != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} quantizers, got {K}"
            )
        if torch.any(speaker_ids < 0) or torch.any(speaker_ids >= self.config.num_speakers):
            raise IndexError(
                f"speaker_ids out of range [0, {self.config.num_speakers})"
            )
        if torch.any(dialect_ids < 0) or torch.any(dialect_ids >= self.config.num_dialects):
            raise IndexError(
                f"dialect_ids out of range [0, {self.config.num_dialects})"
            )

        # 2. Project semantic hidden states
        if S > 0:
            sem_tokens = self.semantic_proj(semantic_hidden_states.to(device))
        else:
            sem_tokens = torch.empty((B, 0, self.config.talker_dim), device=device)

        # 3. Compute Speaker and Dialect Conditioning Vector
        spk_emb = self.speaker_proj(self.speaker_embedding(speaker_ids.to(device))).unsqueeze(1)
        dia_emb = self.dialect_proj(self.dialect_embedding(dialect_ids.to(device))).unsqueeze(1)
        cond = spk_emb + dia_emb # shape: (B, 1, talker_dim)

        # 4. Compute Audio Token Embeddings (sum across all K quantizers)
        audio_emb = torch.zeros((B, T, self.config.talker_dim), device=device)
        for k in range(K):
            audio_emb = audio_emb + self.codebook_embeddings[k](audio_codes[:, k, :])

        # Inject conditioning additively into audio tokens
        audio_emb = audio_emb + cond

        # 5. Concatenate [semantic_prefix, audio_tokens]
        seq = torch.cat([sem_tokens, audio_emb], dim=1) # shape: (B, S + T, talker_dim)
        total_len = seq.shape[1]
        if total_len > self.config.max_seq_len:
            raise ValueError(
                f"Total sequence length {total_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        # Add positional embedding
        positions = torch.arange(total_len, device=device)
        seq = seq + self.pos_embedding(positions).unsqueeze(0)

        # 6. Apply Causal Mask & Transformer Backbone
        mask = self._build_causal_mask(S, T, device)
        hidden = self.transformer(seq, mask=mask, is_causal=False)
        hidden = self.ln_f(hidden)

        # 7. Extract audio token positions and predict codebook logits
        hidden_audio = hidden[:, S:, :] # shape: (B, T, talker_dim)
        logits_list = [head(hidden_audio) for head in self.output_heads]
        # Stack over codebooks -> shape: (B, K, T, codebook_size)
        logits = torch.stack(logits_list, dim=1)

        # 8. Compute Loss if targets provided
        loss = None
        if targets is not None:
            t = targets.to(device)
            loss_sum = 0.0
            for k in range(K):
                loss_k = F.cross_entropy(
                    logits[:, k, :, :].reshape(-1, self.config.codebook_size),
                    t[:, k, :].reshape(-1),
                    ignore_index=self.config.pad_token_id,
                )
                loss_sum = loss_sum + loss_k
            loss = loss_sum / float(K)

        return TalkerOutput(logits=logits, loss=loss)
