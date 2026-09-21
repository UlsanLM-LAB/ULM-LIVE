"""Talker with cached temporal and within-frame RVQ depth modeling."""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F
import yaml


@dataclass
class TalkerConfig:
    semantic_dim: int = 2048
    talker_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 2048
    depth_dim: int = 256
    depth_layers: int = 2
    depth_heads: int = 8
    depth_feedforward: int = 1024
    depth_dropout: float = 0.1
    num_quantizers: int = 32
    codebook_size: int = 2048
    pad_token_id: int = 2048
    bos_token_id: int = 2049
    speaker_embedding_dim: int = 128
    dialect_embedding_dim: int = 64
    num_speakers: int = 256
    num_dialects: int = 16
    conditioning_mode: str = "additive"
    stop_loss_weight: float = 1.0
    stop_pos_weight: float = 1.0
    stop_threshold: float = 0.5
    min_audio_frames: int = 1
    max_audio_frames: int = 750
    acoustic_delay_frames: int = 0
    freeze_thinker: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_eval_batches: int | None = None
    best_metric: str = "total_loss"

    def __post_init__(self):
        if self.num_quantizers not in (8, 16, 32):
            raise ValueError("num_quantizers must be 8, 16, or 32")
        if (
            self.pad_token_id != self.codebook_size
            or self.bos_token_id != self.codebook_size + 1
        ):
            raise ValueError("PAD/BOS IDs must be codebook_size/codebook_size+1")
        if self.acoustic_delay_frames not in (0, 1, 2):
            raise ValueError("acoustic_delay_frames must be 0, 1, or 2")

    @classmethod
    def from_yaml(cls, path):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
            root = raw.get("talker", raw)
        flat = {}
        for k, v in root.items():
            flat.update(v) if isinstance(v, dict) else flat.update({k: v})
        return cls(**{k: v for k, v in flat.items() if k in cls.__dataclass_fields__})

    def to_dict(self):
        return asdict(self)


@dataclass
class TalkerOutput:
    logits: torch.Tensor
    stop_logits: torch.Tensor
    loss: torch.Tensor | None = None
    codec_loss: torch.Tensor | None = None
    stop_loss: torch.Tensor | None = None
    codebook_losses: tuple = ()


@dataclass
class GenerationState:
    layer_kv: list
    key_padding_mask: torch.Tensor
    next_position: int
    condition: torch.Tensor


class Attention(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        assert d % h == 0
        self.h = h
        self.hd = d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = drop

    def split(self, x):
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.hd).transpose(1, 2)

    def forward(self, x, mask=None, padding=None, past=None, cache=False):
        q, k, v = map(self.split, self.qkv(x).chunk(3, -1))
        if past is not None:
            k = torch.cat((past[0], k), 2)
            v = torch.cat((past[1], v), 2)
        a = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        if mask is not None:
            a = a.masked_fill(~mask[None, None], -torch.inf)
        if padding is not None:
            a = a.masked_fill(padding[:, None, None], -torch.inf)
        a = F.softmax(a, -1, dtype=torch.float32).to(a.dtype)
        a = F.dropout(a, self.drop, self.training)
        y = (a @ v).transpose(1, 2).contiguous().view(x.shape)
        return self.out(y), (k, v) if cache else None


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c.talker_dim
        self.n1 = nn.LayerNorm(d)
        self.a = Attention(d, c.num_heads, c.dropout)
        self.n2 = nn.LayerNorm(d)
        self.drop = c.dropout
        self.ff = nn.Sequential(
            nn.Linear(d, c.dim_feedforward),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.dim_feedforward, d),
        )

    def forward(self, x, **kw):
        y, kv = self.a(self.n1(x), **kw)
        x = x + F.dropout(y, self.drop, self.training)
        return x + F.dropout(self.ff(self.n2(x)), self.drop, self.training), kv


class ULMTalker(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TalkerConfig()
        c = self.config
        self.semantic_proj = nn.Linear(c.semantic_dim, c.talker_dim)
        self.speaker_embedding = nn.Embedding(c.num_speakers, c.speaker_embedding_dim)
        self.speaker_proj = nn.Linear(c.speaker_embedding_dim, c.talker_dim)
        self.dialect_embedding = nn.Embedding(c.num_dialects, c.dialect_embedding_dim)
        self.dialect_proj = nn.Linear(c.dialect_embedding_dim, c.talker_dim)
        self.codebook_embeddings = nn.ModuleList(
            nn.Embedding(c.codebook_size + 2, c.talker_dim, padding_idx=c.pad_token_id)
            for _ in range(c.num_quantizers)
        )
        self.pos_embedding = nn.Embedding(c.max_seq_len, c.talker_dim)
        self.temporal_transformer = nn.ModuleList(Block(c) for _ in range(c.num_layers))
        self.temporal_norm = nn.LayerNorm(c.talker_dim)
        self.frame_to_depth = nn.Linear(c.talker_dim, c.depth_dim)
        self.depth_token_embedding = nn.Embedding(
            c.codebook_size + 1, c.depth_dim, padding_idx=c.pad_token_id
        )
        self.depth_bos = nn.Parameter(torch.zeros(c.depth_dim))
        self.depth_position = nn.Embedding(c.num_quantizers, c.depth_dim)
        dl = nn.TransformerEncoderLayer(
            c.depth_dim,
            c.depth_heads,
            c.depth_feedforward,
            c.depth_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.depth_transformer = nn.TransformerEncoder(
            dl, c.depth_layers, enable_nested_tensor=False
        )
        self.depth_norm = nn.LayerNorm(c.depth_dim)
        self.codec_output = nn.Linear(c.depth_dim, c.codebook_size)
        self.stop_head = nn.Linear(c.talker_dim, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            nn.init.zeros_(m.bias) if m.bias is not None else None
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].zero_()

    def _ids(self, s, d):
        if torch.any((s < 0) | (s >= self.config.num_speakers)):
            raise IndexError("speaker_ids out of range")
        if torch.any((d < 0) | (d >= self.config.num_dialects)):
            raise IndexError("dialect_ids out of range")

    def _codes(self, x, inputs):
        if x.ndim != 3 or x.shape[1] != self.config.num_quantizers or x.shape[2] == 0:
            raise ValueError(
                f"invalid codec tensor shape/quantizers; expected [B,{self.config.num_quantizers},T]"
            )
        hi = self.config.bos_token_id if inputs else self.config.pad_token_id
        if torch.any((x < 0) | (x > hi)) or (
            not inputs and torch.any(x == self.config.bos_token_id)
        ):
            raise ValueError("invalid codec token")

    def condition(self, s, d):
        self._ids(s, d)
        return self.speaker_proj(self.speaker_embedding(s)) + self.dialect_proj(
            self.dialect_embedding(d)
        )

    def audio_embed(self, x, cond):
        return (
            sum(e(x[:, k]) for k, e in enumerate(self.codebook_embeddings))
            + cond[:, None]
        )

    def project_semantic(self, sem):
        # Preserve BF16/FP16 while autocast is active; only reconcile dtypes for
        # non-autocast CPU/inference calls where Linear requires an exact match.
        autocast = torch.is_autocast_enabled() or (
            sem.device.type == "cpu" and torch.is_autocast_enabled("cpu")
        )
        return self.semantic_proj(
            sem
            if autocast or sem.dtype == self.semantic_proj.weight.dtype
            else sem.to(self.semantic_proj.weight.dtype)
        )

    def temporal(self, sem, audio, s, d, sem_mask=None, audio_mask=None):
        b, _, t = audio.shape
        sl = sem.shape[1]
        x = torch.cat(
            (self.project_semantic(sem), self.audio_embed(audio, self.condition(s, d))),
            1,
        )
        if x.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        x = x + self.pos_embedding(torch.arange(x.shape[1], device=x.device))[None]
        sv = (
            sem_mask.bool()
            if sem_mask is not None
            else torch.ones(b, sl, dtype=torch.bool, device=x.device)
        )
        if not sv.any(dim=1).all():
            raise ValueError("each sample needs at least one valid semantic token")
        av = (
            audio_mask.bool()
            if audio_mask is not None
            else torch.ones(b, t, dtype=torch.bool, device=x.device)
        )
        padding = ~torch.cat((sv, av), 1)
        mask = torch.zeros(sl + t, sl + t, dtype=torch.bool, device=x.device)
        mask[:sl, :sl] = True
        mask[sl:, :sl] = True
        mask[sl:, sl:] = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()
        for block in self.temporal_transformer:
            x, _ = block(x, mask=mask, padding=padding)
        return self.temporal_norm(x[:, sl:])

    def depth_logits(self, frames, target):
        b, t, _ = frames.shape
        k = self.config.num_quantizers
        prev = torch.empty(
            b, t, k, self.config.depth_dim, device=frames.device, dtype=frames.dtype
        )
        prev[:, :, 0] = self.depth_bos
        if k > 1:
            prev[:, :, 1:] = self.depth_token_embedding(
                target[:, :-1].transpose(1, 2).clamp(0, self.config.pad_token_id)
            ).to(frames.dtype)
        x = (
            self.frame_to_depth(frames)[:, :, None]
            + prev
            + self.depth_position.weight[None, None]
        ).reshape(b * t, k, -1)
        mask = torch.ones(k, k, dtype=torch.bool, device=x.device).triu(1)
        return (
            self.codec_output(self.depth_norm(self.depth_transformer(x, mask=mask)))
            .view(b, t, k, -1)
            .transpose(1, 2)
        )

    def predict_frame(self, frame, sampling_config=None, return_logits=False):
        chosen = []
        all_logits = []
        for q in range(self.config.num_quantizers):
            prior = [self.depth_bos.expand(frame.shape[0], -1)] + [
                self.depth_token_embedding(v) for v in chosen
            ]
            x = (
                torch.stack(prior, 1).to(frame.dtype)
                + self.frame_to_depth(frame)[:, None]
                + self.depth_position.weight[: q + 1]
            )
            mask = torch.ones(q + 1, q + 1, dtype=torch.bool, device=x.device).triu(1)
            logits = self.codec_output(
                self.depth_norm(self.depth_transformer(x, mask=mask)[:, -1])
            )
            all_logits.append(logits)
            if sampling_config is None:
                token = logits.argmax(-1)
            else:
                from .generator import sample_next_tokens

                token = sample_next_tokens(logits[:, None], sampling_config)[:, 0, 0]
            chosen.append(token)
        tokens = torch.stack(chosen, 1)
        return (tokens, torch.stack(all_logits, 1)) if return_logits else tokens

    def forward(
        self,
        semantic_hidden_states,
        audio_codes,
        speaker_ids,
        dialect_ids,
        targets=None,
        semantic_attention_mask=None,
        audio_attention_mask=None,
        stop_targets=None,
    ):
        self._codes(audio_codes, True)
        frames = self.temporal(
            semantic_hidden_states,
            audio_codes,
            speaker_ids,
            dialect_ids,
            semantic_attention_mask,
            audio_attention_mask,
        )
        stop_logits = self.stop_head(frames).squeeze(-1)
        if targets is None:
            _, logits = self.predict_frame(
                frames.reshape(-1, frames.shape[-1]), return_logits=True
            )
            logits = logits.view(
                frames.shape[0], frames.shape[1], self.config.num_quantizers, -1
            ).transpose(1, 2)
            return TalkerOutput(logits, stop_logits)
        self._codes(targets, False)
        logits = self.depth_logits(frames, targets)
        ls = []
        for q in range(self.config.num_quantizers):
            valid = targets[:, q] != self.config.pad_token_id
            ls.append(
                F.cross_entropy(logits[:, q][valid], targets[:, q][valid])
                if valid.any()
                else logits[:, q].sum() * 0
            )
        ls = tuple(ls)
        codec = torch.stack(ls).mean()
        stop = None
        if stop_targets is not None:
            valid = stop_targets >= 0
            if (
                stop_targets.shape != stop_logits.shape
                or not valid.any()
                or torch.any(stop_targets[valid] > 1)
            ):
                raise ValueError("invalid stop targets")
            pos_weight = (
                torch.tensor(
                    [self.config.stop_pos_weight],
                    device=stop_logits.device,
                    dtype=stop_logits.dtype,
                )
                if self.config.stop_pos_weight != 1.0
                else None
            )
            stop = F.binary_cross_entropy_with_logits(
                stop_logits[valid],
                stop_targets[valid].to(stop_logits.dtype),
                pos_weight=pos_weight,
            )
        loss = codec + (
            self.config.stop_loss_weight * stop
            if stop is not None
            else stop_logits.sum() * 0
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Talker loss is NaN or Inf")
        return TalkerOutput(logits, stop_logits, loss, codec, stop, ls)

    def init_generation_state(self, sem, s, d, semantic_attention_mask=None):
        b, sl, _ = sem.shape
        if sl <= 0 or sl >= self.config.max_seq_len:
            raise ValueError("invalid semantic prefix")
        x = (
            self.project_semantic(sem)
            + self.pos_embedding(torch.arange(sl, device=sem.device))[None]
        )
        padding = ~(
            semantic_attention_mask.bool()
            if semantic_attention_mask is not None
            else torch.ones(b, sl, dtype=torch.bool, device=x.device)
        )
        if padding.all(dim=1).any():
            raise ValueError("each sample needs at least one valid semantic token")
        caches = []
        mask = torch.ones(sl, sl, dtype=torch.bool, device=x.device)
        for block in self.temporal_transformer:
            x, kv = block(x, mask=mask, padding=padding, cache=True)
            caches.append(kv)
        return GenerationState(caches, padding, sl, self.condition(s, d))

    def step(self, state, input_frame, sampling_config=None):
        self._codes(input_frame[:, :, None], True)
        if state.next_position >= self.config.max_seq_len:
            raise ValueError("generation exceeds max_seq_len")
        x = (
            self.audio_embed(input_frame[:, :, None], state.condition)
            + self.pos_embedding.weight[state.next_position][None, None]
        )
        padding = torch.cat(
            (
                state.key_padding_mask,
                torch.zeros(x.shape[0], 1, dtype=torch.bool, device=x.device),
            ),
            1,
        )
        caches = []
        for block, past in zip(self.temporal_transformer, state.layer_kv):
            x, kv = block(x, padding=padding, past=past, cache=True)
            caches.append(kv)
        state.layer_kv = caches
        state.key_padding_mask = padding
        state.next_position += 1
        frame = self.temporal_norm(x[:, 0])
        return self.predict_frame(frame, sampling_config), torch.sigmoid(
            self.stop_head(frame).squeeze(-1)
        )

    def parameter_breakdown(self):
        groups = {
            "semantic_projection": [self.semantic_proj],
            "speaker_dialect_conditioning": [
                self.speaker_embedding,
                self.speaker_proj,
                self.dialect_embedding,
                self.dialect_proj,
            ],
            "temporal": [
                self.temporal_transformer,
                self.temporal_norm,
                self.pos_embedding,
            ],
            "depth": [
                self.frame_to_depth,
                self.depth_transformer,
                self.depth_norm,
                self.depth_position,
            ],
            "codec_embeddings": [self.codebook_embeddings, self.depth_token_embedding],
            "output_heads": [self.codec_output],
            "stop_head": [self.stop_head],
        }
        r = {
            n: sum(p.numel() for m in ms for p in m.parameters())
            for n, ms in groups.items()
        }
        r["other"] = self.depth_bos.numel()
        r["total"] = sum(p.numel() for p in self.parameters())
        return r

    def generate(
        self,
        semantic_hidden_states,
        speaker_ids,
        dialect_ids,
        generation_config=None,
        initial_audio_codes=None,
        frame_rate=12.5,
        semantic_attention_mask=None,
    ):
        from .generator import generate_codec_tokens

        return generate_codec_tokens(
            self,
            semantic_hidden_states,
            speaker_ids,
            dialect_ids,
            generation_config,
            initial_audio_codes,
            frame_rate,
            semantic_attention_mask,
        )
