from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class TalkerGenerationConfig:
    """Configuration parameters for Talker autoregressive decoding."""

    max_new_tokens: int = 50  # Number of codec frames to generate (~4.0s at 12.5 Hz)
    max_audio_seconds: float | None = None  # Optional target audio duration in seconds
    temperature: float = 1.0  # Sampling temperature (only used if do_sample=True)
    top_k: int | None = None  # Top-k candidate filtering
    top_p: float | None = None  # Top-p (nucleus) filtering
    do_sample: bool = False  # False for greedy decoding, True for stochastic sampling
    stop_threshold: float | None = None
    min_audio_frames: int | None = None
    max_audio_frames: int | None = None
    return_lengths: bool = False


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int | None = None,
    top_p: float | None = None,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering."""
    filtered_logits = logits.clone()

    if top_k is not None and top_k > 0:
        k = min(top_k, filtered_logits.size(-1))
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = (
            filtered_logits < torch.topk(filtered_logits, k)[0][..., -1, None]
        )
        filtered_logits[indices_to_remove] = filter_value

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits[indices_to_remove] = filter_value

    return filtered_logits


def sample_next_tokens(
    logits: torch.Tensor,
    config: TalkerGenerationConfig,
) -> torch.Tensor:
    """Sample or greedy select next tokens from codebook logits.

    Args:
        logits: Tensor of shape (batch_size, num_quantizers, vocab_size).
        config: TalkerGenerationConfig controlling greedy vs stochastic sampling.

    Returns:
        Tensor of shape (batch_size, num_quantizers, 1), dtype int64.
    """
    B, K, vocab_size = logits.shape

    if not config.do_sample or config.temperature <= 1e-5:
        # Greedy decoding (argmax)
        next_tokens = torch.argmax(logits, dim=-1, keepdim=True)  # (B, K, 1)
        return next_tokens

    # Stochastic sampling
    scaled_logits = logits / max(config.temperature, 1e-5)
    filtered_logits = top_k_top_p_filtering(
        scaled_logits, top_k=config.top_k, top_p=config.top_p
    )

    probs = F.softmax(filtered_logits, dim=-1)  # (B, K, vocab_size)
    flat_probs = probs.reshape(B * K, vocab_size)

    # Sample from multinomial distribution
    sampled = torch.multinomial(flat_probs, num_samples=1)  # (B * K, 1)
    next_tokens = sampled.reshape(B, K, 1).to(dtype=torch.long)
    return next_tokens


def generate_codec_tokens(
    model,
    semantic_hidden_states,
    speaker_ids,
    dialect_ids,
    generation_config=None,
    initial_audio_codes=None,
    frame_rate=12.5,
    semantic_attention_mask=None,
):
    """Incremental generation with real temporal K/V caching and per-sample stop."""
    cfg = generation_config or TalkerGenerationConfig()
    safety_cap = cfg.max_audio_frames or model.config.max_audio_frames
    limit = min(cfg.max_new_tokens, safety_cap)
    threshold = (
        model.config.stop_threshold
        if cfg.stop_threshold is None
        else cfg.stop_threshold
    )
    minimum = (
        model.config.min_audio_frames
        if cfg.min_audio_frames is None
        else cfg.min_audio_frames
    )
    if cfg.max_audio_seconds is not None:
        limit = min(safety_cap, max(1, round(cfg.max_audio_seconds * frame_rate)))
    was_training = model.training
    model.eval()
    state = model.init_generation_state(
        semantic_hidden_states, speaker_ids, dialect_ids, semantic_attention_mask
    )
    b = semantic_hidden_states.shape[0]
    current = torch.full(
        (b, model.config.num_quantizers),
        model.config.bos_token_id,
        dtype=torch.long,
        device=semantic_hidden_states.device,
    )
    produced = []
    if initial_audio_codes is not None:
        if initial_audio_codes.ndim != 3:
            raise ValueError("initial_audio_codes must be [B,K,T]")
        if initial_audio_codes.shape[-1] == 0:
            raise ValueError("initial_audio_codes cannot be empty")
        model.step(state, current, cfg)  # BOS predicts prompt frame zero.
        for i in range(initial_audio_codes.shape[-1] - 1):
            model.step(state, initial_audio_codes[:, :, i], cfg)
        current = initial_audio_codes[:, :, -1]
        produced.extend(initial_audio_codes.unbind(-1))
    finished = torch.zeros(b, dtype=torch.bool, device=current.device)
    lengths = torch.zeros(b, dtype=torch.long, device=current.device)
    with torch.no_grad():
        while len(produced) < limit:
            tokens, stop_prob = model.step(state, current, cfg)
            tokens = torch.where(
                finished[:, None],
                torch.full_like(tokens, model.config.pad_token_id),
                tokens,
            )
            produced.append(tokens)
            current = tokens
            if len(produced) >= minimum:
                newly_finished = (~finished) & (stop_prob >= threshold)
                lengths[newly_finished] = len(produced)
                finished |= newly_finished
            if finished.all():
                break
    model.train(was_training)
    lengths[~finished] = len(produced)
    tokens = torch.stack(produced, -1)
    return (tokens, lengths) if cfg.return_lengths else tokens
