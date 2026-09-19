from pathlib import Path
import pytest
import torch

from ulm_live.talker import TalkerConfig, TalkerGenerationConfig, ULMTalker


def get_test_talker() -> ULMTalker:
    cfg = TalkerConfig(
        semantic_dim=128,
        talker_dim=64,
        num_layers=2,
        num_heads=4,
        dim_feedforward=128,
        num_quantizers=8,
        codebook_size=2048,
        pad_token_id=2048,
        num_speakers=16,
        num_dialects=8,
    )
    return ULMTalker(cfg)


def test_greedy_generation_shape_and_range() -> None:
    talker = get_test_talker()
    B = 2
    S = 4
    sem = torch.randn(B, S, talker.config.semantic_dim)
    spk = torch.tensor([0, 1])
    dia = torch.tensor([0, 0])

    gen_cfg = TalkerGenerationConfig(max_new_tokens=15, do_sample=False)
    tokens = talker.generate(sem, spk, dia, generation_config=gen_cfg)

    assert tokens.ndim == 3
    assert tokens.shape == (B, 8, 15)
    # Validate token range
    assert (tokens >= 0).all()
    assert (tokens < talker.config.codebook_size).all()


def test_deterministic_greedy_output() -> None:
    talker = get_test_talker()
    sem = torch.randn(1, 4, talker.config.semantic_dim)
    spk = torch.tensor([0])
    dia = torch.tensor([0])

    gen_cfg = TalkerGenerationConfig(max_new_tokens=10, do_sample=False)
    tokens1 = talker.generate(sem, spk, dia, generation_config=gen_cfg)
    tokens2 = talker.generate(sem, spk, dia, generation_config=gen_cfg)

    assert torch.equal(tokens1, tokens2)


def test_stochastic_sampling() -> None:
    talker = get_test_talker()
    sem = torch.randn(1, 4, talker.config.semantic_dim)
    spk = torch.tensor([0])
    dia = torch.tensor([0])

    gen_cfg = TalkerGenerationConfig(
        max_new_tokens=12,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
    )
    tokens = talker.generate(sem, spk, dia, generation_config=gen_cfg)
    assert tokens.shape == (1, 8, 12)
    assert (tokens >= 0).all() and (tokens < 2048).all()


def test_max_audio_seconds_conversion() -> None:
    talker = get_test_talker()
    sem = torch.randn(1, 4, talker.config.semantic_dim)
    spk = torch.tensor([0])
    dia = torch.tensor([0])

    # 2.0s at 12.5 Hz = 25 frames
    gen_cfg = TalkerGenerationConfig(max_new_tokens=100, max_audio_seconds=2.0)
    tokens = talker.generate(sem, spk, dia, generation_config=gen_cfg, frame_rate=12.5)
    assert tokens.shape[-1] == 25


def test_speaker_and_dialect_conditioning_in_generation() -> None:
    talker = get_test_talker()
    sem = torch.randn(1, 4, talker.config.semantic_dim)
    gen_cfg = TalkerGenerationConfig(max_new_tokens=10, do_sample=False)

    # Different speaker
    t_spk0 = talker.generate(sem, torch.tensor([0]), torch.tensor([0]), generation_config=gen_cfg)
    t_spk1 = talker.generate(sem, torch.tensor([1]), torch.tensor([0]), generation_config=gen_cfg)
    assert not torch.equal(t_spk0, t_spk1)

    # Different dialect
    t_dia0 = talker.generate(sem, torch.tensor([0]), torch.tensor([0]), generation_config=gen_cfg)
    t_dia1 = talker.generate(sem, torch.tensor([0]), torch.tensor([1]), generation_config=gen_cfg)
    assert not torch.equal(t_dia0, t_dia1)


def test_invalid_conditioning_raises_error() -> None:
    talker = get_test_talker()
    sem = torch.randn(1, 4, talker.config.semantic_dim)

    # Speaker out of range
    with pytest.raises(IndexError):
        talker.generate(sem, torch.tensor([999]), torch.tensor([0]))

    # Dialect out of range
    with pytest.raises(IndexError):
        talker.generate(sem, torch.tensor([0]), torch.tensor([999]))
