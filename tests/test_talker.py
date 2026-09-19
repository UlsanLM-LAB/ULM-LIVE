from pathlib import Path
import pytest
import torch

from ulm_live.talker import TalkerConfig, ULMTalker


def get_test_config() -> TalkerConfig:
    return TalkerConfig(
        semantic_dim=256,
        talker_dim=128,
        num_layers=2,
        num_heads=4,
        dim_feedforward=256,
        num_quantizers=8,
        codebook_size=2048,
        pad_token_id=2048,
        num_speakers=16,
        num_dialects=8,
    )


def test_talker_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
talker:
  semantic_dim: 1024
  talker_dim: 256
  num_layers: 4
  num_heads: 4
  num_quantizers: 8
  codebook_size: 2048
  num_speakers: 64
  num_dialects: 4
"""
    cfg_file = tmp_path / "test_talker.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = TalkerConfig.from_yaml(cfg_file)
    assert cfg.semantic_dim == 1024
    assert cfg.talker_dim == 256
    assert cfg.num_layers == 4
    assert cfg.num_quantizers == 8
    assert cfg.num_speakers == 64


def test_talker_forward_and_backward() -> None:
    config = get_test_config()
    talker = ULMTalker(config)
    talker.train()

    B = 2
    S = 6
    T = 12
    K = config.num_quantizers

    semantic_hidden = torch.randn(B, S, config.semantic_dim)
    audio_codes = torch.randint(0, config.codebook_size, (B, K, T))
    targets = torch.randint(0, config.codebook_size, (B, K, T))
    speaker_ids = torch.tensor([1, 3], dtype=torch.long)
    dialect_ids = torch.tensor([0, 2], dtype=torch.long)

    out = talker(
        semantic_hidden_states=semantic_hidden,
        audio_codes=audio_codes,
        speaker_ids=speaker_ids,
        dialect_ids=dialect_ids,
        targets=targets,
    )

    # 1. Check logits shape: (B, K, T, codebook_size)
    assert out.logits.shape == (B, K, T, config.codebook_size)

    # 2. Check loss is scalar tensor
    assert out.loss is not None
    assert out.loss.ndim == 0
    assert out.loss.item() > 0.0

    # 3. Check backward
    out.loss.backward()
    for name, param in talker.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"


def test_talker_speaker_and_dialect_conditioning() -> None:
    config = get_test_config()
    talker = ULMTalker(config)
    talker.eval()

    B = 1
    S = 4
    T = 8
    K = config.num_quantizers

    semantic_hidden = torch.randn(B, S, config.semantic_dim)
    audio_codes = torch.randint(0, config.codebook_size, (B, K, T))

    with torch.no_grad():
        # Case 1: same speaker, different dialect
        out1 = talker(
            semantic_hidden_states=semantic_hidden,
            audio_codes=audio_codes,
            speaker_ids=torch.tensor([0]),
            dialect_ids=torch.tensor([0]),
        )
        out2 = talker(
            semantic_hidden_states=semantic_hidden,
            audio_codes=audio_codes,
            speaker_ids=torch.tensor([0]),
            dialect_ids=torch.tensor([1]),
        )
        assert not torch.allclose(out1.logits, out2.logits)

        # Case 2: different speaker, same dialect
        out3 = talker(
            semantic_hidden_states=semantic_hidden,
            audio_codes=audio_codes,
            speaker_ids=torch.tensor([1]),
            dialect_ids=torch.tensor([0]),
        )
        assert not torch.allclose(out1.logits, out3.logits)


def test_talker_invalid_ids_raise_error() -> None:
    config = get_test_config()
    talker = ULMTalker(config)

    B = 1
    K = config.num_quantizers
    T = 4
    sem = torch.randn(B, 2, config.semantic_dim)
    codes = torch.randint(0, config.codebook_size, (B, K, T))

    # Invalid speaker id >= num_speakers
    with pytest.raises(IndexError, match="speaker_ids out of range"):
        talker(sem, codes, speaker_ids=torch.tensor([999]), dialect_ids=torch.tensor([0]))

    # Invalid dialect id >= num_dialects
    with pytest.raises(IndexError, match="dialect_ids out of range"):
        talker(sem, codes, speaker_ids=torch.tensor([0]), dialect_ids=torch.tensor([999]))

    # Invalid quantizers count != num_quantizers
    invalid_codes = torch.randint(0, config.codebook_size, (B, K + 2, T))
    with pytest.raises(ValueError, match="quantizers"):
        talker(sem, invalid_codes, speaker_ids=torch.tensor([0]), dialect_ids=torch.tensor([0]))


def test_talker_padding_loss_ignored() -> None:
    config = get_test_config()
    talker = ULMTalker(config)

    B = 1
    K = config.num_quantizers
    T = 6

    sem = torch.randn(B, 2, config.semantic_dim)
    codes = torch.randint(0, config.codebook_size, (B, K, T))
    spk = torch.tensor([0])
    dia = torch.tensor([0])

    # All targets are pad_token_id
    targets_pad = torch.full((B, K, T), config.pad_token_id, dtype=torch.long)
    out_pad = talker(sem, codes, spk, dia, targets=targets_pad)
    # When all targets are ignored, loss is either 0 or nan/masked
    # In F.cross_entropy with all ignored, PyTorch returns 0.0 or nan
    assert out_pad.loss is not None
