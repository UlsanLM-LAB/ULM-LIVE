import pytest
import torch
from ulm_live.talker import TalkerConfig, ULMTalker, TalkerGenerationConfig
from ulm_live.talker.generator import generate_codec_tokens


def test_stop_pos_weight_scales_loss():
    # Verify that stop_pos_weight > 1.0 increases loss when a positive stop frame is present
    c1 = TalkerConfig(
        semantic_dim=16,
        talker_dim=16,
        num_layers=1,
        num_heads=2,
        dim_feedforward=32,
        max_seq_len=32,
        depth_dim=16,
        depth_layers=1,
        depth_heads=2,
        depth_feedforward=32,
        num_quantizers=8,
        num_speakers=2,
        num_dialects=2,
        stop_pos_weight=1.0,
    )
    c5 = TalkerConfig(
        semantic_dim=16,
        talker_dim=16,
        num_layers=1,
        num_heads=2,
        dim_feedforward=32,
        max_seq_len=32,
        depth_dim=16,
        depth_layers=1,
        depth_heads=2,
        depth_feedforward=32,
        num_quantizers=8,
        num_speakers=2,
        num_dialects=2,
        stop_pos_weight=5.0,
    )
    m1 = ULMTalker(c1)
    m5 = ULMTalker(c5)
    m5.load_state_dict(m1.state_dict())

    sem = torch.randn(1, 4, 16)
    codes = torch.randint(0, 2048, (1, 8, 4))
    spk = torch.tensor([0])
    dia = torch.tensor([0])
    st = torch.tensor([[0, 0, 0, 1]])

    out1 = m1(sem, codes, spk, dia, targets=codes, stop_targets=st)
    out5 = m5(sem, codes, spk, dia, targets=codes, stop_targets=st)

    # Positive target has weight 5.0, so stop_loss in m5 must be strictly greater than in m1
    assert out5.stop_loss.item() > out1.stop_loss.item()


def test_generate_codec_tokens_return_details():
    c = TalkerConfig(
        semantic_dim=16,
        talker_dim=16,
        num_layers=1,
        num_heads=2,
        dim_feedforward=32,
        max_seq_len=32,
        depth_dim=16,
        depth_layers=1,
        depth_heads=2,
        depth_feedforward=32,
        num_quantizers=8,
        num_speakers=2,
        num_dialects=2,
    )
    m = ULMTalker(c)
    m.eval()
    sem = torch.randn(1, 4, 16)
    spk = torch.tensor([0])
    dia = torch.tensor([0])
    gen_cfg = TalkerGenerationConfig(max_new_tokens=5, stop_threshold=0.5, return_details=True)

    tokens, lengths, reasons, max_stop_probs = m.generate(
        sem, spk, dia, generation_config=gen_cfg
    )
    assert tokens.ndim == 3
    assert len(reasons) == 1
    assert reasons[0] in ("STOP_PREDICTED", "MAX_FRAMES")
    assert max_stop_probs.ndim == 1
