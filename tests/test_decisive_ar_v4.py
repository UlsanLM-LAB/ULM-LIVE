"""The continuation must consume contiguous, valid, detached full-frame feedback."""

import torch

from scripts.decisive_ar_v4 import (
    feedback_metrics, learner_history, rollout_starts, select_validation,
)
from ulm_live.talker import TalkerCollator, TalkerConfig, ULMTalker


def fixture():
    torch.manual_seed(17)
    cfg = TalkerConfig(semantic_dim=8, talker_dim=16, num_layers=2, num_heads=4,
                       dim_feedforward=32, depth_dim=16, depth_layers=1, depth_heads=4,
                       depth_feedforward=32, num_quantizers=16, dropout=0., depth_dropout=0.)
    model = ULMTalker(cfg).train()
    samples = [{"id": str(i), "semantic_hidden_states": torch.randn(s, 8),
                "semantic_attention_mask": torch.ones(s, dtype=torch.long),
                "audio_codes": torch.randint(0, 2048, (16, a)), "speaker_id": i,
                "dialect_id": 0} for i, (s, a) in enumerate(((3, 5), (7, 13)))]
    return model, TalkerCollator()(samples)


def test_sample_local_ratio_and_pad_exclusion():
    _, batch = fixture()
    mask = batch["audio_attention_mask"]
    starts, counts = rollout_starts(mask, .5)
    lengths = mask.sum(1)
    assert counts.tolist() == torch.round((lengths - 1).float() * .5).long().tolist()
    assert counts[0] > 0 and counts[1] > 0
    metrics = feedback_metrics(batch["audio_codes"], batch["audio_codes"], mask, starts)
    assert metrics["generated_feedback_frames"] == int(counts.sum())
    assert metrics["eligible_valid_frames"] == int((lengths - 1).sum())
    assert metrics["changed_feedback_frames"] == 0
    assert metrics["effective_feedback_ratio"] == int(counts.sum()) / int((lengths - 1).sum())
    assert sum(map(int, metrics["rollout_length_histogram"].values())) == 2


def test_contiguous_full_frame_rollout_is_detached_and_cache_equivalent():
    model, batch = fixture()
    original = batch["audio_codes"]
    mask = batch["audio_attention_mask"]
    starts, counts = rollout_starts(mask, .5)
    mixed, metrics = learner_history(model, batch, .5)
    assert model.training and not mixed.requires_grad
    assert metrics["generated_feedback_frames"] == int(counts.sum())
    for i in range(2):
        length = int(mask[i].sum())
        start = int(starts[i])
        torch.testing.assert_close(mixed[i, :, :start], original[i, :, :start])
        torch.testing.assert_close(mixed[i, :, length:], original[i, :, length:])
        with torch.no_grad():
            model.eval()
            state = model.init_generation_state(batch["semantic_hidden_states"][i:i+1],
                                                batch["speaker_ids"][i:i+1],
                                                batch["dialect_ids"][i:i+1],
                                                batch["semantic_attention_mask"][i:i+1])
            for pos in range(length - 1):
                generated, _ = model.step(state, mixed[i:i+1, :, pos])
                if pos + 1 >= start:
                    torch.testing.assert_close(mixed[i, :, pos + 1], generated[0])
    with torch.no_grad():
        full = model(batch["semantic_hidden_states"], mixed, batch["speaker_ids"],
                     batch["dialect_ids"],
                     semantic_attention_mask=batch["semantic_attention_mask"],
                     audio_attention_mask=mask)
        state = model.init_generation_state(batch["semantic_hidden_states"], batch["speaker_ids"],
                                            batch["dialect_ids"], batch["semantic_attention_mask"])
        for pos in range(mixed.shape[-1]):
            tokens, _ = model.step(state, mixed[:, :, pos])
            for i in range(2):
                if mask[i, pos]:
                    torch.testing.assert_close(tokens[i], full.logits.argmax(-1)[i, :, pos])


def test_validation_selection_is_disjoint_and_deterministic():
    rows = [{"id": f"val-{i:03}", "semantic_path": "s", "codec_path": "c",
             "duration": 2., "text": "한국어 문장"} for i in range(120)]
    first = select_validation(["train-1"], rows)
    assert first == select_validation(["train-1"], rows)
    assert len(first[0]) == 100 and len(first[1]) == 20
    assert set(first[1]).issubset(first[0])
    import pytest
    with pytest.raises(ValueError, match="overlaps"):
        select_validation([first[0][0]], rows)
