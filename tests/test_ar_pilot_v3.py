"""Selection and loss controls used in matched AR pilots."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from ulm_live.talker import TalkerConfig, ULMTalker


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pilot_talker_ar_v3.py"
SPEC = importlib.util.spec_from_file_location("pilot_talker_ar_v3", SCRIPT)
assert SPEC and SPEC.loader
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_subset_selection_distinguishes_single_and_multi_speaker() -> None:
    rows = [
        {"id": f"{speaker}-{n:03d}", "speaker_id": speaker, "duration": 2,
         "text": f"utterance {n}", "semantic_path": "cache.pt", "codec_path": "codes.pt"}
        for speaker in ("a", "b", "c") for n in range(6)
    ]
    single = pilot.select_subset(rows, 4, "single")
    multi = pilot.select_subset(rows, 4, "multi")
    assert len(set(single)) == 4 and len({rows[i]["speaker_id"] for i in single}) == 1
    assert len(set(multi)) == 4 and len({rows[i]["speaker_id"] for i in multi}) > 1
    assert len(pilot.select_subset(rows, 1, "single")) == 1
    with pytest.raises(ValueError, match="largest single speaker"):
        pilot.select_subset(rows, 8, "single")


def test_weight_and_feedback_schedule_controls() -> None:
    device = torch.device("cpu")
    assert pilot.loss_weights("baseline", device).tolist() == [1.0] * 16
    assert pilot.loss_weights("coarse", device)[:4].tolist() == [4.0, 2.0, 1.5, 1.5]
    assert [pilot.feedback_probability("scheduled", step, 100) for step in (1, 11, 31, 61)] == [0.0, 0.1, 0.25, 0.5]


def test_teacher_prefix_cache_matches_serial_steps() -> None:
    torch.manual_seed(7)
    config = TalkerConfig(
        semantic_dim=8, talker_dim=16, num_layers=2, num_heads=4,
        dim_feedforward=32, depth_dim=16, depth_layers=1, depth_heads=4,
        depth_feedforward=32, num_quantizers=8, dropout=0.0, depth_dropout=0.0,
    )
    model = ULMTalker(config).eval()
    inputs = torch.randint(0, 2048, (2, 8, 6))
    inputs[:, :, 0] = config.bos_token_id
    batch = {
        "semantic_hidden_states": torch.randn(2, 3, 8),
        "semantic_attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        "speaker_ids": torch.zeros(2, dtype=torch.long),
        "dialect_ids": torch.zeros(2, dtype=torch.long),
        "audio_codes": inputs,
        "audio_attention_mask": torch.ones(2, 6, dtype=torch.bool),
    }
    with torch.no_grad():
        serial = model.init_generation_state(
            batch["semantic_hidden_states"], batch["speaker_ids"],
            batch["dialect_ids"], batch["semantic_attention_mask"],
        )
        for frame in range(3):
            model.step(serial, inputs[:, :, frame])
        primed = pilot.prime_with_teacher_inputs(model, batch, 3)
        assert torch.equal(serial.next_position, primed.next_position)
        assert torch.equal(serial.key_padding_mask, primed.key_padding_mask)
        for serial_kv, primed_kv in zip(serial.layer_kv, primed.layer_kv, strict=True):
            for a, b in zip(serial_kv, primed_kv, strict=True):
                torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        serial_next, _ = model.step(serial, inputs[:, :, 3])
        primed_next, _ = model.step(primed, inputs[:, :, 3])
        assert torch.equal(serial_next, primed_next)


def test_scheduled_feedback_follows_selected_previous_frames() -> None:
    torch.manual_seed(9)
    config = TalkerConfig(
        semantic_dim=8, talker_dim=16, num_layers=2, num_heads=4,
        dim_feedforward=32, depth_dim=16, depth_layers=1, depth_heads=4,
        depth_feedforward=32, num_quantizers=16, dropout=0.0, depth_dropout=0.0,
    )
    model = ULMTalker(config).train()
    codes = torch.randint(0, 2048, (1, 16, 12))
    codes[:, :, 0] = config.bos_token_id
    batch = {
        "semantic_hidden_states": torch.randn(1, 4, 8),
        "semantic_attention_mask": torch.ones(1, 4),
        "speaker_ids": torch.zeros(1, dtype=torch.long),
        "dialect_ids": torch.zeros(1, dtype=torch.long),
        "audio_codes": codes,
        "audio_attention_mask": torch.ones(1, 12, dtype=torch.bool),
    }
    choices = iter((0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0))
    with patch.object(pilot.torch, "randint", return_value=torch.tensor([1])), \
            patch.object(pilot.torch, "rand", side_effect=lambda *args, **kw: torch.tensor([next(choices)])):
        mixed = pilot.mixed_previous_frames(model, batch, 0.5)
    assert model.training
    torch.testing.assert_close(mixed[:, :, 0:2], codes[:, :, 0:2])
    torch.testing.assert_close(mixed[:, :, 3], codes[:, :, 3])
    with torch.no_grad():
        model.eval()
        state = pilot.prime_with_teacher_inputs(model, batch, 1)
        previous = codes[:, :, 1]
        for offset in range(1, 9):
            generated, _ = model.step(state, previous)
            actual = mixed[:, :, 1 + offset]
            expected = generated if offset % 2 else codes[:, :, 1 + offset]
            torch.testing.assert_close(actual, expected)
            previous = actual
