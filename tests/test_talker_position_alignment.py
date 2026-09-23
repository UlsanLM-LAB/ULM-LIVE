"""Semantic padding must not change a sample's audio positions or logits."""

import torch
import torch.nn.functional as F

from ulm_live.talker import TalkerCollator, TalkerConfig, ULMTalker


def setup():
    torch.manual_seed(20260923)
    config = TalkerConfig(
        semantic_dim=8, talker_dim=16, num_layers=2, num_heads=4,
        dim_feedforward=32, depth_dim=16, depth_layers=1, depth_heads=4,
        depth_feedforward=32, num_quantizers=8, dropout=0.0, depth_dropout=0.0,
    )
    model = ULMTalker(config).eval()
    collator = TalkerCollator()
    samples = []
    for i, semantic_length in enumerate((3, 12, 27)):
        samples.append({
            "id": str(i), "semantic_hidden_states": torch.randn(semantic_length, 8),
            "semantic_attention_mask": torch.ones(semantic_length, dtype=torch.long),
            "audio_codes": torch.randint(0, 2048, (8, 5 + i)),
            "speaker_id": i, "dialect_id": i % 2,
        })
    return model, collator, samples


def logits(model, batch):
    with torch.no_grad():
        return model(batch["semantic_hidden_states"], batch["audio_codes"],
                     batch["speaker_ids"], batch["dialect_ids"],
                     targets=batch["targets"],
                     semantic_attention_mask=batch["semantic_attention_mask"],
                     audio_attention_mask=batch["audio_attention_mask"]).logits


def test_sample_logits_are_independent_of_padding_and_batch_mates():
    model, collator, samples = setup()
    alone = collator([samples[0]])
    base = logits(model, alone)
    for mate in samples[1:]:
        batch = collator([samples[0], mate])
        assert torch.equal(batch["targets"][0, :, :alone["targets"].shape[-1]], alone["targets"][0])
        assert torch.equal(batch["audio_codes"][0, :, :alone["audio_codes"].shape[-1]], alone["audio_codes"][0])
        assert torch.equal(batch["audio_attention_mask"][0, :alone["audio_attention_mask"].shape[-1]], alone["audio_attention_mask"][0])
        actual = logits(model, batch)[0:1, :, :base.shape[2]]
        torch.testing.assert_close(actual, base, atol=1e-5, rtol=1e-5)
    padded = dict(samples[0])
    padded["semantic_hidden_states"] = F.pad(padded["semantic_hidden_states"], (0, 0, 0, 24))
    padded["semantic_attention_mask"] = F.pad(padded["semantic_attention_mask"], (0, 24))
    torch.testing.assert_close(logits(model, collator([padded])), base, atol=1e-5, rtol=1e-5)


def test_full_recomputation_matches_incremental_cache_with_padding():
    model, collator, samples = setup()
    batch = collator(samples[:2])
    with torch.no_grad():
        output = model(batch["semantic_hidden_states"], batch["audio_codes"],
                       batch["speaker_ids"], batch["dialect_ids"],
                       semantic_attention_mask=batch["semantic_attention_mask"],
                       audio_attention_mask=batch["audio_attention_mask"])
        full = output.logits.argmax(-1)
        state = model.init_generation_state(batch["semantic_hidden_states"],
                                            batch["speaker_ids"], batch["dialect_ids"],
                                            batch["semantic_attention_mask"])
        incremental = []
        stop_probs = []
        for frame in range(batch["audio_codes"].shape[-1]):
            token, stop = model.step(state, batch["audio_codes"][:, :, frame])
            incremental.append(token)
            stop_probs.append(stop)
    incremental_tokens = torch.stack(incremental, -1)
    incremental_stop = torch.stack(stop_probs, -1)
    for i in range(len(samples[:2])):
        valid = batch["audio_attention_mask"][i]
        torch.testing.assert_close(incremental_tokens[i, :, valid], full[i, :, valid])
        torch.testing.assert_close(incremental_stop[i, valid], output.stop_logits[i, valid].sigmoid(), atol=1e-5, rtol=1e-5)
