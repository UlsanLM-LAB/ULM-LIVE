import importlib.util
import json
from pathlib import Path
import pytest
import torch
from ulm_live.data.manifest import split_dataset
from ulm_live.data.schema import DatasetItem
from ulm_live.codec.backend import MimiCodec
from ulm_live.talker import (
    TalkerConfig,
    ULMTalker,
    TalkerCollator,
    TalkerDataset,
    TalkerGenerationConfig,
)
from ulm_live.talker.data import apply_acoustic_delay, remove_acoustic_delay
from ulm_live.talker.checkpoint import (
    save_training_checkpoint,
    load_training_checkpoint,
)
from ulm_live.talker.semantic_cache import cache_metadata

SPEC = importlib.util.spec_from_file_location(
    "train_talker", Path(__file__).parents[1] / "scripts/train_talker.py"
)
train = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train)


def cfg(q=8):
    return TalkerConfig(
        semantic_dim=24,
        talker_dim=16,
        num_layers=2,
        num_heads=4,
        dim_feedforward=32,
        dropout=0.0,
        max_seq_len=64,
        depth_dim=16,
        depth_layers=1,
        depth_heads=4,
        depth_feedforward=32,
        depth_dropout=0.0,
        num_quantizers=q,
        num_speakers=4,
        num_dialects=2,
    )


def inputs(c, t=4):
    codes = torch.randint(0, 2048, (1, c.num_quantizers, t))
    x = torch.full_like(codes, c.pad_token_id)
    x[:, :, 0] = c.bos_token_id
    x[:, :, 1:] = codes[:, :, :-1]
    return (
        torch.randn(1, 3, c.semantic_dim),
        x,
        codes,
        torch.tensor([0]),
        torch.tensor([0]),
    )


@pytest.mark.parametrize("q", [8, 16, 32])
def test_supported_quantizer_counts(q):
    c = cfg(q)
    m = ULMTalker(c)
    sem, x, y, s, d = inputs(c, 2)
    assert m(sem, x, s, d, y).logits.shape == (1, q, 2, 2048)


def test_invalid_quantizer_count():
    with pytest.raises(ValueError):
        TalkerConfig(num_quantizers=12)


def test_bos_alignment_first_frame_and_stop_labels():
    codes = torch.arange(24).view(8, 3)
    batch = TalkerCollator()(
        [
            {
                "id": "x",
                "audio_codes": codes,
                "speaker_id": 0,
                "dialect_id": 0,
                "text": "x",
            }
        ]
    )
    assert torch.all(batch["audio_codes"][:, :, 0] == 2049)
    assert torch.equal(batch["targets"][0], codes)
    assert not torch.any(batch["targets"] == 2049)
    assert batch["stop_targets"].tolist() == [[0, 0, 1]]


def test_padding_ignored_and_stop_loss():
    c = cfg()
    m = ULMTalker(c)
    sem, x, y, s, d = inputs(c)
    y[:, :, -1] = c.pad_token_id
    stop = torch.tensor([[0, 0, 1, -1]])
    out = m(sem, x, s, d, y, stop_targets=stop)
    assert torch.isfinite(out.loss) and out.stop_loss is not None


def test_depth_codebook_depends_on_previous_codebook():
    torch.manual_seed(2)
    c = cfg()
    m = ULMTalker(c).eval()
    sem, x, y, s, d = inputs(c)
    frames = m.temporal(sem, x, s, d)
    a = m.depth_logits(frames, y)
    changed = y.clone()
    changed[:, 0] = (changed[:, 0] + 1) % 2048
    b = m.depth_logits(frames, changed)
    assert torch.allclose(a[:, 0], b[:, 0])
    assert not torch.allclose(a[:, 1], b[:, 1])


def test_semantic_padding_is_not_conditioning():
    torch.manual_seed(3)
    c = cfg()
    m = ULMTalker(c).eval()
    sem, x, y, s, d = inputs(c)
    sem = torch.cat((sem, torch.randn(1, 2, c.semantic_dim)), 1)
    mask = torch.tensor([[1, 1, 1, 0, 0]])
    a = m(sem, x, s, d, y, semantic_attention_mask=mask).logits
    sem[:, 3:] = torch.randn_like(sem[:, 3:]) * 1000
    b = m(sem, x, s, d, y, semantic_attention_mask=mask).logits
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_acoustic_delay_round_trip(delay):
    x = torch.randint(0, 2048, (2, 8, 7))
    assert torch.equal(
        remove_acoustic_delay(apply_acoustic_delay(x, delay), delay, 7), x
    )


def test_learned_stop_honors_minimum():
    c = cfg()
    m = ULMTalker(c)
    torch.nn.init.zeros_(m.stop_head.weight)
    m.stop_head.bias.data.fill_(20)
    sem = torch.randn(1, 3, c.semantic_dim)
    out = m.generate(
        sem,
        torch.tensor([0]),
        torch.tensor([0]),
        TalkerGenerationConfig(
            max_new_tokens=10, min_audio_frames=3, stop_threshold=0.5
        ),
    )
    assert out.shape[-1] == 3


def test_batch_generation_tracks_individual_finished_lengths():
    c = cfg()
    m = ULMTalker(c).eval()
    original = m.step
    calls = 0

    def controlled(state, frame, sampling_config=None):
        nonlocal calls
        tokens, _ = original(state, frame, sampling_config)
        calls += 1
        probability = torch.tensor([1.0, 1.0 if calls >= 3 else 0.0])
        return tokens, probability

    m.step = controlled
    config = TalkerGenerationConfig(
        max_new_tokens=5, min_audio_frames=1, stop_threshold=0.5, return_lengths=True
    )
    tokens, lengths = m.generate(
        torch.randn(2, 3, c.semantic_dim),
        torch.tensor([0, 1]),
        torch.tensor([0, 0]),
        config,
    )
    assert lengths.tolist() == [1, 3]
    assert torch.all(tokens[0, :, 1:] == c.pad_token_id)


def test_cached_generation_matches_full_recompute():
    torch.manual_seed(4)
    c = cfg()
    m = ULMTalker(c).eval()
    sem = torch.randn(1, 3, c.semantic_dim)
    s = d = torch.tensor([0])
    state = m.init_generation_state(sem, s, d)
    current = torch.full((1, c.num_quantizers), c.bos_token_id)
    cached = []
    full_inputs = current[:, :, None]
    full = []
    for _ in range(5):
        token, _ = m.step(state, current)
        cached.append(token)
        ref = m(sem, full_inputs, s, d).logits[:, :, -1].argmax(-1)
        full.append(ref)
        assert torch.equal(token, ref)
        current = token
        full_inputs = torch.cat((full_inputs, ref[:, :, None]), -1)
    assert torch.equal(torch.stack(cached, -1), torch.stack(full, -1))


def test_semantic_cache_bfloat16_and_fingerprint(tmp_path):
    codec = tmp_path / "codes.pt"
    torch.save(torch.zeros(8, 2, dtype=torch.long), codec)
    meta = cache_metadata("thinker-A", -1, "tok", 24)
    cache = tmp_path / "sem.pt"
    torch.save(
        {
            "hidden_states": torch.randn(3, 24).bfloat16(),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "metadata": meta,
        },
        cache,
    )
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "id": "x",
                "text": "x",
                "speaker_id": "s",
                "dialect": "u",
                "codec_path": codec.name,
                "semantic_path": cache.name,
            }
        )
        + "\n"
    )
    ds = TalkerDataset(
        manifest,
        num_quantizers=8,
        cache_only=True,
        semantic_cache_metadata={"thinker_fingerprint": meta["thinker_fingerprint"]},
    )
    assert ds[0]["semantic_hidden_states"].dtype == torch.bfloat16
    with pytest.raises(ValueError):
        TalkerDataset(
            manifest,
            num_quantizers=8,
            cache_only=True,
            semantic_cache_metadata={"thinker_fingerprint": "wrong"},
        )[0]


def test_semantic_source_safety(tmp_path):
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({"text": "x"}) + "\n")
    with pytest.raises(ValueError):
        train.validate_semantic_source(p, None, False, False)
    with pytest.raises(ValueError):
        train.validate_semantic_source(p, None, True, False)
    assert not train.validate_semantic_source(p, None, True, True)


def test_scheduler_step_math():
    assert train.optimizer_step_counts(11, 4, 3) == (3, 9)
    assert train.optimizer_step_counts(11, 4, 3, 5) == (3, 5)


def test_validation_default_is_unlimited():
    assert train.parse_args(["--manifest", "x"]).max_eval_batches is None


def test_session_split_known_speakers_and_no_leakage():
    items = []
    for spk in ("a", "b"):
        for session in range(5):
            items.append(
                DatasetItem(
                    f"{spk}{session}",
                    "x.wav",
                    "x",
                    spk,
                    "u",
                    24000,
                    1.0,
                    utterance_id=f"u{spk}{session}",
                    session_id=f"s{spk}{session}",
                    source_audio_id=f"s{spk}{session}",
                )
            )
    tr, va, te = split_dataset(items, "session", 0.6, 0.2, 0.2, 1)
    assert {x.speaker_id for x in va + te} <= {x.speaker_id for x in tr}
    groups = [{x.session_id for x in z} for z in (tr, va, te)]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def test_deterministic_checkpoint_resume(tmp_path):
    torch.manual_seed(8)
    c = cfg()
    base = ULMTalker(c)
    initial = {k: v.clone() for k, v in base.state_dict().items()}
    sem, x, y, s, d = inputs(c, 2)
    stop = torch.tensor([[0, 1]])

    def setup():
        m = ULMTalker(c)
        m.load_state_dict(initial)
        o = torch.optim.AdamW(m.parameters(), lr=1e-3)
        sch = train.build_scheduler(o, 4)
        return m, o, sch

    def step(m, o, sch):
        o.zero_grad()
        loss = m(sem, x, s, d, y, stop_targets=stop).loss
        loss.backward()
        o.step()
        sch.step()
        return loss.item()

    a, oa, sa = setup()
    losses_a = [step(a, oa, sa) for _ in range(4)]
    b, ob, sb = setup()
    losses_b = [step(b, ob, sb) for _ in range(2)]
    path = tmp_path / "r.pt"
    save_training_checkpoint(
        path,
        b,
        ob,
        sb,
        None,
        global_step=2,
        epoch=0,
        micro_batch_position=2,
        accumulation_step=0,
        speaker2id={"s": 0},
        dialect2id={"u": 0},
        training_args={},
        best_validation={},
    )
    r, ore, sre = setup()
    state = load_training_checkpoint(path, r, ore, sre)
    assert state["micro_batch_position"] == 2
    losses_b += [step(r, ore, sre) for _ in range(2)]
    assert losses_a == pytest.approx(losses_b, abs=1e-7)
    for pa, pb in zip(a.parameters(), r.parameters()):
        assert torch.equal(pa, pb)


def test_nan_loss_fails_fast():
    c = cfg()
    m = ULMTalker(c)
    sem, x, y, s, d = inputs(c)
    sem.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        m(sem, x, s, d, y)


@pytest.mark.parametrize("special", [2048, 2049])
def test_mimi_decode_rejects_pad_and_bos(special):
    codec = MimiCodec(load_pretrained=False, device="cpu")
    with pytest.raises(ValueError, match="real codec tokens"):
        codec.decode(torch.full((1, 8, 2), special, dtype=torch.long))
