# ULM-LIVE Talker architecture (pre-full-training)

No serious full Talker training has been completed yet. The implementation below is training-ready infrastructure; it is not evidence of audio quality.

```text
ULM Thinker hidden states [B,S,2048] + semantic mask
        ↓ semantic projection
Temporal Transformer (prefix-causal, incremental K/V cache)
        ↓ current frame z_t [B,T,512]
Depth Transformer (teacher forced during training)
        ↓ q0 → q1|q0 → ... → qK-1|q<K
Mimi codec frame [B,K,T] + stop probability [B,T]
        ↓ remove optional acoustic delay; reject special IDs
Mimi decoder → 24 kHz speech
```

## Sequence contract

Mimi IDs are `0..2047`, PAD is `2048`, and BOS is `2049`. The output vocabulary remains exactly 2048 real Mimi IDs. For codec frames `c[0:T]`, temporal training uses `[BOS] + c[:-1]` as input and all of `c` as targets. Thus frame zero is learned rather than copied or discarded. PAD is ignored by codec and stop losses, and neither PAD nor BOS may cross the Mimi decode boundary.

The stop head labels each final real frame as 1 and preceding real frames as 0. Padded frames use -1 and are ignored. Generation respects `min_audio_frames`, terminates per sample at `stop_threshold`, and retains `max_audio_frames` as a safety cap.

## Temporal and depth ownership

The temporal stack consumes semantic prefix tokens and previous codec frames. Semantic tokens attend bidirectionally only within the prefix; audio tokens attend to valid semantic positions and causal audio history. Semantic and audio padding are key-masked. Its custom pre-norm attention exposes real per-layer K/V tensors. `init_generation_state()` computes the semantic prefix once and `step()` consumes one new frame without replaying history. CPU tests compare cached greedy output with full recomputation.

The depth stack models the residual codebooks inside one frame. During training, position `k` receives the frame state and the true `q[k-1]`; a causal depth mask prevents future codebooks. During inference, it samples/chooses one codebook at a time and feeds that choice into the next depth. This implements `P(q_t,k | semantics, frames_<t, q_t,<k)` without flattening time by K.

Supported K values are 8, 16, and 32. A source tensor may contain more codebooks and is sliced to K; fewer than K is a hard error. `scripts/compare_mimi_quantizers.py` creates small 8q/16q/32q reconstruction sets and runtime/MSE reports for a later controlled experiment.

## Acoustic delay

`acoustic_delay_frames` supports 0, 1, or 2. Each RVQ level `k` is shifted by `k * delay`; standalone forward/inverse transforms preserve the original codec timeline. The default is 0. Delay is removed before decode.

## Conditioning and split policy

Speaker and dialect IDs are learned embeddings. Therefore validation/test speakers must occur in train; unseen IDs would be random embeddings and an invalid quality measurement. The canonical manifest remains immutable while derived split manifests group by `session_id`/`source_audio_id` (falling back to utterance identity), keep each group in exactly one split, and reserve training material for every evaluated speaker. Exact utterance and source-clip leakage are rejected by the integrity checker.

This is known-speaker conditioning, not zero-shot voice cloning. Zero-shot support needs a separately trained reference speaker encoder/embedding extractor and is future work.

## Frozen Thinker cache

Training requires either an explicitly loaded compatible Thinker or a complete semantic cache. `scripts/cache_thinker_hidden.py` stores FP16/BF16 hidden states, mask, sequence length, dtype, hidden layer, tokenizer identity, Thinker identifier, and a fingerprint. Cache-only training requires `--semantic-cache-thinker`; identity, fingerprint, and layer mismatches fail. Normal training has no random/zero/base-model fallback. Synthetic semantics require both `--allow-synthetic-thinker` and `--dry-run`.

Thinker outputs retain their model dtype. Autocast and semantic projection perform the necessary conversion; there is no unconditional FP32 expansion.

## Training and recovery

Scheduler length uses `ceil(micro_batches / gradient_accumulation) * epochs`, unless `max_steps` is explicit. Warmup is an exact step count or a ratio of actual optimizer steps. Validation is unlimited by default and reports total/codec/stop loss, stop accuracy, q0 CE, and mean residual CE. Best checkpoint metric is total validation loss.

Training checkpoints are atomic and only written at accumulation boundaries. They contain model, optimizer, scheduler, scaler, optimizer step, epoch and next micro-batch position, Python/Torch/CUDA RNG, mappings, model config, arguments, and best-validation state. Epoch shuffling is seeded by epoch, allowing a resumed run to recreate and skip the already consumed prefix. A deterministic tiny test checks continuous and resumed parameters/losses.

## Not implemented here

Mimi streaming decoder state, microphone/VAD, barge-in, network transport, and full-duplex serving remain future runtime work. Full training and audio-quality claims also remain pending an AWS preflight and explicit approval.
