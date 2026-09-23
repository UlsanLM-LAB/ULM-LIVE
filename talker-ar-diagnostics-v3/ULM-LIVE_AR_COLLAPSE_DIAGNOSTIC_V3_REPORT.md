# ULM-LIVE AR Collapse Diagnostic v3

Date: 2026-09-23. Instance: `i-0f732bf7d1cc409b4`, `ap-northeast-2`. The 22,150-sample Full Retrain v3 was not run. All percentages below are codec token agreement with cached Mimi targets unless stated otherwise. Pilot step probes use four fixed training examples; final evaluations use twelve.

## Existing pilots and the scale gap

| Fresh post-fix pilot | Steps | Teacher CE | Teacher mean | Teacher Q0 | Forced AR mean | Forced AR Q0 | Normal AR mean | Normal length ratio | Unique outputs / 12 | Normal stops / 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 single speaker | 300 | 0.905 | 100% | 100% | 100% | 100% | 100% | 1.00 | 1 (same utterance) | 12 |
| 32 multi speaker | 1,250 | 0.042 | 100% | 100% | 100% | 100% | 100% | 1.00 | 12 | 12 |
| 512 multi speaker | 4,000 | 1.126 | 76.715% | 96.453% | 11.256% | 13.991% | 11.918% | 2.209 | 12 | see result JSON |
| 2,048 multi speaker | 8,000 | 4.058 | 25.399% | 58.829% | 3.305% | 7.218% | 3.306% | 2.458 | 12 | 7 |

The 2,048 pilot's four-example step probe climbed from 5.480% teacher accuracy at step 2,000 to 25.399% at step 8,000. Forced AR stayed between 1.995% and 2.786%. These are different evaluation sets from the twelve-example final result.

## Position alignment investigation

### Implementation and observed training distribution

The collator pads semantic hidden states from `[L_i, D]` to `[B, max(L_i), D]` and sets trailing `semantic_attention_mask` to zero (`ulm_live/talker/data.py`, lines 263–274). Before the fix, `ULMTalker.temporal()` concatenated `[B, S_max, D]` semantic and `[B, Q, T]` audio embeddings into `[B, S_max+T, D]`, then applied one shared `arange(S_max+T)` position vector. Thus every audio frame started at `S_max`. `init_generation_state()` and `step()` used the single sample's semantic tensor length for cached generation. Attention masked the padded semantic keys but did not remove their contribution to the audio position offset. The temporal attention is bidirectional over semantic tokens, causal over audio tokens, and permits audio queries to read all valid semantic keys. The depth transformer models the 16 quantizers within each frame.

Reproducing epoch 1's seeded `DataLoader` batches (`batch_size=8`) over 22,150 train rows gave **86.713%** shifted samples, median shift **14** positions (prior random-batch estimate: approximately 15), Q1 6, Q3 22, maximum 77. Counts: shift 0: 2,943; 1–4: 1,742; 5–8: 2,232; 9–16: 5,698; 17–24: 5,085; 25–32: 2,768; 33+: 1,682. The complete distribution is in `position_alignment/shift_distribution.json`.

The original diagnostic's `assert torch.equal(target, actual)` failed because `TalkerCollator` pads **audio targets** to the longest audio sequence in the batch. A alone and A with a longer mate therefore had different target tensor widths; A's valid target prefix was unchanged. The diagnostic now slices both logits and targets to A's own collated audio length before the assertion. The regression test also checks A's valid target, audio input, and audio mask prefix directly.

### Old epoch-10 checkpoint: fixed padding shifts

Four unchanged test utterances were evaluated with semantic-mask-zero padding at shifts 0, 4, 8, 16, 24, and 32. All other inputs and targets were fixed. Forced AR generated exactly each utterance's target frame count. The four-utterance means are:

| Shift | Codec CE | TF mean | TF Q0 | TF residual | Forced AR mean | Forced AR Q0 | Generated code entropy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6.3177 | 4.267% | 16.012% | 3.484% | 0.936% | 2.485% | 3.542 |
| 4 | 6.3215 | 4.373% | 14.515% | 3.697% | 1.016% | 3.054% | 3.671 |
| 8 | 6.3222 | 4.099% | 14.253% | 3.422% | 0.946% | 2.485% | 3.540 |
| 16 | 6.3235 | 4.336% | 15.182% | 3.613% | 0.968% | 2.485% | 3.661 |
| 24 | 6.3238 | 4.180% | 14.586% | 3.486% | 0.892% | 2.485% | 3.486 |
| 32 | 6.3240 | 4.214% | 14.018% | 3.560% | 0.909% | 2.485% | 3.604 |

Across the earlier 30-utterance variable-padding probe, shifted vs unshifted unpaired aggregates were CE 5.9741 vs 6.0139, teacher mean 6.983% vs 6.547%, forced AR mean 2.848% vs 2.493%. The shifted set contains more rows per utterance, so these aggregates are not a paired treatment estimate. On 81 paired shifted rows, mean CE delta was -0.00575 and median absolute CE delta 0.00322. On 55 paired AR rows, mean forced-AR delta was +0.204 percentage points and median absolute delta 0.098 points. These changes do not resolve AR collapse.

### Old checkpoint: batch invariance

Sample A (`ulsan_006016`) had semantic length 8 and 26 valid audio frames. With length-24 and length-49 mates, the collator's audio tensor widths became 42 and 75 instead of 26. A's 26-frame target, input, and attention-mask prefixes were bitwise identical in every case (`position_alignment/input_identity.json`); acoustic delay was zero. The old audio start positions were 8, 24, and 49, while the fixed position is 8 in all cases.

| A's batch | Mean abs logit difference vs alone | Max abs difference | Argmax agreement | Q0 agreement | Codec CE |
|---|---:|---:|---:|---:|---:|
| Alone | 0 | 0 | 100% | 100% | 5.9853 |
| With length-24 mate | 0.0891 | 6.1172 | 85.817% | 80.769% | 5.9968 |
| With length-49 mate | 0.0850 | 5.5859 | 86.298% | 80.769% | 5.9999 |

This directly confirms a **batch-dependent position bug**. Its effect on argmax predictions is moderate, while old-checkpoint CE and AR quality changes are small.

### Fix and regression

`ULMTalker.temporal()` now gives valid semantic tokens cumulative sample-local positions and audio frame `t` position `semantic_attention_mask.sum(1) + t`; masked semantic/audio slots have no positional effect. `init_generation_state()` computes the same cumulative semantic positions and starts per-sample cached audio counters at each mask sum. `step()` indexes the position embedding with those counters. `prime_with_teacher_inputs()` applies the same counters to a teacher-forced cached prefix. The change preserves speaker/dialect conditioning, BOS, acoustic delay, depth ordering, and the existing attention pattern. Regression tests compare A alone vs padded batch mates and full recomputation vs incremental KV cache.

On the actual fresh 2,048 baseline checkpoint, A-alone vs the same length-24/49 mates gave FP32 mean absolute logit differences 0.00000104/0.00000114, maxima below 0.000011, and 100% overall/Q0 argmax agreement. Under the production-style BF16 autocast, mean differences were about 0.0088 with 98.8–99.0% overall argmax agreement, consistent with batch-shape numerical rounding; this is roughly tenfold smaller in mean logit difference than the old positional defect. Both precisions and valid-target assertions are preserved in `position_alignment/postfix_batch_invariance*.json`.

| Matched 512 multi pilot, same subset/seed/4,000 steps | Teacher CE | Teacher mean | Teacher Q0 | Forced AR mean | Forced AR Q0 | Normal length ratio |
|---|---:|---:|---:|---:|---:|---:|
| Before position fix | 1.1583 | 75.795% | 95.516% | 12.266% | 14.926% | 2.055 |
| After position fix | 1.1261 | 76.715% | 96.453% | 11.256% | 13.991% | 2.209 |

| Matched 2,048 multi pilot, same subset/seed/8,000 steps | Teacher CE | Teacher mean | Teacher Q0 | Forced AR mean | Forced AR Q0 | Normal length ratio |
|---|---:|---:|---:|---:|---:|---:|
| Before position fix | 4.4002 | 21.839% | 56.993% | 3.853% | 6.121% | 3.343 |
| After position fix | 4.0585 | 25.399% | 58.829% | 3.305% | 7.218% | 2.458 |

The fixed model's single and 32-utterance fresh pilots reached 100% teacher and AR accuracy. The 32-utterance training batches still had shifts in 85.8% of sample exposures (median 7), so the tiny result cannot be explained by a near absence of padding. Memorization overcame this small-set mismatch; that is an inference supported by the observed length distribution and overfit result.

**POSITIONAL ALIGNMENT EFFECT: MODERATE on individual logits/tokens; SMALL on the observed AR collapse. POSITIONAL ALIGNMENT BUG: PARTIAL as an explanation of collapse (implementation defect confirmed, dominant-cause hypothesis not supported).**

## Semantic conditioning

Old epoch-10 checkpoint, 20 test utterances with a same-speaker donor for shuffled semantics:

| Semantic condition | Codec CE | Teacher mean | Teacher Q0 | KL from correct | Mean abs logit difference |
|---|---:|---:|---:|---:|---:|
| Correct | 5.7608 | 9.500% | 21.884% | 0 | 0 |
| Zero | 5.9669 | 8.001% | 14.426% | 0.2679 | 0.4349 |
| Random | 5.9701 | 8.002% | 14.774% | 0.2634 | 0.4403 |
| Shuffled, same speaker | 5.7738 | 9.409% | 20.799% | 0.0119 | 0.0745 |

Zeroing/randomizing semantics changes output materially; swapping a same-speaker utterance changes much less. The fresh post-fix 2,048 baseline shows the same pattern on 20 training utterances:

| Fresh semantic condition | Codec CE | Teacher mean | Teacher Q0 | KL from correct | Mean abs logit difference |
|---|---:|---:|---:|---:|---:|
| Correct | 4.0096 | 26.131% | 60.347% | 0 | 0 |
| Zero | 4.9208 | 16.561% | 30.245% | 0.7607 | 0.9176 |
| Random | 5.2886 | 14.160% | 24.414% | 1.0901 | 1.1131 |
| Shuffled, same speaker | 4.0743 | 25.154% | 57.487% | 0.0501 | 0.2230 |

The model uses semantic features, but same-speaker wrong-text features cause a much smaller loss than removing semantics. **SEMANTIC CONDITIONING: WEAK for utterance identity**, not ignored.

## Forced length, stop, and divergence

The epoch-10 `forced_audio` diagnostic decoded 20 normal/GT-length pairs; normal generated duration averaged 5.286 times target (median 4.636), while every forced output used target length. In the 100-utterance rollout, forced-length codec accuracy equaled normal AR accuracy **2.301%** (Q0 3.927%) over the same target-frame overlap; normal length ratio averaged 4.805. Correcting duration did not correct the codec sequence. **STOP/DURATION: NOT PRIMARY** for codec collapse, although duration is visibly abnormal.

| Old checkpoint epoch, 100 test utterances | Teacher mean | Normal AR mean | Forced-length AR mean | Normal length ratio |
|---:|---:|---:|---:|---:|
| 6 | 6.594% | 2.272% | 2.268% | 5.159 |
| 8 | 6.718% | 2.099% | 2.099% | 5.247 |
| 10 | 6.849% | 2.301% | 2.301% | 4.805 |

In epoch-10 rollout, 99/100 utterances had their first incorrect codec frame at frame zero (median first wrong frame zero). The matched teacher vs AR frame curves are in `rollout/epoch-10-divergence.csv` and `.png`:

| Frames after first wrong | N | Teacher frame accuracy | AR frame accuracy |
|---:|---:|---:|---:|
| 0 | 100 | 24.063% | 16.688% |
| 1 | 100 | 18.875% | 11.313% |
| 2 | 100 | 15.313% | 7.750% |
| 4 | 100 | 7.125% | 2.750% |
| 8 | 100 | 3.875% | 1.063% |
| 16 | 97 | 4.188% | 1.224% |

The first-frame error rate means this old checkpoint alone cannot separate exposure bias from an already weak one-step predictor. The fixed fresh 2,048 pilot still shows a large TF/AR gap.

The old epoch-10 text-variation diagnostic generated 20 different prompts with one speaker: 0/190 output pairs were exactly identical, mean pairwise codec-token agreement was 48.44% (Q0 51.62%), and median generated length was the 100-frame cap. This confirms nonidentical token outputs, not meaningful speech content. The 20-sample hybrid-codebook diagnostic decoded GT, all-predicted, predicted-Q0/GT-residual, and GT-Q0/predicted-residual combinations at several codebook cutoffs. At the Q0-only cutoff, predicted-Q0/GT-residual token agreement was 94.79%, whereas GT-Q0/predicted-residual was 10.77%; these percentages mainly reflect how many GT codebooks each hybrid retains and are not an intelligibility measure. Matched WAVs are under `hybrid/epoch-10/`.

## Capacity and Q0 weighting controls already completed

The existing pre-fix 512-multi runs used identical subset and 4,000-step budget. The 73.9M larger model had teacher 70.533%, forced AR 7.512%; the 40.7M baseline had teacher 75.795%, forced AR 12.266%. Q0×2 reached 99.876% teacher Q0 but only 9.061% forced AR. These results do not support expanding model capacity or upweighting Q0 as the first response to collapse.

## Scheduled sampling investigation

The scheduled path uses the production `init_generation_state()`/`step()` semantics. It primes the KV cache with teacher inputs before a random eight-frame window, then for each frame uses the **actually selected** previous frame to produce the next detached 16-codebook frame. The selected frame is teacher audio with probability `1-p`, generated audio with probability `p`. Training recomputes the full differentiable loss using that mixed input. The schedule is 0% for steps 0–10%, 10% for 10–30%, 25% for 30–60%, and 50% for 60–100%. Accounting for the random window being chosen over each batch's **maximum** audio width, only 10.29% of valid previous-frame inputs fall inside the window on average over the 8,000-step 2,048-sample batch order. Thus the 50% final phase replaces an expected **5.14%** of all valid previous-frame inputs. This bounded exposure matters when interpreting a negative result.

The fresh 32-multi scheduled smoke passed at step 1,250: teacher mean/Q0 and forced/normal AR mean/Q0 all 100%; normal length ratio 1.0; 12 distinct outputs and 12/12 normal stops.

| Step | Baseline TF | Scheduled TF | Baseline forced AR | Scheduled forced AR | Baseline AR Q0 | Scheduled AR Q0 |
|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 5.480% | 5.512% | 2.028% | 2.277% | 7.415% | 7.996% |
| 4,000 | 9.027% | 8.790% | 1.995% | 2.264% | 5.093% | 5.496% |
| 6,000 | 16.045% | 14.883% | 2.689% | 1.504% | 7.206% | 3.829% |
| 8,000 | 25.399% | 23.289% | 2.786% | 2.152% | 8.025% | 3.829% |

The four-example step probes fluctuate and are not the final comparison. On the same twelve final examples at step 8,000, the baseline had 3.305% forced AR mean / 7.218% Q0, 2.458 normal length ratio, and 7/12 stop predictions. Scheduled sampling had 3.560% forced AR mean / 5.511% Q0, 3.240 normal length ratio, and 4/12 stop predictions. Its teacher mean fell from 25.399% to 23.289% (CE 4.0585 to 4.2497). The +0.255 percentage-point forced AR mean change is small, accompanied by worse Q0 and duration, and does not establish an AR improvement.

## Audio artifacts

The post-fix pilots' representative WAV triplets and old-checkpoint forced-length/text/position WAVs are copied locally. All 20 matched utterances in `audio_compare/sample_NNN/` have original, Mimi GT, baseline and scheduled teacher-forced decodes, baseline and scheduled normal AR decodes, text, and per-model metrics. Baseline normal AR agreement is 2.356% mean / 4.658% Q0, mean length ratio 2.264, and 12/20 stop predictions. Scheduled normal AR agreement is 2.802% mean / 4.702% Q0, mean length ratio 2.795, and 6/20 stop predictions. Paired scheduled-minus-baseline mean agreement is +0.446 percentage points (12 better, 7 worse, 1 tied); Q0 is +0.044 points (4 better, 8 worse, 8 tied). All 20 WAV sets were checked for completeness. These are training-subset examples and token metrics; no human intelligibility judgment has been recorded. Valid WAV files alone do not establish intelligible speech.

## Root cause and Full Retrain v3 gate

The implementation bug is fixed and batch-invariance tests pass. The position fix alone did not improve 512 or 2,048 forced AR. The completed small overfits rule out a basic architectural impossibility, a universal stop-head failure, and multi-speaker identity alone as causes. At 512 samples, concentrating 512 utterances in three speakers gave teacher 84.333% / forced AR 12.816%, while 196-speaker sampling gave teacher 75.795% / forced AR 12.266%; speaker variation affects teacher learning more than this AR failure. The completed eight-frame scheduled pilot did not produce a convincing gain: its forced AR mean changed by only +0.255 percentage points, while Q0, stop behavior, and length worsened. Because only about 5.14% of valid previous-frame inputs are replaced in the final phase, this result rejects this specific bounded schedule as a v3 training choice; it does not rule out a better exposure strategy. The weak one-step predictor, limited utterance-specific semantic use, and feedback sensitivity remain plausible contributors. Their relative causal shares are unresolved.

### Conditional Full Retrain v3 design (config only; no training authorization)

| Item | Proposed setting or gate |
|---|---|
| Architecture | Existing sample-local-position Talker: semantic 2048→512, six temporal blocks / eight heads / FFN 2048, 256-dim two-layer depth transformer, 16 Mimi codebooks; 40,706,817 parameters. Do not adopt the unhelpful 73.9M expansion without a matched positive pilot. |
| Data | Same 22,150 cached train utterances; frozen Thinker; semantic mask and codec validation unchanged. |
| Optimization | Fresh model, AdamW, LR 2e-4, weight decay 0.01, batch 8, gradient accumulation 2 (effective 16), warmup 3%, gradient clip 1.0. No old-position checkpoint initialization. |
| Teacher forcing / feedback stages | Start with pure teacher forcing. The diagnostic candidate then uses 0%, 10%, 25%, 50% generated-previous-frame Bernoulli probabilities over the 0–10%, 10–30%, 30–60%, 60–100% step spans. Broader **effective** valid-frame exposure must be validated before adopting this schedule for full training; the current eight-frame pilot reaches only about 5.14% effective replacement in its final phase. |
| Loss | Equal codec codebook weights; stop BCE weight 0.05 and positive weight 5.0. The 512 Q0×2 pilot increased teacher Q0 without improving AR. |
| Stop and duration | Learned stop threshold 0.5, minimum 10 frames, inference safety cap 150 frames for pilot checks; keep model capacity cap 750. Measure normal and forced-length AR separately. |
| Budget | Hard ceiling 20 epochs (about 27,700 optimizer steps at batch 8/accumulation 2); evaluate every 1,000 optimizer steps, save resumable checkpoints at least every 1,000, and stop early if three successive evaluations fail to improve AR while TF rises. This is a design ceiling, not a run request. |
| WAV / selection | Decode the same 20 held-out prompts every 2,000 steps and at each candidate best checkpoint. Select by forced AR mean and Q0 agreement subject to normal length ratio and semantic-shuffle checks; reject checkpoints with collapse or worse listening quality regardless of TF loss. |

This design remains gated on a clear 2,048-sample AR gain, stable duration, semantic conditioning, and better human-audible speech structure. The current scheduled candidate fails the AR/duration gate, and human listening is still needed. None of those outcomes is inferred from TF improvement alone.

## Code verification

Local `pytest -q`: 81 passed, 4 deselected. `python3 -m compileall -q ulm_live scripts`: passed. Ruff on all changed Talker/model/diagnostic/test files: passed. Repository-wide `ruff check .` still reports 96 pre-existing errors in unrelated files; those were left outside this diagnostic change.

**FULL RETRAIN V3 READY: NO (the tested schedule has no convincing AR gain and worsens duration; human listening has not been recorded).**

**PRODUCTION TALKER: NONE.**
