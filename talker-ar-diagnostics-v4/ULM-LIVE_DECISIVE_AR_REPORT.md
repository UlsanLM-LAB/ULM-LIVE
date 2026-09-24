# ULM-LIVE Decisive AR Training Cycle (Frozen Partial Research)

Date: 2026-09-24. Git base HEAD: `be23839a41268500e182e6262558e99213ab5a6d`.

## Question and prior evidence

The question is why teacher-forced Mimi prediction improves while free-running Korean speech collapses. Mimi reconstruction and cache identity already passed in v3; a fresh single utterance and 32-utterance Talker reached 100% teacher and AR accuracy. The sample-local audio position defect was fixed and tested, but matched 512 and 2048 pilots did not improve AR. The old scheduled pilot replaced only about 5.14% of eligible history frames in its final nominal 50% phase, so it did not test sustained learner-state exposure. The 2048 post-fix checkpoint at `outputs/talker-ar-diagnostics-v3/pilots-postfix/2048-multi-baseline/final.pt` is the sole starting checkpoint for both continuations. It is step 8000 and contains no optimizer state.

## Research basis

[Scheduled Sampling](https://arxiv.org/abs/1506.03099) identifies the ground-truth versus generated-history discrepancy. [Huszár](https://arxiv.org/abs/1511.05101) cautions that its usual objective can be inconsistent, so this cycle treats feedback as a controlled diagnostic, not an assumed solution. [DAgger](https://proceedings.mlr.press/v15/ross11a.html) motivates training on states visited by the learner; the present rollout remains only DAgger-inspired because the reference Mimi frame supervises every visited state. [Professor Forcing](https://arxiv.org/abs/1610.09038) offers another way to compare teacher and free-running dynamics but is not added as a third branch.

[AudioLM](https://research.google/blog/audiolm-a-language-modeling-approach-to-audio-generation/), [VALL-E](https://arxiv.org/abs/2301.02111), [MusicGen](https://arxiv.org/abs/2306.05284), and [EnCodec](https://arxiv.org/abs/2210.13438) support treating codec streams and linguistic content as structured signals. The rollout therefore feeds complete 16-codebook frames and preserves the existing within-frame depth ordering. [ELLA-V](https://arxiv.org/abs/2401.07333), [RALL-E](https://arxiv.org/abs/2404.03204), and [Moshi/Mimi](https://arxiv.org/abs/2410.00037) motivate explicit, aligned linguistic conditioning if feedback training fails. Exact codec agreement is a diagnostic metric; it is not proof of speech intelligibility.

## Matched protocol

Both branches restore exactly the same model weights, use new AdamW state at LR `1e-4`, weight decay `0.01`, batch size 8, the same seeded batch order, the same losses, and gradient clip 1.0. Branch A (`2048-cont-tf`) uses ground-truth history for 8000 more steps. Branch B (`2048-cont-rollout`) uses a contiguous per-sample generated suffix with target effective feedback 10% through step +800, 25% through +2800, 40% through +5600, and 50% through +8000. Generated full frames are greedy, detached, and produced by `init_generation_state()` and `step()`; the differentiable forward pass still uses reference next-frame CE. Eligible feedback excludes BOS and PAD. The actual generated and changed fractions are counted separately.

The deterministic held-out validation IDs come from `data/ulsan-full/val.cached.jsonl` and are disjoint from the 2048 training IDs. The same 100 IDs measure TF CE, mean, Q0, and residual accuracy; the first 20 measure forced and normal AR, duration, first-error survival, and audio. Validation IDs are saved in `outputs/talker-decisive-v4/selection.json` on EC2.

### Training exposure normalization

| Prior subset | Steps | Approximate epochs at batch 8 | Interpretation |
|---|---:|---:|---|
| 32 | 1250 | 312.5 | 100% TF and AR overfit |
| 512 | 4000 | 62.5 | 76.715% TF; 11.256% forced AR on prior training examples |
| 2048 | 8000 | 31.25 | 25.399% TF; 3.305% forced AR on prior training examples |

The prior subset comparison is not exposure-normalized. This cycle compares **TF versus rollout at matched total steps** from the same 2048 weights.

### Before the 2048 rollout

The new 32-utterance learner-history smoke reached 100% TF and forced AR at step 1250 before feedback, then retained **99.984% TF** and **99.948% forced and normal AR** at step 1750 after 10%, 25%, and 50% feedback phases began. In the final 250-step smoke interval, 100% of samples received generated history; the measured effective fraction was 34.902% because the interval spans the 25% and 50% phases. The perturbed fraction was 0.112%: a near-memorized 32-sample model mostly reproduces GT, so this smoke validates the path but cannot test exposure-bias benefit. Unit tests cover PAD exclusion, short and long sample suffixes, measured feedback ratios, detached whole-frame generation, semantic-position invariance, KV versus full recomputation, and validation split disjointness.

## Results

The step-8000 base measured on the new held-out validation set: TF CE **7.973**, TF mean **4.131%**, TF Q0 **8.818%**, forced AR mean **2.659%**, forced AR Q0 **4.943%**, normal AR mean **2.672%**, and normal length ratio **2.734**. These differ from the v3 training-subset figures because the IDs are disjoint. The checkpoint SHA-256 is `ac4ea99cc3c45b31cd6d0bdc75948fcdb6577154faaa46591fd8b22187f32882`.

### Branch A — teacher-forcing continuation

| Total step | Train TF mean | Val TF mean | Val TF Q0 | Val TF CE | Val forced AR | Val normal AR | Val AR Q0 | Normal length ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9000 | 36.700% | 3.996% | 8.481% | 8.350 | 2.411% | 2.428% | 4.644% | 2.523 |
| 10000 | 39.785% | 3.863% | 8.560% | 8.697 | 2.527% | 2.530% | 5.073% | 2.629 |
| 11000 | 43.148% | 3.810% | 8.659% | 9.005 | 1.997% | 2.038% | 4.779% | 1.804 |
| 12000 | 46.291% | 3.669% | 8.461% | 9.330 | 1.734% | 1.832% | 5.752% | 1.797 |
| 13000 | 49.440% | 3.642% | 8.441% | 9.636 | 2.132% | 2.261% | 4.371% | 1.726 |
| 14000 | 52.303% | 3.565% | 8.798% | 9.945 | 1.903% | 2.004% | 4.780% | 1.628 |
| 15000 | 54.794% | 3.613% | 8.600% | 10.225 | 2.103% | 2.110% | 4.660% | 1.812 |
| 16000 | 57.665% | 3.523% | 8.361% | 10.494 | 1.879% | 1.888% | 4.943% | 1.330 |

Train TF rises by 26.1 percentage points from the step-8000 base's 100-example probe (31.527%) while held-out TF, CE, and AR worsen. Continued memorization alone did not repair generalization or free-running speech.

### Branch B — contiguous learner-history continuation

Branch B was interrupted at the ULM Live v2 TTS pivot after its third 1000-step evaluation. The resumable `latest.pt`, `history.json`, `rollout.log`, and `frozen_summary.json` remain on EC2 under `outputs/talker-decisive-v4`. A small copy of the [frozen summary](frozen_summary.json) is in this directory. The process received SIGINT; the next 1000-step interval was not evaluated or checkpointed.

| Total step | Val TF mean | Val forced AR | Effective generated-history fraction |
|---:|---:|---:|---:|
| 9000 | 4.353% | 2.883% | 13.004% |
| 10000 | 4.371% | 2.845% | 24.996% |
| 11000 | 4.397% | 2.371% | 27.997% |

At these matched points Branch B did not show decisive AR recovery relative to Branch A. This is a partial diagnostic result, not a completed comparison or a claim about intelligibility. No linguistic pilot was launched.

### First-error survival and speech-level checks

On the new 20 held-out utterances, both the step-8000 base and step-16000 TF continuation had their first incorrect **full 16-codebook frame** at frame 0 in all 20 cases. Thus strict no-error survival is zero from frame 0 for these models; post-error token accuracy is more informative. No matched final A/B plot was produced because B was frozen. An AWS cache search found no Korean ASR weights: **ASR NOT AVAILABLE LOCALLY**. No large ASR model was downloaded, and CER is unavailable. The listening pack remains remote so human intelligibility is not yet verified.
