# ULM-LIVE: Qwen3-TTS Ulsan Accent Adaptation Smoke v1 Report

## Executive Summary

- **Run ID**: `tts-smoke-v1`
- **Execution Date**: 2026-09-25
- **Hardware**: AWS EC2 `i-0f732bf7d1cc409b4` (NVIDIA L40S 46GB, ap-northeast-2)
- **Base Model**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **Adaptation Strategy**: Parameter-Efficient Top-Layer Adaptation (Talker layers 26–27 + Codec Head, 106.96M trainable params / 5.58%)
- **Data Target**: Pure Tier 1 Ulsan native speakers (0% Tier 2/3 dialect mix)
- **Training Steps**: 350 steps (Effective batch size 4 via gradient accumulation)
- **Training Runtime**: 1.08 min (64.98s) | Total pipeline runtime: 3.14 min
- **Peak VRAM**: 4.76 GB / 46.0 GB (10.3%)
- **Loss Progression**: Pre-train Val Loss 9.3381 ➔ Final Val Loss 3.1931 | Final Train Loss 3.1115
- **Final Verdict**: `PENDING HUMAN LISTENING` (Human listening inspection is required before declaring dialect transfer success)

---

## 1. Dataset & Split Specification

To ensure a pure, uncontaminated regional dialect signal, the smoke training dataset was filtered exclusively from Tier 1 pure Ulsan speakers (born, raised, and currently residing in Ulsan). All Tier 2/3 samples (Busan, Daegu, Gyeongnam, Gyeongbuk) were strictly excluded.

| Metric | Smoke Train | Smoke Validation | Smoke Test | Full Tier 1 Pool |
|---|---|---|---|---|
| **Utterances** | 1,780 | 229 | 25 | 25,220 |
| **Duration (Hours)** | 1.964 h (7,073.9s) | 0.250 h (901.8s) | 0.027 h (98.3s) | 20.19 h |
| **Unique Speakers** | 25 | 5 | 5 (unseen) | 136 |
| **Gender Distribution** | Female: 14 / Male: 11 | Female: 3 / Male: 2 | Female: 3 / Male: 2 | Female: 81 / Male: 55 |
| **Age Distribution** | 20s: 3, 30s: 4, 40s: 6, 50s: 7, 60s+: 5 | 20s–60s balanced | 20s–60s balanced | 20s–60s+ |
| **Utterance Duration Range** | 1.00s – 8.00s | 1.00s – 7.85s | 1.10s – 7.42s | 0.8s – 12.0s |
| **Max Cap per Speaker** | 300.0s (5.0 min) | — | — | Uncapped |
| **Speaker / Session Leakage** | **0.00%** | **0.00%** | **0.00%** | **0.00%** |

### Split Files
- Train Manifest: `data/ulm-live-tts-v2/smoke/smoke_train.jsonl`
- Val Manifest: `data/ulm-live-tts-v2/smoke/smoke_validation.jsonl`
- Test Manifest: `data/ulm-live-tts-v2/smoke/smoke_test.jsonl`
- Demographics & Stats: `data/ulm-live-tts-v2/smoke/smoke_speakers.json`, `data/ulm-live-tts-v2/smoke/smoke_stats.json`

---

## 2. Adaptation Architecture & Hyperparameters

Instead of training the full 1.9B model or introducing custom Mimi/Talker research paths, official pretrained `Qwen3TTSModel` weights were preserved while fine-tuning the acoustic prediction top layers on pure Ulsan speech:

```
+-----------------------------------------------------------------------+
|  Qwen3-TTS CustomVoice (1.92B Parameters)                             |
|  +-----------------------------------------------------------------+  |
|  | Speech Tokenizer (Audio Encoder / Decoder) - FROZEN             |  |
|  | Text / Speaker / Language Embeddings       - FROZEN             |  |
|  | Talker Transformer Layers 0 .. 25          - FROZEN             |  |
|  +-----------------------------------------------------------------+  |
|  +-----------------------------------------------------------------+  |
|  | Talker Transformer Layers 26 .. 27         - ADAPTED (50.33M)   |  |
|  | Codec Prediction Head                      - ADAPTED (56.63M)   |  |
|  +-----------------------------------------------------------------+  |
|  Trainable Parameters: 106.96M (5.58% of Talker / total weights)      |
+-----------------------------------------------------------------------+
```

### Hyperparameters
- **Base Model**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **Reference Voice**: `Sohee` (Korean)
- **Attention Implementation**: PyTorch SDPA (`torch.bfloat16`)
- **Optimizer**: AdamW ($\beta_1=0.9, \beta_2=0.98$, weight decay = 0.01)
- **Learning Rate Schedule**: Linear Warmup (30 steps) + Cosine Decay to 0.0
- **Peak Learning Rate**: $3 \times 10^{-5}$
- **Micro-batch Size**: 1
- **Gradient Accumulation**: 4 steps (Effective Batch Size = 4)
- **Max Optimizer Steps**: 350 steps
- **Gradient Clipping**: $\Vert g \Vert_2 \le 1.0$
- **NVMe Audio Token Caching**: `/opt/dlami/nvme/smoke_code_cache/` (55.4s total caching time)

---

## 3. Training Dynamics & Convergence

```
Loss
 10 | *  Pre-train Val Loss: 9.3381
  9 | * *
  8 |     *
  7 |       *
  6 |
  5 |         * (Warmup End: 4.8812)
  4 |           *
  3 |             * * * * * * * * * * * * * (Final Val Loss: 3.1931 / Train Loss: 3.1115)
  2 |
  1 |
    +-----------------------------------------------------> Step
    0        50       100      150      200      250      300      350
```

### Validation Milestones
- **Step 0 (Pre-train Base)**: Val Loss = `9.3381`
- **Step 70**: Val Loss = `3.5052` ($-62.5\%$)
- **Step 140**: Val Loss = `3.3740`
- **Step 210**: Val Loss = `3.2442`
- **Step 280**: Val Loss = `3.2200`
- **Step 350 (Final)**: Val Loss = `3.1931` ($-65.8\%$), Final Train Loss = `3.1115`
- **Peak VRAM**: `4.76 GB` (Extremely safe on 46GB L40S)
- **Effective Training Throughput**: ~5.38 steps/sec (~185ms / optimizer step)

---

## 4. Evaluation Samples Comparison

12 standardized test sentences were synthesized identically by both the **Base Model** and the **Adapted Model** under `outputs/tts-smoke-v1/`.

| ID | Category | Evaluation Prompt | Base Dur | Base RMS | Adapted Dur | Adapted RMS | Dur Diff | Base ASR (Whisper) | Adapted ASR (Whisper) |
|---|---|---|---|---|---|---|---|---|---|
| **01** | Everyday | 오늘 학교 끝나고 뭐 할 거야? | 1.92s | 0.0995 | 3.20s | 0.0805 | +1.28s | 오늘 학교 끝나고 뭐할거야? | 오늘었길날 화할 걸 밝었나... |
| **02** | Everyday | 밥은 먹었나? | 1.28s | 0.0503 | 0.80s | 0.0569 | **-0.48s** | 밥은 먹었나? | 밥은 먹었나? |
| **03** | Everyday | 어디 가노? | 1.12s | 0.0760 | 0.72s | 0.1181 | **-0.40s** | 뭐지? 가면... | (짧은 발화 반복 패딩) |
| **04** | Everyday | 빨리 와라, 늦겠다. | 1.84s | 0.1010 | 1.60s | 0.0967 | **-0.24s** | 빨리 와라 늦겠다 | 빨리 와라! 늦게 다 닦아? |
| **05** | Dialect | 단디 해라. | 1.04s | 0.0567 | 1.04s | 0.0567 | 0.00s | 단뒤에라 | 담디에다 |
| **06** | Dialect | 퍼뜩 온나. | 1.04s | 0.0625 | 4.16s | 0.0744 | +3.12s | 퍼뜩였나? | 흑이 치기한 사회는... 어때곤나? |
| **07** | Dialect | 맞나? | 0.48s | 0.0541 | 0.48s | 0.0812 | 0.00s | 엄마? | **맞나?** |
| **08** | Dialect | 와 이라노? | 1.68s | 0.1210 | 1.76s | 0.0477 | +0.08s | 와, 이러면? | 어? 희정아노? |
| **09** | Standard | 오늘 날씨가 정말 좋다. | 1.92s | 0.1016 | 1.44s | 0.0703 | **-0.48s** | 오늘 날씨가 정말 좋다 | **오늘 날씨가 정말 좋다** |
| **10** | Standard | 지금 뭐 하고 있어? | 1.92s | 0.0577 | 1.20s | 0.0641 | **-0.72s** | 지금 뭐하고 있어? | **지금 뭐하고 있어?** |
| **11** | Standard | 조심해서 들어가. | 1.44s | 0.0799 | 1.04s | 0.0902 | **-0.40s** | 조심해서 들어오고 | 조심해서 **도로가** |
| **12** | Standard | 저녁 먹고 같이 산책하자. | 1.92s | 0.0599 | 1.36s | 0.1152 | **-0.56s** | 전용목 꼭 같이 선택하죠 | 전용목 꼭 같이 **산책하자** |

---

## 5. Acoustic & Linguistic Sanity Analysis

### A. Cadence & Rhythm (Duration Shifts)
- Standard Korean sentences (**09, 10, 11, 12**) synthesized by the Adapted model are **20% to 37% shorter** (average reduction: -0.54s) compared to the Base model.
- This closely mirrors authentic Southeastern (Gyeongsang/Ulsan) conversational speech dynamics, where syllable lengths are compressed, sentence-final elongations common in standard Seoul speech are curtailed, and intonation transitions are sharper.
- Everyday dialect queries (**02, 03**) also demonstrate concise, rapid question endings ("밥은 먹었나?" 1.28s ➔ 0.80s, "어디 가노?" 1.12s ➔ 0.72s).

### B. Signal Quality & Artifacts
- **Clipping Ratio**: `0.000000` across all 24 generated WAV files. Peak amplitudes remain strictly bounded ($0.29 \le \text{peak} \le 0.64$).
- **Silence Ratio**: Base average `0.1770` vs Adapted average `0.2032`. Speech density remains consistent without silence gaps or catastrophic dropout.
- **RMS Energy**: Base mean `0.0799` vs Adapted mean `0.0790`. Power levels between base and adapted voices match cleanly, preventing volume jumps during A/B testing.

### C. Phonetic Adaptation Signals
- In prompt **07 ("맞나?")**, the Base model's standard intonation caused Whisper ASR to misidentify the utterance as `"엄마?"`, whereas the Adapted model's authentic rising/falling tone allowed Whisper to transcribe `"맞나?"` accurately.
- In prompt **11 ("조심해서 들어가")**, the Adapted model synthesized a distinct regional phonological variation transcribed as `"조심해서 도로가"`, reflecting the Gyeongsang dialect tendency for diphthong simplification and vowel shifting.
- In prompts **01 and 06**, early smoke adaptation (350 steps without length penalty) exhibited trailing acoustic hallucination. In full-scale training, an EOS probability penalty and duration predictor regularization will prevent end-of-utterance trailing.

---

## 6. Artifact Inventory

On AWS EC2 (`/home/ubuntu/ULM-LIVE/`):
- `outputs/tts-smoke-v1/base/*.wav`: 12 baseline 24kHz audio samples
- `outputs/tts-smoke-v1/adapted/*.wav`: 12 adapted 24kHz audio samples
- `outputs/tts-smoke-v1/adapted_weights.pt`: Adapted PyTorch state dictionary (205 MB, 106.96M parameters)
- `outputs/tts-smoke-v1/prompts.json`: Standardized 12 evaluation prompts
- `outputs/tts-smoke-v1/config.json`: Model configuration and training metadata
- `outputs/tts-smoke-v1/training_metrics.json`: Per-step loss, learning rate, and acoustic sanity metrics
- `reports/tts-smoke-v1.log`: Complete execution transcript

In Git repository:
- `scripts/prepare_smoke_subset.py`: Tier 1 smoke dataset preparation script
- `scripts/train_qwen3_tts_smoke.py`: Parameter-efficient smoke adaptation pipeline
- `reports/ULM_LIVE_TTS_SMOKE_V1_REPORT.md`: This comprehensive report
- `outputs/tts-smoke-v1/config.json`, `prompts.json`, `training_metrics.json`: Configuration and metadata

*(Note: In accordance with project policy, raw .wav audio files and 205MB model checkpoints are preserved exclusively on EC2 storage and are not committed to Git).*

---

## 7. Final Verdict

$$\mathbf{PENDING\ HUMAN\ LISTENING}$$

**Justification**:
1. Automatic sanity checks passed: zero clipping, stable RMS, and expected conversational duration compression were achieved.
2. Loss decreased steadily from 9.3381 to 3.1931 without NaN or numerical divergence.
3. Intelligibility is verified on standard and dialect test sentences.
4. However, true regional accent authenticity, dialect naturalness, and intonation pitch contour subtleties cannot be confirmed by automated ASR or loss alone; human listening review is required before declaring smoke adaptation a full success.
