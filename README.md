# ULM-Live

Real-time Spoken Language Model runtime and neural audio codec foundations for Ulsan dialect speech generation.

## Architecture Vision

```text
Microphone
   ↓
Audio Tokenizer / Codec (Mimi, 12.5 Hz)
   ↓
ULM-1.7B Thinker (Frozen)
   ↓
Semantic Hidden States [B, S, 2048]
   ↓
ULM Talker (Causal Transformer)  ← [Speaker ID, Dialect ID]
   ↓
Audio Codec Tokens [B, K, T]
   ↓
Streaming Decoder
   ↓
Speech (Ulsan Dialect)
```

## Current Status

```text
[x] Repository foundation
[x] Audio utilities
[x] Audio codec abstraction
[x] WAV → codec → WAV reconstruction
[x] Ulsan dataset pipeline
[x] ULM Thinker adapter
[x] Talker prototype
[ ] Talker training
[ ] Audio generation
[ ] Streaming generation
[ ] Speech input
[ ] Barge-in
[ ] Full duplex
```

For detailed architecture explanations, see [docs/architecture.md](docs/architecture.md).

---

## Neural Audio Codec: Mimi

ULM-Live adopts **Mimi** (`kyutai/mimi`) as its primary neural audio codec backend:

* **Frame Rate:** 12.5 Hz (초당 12.5 프레임으로 LLM autoregressive token prediction 부하 최소화)
* **Sample Rate:** 24,000 Hz
* **Residual Vector Quantization (RVQ):** 32 codebooks (codebook size: 2,048)
* **Approx. Token Rate:** 400.0 tokens/s (32 codebooks 기준)
* **Model Parameters:** ~79.3M
* **Architecture:** Causal convolution & transformer
* **License:** Apache 2.0 (code) / CC-BY 4.0 (model weights)

---

## Dataset Pipeline: AI Hub 경상도 방언 → 울산 방언 음성 데이터 자동 구축

```bash
# Non-destructive metadata inspection
python scripts/inspect_aihub.py --input /path/to/raw_dataset

# Dataset preparation (WAV + JSONL manifests + optional .pt codec tokens)
python scripts/prepare_dataset.py \
  --input /path/to/raw_dataset \
  --output data/ulsan \
  --region ulsan \
  --encode-codec
```

---

## ULM Thinker Adapter

ULM-1.7B (또는 Qwen3-1.7B 기반 모델)에서 semantic hidden states를 추출하기 위한 어댑터입니다.

### Inspect Thinker Model

```bash
python scripts/inspect_thinker.py --model Qwen/Qwen3-1.7B --device cpu
```

Output:

```text
=== ULM Thinker Inspection ===
Model:                 Qwen/Qwen3-1.7B
Hidden size:           2048
Layers:                28
Vocabulary size:       151669
Selected hidden layer: -1
Device:                cpu
Dtype:                 torch.bfloat16

Forward Hidden Test:
  Input tokens:        13
  Hidden state shape:  (1, 13, 2048)
  Status:              OK
```

---

## ULM Talker Prototype

Talker는 Thinker의 semantic hidden state를 prefix conditioning으로 입력받고, 화자(`speaker_id`) 및 방언(`dialect`) 임베딩을 결합하여 오디오 코덱 토큰을 자기회귀적으로 예측합니다.

### 1. Dry-Run Forward and Backward Test

```bash
# Synthetic semantic states test
python scripts/test_talker_forward.py

# Integration test with actual ULM Thinker
python scripts/test_talker_forward.py --thinker Qwen/Qwen3-1.7B
```

Output:

```text
Semantic hidden:       (2, 12, 2048)
Audio codes:           (2, 32, 25)
Speaker ids:           [0, 1]
Dialect ids:           [0, 0]

Talker:
Parameters:            88,336,896 (88.34M)
Hidden size:           512
Layers:                6
Heads:                 8

Output:
Logits shape:          (2, 32, 25, 2048)
Loss:                  7.7227
Backward:              OK
```

### 2. Training Entrypoint

```bash
python scripts/train_talker.py \
  --config configs/talker.yaml \
  --manifest data/ulsan/train.jsonl \
  --thinker Qwen/Qwen3-1.7B \
  --dry-run
```

---

## Installation

Requires Python 3.11+.

```bash
git clone https://github.com/UlsanLM-LAB/ULM-LIVE.git
cd ULM-LIVE

# Create and activate virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -e .
```

---

## Running Tests

Run all unit tests (fast, synthetic fixtures):

```bash
pytest
```

Run integration tests (evaluates real Mimi neural codec and Thinker weights):

```bash
pytest -m integration
```

Verify package compilation:

```bash
python -m compileall ulm_live
```

---

## Project Layout

```text
ULM-Live/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── todo.md
├── configs/
│   ├── codec.yaml
│   ├── dataset.yaml
│   └── talker.yaml
├── docs/
│   └── architecture.md
├── ulm_live/
│   ├── __init__.py
│   ├── codec/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── backend.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── aihub.py
│   │   ├── filters.py
│   │   ├── manifest.py
│   │   ├── schema.py
│   │   └── segment.py
│   ├── thinker/
│   │   ├── __init__.py
│   │   └── adapter.py
│   ├── talker/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   └── model.py
│   └── utils/
│       ├── __init__.py
│       └── audio.py
├── scripts/
│   ├── test_codec.py
│   ├── inspect_codec.py
│   ├── inspect_aihub.py
│   ├── prepare_dataset.py
│   ├── inspect_thinker.py
│   ├── test_talker_forward.py
│   └── train_talker.py
├── tests/
│   ├── __init__.py
│   ├── fixtures/
│   ├── test_audio_utils.py
│   ├── test_codec.py
│   ├── test_dataset_schema.py
│   ├── test_filters.py
│   ├── test_manifest.py
│   ├── test_talker.py
│   ├── test_talker_data.py
│   └── test_thinker.py
└── samples/
    ├── .gitkeep
    └── input.wav
```

---

## Data License & Usage Notice

* 본 프로젝트 코드베이스는 **Apache License 2.0**으로 배포됩니다.
* Pretrained Mimi 코덱 가중치는 Kyutai의 **CC-BY 4.0** 라이선스를 따릅니다.
* AI Hub 경상도 방언 데이터셋은 과학기술정보통신부 및 한국지능정보사회진흥원(NIA)의 **AI-Hub 이용약관** 및 승인 조건에 따릅니다.