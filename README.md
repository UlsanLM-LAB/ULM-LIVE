# ULM-Live

Real-time Spoken Language Model runtime and neural audio codec foundations for Ulsan dialect speech generation.

## Architecture Vision

```text
Microphone
   ↓
Audio Tokenizer / Codec (Mimi, 12.5 Hz)
   ↓
ULM-1.7B Thinker
   ↓
ULM Talker
   ↓
Audio Codec Tokens
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
[ ] ULM-1.7B integration
[ ] Talker
[ ] Streaming
[ ] Full duplex
```

## Neural Audio Codec: Mimi

ULM-Live adopts **Mimi** (`kyutai/mimi`) as its primary neural audio codec backend:

* **Frame Rate:** 12.5 Hz (초당 12.5 프레임으로 LLM autoregressive token prediction 부하 최소화)
* **Sample Rate:** 24,000 Hz
* **Residual Vector Quantization (RVQ):** 32 codebooks (codebook size: 2,048)
* **Approx. Token Rate:** 400.0 tokens/s (32 codebooks 기준)
* **Model Parameters:** ~79.3M
* **Architecture:** Causal convolution & transformer (향후 streaming chunk 추론 지원)
* **License:** Apache 2.0 (code) / CC-BY 4.0 (model weights)

## Dataset Pipeline: AI Hub 경상도 방언 → 울산 방언 음성 데이터 자동 구축

AI Hub 경상도 방언 원천/라벨링 데이터로부터 울산 지역 화자를 자동 판별하고, 오디오 세그멘테이션, 품질 필터링, 데이터 분할(train/val/test), manifest 생성 및 선택적 신경망 코덱 토큰 인코딩을 수행합니다.

### 1. Non-destructive Metadata Inspection

데이터셋을 수정하지 않고 파일 구조, 감지된 메타데이터 필드, 지역 분포를 검사합니다.

```bash
python scripts/inspect_aihub.py --input /path/to/raw_dataset
```

출력 예:

```text
=== AI Hub Dataset Inspection ===
Files discovered:
  JSON files:  1240
  Audio files: 1240

Detected metadata fields:
  Speaker fields:    id, speaker, speaker_id
  Region fields:     birthplace, current_residence, principal_residence
  Transcript fields: dialect_form, standard_form

Regions found:
  ulsan: 320 speaker records
  busan: 540 speaker records
  daegu: 380 speaker records

Scanned 500 JSON files: 1240 unique speakers, 18500 utterances.
```

### 2. Dataset Preparation Pipeline

울산 지역 화자 필터링, 24kHz 모노 변환, 피크 정규화, 음성 세그멘테이션 및 manifest 생성을 수행합니다.

```bash
# Basic preparation (WAV + JSONL manifests)
python scripts/prepare_dataset.py \
  --input /path/to/raw_dataset \
  --output data/ulsan \
  --region ulsan

# Optional: Encode discrete neural codec tokens (.pt) together
python scripts/prepare_dataset.py \
  --input /path/to/raw_dataset \
  --output data/ulsan \
  --region ulsan \
  --encode-codec
```

출력 디렉터리 구조:

```text
data/ulsan/
├── manifest.jsonl
├── train.jsonl
├── val.jsonl
├── test.jsonl
├── summary.json
├── audio/
│   ├── ulsan_000000.wav
│   └── ...
└── codec/               (optional with --encode-codec)
    ├── ulsan_000000.pt  (PyTorch Tensor [1, 32, frames], dtype=int64)
    └── ...
```

### 3. Manifest Schema

`manifest.jsonl` 각 행은 다음 JSON 구조를 가집니다:

```json
{
  "id": "ulsan_000001",
  "audio_path": "audio/ulsan_000001.wav",
  "text": "와 이래 덥노 오늘",
  "speaker_id": "speaker_001",
  "dialect": "ulsan",
  "sample_rate": 24000,
  "duration": 3.82,
  "source": "aihub",
  "codec_path": "codec/ulsan_000001.pt",
  "metadata": {
    "orig_sample_rate": 48000,
    "orig_duration": 3.82,
    "silence_ratio": 0.05
  }
}
```

* **Speaker-Disjoint Split**: 화자별 데이터 누수(speaker leakage)를 방지하기 위해 기본적으로 화자 단위(`split_strategy: speaker`)로 train/val/test를 분할합니다.
* **Dialect vs Speaker Separation**: 방언 정체성(`dialect: ulsan`)과 개인 화자 정체성(`speaker_id`)을 분리하여 모델 조건화(conditioning)를 유연하게 지원합니다.

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

## Quick Start (Codec Testing)

```bash
python scripts/test_codec.py \
  --input samples/input.wav \
  --output outputs/reconstructed.wav
```

Output example:

```text
Input
sample rate: 24000
channels: 1
duration: 2.00s

Codec
backend: mimi
representation shape: (1, 32, 25)
approx token rate: 400.0 tokens/s
encode time: 0.0215s
decode time: 0.0284s

Output
sample rate: 24000
duration: 2.00s
path: outputs/reconstructed.wav
```

## Running Tests

Run all unit tests (fast, synthetic fixtures):

```bash
pytest
```

Run integration tests (requires downloading/evaluating Mimi neural codec weights):

```bash
pytest -m integration
```

Verify compilation:

```bash
python -m compileall ulm_live
```

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
│   └── dataset.yaml
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
│   └── utils/
│       ├── __init__.py
│       └── audio.py
├── scripts/
│   ├── test_codec.py
│   ├── inspect_codec.py
│   ├── inspect_aihub.py
│   └── prepare_dataset.py
├── tests/
│   ├── __init__.py
│   ├── test_audio_utils.py
│   ├── test_codec.py
│   ├── test_dataset_schema.py
│   ├── test_filters.py
│   └── test_manifest.py
└── samples/
    ├── .gitkeep
    └── input.wav
```

## Data License & Usage Notice

* 본 프로젝트 코드베이스는 **Apache License 2.0**으로 배포됩니다.
* Pretrained Mimi 코덱 가중치는 Kyutai의 **CC-BY 4.0** 라이선스를 따릅니다.
* AI Hub 경상도 방언 데이터셋은 과학기술정보통신부 및 한국지능정보사회진흥원(NIA)의 **AI-Hub 이용약관** 및 신청 승인 조건에 따릅니다. 데이터셋을 활용할 때는 해당 라이선스 및 사용 허가 범위를 반드시 준수해야 합니다.