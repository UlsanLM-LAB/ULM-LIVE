# ULM-Live

ULM-1.7B text responses with Korean speech from Qwen3-TTS. The earlier Mimi Talker experiments remain in the repository as frozen research.

## ULM Live v2

The current speech path is `prompt → ULM-1.7B response text → Qwen3-TTS Sohee → 24 kHz WAV`. See [TTS pivot research](docs/research/ULM_LIVE_TTS_PIVOT.md) and [migration results](reports/ULM_LIVE_V2_TTS_MIGRATION.md).

On the EC2 host, install the optional `tts` dependencies into a separate Python 3.12 environment. Keep model weights in its persistent Hugging Face cache. Start one worker, bound to loopback unless a trusted reverse proxy is configured:

```bash
uv venv --python 3.12 .venv-tts
uv pip install --python .venv-tts/bin/python '.[tts]'
export LD_LIBRARY_PATH="$PWD/.venv-tts/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
.venv-tts/bin/uvicorn ulm_live.tts_service:app --host 127.0.0.1 --port 8000 --workers 1
```

The library path keeps the TTS environment's cuDNN ahead of the DLAMI host CUDA libraries on the tested EC2 image.

`POST /v1/audio/speech` accepts `{"input":"안녕하세요"}` and returns `audio/wav`. `POST /v1/chat/speech` accepts `{"prompt":"울산을 소개해 줘"}` and returns a WAV of the ULM answer. Its `X-ULM-Text` header contains the UTF-8 answer encoded as hex. `GET /health` reports readiness after both models load. GPU calls are serialized in one process.

All synthesis and evaluation WAVs stay on EC2. Run `scripts/eval_tts_v2.py --stage zero-shot` or `--stage e2e` with `--output-dir` under `outputs/` there.

## Frozen Mimi Talker research

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
[x] Codec token generation
[x] WAV synthesis pipeline
[ ] Train Talker on Ulsan speech
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

## ULM Talker (pre-full-training architecture)

Talker는 Thinker hidden state를 prefix conditioning으로 받고, cached Temporal Transformer와 RVQ Depth Transformer (`q0 → q1 → …`)로 Mimi 토큰을 예측합니다. BOS/PAD, learned stop, semantic padding mask, 8/16/32 quantizer, acoustic delay 실험을 지원합니다. 아직 본격 Talker 학습 및 음질 검증은 수행되지 않았습니다.

현재 기본 16-quantizer 구성은 40,706,817 parameters입니다 (8q: 32,307,969; 32q: 57,504,513). 16q breakdown은 semantic projection 1,049,088, speaker/dialect 133,120, temporal 19,963,904, depth 1,715,456, codec embeddings 17,318,144, shared codec output 526,336, stop head 513, 기타 256입니다. 이전 32-independent-head prototype은 88,336,896 parameters였으며 해당 체크포인트는 호환되지 않습니다.

### 1. Dry-Run Forward and Backward Test

```bash
# Synthetic states are architecture tests only (never production training).
python scripts/train_talker.py --manifest tests/fixtures/manifest.jsonl \
  --allow-synthetic-thinker --dry-run

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
Parameters:            see the current parameter breakdown below
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

Cache-first training avoids forwarding the frozen 1.7B Thinker every epoch:

```bash
python scripts/cache_thinker_hidden.py \
  --manifest data/ulsan/train.jsonl --thinker /path/to/compatible-thinker \
  --output-dir data/ulsan/semantic_cache --output-manifest data/ulsan/train.cached.jsonl

python scripts/train_talker.py --manifest data/ulsan/train.cached.jsonl \
  --semantic-cache-thinker /path/to/compatible-thinker --bf16
```

The default split is session/source-clip disjoint while retaining every validation/test speaker in train, because current conditioning uses learned speaker IDs. Zero-shot voice cloning is not implemented. See [architecture details](docs/architecture.md).

---

## Offline Speech Generation Pipeline (Phase 4)

ULM Thinker의 semantic hidden state를 입력받아 Talker가 자기회귀적(autoregressive)으로 discrete 오디오 코덱 토큰을 생성하고, Phase 1 Mimi 코덱 디코더를 통해 WAV 파일로 합성하는 엔드투엔드 파이프라인입니다.

> [!NOTE]
> **Audio Quality Notice (Untrained Talker)**
> Phase 4는 음성 합성 파이프라인의 **텐서 및 아키텍처 무결성 검증**(`Talker.generate() → MimiCodec.decode() → WAV`)을 위한 단계입니다. 학습된 체크포인트가 없는 상태(Random init)에서 생성된 오디오는 백색 잡음(white noise) 형태로 출력되며, 이는 정상적인 파이프라인 테스트 결과입니다 (`Audio quality: UNTRAINED / PIPELINE TEST ONLY`).

### 1. Synthetic Semantic States 기반 빠른 파이프라인 검증

```bash
python scripts/test_generation.py --output outputs/test_generation.wav --frames 25
```

Output:
```text
=== Talker Generation & Codec Decode Test ===
Frames:                25 (2.00s audio at 12.5 Hz)
Device:                cuda
Generated codes:       torch.Size([1, 32, 25]) (min=2, max=2047)
Waveform:              torch.Size([1, 1, 48000]) (finite=True)
WAV saved:             outputs/test_generation.wav
Synthesis time:        0.65s (Talker: 0.32s, Codec Decode: 0.33s)
RTF:                   0.32
Pipeline status:       OK (Integrity verified)
```

### 2. End-to-End Text-to-Speech CLI

```bash
# 기본 실행 (Thinker semantic state 추출 → Talker 생성 → Mimi 디코딩 → WAV 저장)
python scripts/generate_audio.py \
  --text "밥 묵었나? 어디 가노?" \
  --speaker ulsan_spk_01 \
  --dialect ulsan \
  --output outputs/generated.wav \
  --max-seconds 2.0
```

주요 CLI 옵션:
* `--thinker`: HuggingFace 모델 식별자 또는 로컬 디렉토리 경로 (기본값: `Qwen/Qwen3-1.7B`)
* `--talker`: 학습된 Talker 체크포인트 (`.pt`) 파일 경로 (미지정 시 random init 모델 사용)
* `--text`: 합성할 입력 텍스트 프롬프트
* `--speaker`: 화자 식별자 (기본값: `default`)
* `--dialect`: 방언 식별자 (기본값: `ulsan`)
* `--output`: 출력 WAV 파일 경로 (기본값: `outputs/generated.wav`)
* `--max-seconds`: 최대 생성 오디오 길이(초) (기본값: `2.0`, 프레임 수: `max_seconds * 12.5`)
* `--temperature`: 샘플링 온도 (기본값: `1.0`, `0.0`일 경우 greedy decoding)
* `--top-k`: Top-K 필터링 (기본값: `50`)
* `--top-p`: Top-P (nucleus) 필터링 (기본값: `0.9`)
* `--greedy`: Greedy argmax 디코딩 플래그
* `--device`: 실행 디바이스 (`cuda` 또는 `cpu`, 기본값: 자동 감지)

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
│   │   ├── checkpoint.py
│   │   ├── data.py
│   │   ├── generator.py
│   │   ├── model.py
│   │   └── synthesizer.py
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
│   ├── train_talker.py
│   ├── test_generation.py
│   └── generate_audio.py
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
│   ├── test_talker_generation.py
│   ├── test_speech_synthesizer.py
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
