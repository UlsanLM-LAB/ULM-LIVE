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
[ ] Ulsan dataset pipeline
[ ] ULM-1.7B integration
[ ] Talker
[ ] Streaming
[ ] Full duplex
```

## Neural Audio Codec: Mimi

ULM-Live Phase 1 adopts **Mimi** (`kyutai/mimi`) as its primary neural audio codec backend:

* **Frame Rate:** 12.5 Hz (초당 12.5 프레임으로 LLM autoregressive token prediction 부하 최소화)
* **Sample Rate:** 24,000 Hz
* **Residual Vector Quantization (RVQ):** 32 codebooks (codebook size: 2,048)
* **Approx. Token Rate:** 400.0 tokens/s (32 codebooks 기준)
* **Model Parameters:** ~79.3M
* **Architecture:** Causal convolution & transformer (향후 streaming chunk 추론 지원)
* **License:** Apache 2.0 (code) / CC-BY 4.0 (model weights)

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

## Quick Start

### 1. Reconstruct Audio (WAV → Codec Tokens → WAV)

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

### 2. Inspect Codec Parameters

```bash
python scripts/inspect_codec.py
```

Output:

```text
=== Neural Audio Codec Inspection ===
Backend:            mimi
Device:             cpu
Sample Rate:        24000 Hz
Frame Rate:         12.50 Hz
Quantizers (RVQ):   32
Approx Token Rate:  400.0 tokens/s
Parameter Count:    79,308,609 (79.31M)
Codebook Size:      2048
Codebook Dim:       256
Hidden Size:        512
```

## Running Tests

Run unit tests (fast, no model weights required):

```bash
pytest
```

Run integration tests (evaluates real Mimi neural codec encode/decode):

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
│   └── codec.yaml
├── ulm_live/
│   ├── __init__.py
│   ├── codec/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── backend.py
│   └── utils/
│       ├── __init__.py
│       └── audio.py
├── scripts/
│   ├── test_codec.py
│   └── inspect_codec.py
├── tests/
│   ├── __init__.py
│   ├── test_audio_utils.py
│   └── test_codec.py
└── samples/
    ├── .gitkeep
    └── input.wav
```

## License

Apache License 2.0. Pretrained Mimi weights are subject to CC-BY 4.0 by Kyutai.