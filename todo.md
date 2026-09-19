# ULM-Live Phase 1, Phase 2, Phase 3 & Phase 4 TODO & Progress

## 1. Phase 1 완료 내역
- [x] Spoken Language Model(ULM-Live)의 기반이 되는 오디오 코덱 인터페이스 및 Mimi 백엔드 구현
- [x] **WAV → Neural Codec Encode → Discrete Codec Tokens → Decode → Reconstructed WAV** 실제 동작 검증
- [x] 단위 테스트 및 통합 테스트 통과, CLI 리포트 출력 준수

## 2. Phase 2 완료 내역: AI Hub 경상도 방언 데이터에서 울산 음성 학습 데이터 자동 생성 파이프라인
- [x] AI Hub 메타데이터 스캔 & 비파괴 조사 CLI (`scripts/inspect_aihub.py`)
- [x] Configurable Mapping Layer (`configs/dataset.yaml`)
- [x] 울산 방언 화자 판별 및 필터링 엔진 (`RegionClassifier`, `FieldMatcher`)
- [x] 오디오 세그멘테이션 및 24kHz 모노 정규화 (`AudioSegmenter`)
- [x] 품질 게이트 (`QualityFilter`: 지속시간, RMS 침묵 비율, 샘플레이트, 텍스트 정제)
- [x] Manifest 직렬화 및 Speaker-disjoint split (`manifest.jsonl`, `summary.json`)
- [x] 선택적 신경망 코덱 인코딩 (`--encode-codec` -> `.pt`)

## 3. Phase 3 완료 내역: ULM Thinker Adapter & Talker Prototype & Tensor Flow (Forward → Loss → Backward)
- [x] ULM Thinker Adapter (`ulm_live/thinker/adapter.py`, `scripts/inspect_thinker.py`)
- [x] Speaker / Dialect Conditioning (`nn.Embedding` + Additive Linear Projection)
- [x] Talker Causal Transformer Backbone (`ulm_live/talker/model.py`, ~88.34M params)
- [x] Codec Token Prediction Head (32개 독립 선형 헤드 per codebook)
- [x] Training Dataset & Collator (`ulm_live/talker/data.py`, teacher-forcing shift-by-1)
- [x] Dry-run forward & backward 파이프라인 검증 (`scripts/test_talker_forward.py`, `scripts/train_talker.py`)
- [x] Thinker-Talker 아키텍처 문서화 (`docs/architecture.md`)

## 4. Phase 4 목표: Talker Autoregressive Generation & Codec Decoder Synthesis (WAV 출력)
- **목표**:
  ULM semantic hidden states를 conditioning으로 받아 Talker가 자기회귀적(autoregressive)으로 multi-codebook 오디오 코덱 토큰을 생성하고, 이를 Phase 1 신경망 코덱 디코더에 직접 전달하여 유효한 WAV 파일(`output.wav`)을 복원하는 End-to-End 추론 파이프라인 구축.
- **성공 기준**:
  음질 튜닝이나 대규모 학습이 아닌, **`Talker.generate() → valid codec token sequence → codec decoder → valid WAV`** 데이터 흐름의 완전한 성립. (랜덤 초기화 모델의 경우 생성 음성이 잡음이어도 파이프라인 검증 성공으로 판정하되 사실을 정직하게 명시)
- **제외 범위**: 대규모 모델 훈련, 실시간 WebSocket 통신, 마이크 스트리밍 입력, full-duplex, barge-in 인터럽션.

### Phase 4 Data Flow
```text
Text Prompt
     ↓
ULM Thinker (Frozen)
     ↓
Semantic Hidden States [1, S, 2048]
     ↓
Talker.generate() (Autoregressive step-by-step decoding)
     - Prefix-causal attention
     - Speaker ID & Dialect ID conditioning
     - Greedy / Temperature / Top-k / Top-p sampling
     - Multi-codebook (32 RVQ codebooks) alignment
     ↓
Discrete Codec Tokens [1, 32, T] (0 <= token < 2048)
     ↓
Phase 1 Mimi Codec Decoder (EncodedAudio)
     ↓
Reconstructed Waveform [1, 1, samples] (24kHz Mono, Finite float32)
     ↓
Generated WAV File (outputs/generated.wav)
```

### Phase 4 세부 작업 목록
- [x] **Talker Autoregressive Generation (`ulm_live/talker/model.py`, `generator.py`)**:
  - `TalkerGenerationConfig`: `max_new_tokens`, `temperature`, `top_k`, `top_p`, `do_sample`, `max_audio_seconds`
  - `ULMTalker.generate()`: 자기회귀적 추론 루프 구현 (기존 causal mask 재사용, step-by-step next token 예측)
  - `sample_next_tokens()`: greedy argmax 및 stochastic sampling (temperature scaling, top-k, top-p filtering)
- [x] **Multi-Codebook Codec Token 정렬 및 특수 토큰 처리**:
  - 생성된 32개 코드북 토큰 `[B, 32, T]`이 Phase 1 `MimiCodec`의 유효 어휘 범위(`0 <= token < 2048`) 내에 존재하도록 보장
  - 시작 더미 토큰과 생성 토큰 정렬
- [x] **Talker Checkpoint 입출력 (`ulm_live/talker/checkpoint.py`)**:
  - `save_talker_checkpoint()` & `load_talker_checkpoint()`
  - `model_state_dict`, `TalkerConfig`, `speaker2id`, `dialect2id`, `codec_metadata` 패키징
- [x] **End-to-End Speech Synthesis Engine (`ulm_live/talker/synthesizer.py`)**:
  - `SpeechSynthesizer`: Thinker, Talker, Codec을 연결하는 고수준 합성 인터페이스
  - `SpeechGenerationResult`: `codec_tokens`, `waveform`, `sample_rate`, `duration`, `timings`, `rtf`
  - 시간 측정(Thinker, Talker, Decoder 시간) 및 Real-Time Factor(RTF) 산출
- [x] **CLI & 스크립트**:
  - `scripts/test_generation.py`: 가상 semantic states 기반 Talker generate → Codec decode → WAV 저장 빠른 테스트
  - `scripts/generate_audio.py`: 텍스트 프롬프트 기반 음성 생성 CLI (미학습 모델 경고 출력, 상세 통계 리포트 출력)
- [x] **Tests (`tests/`)**:
  - `tests/test_talker_generation.py`: greedy, sampling, max tokens, output shape, token range, deterministic behavior
  - `tests/test_speech_synthesizer.py`: checkpoint save/load, synthesizer pipeline, WAV finite value / duration validation
- [x] **Docs & Architecture Record**:
  - `docs/architecture.md`: 훈련(Training)과 추론(Generation)의 차이, 자기회귀 코덱 생성 원리, 디코더 결합, 미학습/학습 모델 차이, 향후 KV-cache TODO 문서화
  - `README.md`: Phase 4 로드맵 갱신 및 `generate_audio.py` 실행 가이드 추가

## 5. Phase 5 로드맵 (Next Steps)
- Talker 모델 사전학습 및 사투리 억양 파인튜닝
- Causal KV-cache를 통한 생성 지연시간 및 연산량 최적화
- Mimi streaming decoder 연동 청크 단위 실시간 오디오 송출
- WebSocket full-duplex 실시간 양방향 음성 대화 서버 구축
