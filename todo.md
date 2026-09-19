# ULM-Live Phase 1, Phase 2 & Phase 3 TODO & Progress

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
- [x] **ULM Thinker Adapter (`ulm_live/thinker/adapter.py`)**:
  - `ULMThinker` 클래스 구현: Hugging Face `AutoModel` 및 `AutoTokenizer` 래핑
  - `hidden_layer: -1` 레이어 선택 지원 및 `freeze: True` 기본 적용
  - `forward_hidden`으로 `(B, S, 2048)` 시맨틱 히든 상태 추출
  - `scripts/inspect_thinker.py` CLI 구현 (Model, Hidden size, Layers, Vocab, Dtype 출력)
- [x] **Speaker / Dialect Conditioning**:
  - `nn.Embedding(num_speakers, 128)` 및 `nn.Embedding(num_dialects, 64)` 독립 분리
  - 선형 사영 후 Additive Projection 방식으로 오디오 토큰 임베딩에 주입
  - "Same speaker + Different dialect", "Same dialect + Different speaker" 독립 제어 검증
- [x] **Talker Causal Transformer Architecture (`ulm_live/talker/model.py`)**:
  - `TalkerConfig`, `TalkerOutput`, `ULMTalker` 구현
  - Semantic Projection: `Linear(2048, 512)`
  - Audio Token Embedding: 32개 코드북 합산 임베딩 (`ModuleList([Embedding(2049, 512) for _ in range(32)])`)
  - Causal Mask: Semantic prefix 양방향 어텐션 + Audio tokens 인과적 어텐션 결합
  - Causal Transformer Encoder: 6 layers, 8 heads, 2048 FFN dimension (~88.34M params)
  - Codec Token Prediction Head: 32개 코드북 독립 선형 분류 헤드 (`ModuleList([Linear(512, 2048) for _ in range(32)])`)
  - Cross-Entropy Loss: Codebook 평균 및 `pad_token_id=2048` 마스킹
- [x] **Training Dataset & Collator (`ulm_live/talker/data.py`)**:
  - Phase 2 `manifest.jsonl` 기반 `TalkerDataset`
  - 결정론적/재현 가능한 화자 및 방언 어휘 매핑 (`build_id_mappings`)
  - `TalkerCollator`: 오디오 토큰 패딩 및 teacher-forcing next-token shifted target 생성
- [x] **실행 스크립트 및 검증 파이프라인**:
  - `scripts/test_talker_forward.py`: Dry-run 테스트 스크립트 구현 및 규격 출력
    ```text
    Semantic hidden: (2, 12, 2048)
    Audio codes:     (2, 32, 25)
    Talker params:   88,336,896 (88.34M)
    Logits shape:    (2, 32, 25, 2048)
    Loss:            7.7227
    Backward:        OK
    ```
  - `scripts/train_talker.py`: 훈련 진입점 스크립트 (AMP, grad accum, dry-run 지원)
- [x] **테스트 및 아키텍처 문서화**:
  - `tests/test_thinker.py`: Thinker 단위 및 실제 모델 통합 테스트
  - `tests/test_talker.py`: Talker forward, backward, shape, conditioning, padding 테스트
  - `tests/test_talker_data.py`: Dataset, Collator, ID 매핑 안정성 테스트
  - `pytest`: 34 passed, 3 deselected in 4.99s
  - `pytest -m integration`: 3 passed in 7.24s (Mimi 코덱 + Qwen3-1.7B Thinker)
  - `docs/architecture.md`: Thinker-Talker 구조 설명, TTS 대비 차별점, semantic vs acoustic, multi-codebook 예측 기법 비교 및 선택 근거 작성
  - `README.md`: Phase 3 로드맵 및 실행 가이드 갱신

## 4. Phase 4 로드맵 (Next Steps)
- 실제 Talker 가중치 사전학습 및 사투리 억양 파인튜닝
- Causal KV-cache를 결합한 autoregressive decoding 루프
- Mimi streaming decoder 연계 실시간 오디오 생성
- WebSocket full-duplex 통신 서버 및 barge-in 인터럽션
