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

## 3. Phase 3 목표: ULM Thinker Adapter & Talker Prototype & Tensor Flow (Forward → Loss → Backward)
- **목표**:
  `ULM-1.7B (Thinker)`의 hidden states를 조건(conditioning)으로 받아 `ULM Talker`가 음향 코덱 토큰(codec tokens)을 자기회귀적으로 예측하고,
  실제 텐서 레벨에서 `Forward → Loss → Backward`가 완전히 성립하도록 구현.
- **제외 범위**: 실제 고품질 음성 합성, 스트리밍, 마이크 입력, WebSocket, full-duplex, barge-in은 이번 Phase에서 구현하지 않음.

### Phase 3 Data Flow
```text
Text / Dialogue Context
       ↓
ULM-1.7B Thinker (Frozen)
       ↓
Semantic Hidden States [B, S, D_thinker]
       ↓
Semantic Projection [B, S, D_talker]
       │
       ├─────────────────────┐
       │                     │
Speaker Embedding [B, D_spk] │
       │                     │
Dialect Embedding [B, D_dia] │
       │                     │
       └──────────┬──────────┘
                  ▼ (Additive Projection)
       Talker Causal Transformer
                  ↓
       Codec Token Logits [B, K, T, Vocab_size]
                  ↓
       Cross Entropy Loss (Teacher-Forcing Next-Token)
                  ↓
       Loss Backward (Gradients propagated to Talker)
```

### Phase 3 세부 작업 목록
- [ ] **ULM Thinker Adapter (`ulm_live/thinker/`)**:
  - `ulm_live/thinker/adapter.py`: `ULMThinker` 클래스 (load, tokenize, forward_hidden, generate_text)
  - Hugging Face `AutoModel` / `AutoTokenizer` 기반, config/CLI로 모델 경로 지원 (`Qwen/Qwen3-1.7B` or `/path/to/ULM-1.7B`)
  - `hidden_layer: -1` 등 유연한 레이어 선택 및 `freeze_thinker: true` 기본 적용
  - `scripts/inspect_thinker.py`: Thinker 모델 구조 및 hidden state inspection CLI
- [ ] **Speaker / Dialect Conditioning**:
  - 화자 정체성과 방언 정체성을 독립적인 임베딩(`nn.Embedding`)으로 분리
  - 울산 억양을 특정 화자의 음색과 동일시하지 않고 `same speaker + different dialect`, `same dialect + different speaker` 지원
  - Additive linear projection 방식으로 토큰 시퀀스에 안정적 주입
- [ ] **Talker Prototype Architecture (`ulm_live/talker/`)**:
  - `configs/talker.yaml`: Talker 하이퍼파라미터 설정
  - `ulm_live/talker/model.py`: `ULMTalker`, `TalkerConfig`, `TalkerOutput`
  - Semantic projection (`Linear(semantic_dim, talker_dim)`)
  - Audio token embedding: K개 codebook 합산 임베딩 (`Embedding(codebook_size + 1, talker_dim)`)
  - Causal transformer: PyTorch causal masked transformer encoder
  - Output head: Multi-codebook independent linear heads (`ModuleList([Linear(dim, vocab_size) for _ in range(K)])`)
  - Loss: Cross-entropy loss with padding ignore
- [ ] **Training Dataset & Collator (`ulm_live/talker/data.py`)**:
  - Phase 2 `manifest.jsonl` 기반 `TalkerDataset` 및 `TalkerCollator`
  - 재현 가능한 결정론적 speaker ID / dialect ID 어휘 매핑 생성 (`speaker2id.json`, `dialect2id.json`)
  - `.pt` 토큰 직접 로드 및 배치 패딩, shift-by-1 타겟 생성
- [ ] **CLI & Scripts**:
  - `scripts/inspect_thinker.py`: 모델 스펙 및 히든 레이어 조사
  - `scripts/test_talker_forward.py`: Dry-run 테스트 및 텐서 플로우 출력 규격 준수
  - `scripts/train_talker.py`: 경량 훈련 진입점 (fp16/bf16, grad accumulation, small batch)
- [ ] **Tests (`tests/`)**:
  - `tests/test_thinker.py`: Thinker 어댑터 단위 테스트 (가중치 없을 때 fallback mock, 로컬 캐시 모델 통합 테스트)
  - `tests/test_talker.py`: Talker 모델 forward, backward, shape 검증, conditioning, loss, padding
  - `tests/test_talker_data.py`: Dataset, Collator, ID 매핑 안정성
- [ ] **Docs & Architecture Record**:
  - `docs/architecture.md`: Thinker-Talker 구조 설명, TTS 대비 차별점, semantic vs acoustic, multi-codebook 예측 기법 비교 및 선택 근거
  - `README.md`: Phase 3 로드맵 업데이트 및 실행 가이드

## 4. Phase 3 완료 기준 (Completion Criteria)
1. `python -m compileall ulm_live` 무결성 통과
2. `pytest` 기존 Phase 1/2 및 Phase 3 신규 테스트 전원 통과
3. `python scripts/test_talker_forward.py` 실행 시:
   - Semantic hidden states, audio codes, speaker ids, dialect ids 입력
   - Talker forward -> logits `[B, K, T, Vocab]` 출력
   - Cross-entropy loss 계산
   - `loss.backward()` 성공 확인
4. 실제 ULM-1.7B 가중치 유무에 따른 명확한 구분 및 통합 테스트
5. 논리적인 Git commits 분할

## 5. Phase 4 로드맵 (Next Steps)
- 실제 Talker 가중치 사전학습 및 사투리 억양 파인튜닝
- Causal KV-cache를 결합한 autoregressive decoding 루프
- Mimi streaming decoder 연계 실시간 오디오 생성
- WebSocket full-duplex 통신 서버 및 barge-in 인터럽션
