# ULM-Live Phase 1 & Phase 2 TODO & Progress

## 1. Phase 1 완료 내역
- [x] Spoken Language Model(ULM-Live)의 기반이 되는 오디오 코덱 인터페이스 및 Mimi 백엔드 구현
- [x] **WAV → Neural Codec Encode → Discrete Codec Tokens → Decode → Reconstructed WAV** 실제 동작 검증
- [x] 단위 테스트 및 통합 테스트 통과, CLI 리포트 출력 준수

## 2. Phase 2 목표: AI Hub 경상도 방언 데이터에서 울산 음성 학습 데이터 자동 생성 파이프라인
- **목표**: AI Hub 경상도 방언 원천/라벨링 데이터로부터 울산 지역 화자를 자동 선별하고, 음성 세그멘테이션, 품질 필터링, 데이터 분할(train/val/test), manifest 생성 및 선택적 신경망 코덱 사전 인코딩 파이프라인 구축.
- **제외 범위**: Talker 학습, ULM-1.7B 연결, 실시간 WebSocket / full-duplex는 이번 단계에서 구현하지 않음.

### 데이터 파이프라인 흐름
```text
AI Hub raw data (JSON metadata + WAV audio)
     ↓
metadata scan & flexible field detection (inspect_aihub.py)
     ↓
Ulsan speaker filtering (configurable region priority & aliases)
     ↓
audio segmentation (start/end timestamp slicing, resample to 24kHz mono)
     ↓
quality filtering (duration, silence, sample rate, text length)
     ↓
manifest generation (JSONL: manifest.jsonl, train/val/test splits, summary.json)
     ↓
optional codec encoding (Phase 1 MimiCodec -> .pt / .safetensors tokens)
     ↓
ULM-Live dataset
```

## 3. Phase 2 구현 파일 목록
- [ ] `configs/dataset.yaml`: 필드 매핑, 지역 별칭(aliases), 지역 우선순위(priority), 품질 필터, split 설정
- [ ] `ulm_live/data/schema.py`: 데이터 모델 dataclass (`SpeakerInfo`, `UtteranceInfo`, `DatasetItem`, `DatasetSummary`)
- [ ] `ulm_live/data/aihub.py`: AI Hub 메타데이터 스캐너, 유연한 필드 감지기 및 화자 지역 매칭 엔진
- [ ] `ulm_live/data/filters.py`: 지속시간(duration), 침묵 비율(silence ratio), 텍스트 길이 등 품질 필터
- [ ] `ulm_live/data/segment.py`: 오디오 세그멘테이션(타임스탬프 기반 슬라이싱, 24kHz 모노 변환, peak normalize) 및 코덱 토큰 인코딩
- [ ] `ulm_live/data/manifest.py`: manifest JSONL 직렬화, speaker leakage 방지 split 생성, 통계 요약기
- [ ] `ulm_live/data/__init__.py`: data 모듈 export
- [ ] `scripts/inspect_aihub.py`: AI Hub 데이터셋 구조 비파괴 조사 CLI
- [ ] `scripts/prepare_dataset.py`: 울산 데이터셋 자동 생성 파이프라인 CLI (`--encode-codec` 옵션 포함)
- [ ] `tests/test_dataset_schema.py`: 스키마 및 매핑 계층 단위 테스트
- [ ] `tests/test_filters.py`: 품질 필터 단위 테스트
- [ ] `tests/test_manifest.py`: 오디오 세그멘테이션, manifest 생성 및 split 단위/통합 테스트
- [ ] `README.md`: Phase 2 사용 가이드, manifest 스키마, 데이터 라이선스 고지

## 4. Phase 2 핵심 설계 원칙
1. **Configurable Mapping Layer**: AI Hub 경상도 방언의 다양한 JSON 스키마 변종(단일/복수 화자, 필드명 차이 등)을 하드코딩하지 않고 yaml 매핑 레이어로 유연하게 처리.
2. **화자 정체성과 방언 정체성의 분리**:
   - `speaker_id` (화자 고유 ID) vs `dialect` ("ulsan")
   - 향후 ULM-1.7B 및 Talker 모델의 conditioning (`speaker_id` + `dialect`) 지원.
3. **Speaker Leakage 방지 Split**:
   - 동일 화자의 발화가 train과 test/val에 섞이지 않도록 화자 단위(`split_strategy: speaker`) 분할을 기본 지원.
4. **선택적 코덱 인코딩 (Optional Codec Encoding)**:
   - 기본은 고품질 24kHz WAV + JSONL manifest 생성.
   - `--encode-codec` 플래그 지정 시 Phase 1 `MimiCodec`을 활용하여 이산 토큰(`codes: [1, 32, frames]`)을 `.pt`로 저장하고 `codec_path` 기록.

## 5. 성공 기준
- `python -m compileall ulm_live` 통과
- `pytest` 기존 Phase 1 테스트 및 신규 Phase 2 테스트 전원 통과
- 합성 데이터셋에 대한 `scripts/inspect_aihub.py` 및 `scripts/prepare_dataset.py` 동작 검증
- 실제 AI Hub 원본 데이터 유무를 정직하게 명시하고 mock fake success 금지
- 논리적 단위의 git commit 완료
