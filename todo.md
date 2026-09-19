# ULM-Live Phase 1 & Phase 2 TODO & Progress

## 1. Phase 1 완료 내역
- [x] Spoken Language Model(ULM-Live)의 기반이 되는 오디오 코덱 인터페이스 및 Mimi 백엔드 구현
- [x] **WAV → Neural Codec Encode → Discrete Codec Tokens → Decode → Reconstructed WAV** 실제 동작 검증
- [x] 단위 테스트 및 통합 테스트 통과, CLI 리포트 출력 준수

## 2. Phase 2 완료 내역: AI Hub 경상도 방언 데이터에서 울산 음성 학습 데이터 자동 생성 파이프라인
- [x] **AI Hub 메타데이터 스캔 & 비파괴 조사 CLI**:
  - `scripts/inspect_aihub.py` 구현
  - JSON 및 오디오 파일 수, 메타데이터 필드, 지역 분포 통계 추출
- [x] **Configurable Mapping Layer**:
  - `configs/dataset.yaml` 작성
  - 지역 우선순위(`region_priority`), 지역 별칭(`region_aliases`), 스키마 필드 후보군(`field_mappings`)
- [x] **울산 방언 화자 판별 및 필터링 엔진**:
  - `RegionClassifier`, `FieldMatcher`, `AIHubParser`
  - 화자 정체성(`speaker_id`)과 방언 정체성(`dialect: ulsan`)의 명확한 분리
- [x] **오디오 세그멘테이션 및 정규화 파이프라인**:
  - `AudioSegmenter`, `slice_waveform`, `match_audio_file`
  - 발화 타임스탬프 슬라이싱, 24kHz 리샘플링, 모노 변환, 피크 정규화(0.95)
- [x] **품질 게이트 (QualityFilter)**:
  - 지속시간(1.0s ~ 20.0s), RMS 기반 침묵 비율(<= 0.6), 샘플레이트(>= 16kHz), 텍스트 길이 및 AI Hub 노이즈 마크 전처리
- [x] **Manifest 직렬화 및 Split 생성**:
  - JSONL 규격: `manifest.jsonl`, `train.jsonl`, `val.jsonl`, `test.jsonl`
  - 데이터셋 통계 요약: `summary.json`
  - Speaker-disjoint split 기본 지원 (동일 화자 발화가 train/val/test 간 누수되지 않도록 방지)
- [x] **선택적 신경망 코덱 인코딩 (`--encode-codec`)**:
  - Phase 1 `MimiCodec`을 활용하여 이산 토큰(`codes: [1, 32, frames]`, dtype=int64)을 `.pt`로 저장하고 manifest에 `codec_path` 기록
- [x] **테스트 및 검증**:
  - `tests/test_dataset_schema.py`: 스키마 및 매핑 계층 단위 테스트
  - `tests/test_filters.py`: 품질 필터 및 침묵 비율 계산 단위 테스트
  - `tests/test_manifest.py`: 세그멘테이션, 직렬화, 화자 분할 누수 방지 테스트
  - `pytest`: 25 passed, 2 deselected in 1.05s
  - `pytest -m integration`: 2 passed in 4.18s
  - 합성 데이터셋 엔드투엔드 CLI 구동 및 `.pt` 토큰 로드 검증

## 3. 구현된 파일 목록
- [x] `configs/dataset.yaml`: 데이터셋 매핑 및 필터 설정
- [x] `ulm_live/data/schema.py`: 데이터 모델 dataclass
- [x] `ulm_live/data/aihub.py`: 메타데이터 스캐너, 파서, 지역 분류기
- [x] `ulm_live/data/filters.py`: 오디오 및 텍스트 품질 필터
- [x] `ulm_live/data/segment.py`: 오디오 세그멘터 및 코덱 토큰 인코더
- [x] `ulm_live/data/manifest.py`: JSONL 매니페스트 직렬화 및 split 분할기
- [x] `ulm_live/data/__init__.py`: data 패키지 export
- [x] `scripts/inspect_aihub.py`: 비파괴 조사 CLI
- [x] `scripts/prepare_dataset.py`: 데이터셋 자동 생성 파이프라인 CLI
- [x] `tests/test_dataset_schema.py`: 스키마 테스트
- [x] `tests/test_filters.py`: 필터 테스트
- [x] `tests/test_manifest.py`: 매니페스트 테스트
- [x] `README.md`: Phase 2 파이프라인 문서화 및 데이터 라이선스 고지

## 4. 검증 결과
- `python -m compileall ulm_live`: 통과 (오류 0건)
- `pytest`: 25 passed in 1.05s
- `pytest -m integration`: 2 passed in 4.18s
- 합성 데이터셋에 대한 `inspect_aihub.py` 및 `prepare_dataset.py --encode-codec` 실행 결과 정상 동작 확인
- **실제 AI Hub 원본 데이터 유무**: 현재 로컬 머신에는 실제 AI Hub 경상도 방언 원천 데이터가 다운로드되어 있지 않음. (실제 데이터셋 다운로드 후 동일한 스크립트로 즉시 처리 가능)

## 5. Phase 3 계획 (Next Steps)
- **ULM-1.7B Thinker 통합**:
  - 경상도/울산 방언 텍스트 및 사투리 의미 토큰(Semantic tokens, codebook 0) 생성 언어 모델 연계
- **ULM Talker 모델 구조 설계 및 학습 준비**:
  - 음향 세부 토큰(Acoustic tokens, RVQ codebooks 1~7/31) 디인터리빙 및 음성 합성 모델 개발
  - `manifest.jsonl` 기반 PyTorch Dataset 및 DataLoader 구축
- **Streaming Pipeline**:
  - 청크 단위 실시간 오디오 디코딩 및 WebSocket 스트리밍 구현
