# ULM-Live Phase 1 TODO

## 1. Phase 1 목표
- Spoken Language Model(ULM-Live)의 기반이 되는 오디오 코덱 인터페이스 및 1개 이상의 실제 신경망 오디오 코덱 백엔드 구현
- 핵심 성공 기준: **WAV → Neural Codec Encode → Discrete Codec Tokens → Decode → Reconstructed WAV** 실제 동작 검증
- Talker, ULM-1.7B Thinker, 대화 데이터셋, WebSocket 등은 구현하지 않고 오직 코덱 파이프라인과 기초 음향 유틸리티에 집중

## 2. Codec Backend 결정 및 조사 요약

### 후보군 비교
| 항목 | Mimi (Kyutai) | EnCodec (Meta) | DAC (Descript) | SNAC / SpeechTokenizer |
| :--- | :--- | :--- | :--- | :--- |
| **프레임 레이트** | **12.5 Hz** (초당 12.5 토큰 프레임) | 75 Hz | ~86 Hz | 가변 / 50 Hz |
| **SLM 적합도** | **최상** (Moshi/SLM용 설계, 컨텍스트 절약) | 보통 (토큰 시퀀스 과다) | 낮음 (토큰 과다) | 보통 |
| **스트리밍 구조** | **Causal** (초저지연 스트리밍 최적화) | Causal | Non-causal (지연 큼) | Non-causal / Semi |
| **라이선스** | Apache 2.0 (코드) / CC-BY 4.0 (가중치) | MIT (코드) / CC-BY-NC 4.0 (가중치) | MIT | Apache 2.0 / MIT |
| **설치 & 유지보수** | Transformers 공식 내장 (`transformers>=4.45`) | Transformers 내장 또는 encodec | descript-audio-codec | 별도 커스텀 패키지 |
| **CPU Fallback** | 완벽 지원 | 완벽 지원 | 지원 | 지원 |
| **Discrete Token** | RVQ 32 (기본 8~32) 코드북 인덱스 분리 | RVQ 32 코드북 | RVQ 9~12 코드북 | Multi-scale RVQ |

### 최종 선택: Mimi (`kyutai/mimi`)
- **선택 이유**:
  1. **Spoken Language Model 최적화**: 12.5 Hz의 초저 프레임 레이트로 ULM-1.7B 및 Talker 모델이 autoregressive token prediction을 실시간으로 수행하기에 최적.
  2. **스트리밍 Causal 설계**: Moshi 실시간 full-duplex 대화 파이프라인에서 검증된 causal transformer/convolution 구조.
  3. **라이선스 및 접근성**: 상업적 이용 가능한 CC-BY 4.0 가중치, Hugging Face `transformers` 내장으로 별도 복잡한 C 확장 컴파일 없이 안정적 구동.
  4. **확장성**: `AudioCodec` 추상 인터페이스를 통해 향후 EnCodec 등 대체 백엔드 플러그인 가능.

## 3. 구현 파일 목록
- `pyproject.toml`: 패키지 메타데이터, 의존성 (`torch`, `transformers`, `scipy`, `pyyaml`, `numpy`), CLI 엔트리포인트
- `requirements.txt`: 기본 및 개발용 의존성 정의
- `configs/codec.yaml`: 기본 코덱 설정 (backend, model_id, sample_rate, num_quantizers, device)
- `ulm_live/__init__.py`: 패키지 초기화 및 버전 정보
- `ulm_live/codec/base.py`: `AudioCodec` ABC, `EncodedAudio` dataclass
- `ulm_live/codec/backend.py`: `MimiCodec` 구현체 (Hugging Face Transformers 기반, CPU/CUDA 자동 처리)
- `ulm_live/codec/__init__.py`: 코덱 클래스 및 팩토리 함수 (`build_codec`) export
- `ulm_live/utils/audio.py`: WAV loading, mono conversion, resampling (polyphase sinc), peak normalization, WAV saving, duration 계산
- `ulm_live/utils/__init__.py`: 오디오 유틸리티 함수 export
- `scripts/test_codec.py`: CLI reconstruction 파이프라인 (인자 파싱, 입출력 통계 및 벤치마크 시간 출력)
- `scripts/inspect_codec.py`: 코덱 정보, 코드북 차원 및 토큰 레이트 분석 스크립트
- `tests/test_audio_utils.py`: audio loading, mono, resampling, normalization, roundtrip 단위 테스트
- `tests/test_codec.py`: 코덱 인스턴스화, Mock/Shape 단위 테스트, 실제 모델 로드 및 encode/decode 통합 테스트 (`@pytest.mark.integration`)
- `README.md`: 프로젝트 개요, Phase 1 구현 현황 체크리스트, 설치 및 실행 가이드
- `LICENSE`: Apache 2.0 라이선스

## 4. 테스트 계획
1. **단위 테스트 (Unit Tests)**:
   - `test_load_and_save_wav`: 합성 사인파 WAV 파일 생성, 저장 및 로드 정확성 확인
   - `test_to_mono`: 멀티채널(스테레오) 텐서의 단일 채널 변환 검증
   - `test_resample`: 44.1kHz -> 24kHz, 16kHz -> 24kHz 고품질 리샘플링 후 길이 및 채널 유지 검증
   - `test_peak_normalize`: 진폭 스케일링 범위 확인
   - `test_codec_interface`: Base AudioCodec 및 EncodedAudio 속성 검증
   - `test_codec_creation`: `build_codec` 팩토리 함수 정상 동작 확인
2. **통합 테스트 (Integration Tests - `@pytest.mark.integration`)**:
   - 실제 `kyutai/mimi` 가중치를 로드하여 1초 합성 음성 인코딩 및 디코딩 파이프라인 검증
   - 인코딩된 토큰 텐서 차원 `[batch, num_quantizers, frames]` 검증
   - 디코딩된 복원 음향의 샘플 레이트 및 waveform shape 검증
3. **CLI 실행 테스트**:
   - `python scripts/test_codec.py --input samples/input.wav --output outputs/reconstructed.wav`
   - 터미널 출력 규격 (Input / Codec / Output 메타데이터 및 encode/decode 소요 시간) 확인

## 5. 성공 기준
- `python -m compileall ulm_live` 에러 없이 통과
- `pytest` 기본 실행 시 단위 테스트 전원 통과 (네트워크/다운로드 없는 순수 유닛 테스트)
- `pytest -m integration` 또는 CLI 실행 시 실제 `kyutai/mimi`를 통한 WAV → Codec Tokens → WAV 복원 완료
- CLI 출력 형식 준수
- 논리적 단위의 git commits 완료

## 6. 남아 있는 문제 (Phase 2+ 로 이관)
- Ulsan 사투리 음성 데이터셋(AI Hub) 다운로드 및 전처리 파이프라인
- ULM-1.7B Thinker (텍스트/음성 토큰 생성 언어 모델) 통합
- ULM Talker (코덱 토큰 디인터리빙 및 음성 합성 모델) 개발
- WebSocket 기반 실시간 풀듀플렉스(full-duplex) 스트리밍 서버 및 클라이언트
- Causal chunk-by-chunk streaming inference 최적화 (Mimi streaming cache 활용)
