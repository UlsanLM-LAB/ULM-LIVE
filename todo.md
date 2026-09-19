# ULM-Live Phase 1 TODO & Progress

## 1. Phase 1 목표
- [x] Spoken Language Model(ULM-Live)의 기반이 되는 오디오 코덱 인터페이스 및 1개 이상의 실제 신경망 오디오 코덱 백엔드 구현
- [x] 핵심 성공 기준: **WAV → Neural Codec Encode → Discrete Codec Tokens → Decode → Reconstructed WAV** 실제 동작 검증
- [x] Talker, ULM-1.7B Thinker, 대화 데이터셋, WebSocket 등은 분리하고 오직 코덱 파이프라인과 기초 음향 유틸리티에 집중

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

## 3. 구현된 파일 목록
- [x] `pyproject.toml`: 패키지 메타데이터, 의존성 (`torch`, `transformers`, `scipy`, `pyyaml`, `numpy`), pytest 설정
- [x] `requirements.txt`: 의존성 정의
- [x] `configs/codec.yaml`: 기본 코덱 설정 (backend: mimi, sample_rate: 24000, num_quantizers: 32)
- [x] `ulm_live/__init__.py`: 패키지 진입점 및 주요 클래스 export
- [x] `ulm_live/codec/base.py`: `AudioCodec` ABC, `EncodedAudio` dataclass
- [x] `ulm_live/codec/backend.py`: `MimiCodec` 구현체 (Hugging Face Transformers 기반, CPU/CUDA 자동 감지)
- [x] `ulm_live/codec/__init__.py`: 코덱 클래스 및 팩토리 함수 (`build_codec`) export
- [x] `ulm_live/utils/audio.py`: WAV loading, mono conversion, resampling (polyphase sinc), peak normalization, WAV saving, duration 계산
- [x] `ulm_live/utils/__init__.py`: 오디오 유틸리티 함수 export
- [x] `scripts/test_codec.py`: CLI reconstruction 파이프라인 (표준 터미널 리포트 출력)
- [x] `scripts/inspect_codec.py`: 코덱 파라미터 및 아키텍처 점검 스크립트
- [x] `tests/test_audio_utils.py`: audio loading, mono, resampling, normalization, roundtrip 단위 테스트
- [x] `tests/test_codec.py`: 코덱 인스턴스화, Mock 단위 테스트 및 Mimi 모델 통합 테스트 (`@pytest.mark.integration`)
- [x] `README.md`: 담백하고 기술적인 프로젝트 문서
- [x] `LICENSE`: Apache 2.0 라이선스

## 4. 검증 결과
- [x] `python -m compileall ulm_live`: 통과 (문법 에러 0건)
- [x] `pytest`: 통과 (11 passed, 2 deselected in 4.60s)
- [x] `pytest -m integration`: 통과 (2 passed, 11 deselected in 5.55s)
- [x] `python scripts/test_codec.py --input samples/input.wav --output outputs/reconstructed.wav`: 통과
  - Input: 24000Hz, 1 channel, 2.00s
  - Codec: Mimi, shape (1, 32, 25), 400.0 tokens/s
  - Output: 24000Hz, 2.00s, reconstructed.wav 생성 확인

## 5. 발견한 제한사항
- **Mimi 고유 샘플레이트 고정**: Mimi는 24kHz 고정 샘플레이트로 훈련되어 있으므로, 16kHz/48kHz 오디오는 인코딩 전 polyphase resampling 및 디코딩 후 필요시 후처리가 수반됨 (유틸리티 레벨에서 자동 처리 완료).
- **첫 웜업 추론 시간**: 모델 초기 로드 및 PyTorch/CUDA JIT 커널 초기화 시 첫 번째 encode에 약 1~2초가 소요되며, 이후 추론은 실시간(초당 수십 ms 이내)으로 수행됨.
- **스트리밍 캐시 추론**: 현재 Phase 1은 파일 전체를 한 번에 인코딩/디코딩하는 오프라인 배치 파이프라인으로 구현됨. 실시간 마이크 입력을 프레임 단위로 처리하기 위해서는 Mimi의 causal conv state 및 streaming cache 관리가 필요함.

## 6. Phase 2 계획
- **Ulsan 사투리 음성 데이터셋(AI Hub) 파이프라인**:
  - 울산 사투리 오디오 정제, 24kHz 모노 변환 및 Mimi discrete token 사전 추출
- **ULM-1.7B Thinker 통합**:
  - 텍스트 입력과 울산 사투리 의미 토큰 연계
- **ULM Talker 개발**:
  - 음향 토큰(Acoustic Tokens, RVQ 1~8) 생성 및 인터리빙
- **Streaming Audio Pipeline**:
  - 청크(chunk) 단위 입출력 및 저지연 WebSocket 스트리밍 구현
