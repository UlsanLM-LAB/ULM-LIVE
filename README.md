# ULM Live

ULM-4B Arm B와 울산 억양으로 적응한 Qwen3-TTS를 연결한 음성 대화 런타임입니다.

현재 MVP 경로:

```text
Microphone
   ↓
Whisper ASR
   ↓
ULM-4B Arm B
   ↓
Qwen3-TTS + Ulsan smoke adapter
   ↓
24 kHz WAV
   ↓
Browser playback
```

## 현재 상태

- ULM-4B Arm B 텍스트 백엔드 연동
- Qwen3-TTS 1.7B CustomVoice 기반 음성 합성
- 울산 Tier 1 음성으로 학습한 smoke adapter 로드 지원
- Whisper Korean STT
- multi-turn history 전달
- 마이크 기반 단일 페이지 Live UI
- 말하는 중 버튼을 다시 누르면 재생/요청을 끊고 새 발화를 시작하는 client-side interruption
- `/health`, `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/chat/speech`

완전한 token/audio streaming WebSocket full-duplex는 후속 단계입니다. 현재 MVP는 한 턴 단위로 음성 인식 → 답변 생성 → 음성 합성을 수행합니다.

## Release assets

큰 모델 파일은 Git 밖에 둡니다.

```text
/home/ubuntu/models/Qwen3.8-4B-Distill
/home/ubuntu/models/ULM-4B-Arm-B
/home/ubuntu/models/ULM-Live-Ulsan-TTS-smoke-v1.pt
```

TTS smoke adapter는 350-step parameter-efficient adaptation 결과이며 Talker layers 26–27과 codec head를 포함합니다. 지역 억양 자연스러움은 human listening 검수가 아직 필요합니다.

## 실행

```bash
cd /home/ubuntu/ULM-LIVE
./scripts/start_live.sh
```

기본 포트는 Live UI/API `8001`, ULM-4B text backend `127.0.0.1:8000`입니다.

종료:

```bash
./scripts/stop_live.sh
```

필요하면 환경변수로 경로를 바꿀 수 있습니다.

```bash
export ULM_MODEL_PATH=/path/to/Qwen3.8-4B-Distill
export ULM_ADAPTER_PATH=/path/to/ULM-4B-Arm-B
export ULM_TTS_ADAPTER_PATH=/path/to/ulsan_tts_adapter.pt
export ULM_ASR_MODEL=openai/whisper-small
./scripts/start_live.sh
```

마이크 UI는 브라우저 보안 정책상 HTTPS 또는 localhost에서 사용하는 것을 권장합니다.

## API

### STT

raw PCM WAV를 body로 보냅니다.

```bash
curl -X POST http://127.0.0.1:8001/v1/audio/transcriptions \
  -H 'Content-Type: audio/wav' \
  --data-binary @speech.wav
```

### TTS

```bash
curl http://127.0.0.1:8001/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"밥은 먹었나?"}' \
  --output reply.wav
```

### ULM → speech

```bash
curl http://127.0.0.1:8001/v1/chat/speech \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"오늘 뭐하노?","history":[],"dialect_strength":2}' \
  --output reply.wav
```

생성된 ULM 텍스트는 UTF-8 hex 형식의 `X-ULM-Text` 헤더에도 포함됩니다.

## Dialect Strength (사투리 강도)

웹 UI의 작은 선택기로 대화의 기본 말투를 바꿀 수 있습니다. 제품 UI에서는 ULM의 방언 모델 성격을 유지하기 위해 3단계만 노출합니다.

| 값 | 이름 | 말투 |
|---|---|---|
| 1 | Mild / 약하게 | 표준어 중심 + 가벼운 울산 표현 |
| 2 | Ulsan / 보통 | 자연스러운 울산 일상 말투 (기본값) |
| 3 | Strong / 강하게 | 울산 어휘·어미를 적극 사용하되 과장과 반복을 피함 |

선택값은 `ulm_dialect_strength` localStorage에 저장되어 새로고침 후에도 유지됩니다.
웹에서 이전에 저장된 0을 포함해 1–3 이외의 값이나 storage 접근 오류는 2로 처리하며, storage 저장이 불가능해도 현재 페이지에서는 선택이 작동합니다.
선택 변경은 다음 chat 요청에 적용됩니다. 대화를 지워도 선택값은 유지합니다.

`POST /v1/chat/speech`의 선택적 `dialect_strength` 필드는 **정수 0–3**, 미지정 시 **2**입니다. 웹 UI는 1–3만 노출하며, API의 0(Standard)은 호환성·직접 호출용으로 유지합니다.
음수·4·문자열·null·boolean·실수는 422로 거부합니다. Live는 이 값을 text backend payload에 그대로 전달하며,
기존 multi-turn user/assistant history에는 강도 지시를 추가하지 않습니다.
Text backend도 `dialect_strength`를 지원하는 버전으로 업데이트되어야 합니다.

현재 구현은 기존 system 지시에 합성하는 **prompt-based text style control**입니다.
서비스 안전·시스템 지시와 사용자의 명시적인 말투·출력 형식 요청을 우선하도록 안내합니다.
학습된 control token이나 정확히 보장되는 강도 제어가 아닙니다.
생성된 문장은 현재 Ulsan TTS가 그대로 읽으며, acoustic model 억양 강도·TTS adapter·voice 설정은 바꾸지 않습니다.

모델을 켜지 않는 관련 테스트:

```bash
pytest -q tests/test_tts_service.py
node --test tests/test_web_dialect_strength.js
```

## TTS adaptation

- Base: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Trainable: Talker layers 26–27 + codec head
- Trainable params: 106.96M
- Validation loss: 9.3381 → 3.1931
- Adapter size: 약 205 MB
- Runtime integration: verified
- Human listening review: pending

세부 결과는 `reports/ULM_LIVE_TTS_SMOKE_V1_REPORT.md`를 참고하세요.

## Legacy research

기존 Mimi + custom Talker 연구 코드는 실험 기록 보존 목적으로 남아 있습니다. 현재 ULM Live MVP의 기본 경로는 Qwen3-TTS입니다.

## License

코드는 저장소의 Apache-2.0 라이선스를 따릅니다. Base model, AI Hub 데이터, Qwen3-TTS 및 기타 외부 모델/데이터는 각각의 원 라이선스와 이용 조건을 별도로 따라야 합니다.
