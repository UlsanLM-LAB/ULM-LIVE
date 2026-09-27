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
  -d '{"prompt":"오늘 뭐하노?","history":[]}' \
  --output reply.wav
```

생성된 ULM 텍스트는 UTF-8 hex 형식의 `X-ULM-Text` 헤더에도 포함됩니다.

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
