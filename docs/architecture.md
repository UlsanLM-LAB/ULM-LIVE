# Thinker-Talker Architecture Specification

`ULM-Live`는 텍스트 기반 대형 언어 모델(Thinker)과 음향 코덱 토큰 자기회귀 예측 모델(Talker)을 계층적으로 결합하여 울산 사투리 음성을 생성하는 Spoken Language Model 아키텍처를 채택합니다.

```text
Microphone
   ↓
Audio Tokenizer / Codec (Mimi, 12.5 Hz)
   ↓
ULM-1.7B Thinker (Frozen)
   ↓
Semantic Hidden States [B, S, 2048]
   ↓
ULM Talker (Causal Transformer)  ← [Speaker ID, Dialect ID]
   ↓
Acoustic Codec Tokens [B, K, T]
   ↓
Streaming Codec Decoder
   ↓
Speech (Ulsan Dialect)
```

---

## 1. 일반 TTS와 Talker의 차이점

| 비교 항목 | 전통적 TTS (Cascaded Pipeline) | ULM-Live Thinker-Talker (End-to-End SLM) |
| :--- | :--- | :--- |
| **입력 인터페이스** | 이산 텍스트 스트링 (Characters / Phonemes) | 언어 모델의 연속형 잠재 표현 (Continuous Hidden States) |
| **맥락 이해도** | 문맥 없는 음절 단위 발음 기호 의존 | 1.7B LLM의 의미적 맥락, 억양 의도, 방언 뉘앙스 직접 전달 |
| **지연 시간 (Latency)**| 텍스트 디코딩 완료 후 음향 합성 시작 (직렬 지연) | 첫 토큰 히든 벡터 즉시 프로젝션 후 오디오 토큰 동시 생성 가능 |
| **화자/방언 제어** | 음향 모델에 화자 임베딩 단순 주입 | 화자 정체성과 방언 억양 공간을 독립적으로 분리하여 제어 |

---

## 2. Text Output 대신 Hidden States를 Conditioning으로 사용하는 이유

텍스트 토큰을 디코딩한 뒤 다시 음향 모델의 입력으로 넣는 방식은 다음과 같은 정보 손실을 유발합니다:
1. **의미적 풍부성 손실**: LLM 내부의 고차원 hidden state(2048차원)에는 단어의 표면형뿐만 아니라 발화자의 감정, 강조점, 전후 맥락에 대한 확률 분포가 함축되어 있습니다. 텍스트로 축약하는 순간 이 풍부한 잠재 정보가 단일 단어 인덱스로 붕괴(information bottleneck)됩니다.
2. **사투리 억양 및 리듬 단서 보존**: 울산 사투리와 같은 방언 발화에서는 텍스트 표기만으로 음조(pitch)와 운율(prosody)을 표현하기 어렵습니다. Thinker 내부의 문맥 표현을 Talker의 conditioning vector로 직접 연결함으로써, 텍스트로는 표현되지 않는 운율 단서를 음향 코덱 토큰 예측에 직접 전달할 수 있습니다.

---

## 3. Semantic Representation과 Acoustic Representation의 분리

* **Semantic Representation (Thinker)**:
  - 언어적 의미, 질문에 대한 답변 내용, 대화의 맥락을 담당.
  - 시간 축이 토큰(subword) 단위이며, 음향 정보는 포함하지 않음.
* **Acoustic Representation (Talker & Mimi Codec)**:
  - 음성 파형의 고주파 세부 질감, 음색(timbre), 성대 진동, 억양을 담당.
  - 시간 축이 12.5 Hz의 고정된 오디오 프레임 단위이며, 32개의 Residual Vector Quantization (RVQ) 코드북으로 분해됨.

Talker는 Thinker의 semantic hidden state를 prefix로 받아, 이를 조건으로 음향 토큰 시퀀스를 한 단계씩 자기회귀적(autoregressive)으로 합성합니다.

---

## 4. Speaker Conditioning과 Dialect Conditioning의 엄격한 분리

### 분리 배경
방언 음성 데이터를 학습할 때 가장 흔히 발생하는 문제는 **"특정 방언의 억양"과 "데이터셋에 포함된 특정 화자의 목소리 음색"이 모델 내부에서 결합(entangled)되는 현상**입니다.
이 경우 울산 사투리를 말하게 하면 데이터셋에 녹음된 특정 노년/중년 화자의 목소리로만 음성이 출력되거나, 젊은 화자의 목소리로는 사투리 억양이 발현되지 않는 왜곡이 발생합니다.

### 구현 방식
* **화자 정체성 (`speaker_id`)**: 화자 고유의 성대 특성, 기본 주파수 대역, 음색을 모델링 (`nn.Embedding(num_speakers, 128)`).
* **방언 정체성 (`dialect`)**: 울산 방언의 어휘적, 운율적 높낮이 규칙을 모델링 (`nn.Embedding(num_dialects, 64)`).
* **Additive Projection**:
  두 임베딩을 선형 사영한 뒤 audio token 시퀀스에 가산(additive) 결합하여, 향후:
  - `Same Speaker + Different Dialect` (동일 인물이 표준어와 울산 사투리를 번갈아 구사)
  - `Same Dialect + Different Speaker` (서로 다른 다양한 목소리로 일관된 울산 억양 구사)
  를 명시적으로 조작할 수 있도록 설계되었습니다.

---

## 5. Codec Token Prediction 방식 비교 및 선택

Phase 1에서 채택한 **Mimi 코덱**은 프레임당 32개의 RVQ 코드북(codebook size 2048)을 가집니다.

| 방식 | 설명 | 장점 | 단점 |
| :--- | :--- | :--- | :--- |
| **Independent Heads per Codebook (선택됨)** | 동일 hidden state에서 K개 코드북을 K개의 선형 분류 헤드로 동시 예측 | **가장 간결하며 시퀀스 길이 증가 없음.** 병렬 손실 계산 및 역전파 용이 | 코드북 간 조건부 종속성(hierarchical dependence)을 단일 스텝 내에서 직접 반영하지 못함 |
| **Flattened Vocabulary / Interleaved** | 코드북 0..31을 시간축으로 직렬화하여 1스텝당 1코드북 예측 | 코드북 간의 순차적 종속성을 완전 모델링 | **시퀀스 길이가 32배로 팽창**하여 연산량 및 어텐션 메모리 1024배 증가 |
| **Delayed Pattern (MusicGen)** | 코드북마다 1타임스텝씩 shift하여 엇갈리게 예측 | 병렬성과 순차성을 절충 | 캐시 관리 및 스트리밍 정렬 구현이 복잡함 |

---

## 6. Training vs Inference (Phase 3 vs Phase 4)

### Training (Phase 3)
* **교사 강요 (Teacher-Forcing)**: 정답 오디오 토큰 시퀀스 $t_0, \dots, t_{T-1}$ 전체를 한 번에 모델에 입력하고, 다음 타임스텝의 타겟 $t_1, \dots, t_T$에 대한 크로스 엔트로피 손실을 병렬로 계산.
* 인과적 마스크(Causal Mask)를 통해 미래 오디오 토큰을 보지 못하도록 제한.

### Inference / Generation (Phase 4)
* **자기회귀 생성 (Autoregressive Generation)**:
  - 시작 시점에는 1프레임의 시작 토큰(BOS)으로 시작.
  - 매 타임스텝마다 트랜스포머를 거쳐 32개 코드북 로짓 `[1, 32, 2048]`을 계산.
  - Greedy(argmax) 또는 Stochastic Sampling(temperature, top-k, top-p)을 통해 32개 코드북 인덱스를 결정.
  - 결정된 토큰을 시퀀스 끝에 붙여(`cur_codes = cat([cur_codes, next_token])`) 다음 스텝 입력으로 재사용.
  - 목표 프레임 수(`max_new_tokens` 또는 `max_audio_seconds * 12.5`)에 도달할 때까지 루프 반복.

```text
Step 0: [Semantic Prefix] + [BOS]               → Predict Codec Token 0
Step 1: [Semantic Prefix] + [BOS, Token 0]       → Predict Codec Token 1
Step 2: [Semantic Prefix] + [BOS, Token 0, 1]    → Predict Codec Token 2
...
Result: [Token 0, Token 1, ..., Token T-1] (32 Codebooks x T Frames)
        ↓
        Phase 1 Mimi Codec Decoder
        ↓
        Synthesized Waveform (24kHz Mono WAV)
```

---

## 7. Trained vs Untrained Talker 모델 구분

* **현재 상태**: Phase 4에서는 **추론 데이터 파이프라인(Inference Data Path)** 의 엔드투엔드 무결성을 검증하는 단계입니다.
* **음질 기대치**:
  - 학습된 Talker 체크포인트가 주어지지 않은 경우(Random initialization), 모델이 출력하는 토큰은 무작위 분포를 가지므로 디코딩된 오디오는 **화이트 노이즈 또는 무의미한 음향 신호**입니다.
  - 이는 파이프라인의 오류가 아니며, 정상적인 추론 텐서 흐름 검증(`PIPELINE TEST ONLY`)에 해당합니다.
  - 실제 명료한 울산 방언 음성은 향후 Phase 5에서 대규모 음성 코퍼스로 Talker 가중치를 본격 훈련한 이후 달성됩니다.

---

## 8. 현재 아키텍처의 한계 및 Future KV-Cache 로드맵

1. **Non-incremental Decoding (현재)**:
   현재 Phase 4는 정확성(correctness)을 최우선으로 하여, 매 타임스텝마다 전체 시퀀스를 다시 인코딩하는 Full-sequence recomputation 방식을 취하고 있습니다. (현재 2초 오디오 기준 RTF 약 0.32~0.65로 실시간보다 빠르나, 긴 오디오 생성 시 $O(T^2)$ 연산량 증가).
2. **TODO: Incremental KV-Cache Generation**:
   향후 실시간 스트리밍 대화를 위해 이미 계산된 과거 키/값 텐서를 캐싱(`past_key_values`)하여 매 스텝 1개의 신규 토큰만 연산하는 $O(T)$ 디코딩 루프로 최적화할 예정입니다.
3. **Mimi Streaming Cache 연계**:
   오프라인 일괄 디코딩(`codec.decode(all_codes)`)을 프레임 단위 또는 80ms 청크 단위 실시간 스트리밍 디코더로 연결하는 작업이 Phase 5에서 진행됩니다.
