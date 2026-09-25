# ULM-LIVE TTS Dataset Report: Qwen3-TTS Ulsan Dialect Adaptation (v2)

Date: 2026-09-25  
Author: Antigravity Assistant  
Target Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`  
Pipeline Mode: CPU-Only Parallel Preprocessing (EC2 & Local)  
Manifest Location: `data/ulm-live-tts-v2/`

---

## 1. Executive Summary & Mandatory Report Metrics

This report documents the curation, acoustic validation, transcript normalization, and speaker/session-disjoint partitioning of the **ULM-LIVE TTS v2 dataset** specifically prepared for adapting `Qwen3-TTS-12Hz-1.7B-CustomVoice` to the **Ulsan regional accent and dialect**.

All preprocessing tasks were executed strictly on **CPU** without allocating GPU resources or interfering with active ULM-4B evaluation workloads.

### Mandatory Metrics Checklist

| Metric (한국어 항목) | Value (수치) | Description & Verification |
| :--- | :--- | :--- |
| **총 화자 수 (Total Speakers)** | **196명** | Active speakers with verified usable speech across 98 sessions |
| **울산 화자 수 (Ulsan Speakers)** | **136명** | **Tier 1 Native Ulsan speakers** (134 Born & Raised Pure Ulsan, 2 Born Ulsan) |
| **경상권 타지역 화자 (Other Gyeongsang)** | **60명** | Tier 2 (27명, 울산 거주/성장 타지역 출생) + Tier 3 (33명, 부산/대구/경남/경북) |
| **총 음성 시간 (Total Duration)** | **28.18시간** (101,456.07초) | Validated mono 24 kHz 16-bit PCM WAV audio |
| **울산 음성 시간 (Ulsan Duration)** | **20.19시간** (72,675.08초) | Duration specifically from Tier 1 native Ulsan dialect speakers |
| **타 경상 음성 시간 (Other Gyeongsang)**| **7.99시간** (28,780.99초) | Supplementary contextual Gyeongsang dialect speech |
| **Usable Utterance 수** | **25,220건** | Acoustically and linguistically verified samples |
| **Dropped Utterance 수** | **2,467건** | Inaudible, masked privacy, laughter/noise, or duration outliers |
| **평균 Duration (Mean Duration)** | **4.023초** | Optimal for neural codec frame chunking |
| **중앙값 Duration (Median Duration)** | **3.850초** | P25 = 2.770s, P75 = 5.080s (Min: 1.00s, Max: 12.00s) |
| **학습에 충분한지 여부 (Sufficiency)** | **YES** | **Exceeds required threshold** (5–15h standard; 20.2h Tier 1 Ulsan available) |

---

## 2. Dataset Sufficiency Analysis (학습 적합성 판정)

### 판정: **YES (충분함)**

### 근거 (Rationale)
1. **음향 모델 적응(Acoustic/Voice Adaptation) 요구량 충족**:
   - `Qwen3-TTS-12Hz-1.7B-CustomVoice`와 같은 최신 거대 음성 생성 모델의 억양/음색 미세조정(Custom Voice Adaptation / Fine-tuning) 시 권장 음성 데이터 양은 **5~15시간**입니다.
   - 본 데이터셋은 **순수 울산 원어민(Tier 1) 발화만 20.19시간(17,892건)**에 달하며, 인접 경상 방언을 포함한 전체 유효 음성은 **28.18시간(25,220건)**으로 모델이 울산 특유의 고저 악센트(Pitch Accent), 어미 억양(예: ~노, ~나, ~데이, ~가꼬), 발화 리듬을 안정적으로 수렴하기에 충분한 규모입니다.
2. **자연스러운 구어체 대화(Conversational Authenticity)**:
   - 본 데이터는 인위적인 스크립트 낭독체가 아니라, AI Hub 경상방언 원천 데이터 중 **100% "사적 대화 > 일상 대화"** 멀티턴 대화 세션에서 추출되었습니다.
   - 화자 간 자연스러운 핑퐁, 감정 표현, 일상 주제(취미, 학교/직장, 여행, 음식 등)가 담겨 있어 실시간 대화 에이전트(ULM-LIVE)의 음성 합성에 이상적입니다.
3. **엄격한 데이터 정제율**:
   - 총 27,687건 중 노이즈, 무음, 비식별화 마스킹, 비음성 음향 태그를 엄격히 걸러내어 **91.1%의 고순도 발화(25,220건)**만 통과시켰습니다.

---

## 3. Data Lineage & Tier Classification

### 화자 지역 계층 분류 (Tier Classification)

AI Hub 전사 레이블(`data/labels/*.json`)의 화자 인적 메타데이터(`birthplace`, `principal_residence`, `current_residence`)를 분석하여 4개 계층으로 세분화하였습니다:

| 계층 (Tier) | 정의 기준 (Criteria) | 화자 수 | 발화 수 | 총 시간 (Hours) | 비중 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Pure Ulsan)** | 출생지 = 울산, 주성장지 = 울산 | 134명 | 17,643건 | 19.93시간 | 70.7% |
| **Tier 1B (Born Ulsan)** | 출생지 = 울산, 주성장지 = 타지역 | 2명 | 249건 | 0.26시간 | 0.9% |
| **Tier 2 (Ulsan-Affinity)** | 출생지 = 경상 타지역, 주성장지/현거주지 = 울산 | 27명 | 3,457건 | 3.73시간 | 13.2% |
| **Tier 3 (Other Gyeongsang)**| 출생지/성장지 = 부산, 대구, 경남, 경북 | 33명 | 3,871건 | 4.27시간 | 15.2% |
| **합계 (Total)** | | **196명** | **25,220건** | **28.18시간** | **100.0%** |

---

## 4. Audio Quality & Transcript Normalization Pipeline

### 4.1 오디오 검증 기준 (Audio Quality Filtering)
- **포맷 표준화**: 24,000 Hz, Mono, 16-bit Signed PCM WAV.
- **길이 제약**: $[1.0\text{s}, 12.0\text{s}]$ (1초 미만 파편 발화 159건, 12초 초과 장문 발화 69건 제거).
- **무음 검출**: RMS 에너지 측정 ($RMS < 0.005$ 제거, 전 샘플 유효 음성 레벨 확인).
- **클리핑 검출**: 피크 진폭 $\ge 0.999$ 샘플 비율이 1% 초과 시 제거 (0건 감지).
- **수치적 무결성**: NaN, Inf, 0-byte 파일 검증 (0건).

### 4.2 전사 정규화 및 결측 제거 (Transcript Normalization)
- **비음성 음향 태그 배제**: `{laughing}`, `{clearing}` 등 287건 제거. (텍스트만 남기고 음성에 웃음소리가 섞이는 왜곡 방지)
- **청취 불가 구간 배제**: `(())`, `((서))` 등 전사 불능 마커 포함 588건 제거.
- **개인정보 비식별화 마스킹 배제**: `#이름#`, `&회사명&` 등 마스킹 토큰 포함 1,291건 제거. (묵음 처리 또는 불일치 방지)
- **발화 연장 부호 정제**: 물결표(`~`) 제거 (`그~` $\rightarrow$ `그`, `아~` $\rightarrow$ `아`).
- **말더듬/간투사 정제**: 하이픈 표기(`-태- 태교` $\rightarrow$ `태교`) 정규화.
- **구두점 및 공백 정리**: 중복 공백 단일화 및 문장 끝 마침표 보정.

### 정제 통계 요약 (Drop Breakdown)

```
Total Source Utterances: 27,687
├── Retained Usable Utterances: 25,220 (91.09%)
└── Dropped Utterances: 2,467 (8.91%)
    ├── #이름# 등 비식별화 마스킹: 1,291건
    ├── (()) 등 청취 불가 마커: 588건
    ├── {laughing} 등 비음성 태그: 287건
    ├── 1.0초 미만 너무 짧은 발화: 159건
    ├── 12.0초 초과 너무 긴 발화: 69건
    └── 한글 문자 2자 미만 단편 발화: 73건
```

---

## 5. Speaker & Session Disjoint Partitioning (Zero Leakage)

동일 화자 또는 동일 대화 세션의 발화가 훈련셋과 평가셋에 교차하여 과적합 또는 평가 왜곡이 발생하는 현상을 방지하기 위해 **Session-Level Disjoint Stratified Split**을 적용하였습니다.

### 분할 규칙 (Split Rules)
1. 한 대화 세션의 모든 발화와 화자는 오직 단 하나의 Split(Train, Validation, Test 중 하나)에만 배정됩니다.
2. 대화 세션의 화자 구성(순수 울산 세션, 복합 세션, 일반 경상 세션)에 따라 계층화(Stratified) 추출되었습니다.
3. 검증 결과:
   - $\text{Train Speakers} \cap \text{Val Speakers} = \emptyset$ (0명)
   - $\text{Train Speakers} \cap \text{Test Speakers} = \emptyset$ (0명)
   - $\text{Val Speakers} \cap \text{Test Speakers} = \emptyset$ (0명)
   - $\text{Train Sessions} \cap \text{Val Sessions} = \emptyset$ (0건)
   - $\text{Train Sessions} \cap \text{Test Sessions} = \emptyset$ (0건)
   - $\text{Val Sessions} \cap \text{Test Sessions} = \emptyset$ (0건)

### 데이터 분할 통계 (Split Breakdown)

| Split | Sessions | Total Speakers | Tier 1 Ulsan | Utterances | Hours | Duration Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 82세션 | 164명 | 114명 | 20,943건 | **23.41시간** (84,288.75s) | 83.1% |
| **Validation** | 8세션 | 16명 | 11명 | 2,167건 | **2.41시간** (8,674.84s) | 8.6% |
| **Test** | 8세션 | 16명 | 11명 | 2,110건 | **2.36시간** (8,492.48s) | 8.3% |
| **Total** | **98세션** | **196명** | **136명** | **25,220건** | **28.18시간** (101,456.07s) | **100.0%** |

---

## 6. Demographics & Speaker Duration Distribution

### 6.1 성별 및 연령대 분포 (Demographics)

| 구분 | 범주 | 발화 수 | 총 시간 (Hours) | 비율 (%) |
| :--- | :--- | :--- | :--- | :--- |
| **성별 (Gender)** | 여성 | 19,491건 | 21.74시간 | 77.1% |
| | 남성 | 5,729건 | 6.44시간 | 22.9% |
| **연령대 (Age)** | 20대 | 9,786건 | 11.27시간 | 40.0% |
| | 10대 | 7,034건 | 7.85시간 | 27.9% |
| | 30대 | 5,081건 | 5.70시간 | 20.2% |
| | 50대 | 2,122건 | 2.16시간 | 7.7% |
| | 40대 | 604건 | 0.64시간 | 2.3% |
| | 60대 이상 | 593건 | 0.55시간 | 2.0% |

### 6.2 화자별 음성 시간 통계 (Duration per Speaker Summary)
- 화자당 평균 음성 시간: **8.63분** (517.6초)
- 화자당 중앙값 음성 시간: **8.42분** (505.2초)
- 화자당 최소/최대 시간: **3.82분 ~ 13.97분** (229.2초 ~ 838.2초)
- 상위 발화 시간 화자 목록 (`data/ulm-live-tts-v2/speakers.json` 발췌):

| Speaker ID | Session | Gender | Age | Region / Tier | Split | Utterances | Duration |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `DKSR20001168_2` | DKSR20001168 | 여성 | 50대 | 울산 (Tier 1 Pure) | Train | 255 | 13.97분 (838.2s) |
| `DKSR20000962_2` | DKSR20000962 | 여성 | 20대 | 울산 (Tier 1 Pure) | Train | 187 | 13.86분 (831.3s) |
| `DKSR20001511_1` | DKSR20001511 | 여성 | 10대 | 울산 (Tier 1 Pure) | Val | 175 | 13.53분 (812.0s) |
| `DKSR20001618_1` | DKSR20001618 | 여성 | 30대 | 울산 (Tier 1 Pure) | Val | 175 | 13.02분 (781.4s) |
| `DKSR20001536_1` | DKSR20001536 | 남성 | 20대 | 울산 (Tier 1 Pure) | Train | 199 | 12.84분 (770.3s) |
| `DKSR20000971_1` | DKSR20000971 | 여성 | 20대 | 울산 (Tier 1 Pure) | Train | 188 | 12.75분 (764.9s) |
| `DKSR20001421_1` | DKSR20001421 | 여성 | 10대 | 울산 (Tier 1 Pure) | Test | 191 | 12.50분 (749.8s) |
| `DKSR20001150_2` | DKSR20001150 | 남성 | 50대 | 울산 (Tier 1 Pure) | Train | 158 | 12.43분 (746.0s) |

---

## 7. Deliverables & Artifacts Structure

산출물은 `ULM-LIVE` 리포지토리의 `data/ulm-live-tts-v2/`에 배포되었습니다:

```
data/ulm-live-tts-v2/
├── train.jsonl          # 20,943개 발화 (23.41시간, 164명 화자, Zero Leakage)
├── validation.jsonl     # 2,167개 발화 (2.41시간, 16명 화자)
├── test.jsonl           # 2,110개 발화 (2.36시간, 16명 화자)
├── speakers.json        # 196명 전체 화자별 인적사항, 지역 Tier, 총 음성 시간 메타데이터
└── stats.json           # 발화수, 음성 시간, 길이 분위수, 손실 사유별 통계 JSON
```

### JSONL 레코드 스키마 예시
```json
{
  "id": "ulsan_000000",
  "utterance_id": "DKSR20000953.1.1.1",
  "session_id": "DKSR20000953",
  "speaker_id": "DKSR20000953_1",
  "rel_audio_path": "data/ulsan-full/audio/000000_DKSR20000953.1.1.1.wav",
  "text": "우리가 인제.",
  "raw_text": "우리가 인제",
  "standard_text": "우리가 인제",
  "duration": 4.87,
  "sample_rate": 24000,
  "channels": 1,
  "gender": "여성",
  "age": "60대 이상",
  "dialect": "ulsan",
  "region": "ulsan",
  "birthplace": "울산",
  "principal_residence": "울산",
  "current_residence": "울산",
  "tier": "Tier1_PureUlsan",
  "is_ulsan_tier1": true,
  "category": "경상방언 > 사적 대화 > 일상 대화",
  "topic": "건강",
  "instruct": "자연스러운 경상도 울산 억양으로 발화해주세요.",
  "rms": 0.13159,
  "peak": 0.95,
  "exact_duration": 4.87
}
```

---

## 8. Qwen3-TTS Fine-Tuning Execution Guide (Next Phase)

GPU 평가 작업이 종료된 후 후속 TTS 학습 시 다음 구성을 권장합니다:

1. **모델 베이스**: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
2. **학습 모드**:
   - 옵션 A (Ulsan Pure Adaptation): `is_ulsan_tier1 == true`인 발화(17,892건, 20.19시간)만 필터링하여 순수 울산 방언 음색 특화.
   - 옵션 B (Full Multi-Dialect Conditioned): 전체 25,220건을 활용하되, `instruct="자연스러운 경상도 울산 억양으로 발화해주세요."` 컨디셔닝을 부여하여 악센트 가이드 학습.
3. **권장 하이퍼파라미터**:
   - Batch size: 8~16 per GPU (with gradient accumulation to reach effective batch 32~64)
   - Max audio tokens: 1,200 (12.0s duration at 12Hz/25Hz codec)
   - Learning rate: $2 \times 10^{-5}$ with Cosine Annealing schedule
   - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.98, \text{weight\_decay}=0.01$)
   - Loss: Cross-entropy on codec tokens with padding token masking
4. **주의사항**:
   - 원본 오디오 파일(`.wav`)은 용량 보호를 위해 Git에 커밋하지 않고 EC2 로컬 경로(`data/ulsan-full/audio/`)에 보존됩니다.
   - Git에는 파이프라인 스크립트(`scripts/prepare_qwen3_tts_dataset.py`), 본 보고서(`reports/ULM_LIVE_TTS_DATASET_REPORT.md`), 및 통계(`stats.json`, `speakers.json`)만 관리합니다.
