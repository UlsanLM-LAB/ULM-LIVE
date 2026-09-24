"""Keep TTS evaluation WAVs and measurements on the EC2 host."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ulm_live.tts_service import Runtime, TTS_MODEL


TEXTS = [
    "안녕하세요. 울산에 오신 걸 환영합니다.",
    "오늘 날씨가 참 좋네요. 태화강 따라 걸어 볼까요?",
    "점심은 드셨어요? 아직이면 같이 먹으러 가요.",
    "아이고, 비가 오네. 우산 챙겼나?",
    "버스가 조금 늦네요. 잠깐만 기다려 주세요.",
    "고래박물관은 어디로 가면 되나요?",
    "주말에 대왕암공원에 가 봤어요? 바다가 참 예쁘더라고요.",
    "그거 오늘까지 끝낼 수 있겠나? 무리하지 마세요.",
    "어제는 정말 수고 많았어요. 푹 쉬세요.",
    "잠깐만요. 제가 다시 한번 확인해 볼게요.",
    "지금 몇 시예요? 약속에 늦을 것 같아요.",
    "이 길로 쭉 가면 울산역이 나오나요?",
    "아침에 일찍 일어났더니 좀 피곤하네요.",
    "커피 한 잔 마실래요? 따뜻한 걸로 드릴까요?",
    "괜찮아요. 천천히 말씀해 주세요.",
    "그 이야기를 들으니 정말 반갑네요.",
    "오늘 저녁에 시간 되면 같이 산책할까요?",
    "이거 어떻게 쓰는 건지 알려 주실 수 있나요?",
    "울산 바다는 바람이 시원해서 좋더라.",
    "내일 다시 연락드릴게요. 좋은 하루 보내세요.",
]

PROMPTS = [
    "울산에 처음 온 사람에게 짧게 인사해 줘.",
    "태화강 산책을 추천해 줘.",
    "오늘 날씨가 좋을 때 할 일을 하나 추천해 줘.",
    "친구가 피곤하다고 하면 어떻게 말할래?",
    "점심 메뉴를 고민하는 친구에게 답해 줘.",
    "울산역 가는 길을 모르면 어떻게 도움을 청할래?",
    "대왕암공원에 대해 한 문장으로 소개해 줘.",
    "비가 와서 우산이 없는 친구를 걱정해 줘.",
    "약속에 늦는다고 연락 온 친구에게 답해 줘.",
    "커피를 권하는 자연스러운 말 한마디 해 줘.",
    "고래박물관에 가고 싶다는 사람에게 답해 줘.",
    "퇴근한 친구에게 따뜻하게 인사해 줘.",
    "일이 너무 많다는 동료에게 한마디 해 줘.",
    "주말 계획을 묻는 친구에게 답해 줘.",
    "울산 바다를 좋아하는 이유를 짧게 말해 줘.",
    "버스를 기다리는 사람에게 안부를 물어 줘.",
    "길을 잃었다는 사람에게 차분하게 답해 줘.",
    "저녁 산책을 함께하자고 제안해 줘.",
    "처음 만난 사람에게 이름을 물어봐 줘.",
    "대화를 마치며 좋은 하루 보내라고 해 줘.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("zero-shot", "e2e"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "e2e":
        runtime = Runtime()
        prompts = PROMPTS
    else:
        tts = Qwen3TTSModel.from_pretrained(
            TTS_MODEL, device_map="cuda:0", dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        prompts = TEXTS

    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as report:
        for index, prompt in enumerate(prompts, 1):
            started = time.monotonic()
            text = runtime.chat(prompt) if args.stage == "e2e" else prompt
            if not text:
                raise RuntimeError(f"Empty ULM response for sample {index}")
            if args.stage == "e2e":
                wav = runtime.speech(text)
                from io import BytesIO
                audio, rate = sf.read(BytesIO(wav), dtype="float32")
            else:
                waves, rate = tts.generate_custom_voice(
                    text=text, language="Korean", speaker="Sohee"
                )
                audio = np.asarray(waves[0], dtype=np.float32)
            if (audio.ndim != 1 or len(audio) == 0 or rate != 24000
                    or not np.isfinite(audio).all()
                    or np.max(np.abs(audio)) < 1e-4):
                raise RuntimeError(f"Invalid audio for sample {index}")
            filename = f"{index:02d}.wav"
            sf.write(args.output_dir / filename, audio, rate, subtype="PCM_16")
            row = {
                "index": index, "prompt": prompt, "spoken_text": text,
                "file": filename, "sample_rate": rate,
                "duration_seconds": round(len(audio) / rate, 3),
                "rms": round(float(np.sqrt(np.mean(audio ** 2))), 6),
                "peak": round(float(np.max(np.abs(audio))), 6),
                "latency_seconds": round(time.monotonic() - started, 3),
            }
            report.write(json.dumps(row, ensure_ascii=False) + "\n")
            report.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
