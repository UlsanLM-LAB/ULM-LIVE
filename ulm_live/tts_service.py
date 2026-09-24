"""ULM text and Qwen3-TTS speech service. Run with one Uvicorn worker."""

from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
ULM_MODEL = "/home/ubuntu/ULM-1.7B/outputs/ulm-1.7b-phase4-best-merged"


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=500)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)


class Runtime:
    def __init__(self) -> None:
        model_path = os.environ.get("ULM_TEXT_MODEL", ULM_MODEL)
        # ULM's Transformers 5 checkpoint stores extra_special_tokens as a list;
        # Qwen TTS currently pins Transformers 4, whose loader expects a mapping.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, extra_special_tokens={})
        self.text_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="cuda:0"
        ).eval()
        self.tts = Qwen3TTSModel.from_pretrained(
            os.environ.get("ULM_TTS_MODEL", TTS_MODEL),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )

    @torch.inference_mode()
    def chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "울산 지역어 대화 assistant로서 의미와 사실성을 우선한다. 의미를 보존하고 약한 울산 지역색만 사용한다."},
            {"role": "user", "content": prompt},
        ]
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_tensors="pt",
        ).to(self.text_model.device)
        output = self.text_model.generate(
            tokens, attention_mask=torch.ones_like(tokens),
            max_new_tokens=64, do_sample=True,
            temperature=0.7, top_p=0.9, top_k=20,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(output[0, tokens.shape[-1]:], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def speech(self, text: str) -> bytes:
        wavs, rate = self.tts.generate_custom_voice(
            text=text, language="Korean", speaker="Sohee"
        )
        if not wavs or rate <= 0 or not np.isfinite(wavs[0]).all() or len(wavs[0]) == 0:
            raise RuntimeError("TTS returned invalid audio")
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = await asyncio.to_thread(Runtime)
    app.state.gpu_lock = asyncio.Lock()
    yield


app = FastAPI(title="ULM Live v2", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready", "text_model": "ULM-1.7B", "tts_model": TTS_MODEL}


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest) -> Response:
    async with app.state.gpu_lock:
        wav = await asyncio.to_thread(app.state.runtime.speech, request.input)
    return Response(wav, media_type="audio/wav")


@app.post("/v1/chat/speech")
async def chat_speech(request: ChatRequest) -> Response:
    async with app.state.gpu_lock:
        answer = await asyncio.to_thread(app.state.runtime.chat, request.prompt)
        if not answer:
            raise HTTPException(status_code=502, detail="Text model returned an empty response")
        wav = await asyncio.to_thread(app.state.runtime.speech, answer)
    return Response(wav, media_type="audio/wav", headers={"X-ULM-Text": answer.encode("utf-8").hex()})
