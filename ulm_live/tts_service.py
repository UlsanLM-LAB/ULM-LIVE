"""ULM text and Qwen3-TTS speech service. Run with one Uvicorn worker."""

from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager
from urllib.parse import urljoin

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import torch
from qwen_tts import Qwen3TTSModel


TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
ULM_BACKEND_URL = "http://127.0.0.1:8000/v1/chat/completions"


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=500)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)


class Runtime:
    def __init__(self) -> None:
        self.text_backend_url = os.environ.get("ULM_BACKEND_URL", ULM_BACKEND_URL)
        self.tts = Qwen3TTSModel.from_pretrained(
            os.environ.get("ULM_TTS_MODEL", TTS_MODEL),
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.http = httpx.Client(timeout=120)

    def chat(self, prompt: str) -> str:
        response = self.http.post(
            self.text_backend_url,
            json={"messages": [{"role": "user", "content": prompt}], "stream": False},
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise ValueError("ULM response has no choices")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
            raise ValueError("ULM response ended without EOS")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("ULM response has no text content")
        return content.strip()

    def text_health(self) -> dict:
        response = self.http.get(urljoin(self.text_backend_url, "/health"))
        response.raise_for_status()
        status = response.json()
        if (not isinstance(status, dict) or status.get("model_loaded") is not True
                or not isinstance(status.get("model"), str)
                or not isinstance(status.get("model_path"), str)):
            raise ValueError("ULM text backend is not ready")
        return status

    def close(self) -> None:
        self.http.close()

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
    try:
        yield
    finally:
        app.state.runtime.close()


app = FastAPI(title="ULM Live v2", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        backend = await asyncio.to_thread(app.state.runtime.text_health)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="ULM text backend unavailable") from exc
    return {
        "status": "ready", "text_model": backend["model"],
        "text_model_path": backend["model_path"], "tts_model": TTS_MODEL,
    }


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest) -> Response:
    async with app.state.gpu_lock:
        wav = await asyncio.to_thread(app.state.runtime.speech, request.input)
    return Response(wav, media_type="audio/wav")


@app.post("/v1/chat/speech")
async def chat_speech(request: ChatRequest) -> Response:
    async with app.state.gpu_lock:
        try:
            answer = await asyncio.to_thread(app.state.runtime.chat, request.prompt)
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise HTTPException(status_code=502, detail="ULM text backend failed") from exc
        if not answer:
            raise HTTPException(status_code=502, detail="Text model returned an empty response")
        wav = await asyncio.to_thread(app.state.runtime.speech, answer)
    return Response(wav, media_type="audio/wav", headers={"X-ULM-Text": answer.encode("utf-8").hex()})
