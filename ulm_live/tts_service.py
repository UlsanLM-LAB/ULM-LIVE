"""ULM-4B + Qwen3-TTS + Whisper live voice service.

The service keeps the text model in a separate process and exposes a browser UI,
raw-WAV transcription, speech synthesis, and multi-turn chat-to-speech.
Run with one Uvicorn worker so the GPU runtime is not duplicated.
"""

from __future__ import annotations

import asyncio
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urljoin

import httpx
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field

TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
ASR_MODEL = "openai/whisper-small"
ULM_BACKEND_URL = "http://127.0.0.1:8000/v1/chat/completions"
WEB_INDEX = Path(__file__).resolve().parents[1] / "web" / "index.html"


class SpeechRequest(BaseModel):
    input: str = Field(min_length=1, max_length=500)


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)
    dialect_strength: int = Field(default=2, ge=0, le=3, strict=True)


class Runtime:
    def __init__(self) -> None:
        from qwen_tts import Qwen3TTSModel

        self.text_backend_url = os.environ.get("ULM_BACKEND_URL", ULM_BACKEND_URL)
        self.tts_model_name = os.environ.get("ULM_TTS_MODEL", TTS_MODEL)
        self.tts_adapter_path = os.environ.get("ULM_TTS_ADAPTER_PATH", "").strip()
        self.asr_model_name = os.environ.get("ULM_ASR_MODEL", ASR_MODEL)
        self.asr = None
        self.asr_processor = None

        self.tts = Qwen3TTSModel.from_pretrained(
            self.tts_model_name,
            device_map="cuda:0",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.tts_adapter_loaded = False
        if self.tts_adapter_path:
            self._load_tts_adapter(Path(self.tts_adapter_path))

        self.http = httpx.Client(timeout=120)

    def _load_tts_adapter(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"ULM TTS adapter not found: {path}")
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        if not isinstance(state, dict) or not state:
            raise RuntimeError("ULM TTS adapter is empty or invalid")

        talker = self.tts.model.talker
        current = talker.state_dict()
        bad = [
            key
            for key, value in state.items()
            if key not in current or tuple(current[key].shape) != tuple(value.shape)
        ]
        if bad:
            raise RuntimeError(f"ULM TTS adapter is incompatible: {bad[:5]}")

        incompatible = talker.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected ULM TTS adapter keys: {incompatible.unexpected_keys[:5]}"
            )
        talker.eval()
        self.tts_adapter_loaded = True

    def chat(self, prompt: str, history: list[ChatMessage], dialect_strength: int = 2) -> str:
        messages = [
            {"role": item.role, "content": item.content}
            for item in history[-12:]
        ]
        messages.append({"role": "user", "content": prompt})
        response = self.http.post(
            self.text_backend_url,
            json={
                "model": "ulm-4b",
                "messages": messages,
                "dialect_strength": dialect_strength,
                "stream": False,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 150,
            },
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise ValueError("ULM response has no choices")
        choice = data["choices"][0]
        if not isinstance(choice, dict) or choice.get("finish_reason") not in {"stop", "length"}:
            raise ValueError("ULM response ended unexpectedly")
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("ULM response has no text content")
        return content.strip()

    def text_health(self) -> dict:
        response = self.http.get(urljoin(self.text_backend_url, "/health"))
        response.raise_for_status()
        status = response.json()
        if (
            not isinstance(status, dict)
            or status.get("model_loaded") is not True
            or not isinstance(status.get("model"), str)
            or not isinstance(status.get("model_path"), str)
        ):
            raise ValueError("ULM text backend is not ready")
        return status

    def close(self) -> None:
        self.http.close()
        self.asr = None
        self.asr_processor = None

    @torch.inference_mode()
    def speech(self, text: str) -> bytes:
        wavs, rate = self.tts.generate_custom_voice(
            text=text,
            language="Korean",
            speaker=os.environ.get("ULM_TTS_SPEAKER", "Sohee"),
        )
        if not wavs or rate <= 0 or not np.isfinite(wavs[0]).all() or len(wavs[0]) == 0:
            raise RuntimeError("TTS returned invalid audio")
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def _ensure_asr(self) -> None:
        if self.asr is not None and self.asr_processor is not None:
            return
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.asr_processor = AutoProcessor.from_pretrained(self.asr_model_name)
        self.asr = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.asr_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if torch.cuda.is_available():
            self.asr.to("cuda:0")
        self.asr.eval()

    @torch.inference_mode()
    def transcribe(self, wav_bytes: bytes) -> str:
        self._ensure_asr()
        audio, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Invalid audio")
        max_seconds = float(os.environ.get("ULM_ASR_MAX_SECONDS", "30"))
        if len(audio) / float(rate) > max_seconds:
            raise ValueError(f"Audio is longer than {max_seconds:g} seconds")
        if rate != 16000:
            size = max(1, round(len(audio) * 16000 / rate))
            tensor = torch.from_numpy(audio).view(1, 1, -1)
            audio = (
                F.interpolate(tensor, size=size, mode="linear", align_corners=False)
                .view(-1)
                .numpy()
            )

        inputs = self.asr_processor(audio, sampling_rate=16000, return_tensors="pt")
        features = inputs.input_features
        device = next(self.asr.parameters()).device
        dtype = next(self.asr.parameters()).dtype
        features = features.to(device=device, dtype=dtype)
        generated = self.asr.generate(
            features,
            language="ko",
            task="transcribe",
            max_new_tokens=128,
        )
        text = self.asr_processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()
        if not text:
            raise ValueError("ASR returned empty text")
        return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime = await asyncio.to_thread(Runtime)
    app.state.gpu_lock = asyncio.Lock()
    try:
        yield
    finally:
        app.state.runtime.close()


app = FastAPI(title="ULM Live", version="0.3.0", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    if not WEB_INDEX.is_file():
        raise HTTPException(status_code=404, detail="ULM Live web UI is missing")
    return HTMLResponse(WEB_INDEX.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict[str, object]:
    try:
        backend = await asyncio.to_thread(app.state.runtime.text_health)
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="ULM text backend unavailable") from exc
    runtime: Runtime = app.state.runtime
    return {
        "status": "ready",
        "text_model": backend["model"],
        "text_model_path": backend["model_path"],
        "tts_model": runtime.tts_model_name,
        "tts_adapter_loaded": runtime.tts_adapter_loaded,
        "tts_adapter_path": runtime.tts_adapter_path or None,
        "asr_model": runtime.asr_model_name,
        "asr_loaded": runtime.asr is not None,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(request: Request) -> dict[str, str]:
    wav_bytes = await request.body()
    if not wav_bytes:
        raise HTTPException(status_code=400, detail="Empty audio body")
    try:
        async with app.state.gpu_lock:
            text = await asyncio.to_thread(app.state.runtime.transcribe, wav_bytes)
    except (ValueError, RuntimeError, sf.LibsndfileError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"text": text}


@app.post("/v1/audio/speech")
async def speech(request: SpeechRequest) -> Response:
    async with app.state.gpu_lock:
        wav = await asyncio.to_thread(app.state.runtime.speech, request.input)
    return Response(wav, media_type="audio/wav")


@app.post("/v1/chat/speech")
async def chat_speech(request: ChatRequest) -> Response:
    async with app.state.gpu_lock:
        try:
            answer = await asyncio.to_thread(
                app.state.runtime.chat, request.prompt, request.history, request.dialect_strength
            )
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            raise HTTPException(status_code=502, detail="ULM text backend failed") from exc
        if not answer:
            raise HTTPException(status_code=502, detail="Text model returned an empty response")
        wav = await asyncio.to_thread(app.state.runtime.speech, answer)
    return Response(
        wav,
        media_type="audio/wav",
        headers={
            "X-ULM-Text": answer.encode("utf-8").hex(),
            "Access-Control-Expose-Headers": "X-ULM-Text",
        },
    )
