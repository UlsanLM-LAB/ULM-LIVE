import io
import wave
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("qwen_tts")

from fastapi.testclient import TestClient
from ulm_live import tts_service


def _wav() -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\0\0" * 1600)
    return out.getvalue()


def test_live_routes(monkeypatch):
    class FakeRuntime:
        def __init__(self):
            self.tts_model_name = "fake-tts"
            self.tts_adapter_loaded = True
            self.tts_adapter_path = "/models/ulsan.pt"
            self.asr_model_name = "fake-asr"
            self.asr = object()

        def text_health(self):
            return {"model": "ULM-4B", "model_path": "/models/ulm4b", "model_loaded": True}

        def chat(self, prompt, history):
            assert history == []
            return "안녕하세요" if prompt != "empty" else ""

        def speech(self, text):
            assert text in ("반갑습니다", "안녕하세요")
            return _wav()

        def transcribe(self, wav_bytes):
            assert wav_bytes.startswith(b"RIFF")
            return "밥은 먹었나?"

        def close(self):
            pass

    monkeypatch.setattr(tts_service, "Runtime", FakeRuntime)
    with TestClient(tts_service.app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ready"
        assert health["text_model"] == "ULM-4B"
        assert health["tts_adapter_loaded"] is True
        assert health["asr_loaded"] is True
        assert client.get("/").status_code == 200

        response = client.post(
            "/v1/audio/transcriptions",
            content=_wav(),
            headers={"content-type": "audio/wav"},
        )
        assert response.status_code == 200
        assert response.json()["text"] == "밥은 먹었나?"

        response = client.post("/v1/audio/speech", json={"input": "반갑습니다"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content.startswith(b"RIFF")

        response = client.post("/v1/chat/speech", json={"prompt": "인사해 줘", "history": []})
        assert response.status_code == 200
        assert bytes.fromhex(response.headers["x-ulm-text"]).decode() == "안녕하세요"

        assert client.post("/v1/audio/speech", json={"input": ""}).status_code == 422
        assert client.post("/v1/chat/speech", json={"prompt": "empty", "history": []}).status_code == 502


def test_text_backend_contract_without_loading_tts():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={"model_loaded": True, "model": "ULM-4B", "model_path": "/models/ulm4b"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "안녕하세요"}}]},
        )

    runtime = SimpleNamespace(
        text_backend_url="http://127.0.0.1:8000/v1/chat/completions",
        http=httpx.Client(transport=httpx.MockTransport(respond)),
        asr=None,
        asr_processor=None,
    )
    try:
        assert tts_service.Runtime.text_health(runtime)["model_path"] == "/models/ulm4b"
        history = [tts_service.ChatMessage(role="assistant", content="전에 이야기했제.")]
        assert tts_service.Runtime.chat(runtime, "인사해 줘", history) == "안녕하세요"
        assert requests[0].url.path == "/health"
        assert requests[1].url.path == "/v1/chat/completions"
        payload = requests[1].read().decode()
        assert '"stream":false' in payload
        assert '"model":"ulm-4b"' in payload
        assert "전에 이야기했제." in payload
    finally:
        runtime.http.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"choices": []},
        {"choices": [{"finish_reason": "tool_calls", "message": {"content": "미완성"}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
    ],
)
def test_text_backend_rejects_invalid_response(payload):
    runtime = SimpleNamespace(
        text_backend_url="http://127.0.0.1:8000/v1/chat/completions",
        http=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        ),
    )
    try:
        with pytest.raises(ValueError):
            tts_service.Runtime.chat(runtime, "인사해 줘", [])
    finally:
        runtime.http.close()
