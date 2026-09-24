import io
import wave

import pytest
import httpx

pytest.importorskip("fastapi")
pytest.importorskip("qwen_tts")

from fastapi.testclient import TestClient
from ulm_live import tts_service


def _wav() -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 240)
    return out.getvalue()


def test_speech_and_chat_routes(monkeypatch):
    class FakeRuntime:
        def text_health(self):
            return {"model": "ULM-1.7B", "model_path": "/models/phase4", "model_loaded": True}

        def chat(self, prompt):
            return "안녕하세요" if prompt != "empty" else ""

        def speech(self, text):
            assert text in ("반갑습니다", "안녕하세요")
            return _wav()

        def close(self):
            pass

    monkeypatch.setattr(tts_service, "Runtime", FakeRuntime)
    with TestClient(tts_service.app) as client:
        assert client.get("/health").json()["status"] == "ready"
        response = client.post("/v1/audio/speech", json={"input": "반갑습니다"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert response.content.startswith(b"RIFF")

        response = client.post("/v1/chat/speech", json={"prompt": "인사해 줘"})
        assert response.status_code == 200
        assert bytes.fromhex(response.headers["x-ulm-text"]).decode() == "안녕하세요"

        assert client.post("/v1/audio/speech", json={"input": ""}).status_code == 422
        assert client.post("/v1/chat/speech", json={"prompt": "empty"}).status_code == 502


def test_text_backend_contract_without_loading_tts():
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "model_loaded": True, "model": "ULM-1.7B", "model_path": "/models/phase4",
            })
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": "안녕하세요"},
        }]})

    runtime = tts_service.Runtime.__new__(tts_service.Runtime)
    runtime.text_backend_url = "http://127.0.0.1:8000/v1/chat/completions"
    runtime.http = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        assert runtime.text_health()["model_path"] == "/models/phase4"
        assert runtime.chat("인사해 줘") == "안녕하세요"
        assert requests[0].url.path == "/health"
        assert requests[1].url.path == "/v1/chat/completions"
        assert requests[1].read().decode().find('"stream":false') >= 0
    finally:
        runtime.close()


@pytest.mark.parametrize("payload", [
    {"choices": []},
    {"choices": [{"finish_reason": "length", "message": {"content": "미완성"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": None}}]},
])
def test_text_backend_rejects_incomplete_response(payload):
    runtime = tts_service.Runtime.__new__(tts_service.Runtime)
    runtime.text_backend_url = "http://127.0.0.1:8000/v1/chat/completions"
    runtime.http = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload)
    ))
    try:
        with pytest.raises(ValueError):
            runtime.chat("인사해 줘")
    finally:
        runtime.close()
