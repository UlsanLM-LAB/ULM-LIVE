import io
import wave

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
        writer.setframerate(24000)
        writer.writeframes(b"\0\0" * 240)
    return out.getvalue()


def test_speech_and_chat_routes(monkeypatch):
    class FakeRuntime:
        def chat(self, prompt):
            return "안녕하세요" if prompt != "empty" else ""

        def speech(self, text):
            assert text in ("반갑습니다", "안녕하세요")
            return _wav()

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
