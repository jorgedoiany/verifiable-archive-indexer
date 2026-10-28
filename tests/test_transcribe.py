import wave

import httpx
import pytest

from app.config import Settings
from app.pipeline import transcribe
from app.pipeline.transcribe import TranscribeError, transcribe_file

URL = "https://api.meta.ai/v1/asr/transcribe"

SAMPLE = {
    "sessionId": "abc",
    "transcript": "Hola mundo.",
    "audioDurationMs": 1000,
    "turns": [
        {
            "turnId": 1,
            "startMs": 0,
            "endMs": 900,
            "transcript": "Hola mundo.",
            "speaker": "A",
        }
    ],
}


@pytest.fixture
def settings():
    return Settings(
        api_key="test-key",
        api_base="https://api.meta.ai/v1",
        spark_model="muse-spark-1.3",
        wonder_model="muse-wonder-1.0",
        asr_model="muse-voice-transcribe-1.0",
        sam_model="sam-3.1",
    )


def make_wav(path, seconds=1, rate=24_000, channels=1):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * rate * seconds * channels)
    return path


def test_success_sends_documented_request(tmp_path, settings, respx_mock):
    wav = make_wav(tmp_path / "a.wav")
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=SAMPLE))
    result = transcribe_file(wav, settings)
    assert result.turns[0].transcript == "Hola mundo."
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key"
    body = request.content
    assert b'name="request"' in body
    assert b"DIARIZATION" in body
    assert b'"Spanish"' in body
    assert b"muse-voice-transcribe-1.0" in body
    assert b'name="audio"' in body


def test_retries_on_429_then_succeeds(tmp_path, settings, respx_mock):
    wav = make_wav(tmp_path / "a.wav")
    route = respx_mock.post(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=SAMPLE)]
    )
    waits = []
    result = transcribe_file(wav, settings, sleep=waits.append)
    assert result.audio_duration_ms == 1000
    assert route.call_count == 2
    assert waits == [1]


def test_gives_up_after_retries(tmp_path, settings, respx_mock):
    wav = make_wav(tmp_path / "a.wav")
    respx_mock.post(URL).mock(return_value=httpx.Response(500, text="boom"))
    with pytest.raises(TranscribeError) as info:
        transcribe_file(wav, settings, max_retries=1, sleep=lambda s: None)
    assert info.value.status_code == 500


def test_client_error_is_not_retried(tmp_path, settings, respx_mock):
    wav = make_wav(tmp_path / "a.wav")
    route = respx_mock.post(URL).mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(TranscribeError):
        transcribe_file(wav, settings, sleep=lambda s: None)
    assert route.call_count == 1


def test_bad_response_shape_is_wrapped(tmp_path, settings, respx_mock):
    wav = make_wav(tmp_path / "a.wav")
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json={"oops": 1}))
    with pytest.raises(TranscribeError):
        transcribe_file(wav, settings)


def test_rejects_stereo(tmp_path):
    wav = make_wav(tmp_path / "s.wav", channels=2)
    with pytest.raises(ValueError):
        transcribe.validate_wav(wav)


def test_rejects_unsupported_rate(tmp_path):
    wav = make_wav(tmp_path / "r.wav", rate=44_100)
    with pytest.raises(ValueError):
        transcribe.validate_wav(wav)


def test_rejects_too_long(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe, "MAX_SECONDS", 1)
    wav = make_wav(tmp_path / "l.wav", seconds=2)
    with pytest.raises(ValueError):
        transcribe.validate_wav(wav)


def test_rejects_too_big(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe, "MAX_BYTES", 100)
    wav = make_wav(tmp_path / "b.wav", seconds=1)
    with pytest.raises(ValueError):
        transcribe.validate_wav(wav)
