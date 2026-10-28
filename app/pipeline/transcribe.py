"""Client for the Muse Voice Transcribe file endpoint (POST /v1/asr/transcribe)."""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path
from typing import Callable, Sequence

import httpx
from pydantic import ValidationError

from app.config import Settings
from app.pipeline.models import Transcript, parse_transcript

MAX_BYTES = 32 * 1024 * 1024  # documented request-body cap
MAX_SECONDS = 10 * 60  # documented audio cap
ALLOWED_RATES = (16_000, 24_000)
RETRY_STATUSES = {429, 500}


class TranscribeError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def validate_wav(path: Path) -> None:
    """Fail fast on files the endpoint would reject."""
    size = path.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(f"file is {size} bytes; the limit is {MAX_BYTES}")
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            rate = wav.getframerate()
            frames = wav.getnframes()
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"not a readable WAV file: {exc}") from exc
    if channels != 1 or width != 2 or rate not in ALLOWED_RATES:
        raise ValueError(
            "audio must be mono, 16-bit PCM at 16 kHz or 24 kHz "
            f"(got {channels} ch, {width * 8}-bit, {rate} Hz)"
        )
    if frames / rate > MAX_SECONDS:
        raise ValueError(f"audio is longer than {MAX_SECONDS} seconds")


def transcribe_file(
    path: Path,
    settings: Settings,
    *,
    mode: str = "DIARIZATION",
    language_bias: Sequence[str] = ("Spanish",),
    keywords: Sequence[str] = (),
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> Transcript:
    """Transcribe a WAV file and return validated turn-level results."""
    path = Path(path)
    validate_wav(path)

    request_part: dict = {
        "mode": mode,
        "model": settings.asr_model,
        "audioEncoding": "WAV",
    }
    if language_bias:
        request_part["languageBias"] = list(language_bias)
    if keywords:
        request_part["keywords"] = list(keywords)

    url = f"{settings.api_base}/asr/transcribe"
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    audio_bytes = path.read_bytes()

    with httpx.Client(timeout=120.0) as client:
        response: httpx.Response | None = None
        for attempt in range(max_retries + 1):
            response = client.post(
                url,
                headers=headers,
                files={
                    "request": (
                        None,
                        json.dumps(request_part),
                        "application/json",
                    ),
                    "audio": (path.name, audio_bytes, "audio/wav"),
                },
            )
            if response.status_code in RETRY_STATUSES and attempt < max_retries:
                sleep(2**attempt)
                continue
            break

    assert response is not None
    if response.status_code != 200:
        raise TranscribeError(
            f"transcription failed with HTTP {response.status_code}: "
            f"{response.text[:200]}",
            status_code=response.status_code,
        )
    try:
        return parse_transcript(response.json())
    except (ValueError, ValidationError) as exc:
        raise TranscribeError(f"unexpected response shape: {exc}") from exc
