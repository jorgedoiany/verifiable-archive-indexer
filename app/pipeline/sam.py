"""Client for SAM 3.1: does a concept appear in an uploaded video, and where?

SAM bills per video frame ($0.20 per 1,000, minimum 50 frames per request),
so callers should send a low-fps copy of the video and cap concepts per clip.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from app.config import Settings
from app.pipeline.extract import output_text
from app.pipeline.sam_parse import Detection, SamParseError, parse_sam_text

PRICE_PER_1000_FRAMES = 0.20
MIN_BILLED_FRAMES = 50
MAX_CONCEPT_CHARS = 60
RETRY_STATUSES = {429, 500}
_SEPARATORS = re.compile(r"[,;+\n\r]")


class SamCallError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class SamResult:
    concept: str
    detections: tuple[Detection, ...]
    frames_billed: int | None


def clean_concept(concept: str) -> str:
    """SAM takes one short noun phrase per request; lists return no matches."""
    text = " ".join(concept.split())
    if not text:
        raise ValueError("concept must not be empty")
    if _SEPARATORS.search(concept):
        raise ValueError(
            "send one concept per request: no commas, semicolons, plus signs "
            "or newlines"
        )
    if len(text) > MAX_CONCEPT_CHARS:
        raise ValueError(f"concept is longer than {MAX_CONCEPT_CHARS} characters")
    return text


def estimate_cost(frames: int, requests: int = 1) -> float:
    """Estimated dollars for `requests` video requests of `frames` frames each."""
    if frames < 0 or requests < 0:
        raise ValueError("frames and requests must not be negative")
    billed = max(frames, MIN_BILLED_FRAMES) * requests
    return billed * PRICE_PER_1000_FRAMES / 1000


def segment_concept(
    file_id: str,
    concept: str,
    settings: Settings,
    *,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> SamResult:
    """Run one concept over an uploaded video (non-streaming) and parse the boxes."""
    concept = clean_concept(concept)
    payload = {
        "model": settings.sam_model,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": concept},
                    {"type": "input_video", "file_id": file_id},
                ],
            }
        ],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
    }
    url = f"{settings.api_base}/responses"
    headers = {
        "Authorization": f"Bearer {settings.api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=600.0) as client:
        response: httpx.Response | None = None
        for attempt in range(max_retries + 1):
            response = client.post(url, headers=headers, json=payload)
            if response.status_code in RETRY_STATUSES and attempt < max_retries:
                sleep(2**attempt)
                continue
            break

    assert response is not None
    if not response.is_success:
        raise SamCallError(
            f"segmentation failed with HTTP {response.status_code}: "
            f"{response.text[:200]}",
            status_code=response.status_code,
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise SamCallError("segmentation response is not valid JSON") from exc
    if not isinstance(body, dict):
        raise SamCallError("segmentation response is not a JSON object")
    if body.get("status") == "incomplete":
        raise SamCallError("segmentation response is incomplete")

    try:
        detections = parse_sam_text(output_text(body))  # empty text = no matches
    except SamParseError as exc:
        raise SamCallError(f"unreadable segmentation output: {exc}") from exc

    usage = body.get("usage")
    billed = usage.get("video_frames_processed") if isinstance(usage, dict) else None
    return SamResult(
        concept=concept,
        detections=tuple(detections),
        frames_billed=billed if isinstance(billed, int) else None,
    )
