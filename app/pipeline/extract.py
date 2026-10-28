"""Stage 2: ask the vision model for claims with evidence, then check them offline.

The response shape (status, output[].content[].text) follows the usual
Responses API layout and has not been confirmed against a live call yet.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx

from app.config import Settings
from app.pipeline.claims import ClaimSet
from app.pipeline.extract_parse import (
    apply_checks,
    build_extraction_prompt,
    parse_model_output,
)
from app.pipeline.files_api import FilesError, delete_file, upload_video
from app.pipeline.models import Transcript

RETRY_STATUSES = {429, 500}
# Reasoning shares the output budget; the docs advise at least 4000.
DEFAULT_MAX_OUTPUT_TOKENS = 8_000


class ExtractCallError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def output_text(body: dict) -> str:
    direct = body.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        for part in item.get("content") or []:
            if (
                isinstance(part, dict)
                and part.get("type") in ("output_text", "text")
                and isinstance(part.get("text"), str)
            ):
                parts.append(part["text"])
    return "".join(parts)


def call_model(
    file_id: str,
    prompt: str,
    settings: Settings,
    *,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_retries: int = 2,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Send the uploaded video and the prompt to the vision model; return its text."""
    payload = {
        "model": settings.spark_model,
        "max_output_tokens": max_output_tokens,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": file_id},
                ],
            }
        ],
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
        raise ExtractCallError(
            f"model call failed with HTTP {response.status_code}: "
            f"{response.text[:200]}",
            status_code=response.status_code,
        )
    try:
        body = response.json()
    except ValueError as exc:
        raise ExtractCallError("model response is not valid JSON") from exc
    if not isinstance(body, dict):
        raise ExtractCallError("model response is not a JSON object")
    if body.get("status") == "incomplete":
        details = body.get("incomplete_details")
        reason = details.get("reason") if isinstance(details, dict) else None
        raise ExtractCallError(
            f"response incomplete ({reason}); raise max_output_tokens"
        )
    text = output_text(body)
    if not text.strip():
        raise ExtractCallError("model returned no text")
    return text


def extract_claims(
    file_id: str, transcript: Transcript, settings: Settings, **call_kwargs
) -> tuple[ClaimSet, list[str]]:
    """Call the model, parse its claims, and apply the checks that need no model."""
    text = call_model(
        file_id, build_extraction_prompt(transcript), settings, **call_kwargs
    )
    claim_set, errors = parse_model_output(text)
    checked = apply_checks(claim_set, transcript, transcript.audio_duration_ms / 1000)
    return checked, errors


def run_extract_stage(
    silent_video: Path,
    transcript: Transcript,
    work_dir: Path,
    settings: Settings,
    *,
    expires_seconds: int = 86_400,
) -> tuple[ClaimSet, list[str], str, Path]:
    """Upload the audio-free video, extract claims, write <stem>.claims.json.

    The uploaded file is kept so later stages can reuse its ID; the caller
    deletes it at the end. If this stage fails, the file is deleted here.
    """
    silent_video = Path(silent_video)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    file_id = upload_video(silent_video, settings, expires_seconds=expires_seconds)
    try:
        claim_set, errors = extract_claims(file_id, transcript, settings)
    except BaseException:
        try:
            delete_file(file_id, settings)
        except FilesError:
            pass
        raise

    stem = silent_video.stem.removesuffix(".silent")
    out = work_dir / f"{stem}.claims.json"
    out.write_text(
        json.dumps(
            {"claims": claim_set.model_dump(mode="json")["claims"], "parse_errors": errors},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return claim_set, errors, file_id, out
