"""Live smoke test against the Meta Model API with one short clip.

Usage: python scripts/smoke_test.py data/clips/my_clip.mp4

Prints one line per step (OK or FAIL). Keep the clip under 60 seconds; it
costs a few cents. The API key is read from .env and never printed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_settings
from app.pipeline.extract import _output_text
from app.pipeline.files_api import delete_file, upload_video
from app.pipeline.media import strip_audio
from app.pipeline.transcript_stage import run_transcript_stage


def step(name, fn) -> bool:
    try:
        print(f"OK   {name}: {fn()}")
        return True
    except Exception as exc:  # report every failure on one line
        print(f"FAIL {name}: {type(exc).__name__}: {str(exc)[:200]}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()

    settings = load_settings()
    work = Path("data/work")
    headers = {"Authorization": f"Bearer {settings.api_key}"}
    state: dict = {}

    def list_models() -> str:
        response = httpx.get(f"{settings.api_base}/models", headers=headers, timeout=60)
        response.raise_for_status()
        ids = [m.get("id") for m in response.json().get("data", [])]
        return ", ".join(str(i) for i in ids[:12]) or "no models listed"

    if not step("models", list_models):
        return 1

    def transcribe() -> str:
        transcript, _ = run_transcript_stage(args.video, work, settings)
        first = transcript.turns[0].transcript[:80] if transcript.turns else "(no turns)"
        return (
            f"{len(transcript.turns)} turns, {transcript.audio_duration_ms} ms, "
            f"first: {first!r}"
        )

    step("transcribe", transcribe)

    def upload() -> str:
        silent = strip_audio(args.video, work / f"{args.video.stem}.silent.mp4")
        state["file_id"] = upload_video(silent, settings, expires_seconds=3600)
        return state["file_id"]

    if not step("upload", upload):
        return 1

    def vision() -> str:
        payload = {
            "model": settings.spark_model,
            "max_output_tokens": 4000,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "In one sentence, describe what is visible in this video.",
                        },
                        {"type": "input_file", "file_id": state["file_id"]},
                    ],
                }
            ],
        }
        response = httpx.post(
            f"{settings.api_base}/responses",
            headers={**headers, "Content-Type": "application/json"},
            json=payload,
            timeout=600,
        )
        response.raise_for_status()
        body = response.json()
        types = [i.get("type") for i in body.get("output") or [] if isinstance(i, dict)]
        text = _output_text(body)
        return (
            f"status={body.get('status')}, keys={sorted(body)[:12]}, "
            f"output types={types}, usage={body.get('usage')}, text={text[:160]!r}"
        )

    try:
        step("vision", vision)
    finally:
        step("delete", lambda: delete_file(state["file_id"], settings) or "deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
