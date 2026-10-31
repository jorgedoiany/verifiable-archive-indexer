"""Stage 1: video -> speech audio -> turn-level transcript saved as JSON."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from app.config import Settings
from app.pipeline.media import extract_audio, has_audio, probe_duration
from app.pipeline.models import Transcript
from app.pipeline.transcribe import transcribe_file


def run_transcript_stage(
    video: Path,
    work_dir: Path,
    settings: Settings,
    *,
    keywords: Sequence[str] = (),
) -> tuple[Transcript, Path]:
    """Transcribe the clip's speech and write <stem>.transcript.json.

    A clip with no audio track has nothing to transcribe: it gets an empty
    transcript and the API is not called.
    """
    video = Path(video)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if has_audio(video):
        wav = extract_audio(video, work_dir / f"{video.stem}.wav")
        transcript = transcribe_file(wav, settings, keywords=keywords)
    else:
        transcript = Transcript(
            transcript="",
            audio_duration_ms=round(probe_duration(video) * 1000),
            turns=[],
        )

    out = work_dir / f"{video.stem}.transcript.json"
    out.write_text(
        transcript.model_dump_json(by_alias=True, indent=2), encoding="utf-8"
    )
    return transcript, out
