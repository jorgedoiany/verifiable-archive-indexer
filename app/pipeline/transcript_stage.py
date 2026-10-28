"""Stage 1: video -> speech audio -> turn-level transcript saved as JSON."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.config import Settings
from app.pipeline.media import extract_audio
from app.pipeline.models import Transcript
from app.pipeline.transcribe import transcribe_file


def run_transcript_stage(
    video: Path,
    work_dir: Path,
    settings: Settings,
    *,
    keywords: Sequence[str] = (),
) -> tuple[Transcript, Path]:
    """Extract audio, transcribe it, and write <stem>.transcript.json."""
    video = Path(video)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    wav = extract_audio(video, work_dir / f"{video.stem}.wav")
    transcript = transcribe_file(wav, settings, keywords=keywords)

    out = work_dir / f"{video.stem}.transcript.json"
    out.write_text(
        transcript.model_dump_json(by_alias=True, indent=2), encoding="utf-8"
    )
    return transcript, out
