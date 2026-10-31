"""Run every stage on one clip: transcript, extraction, verification."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.pipeline.extract import run_extract_stage
from app.pipeline.files_api import FilesError, delete_file
from app.pipeline.media import strip_audio
from app.pipeline.models import Transcript
from app.pipeline.transcript_stage import run_transcript_stage
from app.pipeline.verify import DEFAULT_MAX_CONCEPTS
from app.pipeline.verify_stage import (
    DEFAULT_FPS,
    DEFAULT_MAX_COST,
    VerifyOutcome,
    run_verify_stage,
)


@dataclass(frozen=True, slots=True)
class PipelineResult:
    transcript: Transcript
    parse_errors: list[str]
    verified: VerifyOutcome
    status_counts: dict[str, int]
    summary_path: Path


def run_pipeline(
    video: Path,
    work_dir: Path,
    settings: Settings,
    *,
    fps: int = DEFAULT_FPS,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    max_cost: float = DEFAULT_MAX_COST,
    prompt_variant: str = "basic",
) -> PipelineResult:
    """Run the three stages and write <stem>.result.json next to the stage files."""
    video = Path(video)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    transcript, transcript_path = run_transcript_stage(video, work_dir, settings)

    silent = strip_audio(video, work_dir / f"{video.stem}.silent.mp4")
    claim_set, parse_errors, file_id, claims_path = run_extract_stage(
        silent, transcript, work_dir, settings, prompt_variant=prompt_variant
    )
    try:
        delete_file(file_id, settings)  # SAM uses its own low-fps upload
    except FilesError:
        pass  # the file expires on its own

    outcome = run_verify_stage(
        video,
        claim_set,
        work_dir,
        settings,
        fps=fps,
        max_concepts=max_concepts,
        max_cost=max_cost,
    )

    counts = dict(Counter(c.status.value for c in outcome.claim_set.claims))
    summary_path = work_dir / f"{video.stem}.result.json"
    summary = {
        "clip": video.name,
        "prompt_variant": prompt_variant,
        "audio_duration_ms": transcript.audio_duration_ms,
        "transcript_turns": len(transcript.turns),
        "claims_total": len(outcome.claim_set.claims),
        "status_counts": counts,
        "parse_errors": len(parse_errors),
        "sam": {
            "concepts": list(outcome.plan.concepts),
            "frames_billed": outcome.frames_billed,
            "actual_cost": outcome.actual_cost,
            "errors": len(outcome.errors),
        },
        "files": {
            "transcript": str(transcript_path),
            "claims": str(claims_path),
            "verified": str(outcome.output),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return PipelineResult(transcript, parse_errors, outcome, counts, summary_path)
