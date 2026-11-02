"""Load a pipeline run (claims and transcript) from the work directory."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.models import Transcript, parse_transcript

STEM = re.compile(r"[A-Za-z0-9_-]{1,64}")


class _Verified(BaseModel):
    claims: list[Claim]


@dataclass(frozen=True, slots=True)
class RunData:
    stem: str
    claim_set: ClaimSet
    transcript: Transcript


def is_valid_stem(stem: str) -> bool:
    return STEM.fullmatch(stem) is not None


def list_stems(work_dir: Path) -> list[str]:
    work_dir = Path(work_dir)
    stems = []
    for path in sorted(work_dir.glob("*.verified.json")):
        stem = path.name.removesuffix(".verified.json")
        if is_valid_stem(stem) and (work_dir / f"{stem}.transcript.json").exists():
            stems.append(stem)
    return stems


def load_run(stem: str, work_dir: Path) -> RunData:
    if not is_valid_stem(stem):
        raise FileNotFoundError(f"invalid clip name: {stem!r}")
    work_dir = Path(work_dir)
    verified = work_dir / f"{stem}.verified.json"
    transcript = work_dir / f"{stem}.transcript.json"
    for path in (verified, transcript):
        if not path.exists():
            raise FileNotFoundError(f"missing file: {path}")
    report = _Verified.model_validate_json(verified.read_text(encoding="utf-8"))
    return RunData(
        stem=stem,
        claim_set=ClaimSet(claims=report.claims),
        transcript=parse_transcript(
            json.loads(transcript.read_text(encoding="utf-8"))
        ),
    )
