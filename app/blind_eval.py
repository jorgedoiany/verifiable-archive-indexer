"""Pooled measurements for a batch run, computed against frozen ground truth.

Pooling adds the counts of all clips before dividing, so a clip whose reference
has a single word cannot swing the figure. Nothing here looks at the content of
a claim: only counts are produced.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from app.judge_report import wilson_interval
from app.pipeline.claims import Claim, ClaimCategory, ClaimStatus
from app.pipeline.evaluate import claim_matches_item, load_ground_truth, normalize_text
from app.pipeline.models import parse_transcript


class _Verified(BaseModel):
    claims: list[Claim]


@dataclass(frozen=True, slots=True)
class ClipMeasure:
    stem: str
    reference_words: int
    hypothesis_words: int
    edits: int
    truth_objects: int
    matched_by_any_claim: int
    matched_by_checks_agree: int
    object_claims: int
    frames_billed: int
    cost: float


def word_edit_stats(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (word edits, reference words); extra words on an empty reference count as edits."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1], len(ref)


def measure_clip(stem: str, work_dir: Path, truth_dir: Path) -> ClipMeasure:
    work_dir = Path(work_dir)
    verified = _Verified.model_validate_json(
        (work_dir / f"{stem}.verified.json").read_text(encoding="utf-8")
    )
    transcript = parse_transcript(
        json.loads((work_dir / f"{stem}.transcript.json").read_text(encoding="utf-8"))
    )
    result = json.loads((work_dir / f"{stem}.result.json").read_text(encoding="utf-8"))
    truth = load_ground_truth(Path(truth_dir) / f"{stem}.json")

    object_claims = [c for c in verified.claims if c.category == ClaimCategory.OBJECT]
    agreeing = [c for c in object_claims if c.status == ClaimStatus.VERIFIED]
    truth_objects = [i for i in truth.items if i.category == ClaimCategory.OBJECT]
    edits, reference_words = word_edit_stats(truth.transcript, transcript.transcript)

    return ClipMeasure(
        stem=stem,
        reference_words=reference_words,
        hypothesis_words=len(normalize_text(transcript.transcript).split()),
        edits=edits,
        truth_objects=len(truth_objects),
        matched_by_any_claim=sum(
            any(claim_matches_item(c, item) for c in object_claims)
            for item in truth_objects
        ),
        matched_by_checks_agree=sum(
            any(claim_matches_item(c, item) for c in agreeing)
            for item in truth_objects
        ),
        object_claims=len(object_claims),
        frames_billed=int(result["sam"]["frames_billed"]),
        cost=float(result["sam"]["actual_cost"]),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _interval(numerator: int, denominator: int) -> list[float] | None:
    interval = wilson_interval(numerator, denominator)
    return None if interval is None else [interval[0], interval[1]]


def pool(measures: Sequence[ClipMeasure]) -> dict:
    """Add the counts of every clip, then divide."""
    reference = sum(m.reference_words for m in measures)
    edits = sum(m.edits for m in measures)
    truth_objects = sum(m.truth_objects for m in measures)
    by_any = sum(m.matched_by_any_claim for m in measures)
    by_agree = sum(m.matched_by_checks_agree for m in measures)
    silent = [m for m in measures if m.reference_words == 0]
    return {
        "clips": len(measures),
        "reference_words": reference,
        "edits": edits,
        "word_error_rate": _ratio(edits, reference),
        "silent_clips": len(silent),
        "invented_words_on_silent_clips": sum(m.hypothesis_words for m in silent),
        "truth_objects": truth_objects,
        "matched_by_any_claim": by_any,
        "coverage_any_claim": _ratio(by_any, truth_objects),
        "coverage_any_claim_interval_95": _interval(by_any, truth_objects),
        "matched_by_checks_agree": by_agree,
        "coverage_checks_agree": _ratio(by_agree, truth_objects),
        "coverage_checks_agree_interval_95": _interval(by_agree, truth_objects),
        "object_claims": sum(m.object_claims for m in measures),
        "frames_billed": sum(m.frames_billed for m in measures),
        "cost": round(sum(m.cost for m in measures), 4),
    }
