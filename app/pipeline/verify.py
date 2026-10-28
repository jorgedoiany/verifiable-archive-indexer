"""Stage 3 decision logic: turn SAM results into claim statuses.

Pure functions with no network access: the stage runner calls SAM and feeds
the results here. Only OBJECT claims that still have status PROPOSED and carry
visual evidence are checked; other categories are left for human review.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from app.pipeline.claims import (
    Claim,
    ClaimCategory,
    ClaimSet,
    ClaimStatus,
    VisualEvidence,
)
from app.pipeline.sam import SamResult, clean_concept, estimate_cost
from app.pipeline.sam_parse import frames_present

DEFAULT_MAX_CONCEPTS = 12
DEFAULT_MIN_CONFIDENCE = 0.5
DEFAULT_MIN_FRAMES = 2


@dataclass(frozen=True, slots=True)
class VerifyPlan:
    concepts: tuple[str, ...]
    capped_claim_ids: tuple[str, ...]
    estimated_cost: float


def concept_key(concept: str) -> str:
    return concept.casefold()


def _visual(claim: Claim) -> list[VisualEvidence]:
    return [e for e in claim.evidence if isinstance(e, VisualEvidence)]


def needs_sam(claim: Claim) -> bool:
    return (
        claim.status == ClaimStatus.PROPOSED
        and claim.category == ClaimCategory.OBJECT
        and bool(_visual(claim))
    )


def concept_for(claim: Claim) -> str | None:
    try:
        return clean_concept(claim.value)
    except ValueError:
        return None


def plan_verification(
    claim_set: ClaimSet,
    *,
    duration_s: float,
    fps: float,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
) -> VerifyPlan:
    """List the concepts to send to SAM and estimate what they will cost."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    chosen: dict[str, str] = {}
    skipped_keys: set[str] = set()
    for claim in claim_set.claims:
        if not needs_sam(claim):
            continue
        concept = concept_for(claim)
        if concept is None:
            continue
        key = concept_key(concept)
        if key in chosen or key in skipped_keys:
            continue
        if len(chosen) < max_concepts:
            chosen[key] = concept
        else:
            skipped_keys.add(key)

    capped: list[str] = []
    for claim in claim_set.claims:
        if not needs_sam(claim):
            continue
        concept = concept_for(claim)
        if concept is not None and concept_key(concept) in skipped_keys:
            capped.append(claim.claim_id)

    frames = math.ceil(duration_s * fps)
    return VerifyPlan(
        concepts=tuple(chosen.values()),
        capped_claim_ids=tuple(capped),
        estimated_cost=estimate_cost(frames, requests=len(chosen)),
    )


def _with_note(claim: Claim, text: str, status: ClaimStatus | None = None) -> Claim:
    notes = f"{claim.notes}; {text}" if claim.notes else text
    update: dict = {"notes": notes}
    if status is not None:
        update["status"] = status
    return claim.model_copy(update=update)


def _frames_in_ranges(
    frames: Sequence[int], ranges: Sequence[VisualEvidence], fps: float
) -> list[int]:
    pad = 1 / fps  # tolerate one frame of slack on each side
    return [
        frame
        for frame in frames
        if any(r.start_s - pad <= frame / fps <= r.end_s + pad for r in ranges)
    ]


def verify_claims(
    claim_set: ClaimSet,
    results: Mapping[str, SamResult],
    *,
    fps: float,
    errors: Mapping[str, str] | None = None,
    capped_ids: Sequence[str] = (),
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> ClaimSet:
    """Set statuses from SAM results; `results` and `errors` are keyed by concept_key."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    errors = errors or {}
    capped = set(capped_ids)
    updated: list[Claim] = []

    for claim in claim_set.claims:
        if claim.status != ClaimStatus.PROPOSED or not _visual(claim):
            updated.append(claim)
            continue
        if claim.category != ClaimCategory.OBJECT:
            updated.append(
                _with_note(
                    claim,
                    "no automatic visual check for this category; needs review",
                )
            )
            continue
        concept = concept_for(claim)
        if concept is None:
            updated.append(
                _with_note(
                    claim,
                    "value is not a single short noun phrase; needs review",
                )
            )
            continue
        key = concept_key(concept)
        if claim.claim_id in capped:
            updated.append(_with_note(claim, "not checked: concept cap reached"))
        elif key in errors:
            updated.append(_with_note(claim, f"visual check failed: {errors[key]}"))
        elif key in results:
            present = frames_present(
                results[key].detections, min_confidence=min_confidence
            )
            in_range = _frames_in_ranges(present, _visual(claim), fps)
            if len(in_range) >= min_frames:
                text = f"SAM found '{concept}' in {len(in_range)} frames inside the cited range"
                updated.append(_with_note(claim, text, ClaimStatus.VERIFIED))
            elif present:
                text = f"SAM found '{concept}' elsewhere in the clip, not in the cited range"
                updated.append(_with_note(claim, text, ClaimStatus.DISPUTED))
            else:
                text = f"SAM found no match for '{concept}' anywhere in the clip"
                updated.append(_with_note(claim, text, ClaimStatus.UNSUPPORTED))
        else:
            updated.append(_with_note(claim, "not checked: no SAM result available"))
    return ClaimSet(claims=updated)
