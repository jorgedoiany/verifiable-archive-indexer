"""Compare pipeline output with a hand-labeled ground truth.

The ground truth lists what a perfect catalog of the clip would contain and the
reference transcript. Metrics: word error rate of the transcription, precision
of all claims (the model alone), precision of the verified claims, recall of
the verified claims, and how many correct claims were wrongly rejected.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.pipeline.claims import (
    Claim,
    ClaimCategory,
    ClaimSet,
    ClaimStatus,
    VisualEvidence,
    normalize_text,
)


class TruthItem(BaseModel):
    category: ClaimCategory
    names: list[str] = Field(min_length=1)
    start_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _range_is_complete_and_ordered(self) -> TruthItem:
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("start_s and end_s must be given together")
        if (
            self.start_s is not None
            and self.end_s is not None
            and self.end_s < self.start_s
        ):
            raise ValueError("end_s must not be earlier than start_s")
        return self


class GroundTruth(BaseModel):
    clip: str
    transcript: str
    items: list[TruthItem] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Metrics:
    wer: float | None
    claims_total: int
    precision_all: float | None
    verified_total: int
    precision_verified: float | None
    truth_total: int
    recall_verified: float | None
    rejected_total: int
    rejected_but_true: int
    review_queue: int
    by_status: dict[str, int]


def load_ground_truth(path: Path) -> GroundTruth:
    return GroundTruth.model_validate_json(Path(path).read_text(encoding="utf-8"))


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Word-level edit distance over the reference length; None without a reference."""
    ref = normalize_text(reference).split()
    hyp = normalize_text(hypothesis).split()
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[-1] / len(ref)


def _contains_phrase(haystack: str, needle: str) -> bool:
    return bool(needle) and f" {needle} " in f" {haystack} "


def name_matches(value: str, names: Sequence[str]) -> bool:
    """Whole-word containment in either direction, ignoring case and accents."""
    normalized = normalize_text(value)
    for name in names:
        candidate = normalize_text(name)
        if _contains_phrase(normalized, candidate) or _contains_phrase(
            candidate, normalized
        ):
            return True
    return False


def claim_matches_item(claim: Claim, item: TruthItem) -> bool:
    if claim.category != item.category or not name_matches(claim.value, item.names):
        return False
    if item.start_s is None:
        return True
    ranges = [e for e in claim.evidence if isinstance(e, VisualEvidence)]
    if not ranges:
        return True  # speech-only claim: the truth time range does not apply
    start = item.start_s
    end = item.end_s if item.end_s is not None else item.start_s
    return any(r.start_s <= end and r.end_s >= start for r in ranges)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate(
    claim_set: ClaimSet, transcript_text: str, truth: GroundTruth
) -> Metrics:
    claims = claim_set.claims

    def is_correct(claim: Claim) -> bool:
        return any(claim_matches_item(claim, item) for item in truth.items)

    verified = [c for c in claims if c.status == ClaimStatus.VERIFIED]
    rejected = [
        c for c in claims if c.status in (ClaimStatus.UNSUPPORTED, ClaimStatus.DISPUTED)
    ]
    matched_items = {
        index
        for index, item in enumerate(truth.items)
        if any(claim_matches_item(c, item) for c in verified)
    }
    by_status = Counter(c.status.value for c in claims)

    return Metrics(
        wer=word_error_rate(truth.transcript, transcript_text),
        claims_total=len(claims),
        precision_all=_ratio(sum(is_correct(c) for c in claims), len(claims)),
        verified_total=len(verified),
        precision_verified=_ratio(sum(is_correct(c) for c in verified), len(verified)),
        truth_total=len(truth.items),
        recall_verified=_ratio(len(matched_items), len(truth.items)),
        rejected_total=len(rejected),
        rejected_but_true=sum(is_correct(c) for c in rejected),
        review_queue=by_status.get(ClaimStatus.PROPOSED.value, 0),
        by_status=dict(by_status),
    )


def metrics_to_dict(metrics: Metrics) -> dict:
    return asdict(metrics)
