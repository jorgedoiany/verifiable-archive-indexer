"""View models for the review page: pure functions, no web framework."""
from __future__ import annotations

from dataclasses import dataclass

from app.pipeline.claims import (
    Claim,
    ClaimSet,
    ClaimStatus,
    SpeechEvidence,
    VisualEvidence,
)
from app.pipeline.models import Transcript
from app.pipeline.review import ReviewStore

STATE_ORDER = ("pending", "verified", "accepted", "edited", "rejected")


@dataclass(frozen=True, slots=True)
class EvidenceView:
    kind: str
    text: str
    seek: float | None


@dataclass(frozen=True, slots=True)
class ClaimView:
    position: int
    claim_id: str
    category: str
    value: str
    original_value: str | None
    state: str
    auto_status: str
    notes: str | None
    review_note: str | None
    confidence: float | None
    decided_at: str | None
    evidence: tuple[EvidenceView, ...]


def state_of(claim: Claim, store: ReviewStore) -> str:
    decision = store.decisions.get(claim.claim_id)
    if decision is not None:
        return decision.decision.value
    return "verified" if claim.status == ClaimStatus.VERIFIED else "pending"


def _evidence_views(claim: Claim, transcript: Transcript) -> tuple[EvidenceView, ...]:
    views: list[EvidenceView] = []
    for item in claim.evidence:
        if isinstance(item, SpeechEvidence):
            turn = transcript.turn_by_id(item.turn_id)
            if turn is None:
                span, seek = " (turn not found)", None
            else:
                span = f" ({turn.start_ms / 1000:.1f}-{turn.end_ms / 1000:.1f} s)"
                seek = turn.start_ms / 1000
            views.append(
                EvidenceView("speech", f'Turn {item.turn_id}{span}: "{item.quote}"', seek)
            )
        elif isinstance(item, VisualEvidence):
            text = f"Visual {item.start_s:.1f}-{item.end_s:.1f} s"
            if item.description:
                text += f": {item.description}"
            views.append(EvidenceView("visual", text, item.start_s))
    return tuple(views)


def build_views(
    claim_set: ClaimSet, transcript: Transcript, store: ReviewStore
) -> list[ClaimView]:
    """One view per claim, ordered: needs review first, rejected last."""
    views: list[ClaimView] = []
    for position, claim in enumerate(claim_set.claims):
        decision = store.decisions.get(claim.claim_id)
        edited = decision is not None and decision.value is not None
        value = decision.value if edited and decision and decision.value else claim.value
        views.append(
            ClaimView(
                position=position,
                claim_id=claim.claim_id,
                category=claim.category.value,
                value=value,
                original_value=claim.value if edited else None,
                state=state_of(claim, store),
                auto_status=claim.status.value,
                notes=claim.notes,
                review_note=decision.note if decision else None,
                confidence=claim.model_confidence,
                decided_at=decision.reviewed_at if decision else None,
                evidence=_evidence_views(claim, transcript),
            )
        )
    return sorted(views, key=lambda v: (STATE_ORDER.index(v.state), v.position))


def count_states(views: list[ClaimView]) -> dict[str, int]:
    counts = dict.fromkeys(STATE_ORDER, 0)
    for view in views:
        counts[view.state] += 1
    return counts
