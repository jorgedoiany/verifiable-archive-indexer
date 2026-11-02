"""Archivist review: decisions on claims, stored beside the pipeline output."""
from __future__ import annotations

import os
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from app.pipeline.claims import Claim, ClaimSet, ClaimStatus


class Decision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"


class ReviewDecision(BaseModel):
    claim_id: str = Field(min_length=1)
    decision: Decision
    value: str | None = Field(default=None, max_length=200)
    note: str | None = Field(default=None, max_length=500)
    reviewed_at: str

    @model_validator(mode="after")
    def _value_only_when_edited(self) -> ReviewDecision:
        if self.decision == Decision.EDITED:
            if not self.value or not self.value.strip():
                raise ValueError("an edit needs a non-empty value")
            self.value = self.value.strip()
        else:
            self.value = None
        return self


class ReviewStore(BaseModel):
    decisions: dict[str, ReviewDecision] = Field(default_factory=dict)


def load_store(path: Path) -> ReviewStore:
    path = Path(path)
    if not path.exists():
        return ReviewStore()
    return ReviewStore.model_validate_json(path.read_text(encoding="utf-8"))


def save_store(path: Path, store: ReviewStore) -> None:
    """Write atomically so a crash never leaves a half-written review file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(store.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp, path)


def record_decision(
    store: ReviewStore,
    claim_set: ClaimSet,
    claim_id: str,
    decision: Decision,
    *,
    value: str | None = None,
    note: str | None = None,
    now: datetime | None = None,
) -> ReviewStore:
    """Return a new store with the decision recorded; the old one is unchanged."""
    if claim_id not in {claim.claim_id for claim in claim_set.claims}:
        raise KeyError(f"unknown claim: {claim_id}")
    stamp = (now or datetime.now(UTC)).isoformat()
    entry = ReviewDecision(
        claim_id=claim_id,
        decision=decision,
        value=value,
        note=note,
        reviewed_at=stamp,
    )
    return ReviewStore(decisions={**store.decisions, claim_id: entry})


def undo_decision(store: ReviewStore, claim_id: str) -> ReviewStore:
    remaining = {k: v for k, v in store.decisions.items() if k != claim_id}
    return ReviewStore(decisions=remaining)


def pending_claims(claim_set: ClaimSet, store: ReviewStore) -> list[Claim]:
    """Claims that automatic checks did not verify and nobody has decided yet."""
    return [
        claim
        for claim in claim_set.claims
        if claim.status != ClaimStatus.VERIFIED
        and claim.claim_id not in store.decisions
    ]
