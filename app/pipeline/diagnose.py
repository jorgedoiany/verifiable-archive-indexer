"""One line per claim and per ground-truth item, to see why metrics came out as they did."""
from __future__ import annotations

from app.pipeline.claims import ClaimSet, ClaimStatus, VisualEvidence
from app.pipeline.evaluate import GroundTruth, claim_matches_item


def describe_run(claim_set: ClaimSet, truth: GroundTruth) -> list[str]:
    lines = ["CLAIMS: id | category | status | vs truth | value | evidence | notes"]
    for claim in claim_set.claims:
        correct = any(claim_matches_item(claim, item) for item in truth.items)
        spans = "; ".join(
            f"{e.start_s:.0f}-{e.end_s:.0f}s"
            for e in claim.evidence
            if isinstance(e, VisualEvidence)
        )
        kinds = "+".join(sorted({e.kind for e in claim.evidence}))
        lines.append(
            f"{claim.claim_id} | {claim.category.value} | {claim.status.value} | "
            f"{'TRUE' if correct else 'WRONG'} | {claim.value!r} | "
            f"{kinds} {spans} | {claim.notes or ''}"
        )

    lines.append("TRUTH ITEMS: state | category | first name")
    verified = [c for c in claim_set.claims if c.status == ClaimStatus.VERIFIED]
    for item in truth.items:
        if any(claim_matches_item(c, item) for c in verified):
            state = "verified"
        elif any(claim_matches_item(c, item) for c in claim_set.claims):
            state = "claimed, not verified"
        else:
            state = "no claim"
        lines.append(f"{state} | {item.category.value} | {item.names[0]}")
    return lines
