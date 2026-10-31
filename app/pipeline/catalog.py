"""Final catalog: what an archivist can trust, with provenance, in open formats."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.pipeline.claims import (
    Claim,
    ClaimSet,
    ClaimStatus,
    SpeechEvidence,
    VisualEvidence,
)
from app.pipeline.review import Decision, ReviewStore, load_store

CSV_COLUMNS = (
    "claim_id",
    "category",
    "value",
    "final_status",
    "origin",
    "model_confidence",
    "evidence",
    "original_value",
    "notes",
    "review_note",
)


CATALOG_NOTE = (
    "final_status 'checks_agree' means independent automatic checks agreed with the "
    "model's claim. It is not a guarantee that the claim is correct. "
    "'accepted' and 'edited' come from a human reviewer; 'pending' has "
    "not been reviewed."
)


class CatalogEntry(BaseModel):
    claim_id: str
    category: str
    value: str
    final_status: Literal["checks_agree", "accepted", "edited", "pending"]
    origin: Literal["automatic checks", "archivist", "unreviewed"]
    model_confidence: float | None = None
    original_value: str | None = None
    evidence: list[dict]
    notes: str | None = None
    review_note: str | None = None


class Catalog(BaseModel):
    clip: str
    entries: list[CatalogEntry]
    rejected_claim_ids: list[str]
    pending_claim_ids: list[str]
    note: str = CATALOG_NOTE


class _Report(BaseModel):
    claims: list[Claim]


def _evidence_dicts(claim: Claim) -> list[dict]:
    out: list[dict] = []
    for item in claim.evidence:
        if isinstance(item, SpeechEvidence):
            out.append({"type": "speech", "turn_id": item.turn_id, "quote": item.quote})
        elif isinstance(item, VisualEvidence):
            out.append(
                {
                    "type": "visual",
                    "start_s": item.start_s,
                    "end_s": item.end_s,
                    "description": item.description,
                }
            )
    return out


def build_catalog(
    clip: str,
    claim_set: ClaimSet,
    store: ReviewStore,
    *,
    include_pending: bool = False,
) -> Catalog:
    entries: list[CatalogEntry] = []
    rejected: list[str] = []
    pending: list[str] = []

    for claim in claim_set.claims:
        decision = store.decisions.get(claim.claim_id)
        original: str | None = None
        review_note: str | None = None

        if decision is not None and decision.decision == Decision.REJECTED:
            rejected.append(claim.claim_id)
            continue
        if decision is not None:
            edited = decision.decision == Decision.EDITED and bool(decision.value)
            status = "edited" if edited else "accepted"
            origin = "archivist"
            value = decision.value if edited and decision.value else claim.value
            original = claim.value if edited else None
            review_note = decision.note
        elif claim.status == ClaimStatus.VERIFIED:
            status, origin, value = "checks_agree", "automatic checks", claim.value
        else:
            pending.append(claim.claim_id)
            if not include_pending:
                continue
            status, origin, value = "pending", "unreviewed", claim.value

        entries.append(
            CatalogEntry(
                claim_id=claim.claim_id,
                category=claim.category.value,
                value=value,
                final_status=status,
                origin=origin,
                model_confidence=claim.model_confidence,
                original_value=original,
                evidence=_evidence_dicts(claim),
                notes=claim.notes,
                review_note=review_note,
            )
        )
    return Catalog(
        clip=clip,
        entries=entries,
        rejected_claim_ids=rejected,
        pending_claim_ids=pending,
    )


def describe_evidence(evidence: list[dict]) -> str:
    parts: list[str] = []
    for item in evidence:
        if item["type"] == "speech":
            parts.append(f"speech turn {item['turn_id']}: '{item['quote']}'")
        else:
            parts.append(f"visual {item['start_s']:.1f}-{item['end_s']:.1f}s")
    return " | ".join(parts)


def _safe_cell(value: object) -> str:
    """Neutralize spreadsheet formulas in text that came from a model or a person."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def catalog_to_json(catalog: Catalog) -> str:
    return catalog.model_dump_json(indent=2)


def catalog_to_csv(catalog: Catalog) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for entry in catalog.entries:
        cells = (
            entry.claim_id,
            entry.category,
            entry.value,
            entry.final_status,
            entry.origin,
            entry.model_confidence,
            describe_evidence(entry.evidence),
            entry.original_value,
            entry.notes,
            entry.review_note,
        )
        writer.writerow([_safe_cell(cell) for cell in cells])
    return buffer.getvalue()


def export_catalog(
    stem: str, work_dir: Path, *, include_pending: bool = False
) -> tuple[Catalog, Path, Path]:
    """Write <stem>.catalog.json and <stem>.catalog.csv from a verified run."""
    work_dir = Path(work_dir)
    verified_path = work_dir / f"{stem}.verified.json"
    if not verified_path.exists():
        raise FileNotFoundError(f"missing file: {verified_path}")
    report = _Report.model_validate_json(verified_path.read_text(encoding="utf-8"))
    store = load_store(work_dir / f"{stem}.review.json")
    catalog = build_catalog(
        stem, ClaimSet(claims=report.claims), store, include_pending=include_pending
    )
    json_path = work_dir / f"{stem}.catalog.json"
    csv_path = work_dir / f"{stem}.catalog.csv"
    json_path.write_text(catalog_to_json(catalog), encoding="utf-8")
    csv_path.write_text(catalog_to_csv(catalog), encoding="utf-8", newline="")
    return catalog, json_path, csv_path
