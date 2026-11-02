import csv
import io
import json
from datetime import UTC, datetime

import pytest

from app.pipeline.catalog import (
    build_catalog,
    catalog_to_csv,
    catalog_to_json,
    describe_evidence,
    export_catalog,
)
from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.review import (
    Decision,
    ReviewDecision,
    ReviewStore,
    record_decision,
    save_store,
)

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def make_claim(claim_id, status="proposed", value="bicycle", speech=False):
    if speech:
        evidence = [{"kind": "speech", "turn_id": 3, "quote": "mi bicicleta"}]
    else:
        evidence = [{"kind": "visual", "start_s": 1.0, "end_s": 2.0}]
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": "object",
            "value": value,
            "evidence": evidence,
            "status": status,
            "model_confidence": 0.8,
        }
    )


CLAIMS = ClaimSet(
    claims=[
        make_claim("a", "verified"),
        make_claim("b"),
        make_claim("c", "unsupported"),
        make_claim("d", "disputed"),
    ]
)


def decided_store():
    store = record_decision(
        ReviewStore(), CLAIMS, "b", Decision.ACCEPTED, note="looks right", now=NOW
    )
    store = record_decision(store, CLAIMS, "c", Decision.EDITED, value="red chair", now=NOW)
    return record_decision(store, CLAIMS, "d", Decision.REJECTED, now=NOW)


def test_catalog_separates_automatic_archivist_and_rejected():
    catalog = build_catalog("clip", CLAIMS, decided_store())
    by_id = {e.claim_id: e for e in catalog.entries}
    assert list(by_id) == ["a", "b", "c"]
    assert (by_id["a"].final_status, by_id["a"].origin) == ("checks_agree", "automatic checks")
    assert (by_id["b"].final_status, by_id["b"].origin) == ("accepted", "archivist")
    assert by_id["b"].review_note == "looks right"
    assert by_id["c"].final_status == "edited"
    assert by_id["c"].value == "red chair"
    assert by_id["c"].original_value == "bicycle"
    assert catalog.rejected_claim_ids == ["d"]
    assert catalog.pending_claim_ids == []


def test_unreviewed_claims_are_listed_but_not_exported_by_default():
    catalog = build_catalog("clip", CLAIMS, ReviewStore())
    assert [e.claim_id for e in catalog.entries] == ["a"]
    assert catalog.pending_claim_ids == ["b", "c", "d"]


def test_including_pending_marks_them_clearly():
    catalog = build_catalog("clip", CLAIMS, ReviewStore(), include_pending=True)
    pending = [e for e in catalog.entries if e.final_status == "pending"]
    assert [e.claim_id for e in pending] == ["b", "c", "d"]
    assert all(e.origin == "unreviewed" for e in pending)


def test_a_decision_for_an_unknown_claim_is_ignored():
    stale = ReviewStore(
        decisions={
            "zzz": ReviewDecision(
                claim_id="zzz", decision=Decision.REJECTED, reviewed_at=NOW.isoformat()
            )
        }
    )
    catalog = build_catalog("clip", CLAIMS, stale)
    assert catalog.rejected_claim_ids == []


def test_csv_has_a_header_and_neutralizes_formulas():
    claims = ClaimSet(
        claims=[
            make_claim("a", "verified", value="=SUM(A1)"),
            make_claim("s", "verified", value="bike", speech=True),
        ]
    )
    text = catalog_to_csv(build_catalog("clip", claims, ReviewStore()))
    rows = list(csv.reader(io.StringIO(text)))
    assert rows[0][:4] == ["claim_id", "category", "value", "final_status"]
    assert rows[1][2] == "'=SUM(A1)"
    assert "visual 1.0-2.0s" in rows[1][6]
    assert "speech turn 3: 'mi bicicleta'" in rows[2][6]
    assert {len(row) for row in rows} == {10}


def test_json_export_is_valid_and_keeps_provenance():
    data = json.loads(catalog_to_json(build_catalog("clip", CLAIMS, decided_store())))
    assert data["clip"] == "clip"
    assert data["rejected_claim_ids"] == ["d"]
    assert data["entries"][1]["origin"] == "archivist"


def test_describe_evidence_formats_both_kinds():
    text = describe_evidence(
        [
            {"type": "speech", "turn_id": 1, "quote": "hola"},
            {"type": "visual", "start_s": 4.0, "end_s": 7.25, "description": None},
        ]
    )
    assert text == "speech turn 1: 'hola' | visual 4.0-7.2s"


def test_export_catalog_reads_the_run_and_the_review_from_disk(tmp_path):
    report = {"claims": [c.model_dump(mode="json") for c in CLAIMS.claims], "sam": {}}
    (tmp_path / "clip.verified.json").write_text(json.dumps(report), encoding="utf-8")
    save_store(tmp_path / "clip.review.json", decided_store())

    catalog, json_path, csv_path = export_catalog("clip", tmp_path)
    assert [e.claim_id for e in catalog.entries] == ["a", "b", "c"]
    assert json_path.exists() and csv_path.exists()
    assert len(list(csv.reader(io.StringIO(csv_path.read_text(encoding="utf-8"))))) == 4


def test_export_catalog_needs_a_verified_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_catalog("clip", tmp_path)


def test_exports_say_checks_agree_and_carry_the_disclaimer():
    catalog = build_catalog("clip", CLAIMS, ReviewStore())
    assert catalog.entries[0].final_status == "checks_agree"
    assert "not a guarantee" in catalog.note
    data = json.loads(catalog_to_json(catalog))
    assert "verified" not in {entry["final_status"] for entry in data["entries"]}
    assert "not a guarantee" in data["note"]
    assert "checks_agree" in catalog_to_csv(catalog)


def test_an_automatic_miss_never_discards_a_claim():
    catalog = build_catalog("clip", CLAIMS, ReviewStore(), include_pending=True)
    assert catalog.rejected_claim_ids == []
    pending = {e.claim_id for e in catalog.entries if e.final_status == "pending"}
    assert {"c", "d"} <= pending
