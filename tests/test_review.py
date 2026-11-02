from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.review import (
    Decision,
    ReviewStore,
    load_store,
    pending_claims,
    record_decision,
    save_store,
    undo_decision,
)

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)


def make_claim(claim_id, status="proposed", value="bicycle"):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": "object",
            "value": value,
            "evidence": [{"kind": "visual", "start_s": 1.0, "end_s": 2.0}],
            "status": status,
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


def test_accept_records_a_timestamped_decision():
    store = record_decision(ReviewStore(), CLAIMS, "b", Decision.ACCEPTED, now=NOW)
    entry = store.decisions["b"]
    assert entry.decision == Decision.ACCEPTED
    assert entry.reviewed_at == "2026-10-30T12:00:00+00:00"
    assert entry.value is None


def test_edit_needs_a_value_and_trims_it():
    store = record_decision(
        ReviewStore(), CLAIMS, "b", Decision.EDITED, value="  red bicycle ", now=NOW
    )
    assert store.decisions["b"].value == "red bicycle"
    for bad in (None, "", "   "):
        with pytest.raises(ValidationError):
            record_decision(ReviewStore(), CLAIMS, "b", Decision.EDITED, value=bad)


def test_value_is_ignored_unless_the_decision_is_an_edit():
    store = record_decision(
        ReviewStore(), CLAIMS, "b", Decision.ACCEPTED, value="ignored", now=NOW
    )
    assert store.decisions["b"].value is None


def test_value_and_note_have_length_limits():
    with pytest.raises(ValidationError):
        record_decision(ReviewStore(), CLAIMS, "b", Decision.EDITED, value="x" * 201)
    with pytest.raises(ValidationError):
        record_decision(ReviewStore(), CLAIMS, "b", Decision.ACCEPTED, note="x" * 501)


def test_unknown_claim_is_an_error():
    with pytest.raises(KeyError):
        record_decision(ReviewStore(), CLAIMS, "zzz", Decision.ACCEPTED)


def test_recording_does_not_mutate_the_previous_store():
    first = ReviewStore()
    second = record_decision(first, CLAIMS, "b", Decision.ACCEPTED, now=NOW)
    assert first.decisions == {}
    assert "b" in second.decisions


def test_undo_removes_a_decision_and_ignores_missing_ones():
    store = record_decision(ReviewStore(), CLAIMS, "b", Decision.REJECTED, now=NOW)
    assert undo_decision(store, "b").decisions == {}
    assert undo_decision(store, "zzz").decisions == store.decisions


def test_pending_excludes_verified_and_decided_claims():
    assert [c.claim_id for c in pending_claims(CLAIMS, ReviewStore())] == ["b", "c", "d"]
    store = record_decision(ReviewStore(), CLAIMS, "b", Decision.REJECTED, now=NOW)
    assert [c.claim_id for c in pending_claims(CLAIMS, store)] == ["c", "d"]


def test_save_and_load_round_trip_without_leftovers(tmp_path):
    path = tmp_path / "work" / "clip.review.json"
    store = record_decision(
        ReviewStore(), CLAIMS, "c", Decision.EDITED, value="red chair", note="n", now=NOW
    )
    save_store(path, store)
    assert load_store(path) == store
    assert list(path.parent.glob("*.tmp")) == []


def test_missing_review_file_means_no_decisions(tmp_path):
    assert load_store(tmp_path / "nope.json") == ReviewStore()
