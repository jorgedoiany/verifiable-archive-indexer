import json

import pytest

from app import search as search_module
from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.review import Decision, ReviewStore, record_decision, save_store
from app.search import matches, search_catalog

TRANSCRIPT = {
    "transcript": "Estamos en Santo Domingo en el Malecon",
    "audioDurationMs": 6000,
    "turns": [
        {
            "turnId": 1,
            "startMs": 1500,
            "endMs": 5000,
            "transcript": "Estamos en Santo Domingo en el Malecon",
        }
    ],
}
SPEECH = [{"kind": "speech", "turn_id": 1, "quote": "Santo Domingo"}]


def visual(start, end):
    return [{"kind": "visual", "start_s": start, "end_s": end}]


def make_claim(claim_id, category, value, status, evidence):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value,
            "evidence": evidence,
            "status": status,
        }
    )


CLAIMS = [
    make_claim("a", "object", "red bicycle", "verified", visual(4.0, 7.0)),
    make_claim("b", "object", "chair", "unsupported", visual(1.0, 2.0)),
    make_claim("c", "place", "Santo Domingo", "proposed", SPEECH),
    make_claim("d", "scene", "Malecón", "verified", SPEECH),
]


def write_run(work, stem, claims):
    work.mkdir(parents=True, exist_ok=True)
    report = {"claims": [claim.model_dump(mode="json") for claim in claims]}
    (work / f"{stem}.verified.json").write_text(json.dumps(report), encoding="utf-8")
    (work / f"{stem}.transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")


@pytest.fixture
def work(tmp_path):
    folder = tmp_path / "work"
    write_run(folder, "clip_a", CLAIMS)
    return folder


def values(result):
    return [hit.view.value for hit in result.hits]


@pytest.mark.parametrize(
    ("value", "query", "expected"),
    [
        ("stack of books", "book", True),
        ("stack of BOOKS", "books", True),
        ("Malecón", "malecon", True),
        ("red bicycle", "bicycle red", True),
        ("red bicycle", "red chair", False),
        ("scarf", "car", False),
        ("libros", "books", False),
        ("anything", "", True),
        ("anything", "  ...  ", True),
    ],
)
def test_matches(value, query, expected):
    assert matches(value, query) is expected


def test_only_claims_the_checks_or_the_archivist_confirmed_are_listed(work):
    result = search_catalog(work)
    assert values(result) == ["red bicycle", "Malecón"]
    assert (result.total, result.clips_searched, result.clips_unreadable) == (2, 1, 0)


def test_claims_that_need_review_appear_only_when_asked(work):
    result = search_catalog(work, include_pending=True)
    assert values(result) == ["red bicycle", "chair", "Santo Domingo", "Malecón"]
    pending = [hit.view.state for hit in result.hits]
    assert pending == ["verified", "pending", "pending", "verified"]


def test_rejected_claims_never_appear(work):
    store = record_decision(ReviewStore(), ClaimSet(claims=CLAIMS), "b", Decision.REJECTED)
    save_store(work / "clip_a.review.json", store)
    result = search_catalog(work, include_pending=True)
    assert "chair" not in values(result)
    assert result.total == 3


def test_accepted_and_edited_claims_are_found_by_their_final_value(work):
    claim_set = ClaimSet(claims=CLAIMS)
    store = record_decision(ReviewStore(), claim_set, "c", Decision.ACCEPTED)
    store = record_decision(store, claim_set, "b", Decision.EDITED, value="red chair")
    save_store(work / "clip_a.review.json", store)

    santo = search_catalog(work, "santo")
    assert [h.view.state for h in santo.hits] == ["accepted"]

    red = search_catalog(work, "red")
    assert values(red) == ["red bicycle", "red chair"]
    edited = red.hits[1].view
    assert (edited.state, edited.original_value) == ("edited", "chair")


def test_filter_by_category_and_reject_unknown_ones(work):
    result = search_catalog(work, category="place", include_pending=True)
    assert values(result) == ["Santo Domingo"]
    with pytest.raises(ValueError):
        search_catalog(work, category="colour")


def test_several_clips_are_listed_in_clip_order(work):
    write_run(
        work,
        "clip_b",
        [make_claim("x", "object", "red kettle", "verified", visual(2.0, 3.0))],
    )
    result = search_catalog(work, "red")
    assert [(h.stem, h.view.value) for h in result.hits] == [
        ("clip_a", "red bicycle"),
        ("clip_b", "red kettle"),
    ]
    assert result.clips_searched == 2


def test_results_are_capped_but_the_total_is_kept(work, monkeypatch):
    monkeypatch.setattr(search_module, "MAX_RESULTS", 1)
    result = search_catalog(work)
    assert len(result.hits) == 1
    assert result.total == 2


def test_an_unreadable_run_is_counted_and_the_others_still_searched(work):
    (work / "clip_c.verified.json").write_text("{broken", encoding="utf-8")
    (work / "clip_c.transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    result = search_catalog(work)
    assert result.clips_unreadable == 1
    assert result.clips_searched == 1
    assert result.total == 2


def test_a_folder_without_runs_has_no_results(tmp_path):
    result = search_catalog(tmp_path / "nothing")
    assert (result.total, result.clips_searched) == (0, 0)


def test_clips_with_matches_counts_distinct_clips(work):
    write_run(
        work,
        "clip_b",
        [make_claim("x", "object", "red kettle", "verified", visual(2.0, 3.0))],
    )
    both = search_catalog(work, "red")
    assert (both.total, both.clips_with_matches, both.clips_searched) == (2, 2, 2)
    one = search_catalog(work, "bicycle")
    assert (one.total, one.clips_with_matches, one.clips_searched) == (1, 1, 2)
    none = search_catalog(work, "zebra")
    assert (none.total, none.clips_with_matches) == (0, 0)
