import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.judging import (
    ItemView,
    JudgingDataError,
    JudgmentStore,
    Verdict,
    build_item_view,
    build_sheet,
    cited_range,
    load_claims,
    load_judgments,
    load_sheet,
    next_unjudged,
    progress,
    record_judgment,
    save_judgments,
)
from app.pipeline.claims import Claim

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
SPEECH = {"kind": "speech", "turn_id": 1, "quote": "hola"}


def visual(start, end):
    return {"kind": "visual", "start_s": start, "end_s": end}


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


RUN_A = [
    make_claim("a1", "object", "bicycle", "verified", [visual(4.0, 7.0)]),
    make_claim("a2", "place", "Santo Domingo", "proposed", [SPEECH]),
    make_claim("a3", "object", "chair", "unsupported", [visual(1.0, 2.0)]),
]
RUN_B = [
    make_claim("b1", "object", "lamp", "disputed", [SPEECH, visual(10.0, 12.0)]),
    make_claim("b2", "object", "table", "proposed", [visual(0.0, 3.0)]),
]


def write_run(work, stem, claims):
    work.mkdir(parents=True, exist_ok=True)
    report = {"claims": [claim.model_dump(mode="json") for claim in claims]}
    (work / f"{stem}.verified.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def work(tmp_path):
    folder = tmp_path / "work"
    write_run(folder, "blind_01", RUN_A)
    write_run(folder, "blind_02", RUN_B)
    return folder


def test_the_sheet_has_only_object_claims_anonymous_ids_and_no_verdicts(work):
    sheet = build_sheet(work, seed=3)
    assert sorted((i.stem, i.claim_id) for i in sheet.items) == [
        ("blind_01", "a1"),
        ("blind_01", "a3"),
        ("blind_02", "b1"),
        ("blind_02", "b2"),
    ]
    assert [i.item_id for i in sheet.items] == ["j001", "j002", "j003", "j004"]
    assert set(sheet.items[0].model_dump()) == {"item_id", "stem", "claim_id"}


def test_the_same_seed_gives_the_same_order(tmp_path, work):
    other = tmp_path / "other"
    write_run(other, "blind_01", RUN_A)
    write_run(other, "blind_02", RUN_B)
    assert build_sheet(work, seed=5) == build_sheet(other, seed=5)


def test_the_sheet_is_saved_never_redrawn_and_kept_out_of_git(work):
    sheet = build_sheet(work, seed=1)
    assert load_sheet(work) == sheet
    assert (work / "judging" / ".gitignore").read_text(encoding="utf-8") == "*\n"
    before = (work / "judging" / "sheet.json").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_sheet(work, seed=2)
    assert (work / "judging" / "sheet.json").read_text(encoding="utf-8") == before


def test_no_object_claims_is_an_error(tmp_path):
    folder = tmp_path / "work"
    write_run(folder, "blind_01", [RUN_A[1]])
    with pytest.raises(ValueError, match="no object claims"):
        build_sheet(folder, seed=1)
    assert not (folder / "judging").exists()


def test_files_with_invalid_names_are_ignored(work):
    write_run(work, "bad name", RUN_B)
    sheet = build_sheet(work, seed=1)
    assert {item.stem for item in sheet.items} == {"blind_01", "blind_02"}


def test_cited_range_spans_the_visual_evidence_only():
    assert cited_range(RUN_B[0]) == (10.0, 12.0)
    two = make_claim("x", "object", "x", "proposed", [visual(2.0, 3.0), visual(8.0, 9.5)])
    assert cited_range(two) == (2.0, 9.5)
    speech_only = make_claim("y", "object", "y", "proposed", [SPEECH])
    assert cited_range(speech_only) is None


def test_load_claims_rejects_invalid_names_and_missing_runs(work):
    with pytest.raises(FileNotFoundError):
        load_claims(work, "../etc")
    with pytest.raises(FileNotFoundError):
        load_claims(work, "blind_99")
    assert [c.claim_id for c in load_claims(work, "blind_01").claims] == [
        "a1",
        "a2",
        "a3",
    ]


def test_a_judgment_is_recorded_and_can_be_changed(work):
    sheet = build_sheet(work, seed=1)
    item_id = sheet.items[0].item_id
    store = record_judgment(
        JudgmentStore(), sheet, item_id, Verdict.EXISTS, note="  clear  ", now=NOW
    )
    entry = store.judgments[item_id]
    assert (entry.verdict, entry.note) == (Verdict.EXISTS, "clear")
    assert entry.judged_at == "2026-11-01T12:00:00+00:00"
    changed = record_judgment(store, sheet, item_id, Verdict.NOT_EXISTS, note="   ", now=NOW)
    assert changed.judgments[item_id].verdict == Verdict.NOT_EXISTS
    assert changed.judgments[item_id].note is None
    assert store.judgments[item_id].verdict == Verdict.EXISTS  # the old store is untouched


def test_unknown_items_and_long_notes_are_rejected(work):
    sheet = build_sheet(work, seed=1)
    with pytest.raises(KeyError):
        record_judgment(JudgmentStore(), sheet, "j999", Verdict.EXISTS)
    with pytest.raises(ValidationError):
        record_judgment(JudgmentStore(), sheet, "j001", Verdict.EXISTS, note="x" * 201)


def test_judgments_round_trip_without_leftovers(work):
    assert load_judgments(work) == JudgmentStore()
    sheet = build_sheet(work, seed=1)
    store = record_judgment(JudgmentStore(), sheet, "j002", Verdict.UNSURE, now=NOW)
    save_judgments(work, store)
    assert load_judgments(work) == store
    assert list((work / "judging").glob("*.tmp")) == []


def test_next_unjudged_skips_judged_items_and_wraps_around(work):
    sheet = build_sheet(work, seed=1)
    store = JudgmentStore()
    assert next_unjudged(sheet, store).item_id == "j001"
    store = record_judgment(store, sheet, "j001", Verdict.EXISTS, now=NOW)
    assert next_unjudged(sheet, store).item_id == "j002"
    assert next_unjudged(sheet, store, after="j004").item_id == "j002"
    assert next_unjudged(sheet, store, after="j002").item_id == "j003"
    assert next_unjudged(sheet, store, after="zzz").item_id == "j002"
    for item_id in ("j002", "j003", "j004"):
        store = record_judgment(store, sheet, item_id, Verdict.UNSURE, now=NOW)
    assert next_unjudged(sheet, store) is None


def test_progress_counts_only_items_of_the_sheet(work):
    sheet = build_sheet(work, seed=1)
    store = record_judgment(JudgmentStore(), sheet, "j001", Verdict.EXISTS, now=NOW)
    assert progress(sheet, store) == (1, 4)
    assert progress(sheet, JudgmentStore()) == (0, 4)


def test_item_view_shows_the_claim_range_position_and_neighbours(work):
    sheet = build_sheet(work, seed=1)
    store = record_judgment(
        JudgmentStore(), sheet, "j002", Verdict.EXISTS, note="ok", now=NOW
    )
    first = build_item_view(work, sheet, store, "j001")
    assert (first.position, first.total, first.prev_id, first.next_id) == (1, 4, None, "j002")
    last = build_item_view(work, sheet, store, "j004")
    assert (last.position, last.prev_id, last.next_id) == (4, "j003", None)

    lamp_id = next(i.item_id for i in sheet.items if i.claim_id == "b1")
    lamp = build_item_view(work, sheet, store, lamp_id)
    assert (lamp.stem, lamp.value, lamp.start_s, lamp.end_s) == ("blind_02", "lamp", 10.0, 12.0)

    judged = build_item_view(work, sheet, store, "j002")
    assert (judged.verdict, judged.note) == ("exists", "ok")
    assert build_item_view(work, sheet, JudgmentStore(), "j002").verdict is None


def test_an_object_claim_without_visual_evidence_has_no_range(tmp_path):
    folder = tmp_path / "work"
    guitar = make_claim("c1", "object", "guitar", "proposed", [SPEECH])
    write_run(folder, "blind_01", [guitar])
    sheet = build_sheet(folder, seed=1)
    view = build_item_view(folder, sheet, JudgmentStore(), "j001")
    assert (view.start_s, view.end_s) == (None, None)


def test_unknown_items_and_vanished_claims(work):
    sheet = build_sheet(work, seed=1)
    with pytest.raises(KeyError):
        build_item_view(work, sheet, JudgmentStore(), "j999")
    write_run(work, "blind_01", [RUN_A[1]])  # the run no longer has those claims
    victim = next(i for i in sheet.items if i.stem == "blind_01")
    with pytest.raises(JudgingDataError):
        build_item_view(work, sheet, JudgmentStore(), victim.item_id)


def test_the_item_view_has_no_verifier_fields():
    fields = set(ItemView.__dataclass_fields__)
    assert not fields & {"status", "auto_status", "notes", "confidence", "model_confidence"}
