import json

import pytest

from app.judge_report import (
    GroupStats,
    summarize_judgments,
    summary_to_dict,
    wilson_interval,
)
from app.judging import (
    JudgingDataError,
    JudgmentStore,
    Verdict,
    build_sheet,
    record_judgment,
    save_judgments,
)
from app.pipeline.claims import Claim


def make_claim(claim_id, status, category="object"):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": f"thing {claim_id}",
            "evidence": [{"kind": "visual", "start_s": 1.0, "end_s": 2.0}],
            "status": status,
        }
    )


CLAIMS = [
    make_claim("a1", "verified"),
    make_claim("a2", "unsupported"),
    make_claim("a3", "disputed"),
    make_claim("a4", "proposed"),
    make_claim("p1", "proposed", category="place"),
]


def write_run(work, claims):
    work.mkdir(parents=True, exist_ok=True)
    report = {"claims": [claim.model_dump(mode="json") for claim in claims]}
    (work / "blind_01.verified.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def work(tmp_path):
    folder = tmp_path / "work"
    write_run(folder, CLAIMS)
    return folder


def judge(work, verdicts):
    sheet = build_sheet(work, seed=1)
    ids = {item.claim_id: item.item_id for item in sheet.items}
    store = JudgmentStore()
    for claim_id, verdict in verdicts.items():
        store = record_judgment(store, sheet, ids[claim_id], verdict)
    save_judgments(work, store)


def test_wilson_interval_matches_known_values():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.2775, abs=1e-3)
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-3)
    assert high == pytest.approx(0.7634, abs=1e-3)
    assert wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert wilson_interval(0, 0) is None


def test_group_stats_shares():
    stats = GroupStats(judged=5, exists=2, not_exists=2, unsure=1)
    assert stats.decided == 4
    assert stats.share_existing == pytest.approx(0.5)
    assert stats.share_counting_unsure_as_missing == pytest.approx(0.4)
    empty = GroupStats()
    assert empty.share_existing is None
    assert empty.share_counting_unsure_as_missing is None
    assert empty.interval() is None


def test_summary_groups_judgments_by_the_automatic_status(work):
    judge(
        work,
        {
            "a1": Verdict.EXISTS,
            "a2": Verdict.EXISTS,
            "a3": Verdict.UNSURE,
            "a4": Verdict.NOT_EXISTS,
        },
    )
    summary = summarize_judgments(work)
    assert (summary.items_total, summary.items_judged) == (4, 4)

    overall = summary.groups["all"]
    assert (overall.judged, overall.exists, overall.not_exists, overall.unsure) == (4, 2, 1, 1)
    assert overall.share_existing == pytest.approx(2 / 3)
    assert overall.share_counting_unsure_as_missing == pytest.approx(0.5)

    agree = summary.groups["checks_agree"]
    assert (agree.judged, agree.exists) == (1, 1)
    assert agree.share_existing == 1.0

    miss = summary.groups["automatic_miss"]
    assert (miss.judged, miss.exists, miss.unsure) == (2, 1, 1)
    assert miss.share_existing == 1.0

    unchecked = summary.groups["unchecked"]
    assert (unchecked.judged, unchecked.not_exists) == (1, 1)
    assert unchecked.share_existing == 0.0


def test_a_partial_judgment_is_reported_as_partial(work):
    judge(work, {"a1": Verdict.EXISTS})
    summary = summarize_judgments(work)
    assert (summary.items_total, summary.items_judged) == (4, 1)
    assert summary.groups["automatic_miss"].judged == 0
    assert summary.groups["automatic_miss"].share_existing is None


def test_the_dict_form_carries_the_interval_only_when_there_is_data(work):
    judge(work, {"a1": Verdict.EXISTS, "a2": Verdict.NOT_EXISTS})
    data = summary_to_dict(summarize_judgments(work))
    assert data["items_judged"] == 2
    assert len(data["groups"]["all"]["interval_95"]) == 2
    assert data["groups"]["unchecked"]["interval_95"] is None
    json.dumps(data)  # it must be serializable


def test_missing_sheet_and_vanished_claims_are_errors(tmp_path, work):
    with pytest.raises(FileNotFoundError):
        summarize_judgments(tmp_path / "empty")
    judge(work, {"a1": Verdict.EXISTS})
    write_run(work, [CLAIMS[4]])  # the run lost its claims
    with pytest.raises(JudgingDataError):
        summarize_judgments(work)
