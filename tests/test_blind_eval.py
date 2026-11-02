import json

import pytest

from app.blind_eval import ClipMeasure, measure_clip, pool, word_edit_stats
from app.pipeline.claims import Claim
from app.pipeline.evaluate import GroundTruth
from app.pipeline.models import parse_transcript


def test_word_edit_stats():
    assert word_edit_stats("hola mundo cruel", "hola mundo") == (1, 3)
    assert word_edit_stats("a b c", "a x c") == (1, 3)
    assert word_edit_stats("Qué día", "que dia") == (0, 2)
    assert word_edit_stats("", "a b") == (2, 0)
    assert word_edit_stats("", "") == (0, 0)


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


def visual(start, end):
    return [{"kind": "visual", "start_s": start, "end_s": end}]


def write_clip(tmp_path, claims, transcript_text, truth, cost=0.04, frames=200):
    work = tmp_path / "work"
    truth_dir = tmp_path / "truth"
    work.mkdir(exist_ok=True)
    truth_dir.mkdir(exist_ok=True)
    (work / "clip_a.verified.json").write_text(
        json.dumps({"claims": [c.model_dump(mode="json") for c in claims]}),
        encoding="utf-8",
    )
    transcript = parse_transcript(
        {"transcript": transcript_text, "audioDurationMs": 1000, "turns": []}
    )
    (work / "clip_a.transcript.json").write_text(
        transcript.model_dump_json(by_alias=True), encoding="utf-8"
    )
    (work / "clip_a.result.json").write_text(
        json.dumps({"sam": {"frames_billed": frames, "actual_cost": cost}}),
        encoding="utf-8",
    )
    (truth_dir / "clip_a.json").write_text(
        GroundTruth.model_validate(truth).model_dump_json(), encoding="utf-8"
    )
    return work, truth_dir


TRUTH = {
    "clip": "clip_a.mp4",
    "transcript": "hola mundo cruel",
    "items": [
        {"category": "object", "names": ["bicycle"], "ranges": [[3, 8]]},
        {"category": "object", "names": ["table"], "ranges": [[0, 10]]},
        {"category": "object", "names": ["lamp"], "ranges": [[0, 10]]},
        {"category": "place", "names": ["Santo Domingo"]},
    ],
}


def test_measure_clip_counts_matches_by_any_claim_and_by_agreeing_checks(tmp_path):
    claims = [
        make_claim("a", "object", "red bicycle", "verified", visual(4, 7)),
        make_claim("b", "object", "table", "unsupported", visual(1, 2)),
        make_claim("c", "object", "chair", "verified", visual(1, 2)),
        make_claim("d", "place", "Santo Domingo", "proposed", [
            {"kind": "speech", "turn_id": 1, "quote": "x"}
        ]),
    ]
    work, truth_dir = write_clip(tmp_path, claims, "hola mundo", TRUTH)
    measure = measure_clip("clip_a", work, truth_dir)
    assert (measure.reference_words, measure.hypothesis_words, measure.edits) == (3, 2, 1)
    assert measure.truth_objects == 3  # the place is not an object
    assert measure.matched_by_any_claim == 2  # bicycle and table
    assert measure.matched_by_checks_agree == 1  # only the bicycle was agreed
    assert measure.object_claims == 3
    assert (measure.frames_billed, measure.cost) == (200, 0.04)


def test_a_claim_in_the_wrong_time_range_does_not_match(tmp_path):
    claims = [make_claim("a", "object", "bicycle", "verified", visual(20, 25))]
    work, truth_dir = write_clip(tmp_path, claims, "hola", TRUTH)
    measure = measure_clip("clip_a", work, truth_dir)
    assert measure.matched_by_any_claim == 0
    assert measure.matched_by_checks_agree == 0


def test_a_missing_run_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        measure_clip("nope", tmp_path, tmp_path)


def make_measure(stem, **overrides):
    values = {
        "stem": stem,
        "reference_words": 10,
        "hypothesis_words": 10,
        "edits": 0,
        "truth_objects": 4,
        "matched_by_any_claim": 2,
        "matched_by_checks_agree": 1,
        "object_claims": 5,
        "frames_billed": 200,
        "cost": 0.04,
    }
    values.update(overrides)
    return ClipMeasure(**values)


def test_pooling_adds_the_counts_before_dividing():
    short = make_measure("a", reference_words=1, hypothesis_words=2, edits=1)
    long = make_measure("b", reference_words=99, hypothesis_words=99, edits=0)
    total = pool([short, long])
    assert total["word_error_rate"] == pytest.approx(0.01)  # not (100% + 0%) / 2
    assert total["reference_words"] == 100
    assert total["truth_objects"] == 8
    assert total["coverage_any_claim"] == pytest.approx(0.5)
    assert total["coverage_checks_agree"] == pytest.approx(0.25)
    assert len(total["coverage_any_claim_interval_95"]) == 2
    assert (total["frames_billed"], total["cost"]) == (400, 0.08)
    assert total["object_claims"] == 10


def test_clips_without_speech_report_invented_words_separately():
    quiet = make_measure("a", reference_words=0, hypothesis_words=3, edits=3)
    talking = make_measure("b", reference_words=10, hypothesis_words=10, edits=1)
    total = pool([quiet, talking])
    assert total["silent_clips"] == 1
    assert total["invented_words_on_silent_clips"] == 3
    assert total["word_error_rate"] == pytest.approx(4 / 10)  # insertions are edits


def test_pooling_nothing_has_no_rates():
    total = pool([make_measure("a", reference_words=0, edits=0, truth_objects=0,
                               matched_by_any_claim=0, matched_by_checks_agree=0)])
    assert total["word_error_rate"] is None
    assert total["coverage_any_claim"] is None
    assert total["coverage_any_claim_interval_95"] is None
