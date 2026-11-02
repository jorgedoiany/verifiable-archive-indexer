import pytest
from pydantic import ValidationError

from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.evaluate import (
    GroundTruth,
    evaluate,
    load_ground_truth,
    metrics_to_dict,
    name_matches,
    word_error_rate,
)

TRUTH = GroundTruth.model_validate(
    {
        "clip": "clip.mp4",
        "transcript": "Estamos en el Malecón",
        "items": [
            {"category": "object", "names": ["bicycle"], "start_s": 3, "end_s": 8},
            {"category": "object", "names": ["table"], "start_s": 0, "end_s": 10},
            {"category": "place", "names": ["Malecón"]},
        ],
    }
)


def make_claim(
    claim_id, value, category="object", status="proposed", start=4.0, end=7.0,
    speech=False,
):
    if speech:
        evidence = [{"kind": "speech", "turn_id": 1, "quote": value}]
    else:
        evidence = [{"kind": "visual", "start_s": start, "end_s": end}]
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value,
            "evidence": evidence,
            "status": status,
        }
    )


def test_word_error_rate():
    assert word_error_rate("hola mundo cruel", "hola mundo") == pytest.approx(1 / 3)
    assert word_error_rate("a b c", "a x c") == pytest.approx(1 / 3)
    assert word_error_rate("a b", "a b c") == pytest.approx(1 / 2)
    assert word_error_rate("Qué día", "que dia") == 0
    assert word_error_rate("", "algo") is None


def test_name_matching_is_whole_word_and_symmetric():
    assert name_matches("red bicycle", ["bicycle"])
    assert name_matches("bicycle", ["red bicycle"])
    assert name_matches("MALECON", ["Malecón"])
    assert not name_matches("cat", ["category"])
    assert not name_matches("", ["bicycle"])


def test_metrics_on_a_mixed_result():
    claims = ClaimSet(
        claims=[
            make_claim("c1", "bicycle", status="verified"),
            make_claim("c2", "chair", status="verified"),
            make_claim("c3", "table", status="unsupported"),
            make_claim("c4", "Malecón", category="place", speech=True),
        ]
    )
    metrics = evaluate(claims, "Estamos en el Malecon", TRUTH)
    assert metrics.wer == 0
    assert metrics.claims_total == 4
    assert metrics.precision_all == pytest.approx(0.75)
    assert metrics.verified_total == 2
    assert metrics.precision_verified == pytest.approx(0.5)
    assert metrics.truth_total == 3
    assert metrics.recall_verified == pytest.approx(1 / 3)
    assert metrics.rejected_total == 1
    assert metrics.rejected_but_true == 1
    assert metrics.review_queue == 1
    assert metrics.by_status == {"verified": 2, "unsupported": 1, "proposed": 1}
    assert metrics_to_dict(metrics)["precision_verified"] == pytest.approx(0.5)


def test_time_range_must_overlap_the_truth():
    far = ClaimSet(
        claims=[make_claim("c1", "bicycle", status="verified", start=20, end=25)]
    )
    assert evaluate(far, "", TRUTH).precision_verified == 0


def test_category_must_match():
    wrong = ClaimSet(
        claims=[make_claim("c1", "bicycle", category="place", status="verified")]
    )
    assert evaluate(wrong, "", TRUTH).precision_verified == 0


def test_empty_result_has_undefined_precision_and_zero_recall():
    metrics = evaluate(ClaimSet(), "", TRUTH)
    assert metrics.precision_all is None
    assert metrics.precision_verified is None
    assert metrics.recall_verified == 0


def test_truth_item_validation():
    with pytest.raises(ValidationError):
        GroundTruth.model_validate(
            {"clip": "c", "transcript": "t", "items": [{"category": "object", "names": []}]}
        )
    with pytest.raises(ValidationError):
        GroundTruth.model_validate(
            {
                "clip": "c",
                "transcript": "t",
                "items": [{"category": "object", "names": ["x"], "start_s": 1}],
            }
        )
    with pytest.raises(ValidationError):
        GroundTruth.model_validate(
            {
                "clip": "c",
                "transcript": "t",
                "items": [
                    {"category": "object", "names": ["x"], "start_s": 5, "end_s": 1}
                ],
            }
        )


def test_load_ground_truth_from_file(tmp_path):
    path = tmp_path / "truth.json"
    path.write_text(TRUTH.model_dump_json(), encoding="utf-8")
    assert load_ground_truth(path) == TRUTH


def test_item_with_several_ranges_matches_any_of_them():
    truth = GroundTruth.model_validate(
        {
            "clip": "c",
            "transcript": "t",
            "items": [
                {
                    "category": "object",
                    "names": ["air conditioner"],
                    "ranges": [[0, 6], [35, 40]],
                }
            ],
        }
    )
    early = ClaimSet(
        claims=[make_claim("c1", "air conditioner", status="verified", start=1, end=4)]
    )
    late = ClaimSet(
        claims=[make_claim("c2", "air conditioner", status="verified", start=36, end=39)]
    )
    middle = ClaimSet(
        claims=[make_claim("c3", "air conditioner", status="verified", start=15, end=20)]
    )
    assert evaluate(early, "", truth).precision_verified == 1.0
    assert evaluate(late, "", truth).precision_verified == 1.0
    assert evaluate(middle, "", truth).precision_verified == 0
    # one object counts once in the recall denominator, however many ranges it has
    assert evaluate(early, "", truth).truth_total == 1
    assert evaluate(early, "", truth).recall_verified == 1.0


def test_invalid_range_is_rejected():
    with pytest.raises(ValidationError):
        GroundTruth.model_validate(
            {
                "clip": "c",
                "transcript": "t",
                "items": [
                    {"category": "object", "names": ["x"], "ranges": [[5, 1]]}
                ],
            }
        )


def test_clip_without_speech_reports_invented_words_instead_of_wer():
    silent = GroundTruth.model_validate({"clip": "c", "transcript": "", "items": []})
    quiet = evaluate(ClaimSet(), "", silent)
    assert quiet.wer is None
    assert quiet.spurious_words == 0
    noisy = evaluate(ClaimSet(), "Gracias por ver, ¡hasta pronto!", silent)
    assert noisy.wer is None
    assert noisy.spurious_words == 5
    assert evaluate(ClaimSet(), "hola", TRUTH).spurious_words is None
