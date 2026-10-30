import pytest

from app.pipeline.claims import Claim, ClaimSet, ClaimStatus
from app.pipeline.sam import SamResult, estimate_cost
from app.pipeline.sam_parse import Detection
from app.pipeline.verify import plan_verification, verify_claims

FPS = 5


def make_claim(
    claim_id, value="bicycle", category="object", start=4.0, end=7.0, status="proposed"
):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value,
            "evidence": [{"kind": "visual", "start_s": start, "end_s": end}],
            "status": status,
        }
    )


def make_detection(frame, confidence=0.9):
    return Detection(
        frame=frame,
        object_id=0,
        x1=0,
        y1=0,
        x2=1,
        y2=1,
        width=10,
        height=10,
        confidence=confidence,
    )


def make_result(concept, *detections):
    return SamResult(concept=concept, detections=tuple(detections), frames_billed=205)


def only(claim_set):
    return claim_set.claims[0]


def test_plan_collects_unique_object_concepts():
    claims = ClaimSet(
        claims=[
            make_claim("a", "Bicycle"),
            make_claim("b", "bicycle"),
            make_claim("c", "chair"),
            make_claim("d", "living room", category="scene"),
            make_claim("e", "table", status="verified"),
            make_claim("f", "bus, bicycle"),
        ]
    )
    plan = plan_verification(claims, duration_s=41.0, fps=FPS)
    assert plan.concepts == ("Bicycle", "chair")
    assert plan.capped_claim_ids == ()
    assert plan.estimated_cost == pytest.approx(estimate_cost(205, requests=2))


def test_plan_applies_the_concept_cap():
    claims = ClaimSet(
        claims=[make_claim("a", "Bicycle"), make_claim("b", "bicycle"), make_claim("c", "chair")]
    )
    plan = plan_verification(claims, duration_s=41.0, fps=FPS, max_concepts=1)
    assert plan.concepts == ("Bicycle",)
    assert plan.capped_claim_ids == ("c",)


def test_plan_rejects_bad_fps():
    with pytest.raises(ValueError):
        plan_verification(ClaimSet(), duration_s=10.0, fps=0)


def test_verified_when_found_inside_the_cited_range():
    claims = ClaimSet(claims=[make_claim("a")])
    results = {"bicycle": make_result("bicycle", make_detection(22), make_detection(23))}
    claim = only(verify_claims(claims, results, fps=FPS))
    assert claim.status == ClaimStatus.VERIFIED
    assert "2 frames" in claim.notes


def test_disputed_when_found_only_elsewhere():
    claims = ClaimSet(claims=[make_claim("a")])
    results = {"bicycle": make_result("bicycle", make_detection(100), make_detection(101))}
    claim = only(verify_claims(claims, results, fps=FPS))
    assert claim.status == ClaimStatus.DISPUTED
    assert "elsewhere" in claim.notes


def test_unsupported_when_not_found_anywhere():
    claims = ClaimSet(claims=[make_claim("a")])
    claim = only(verify_claims(claims, {"bicycle": make_result("bicycle")}, fps=FPS))
    assert claim.status == ClaimStatus.UNSUPPORTED


def test_low_confidence_detections_are_ignored():
    claims = ClaimSet(claims=[make_claim("a")])
    results = {
        "bicycle": make_result(
            "bicycle", make_detection(22, 0.2), make_detection(23, 0.2)
        )
    }
    claim = only(verify_claims(claims, results, fps=FPS))
    assert claim.status == ClaimStatus.UNSUPPORTED


def test_sam_errors_leave_the_claim_proposed():
    claims = ClaimSet(claims=[make_claim("a")])
    claim = only(verify_claims(claims, {}, fps=FPS, errors={"bicycle": "HTTP 500"}))
    assert claim.status == ClaimStatus.PROPOSED
    assert "failed" in claim.notes


def test_capped_claims_stay_proposed_with_a_note():
    claims = ClaimSet(claims=[make_claim("a")])
    claim = only(verify_claims(claims, {}, fps=FPS, capped_ids=["a"]))
    assert claim.status == ClaimStatus.PROPOSED
    assert "cap" in claim.notes


def test_non_object_claims_are_left_for_review():
    claims = ClaimSet(claims=[make_claim("p", "kitchen", category="place")])
    claim = only(verify_claims(claims, {}, fps=FPS))
    assert claim.status == ClaimStatus.PROPOSED
    assert "no automatic visual check" in claim.notes


def test_already_decided_claims_are_unchanged():
    claims = ClaimSet(claims=[make_claim("a", status="verified")])
    claim = only(verify_claims(claims, {}, fps=FPS))
    assert claim.status == ClaimStatus.VERIFIED
    assert claim.notes is None


def test_default_cap_allows_a_full_inventory():
    claims = ClaimSet(claims=[make_claim(f"c{i}", f"object {i}") for i in range(25)])
    plan = plan_verification(claims, duration_s=41.0, fps=FPS)
    assert len(plan.concepts) == 25
    assert plan.capped_claim_ids == ()
