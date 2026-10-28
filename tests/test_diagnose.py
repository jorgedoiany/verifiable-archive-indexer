from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.diagnose import describe_run
from app.pipeline.evaluate import GroundTruth

TRUTH = GroundTruth.model_validate(
    {
        "clip": "c",
        "transcript": "t",
        "items": [
            {"category": "object", "names": ["bicycle"], "start_s": 3, "end_s": 8},
            {"category": "object", "names": ["table"], "start_s": 0, "end_s": 10},
            {"category": "place", "names": ["Malecon"]},
        ],
    }
)


def make_claim(claim_id, value, status):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": "object",
            "value": value,
            "evidence": [{"kind": "visual", "start_s": 4.0, "end_s": 7.0}],
            "status": status,
            "notes": "a note",
        }
    )


def test_marks_each_claim_and_each_truth_item():
    claims = ClaimSet(
        claims=[
            make_claim("c1", "bicycle", "verified"),
            make_claim("c2", "unicorn", "verified"),
            make_claim("c3", "table", "unsupported"),
        ]
    )
    lines = describe_run(claims, TRUTH)
    assert any(
        line.startswith("c1 |") and "TRUE" in line and "4-7s" in line for line in lines
    )
    assert any(line.startswith("c2 |") and "WRONG" in line for line in lines)
    assert "verified | object | bicycle" in lines
    assert "claimed, not verified | object | table" in lines
    assert "no claim | place | Malecon" in lines
