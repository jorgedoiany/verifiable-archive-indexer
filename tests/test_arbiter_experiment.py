import json

import pytest

from app.pipeline.arbiter import ArbiterAnswer
from app.pipeline.arbiter_experiment import (
    build_probes,
    disagreements,
    load_not_real,
    merge_scores,
    score,
)
from app.pipeline.claims import Claim, ClaimSet


def make_claim(claim_id, status, category="object", visual=True, value=None):
    if visual:
        evidence = [{"kind": "visual", "start_s": 4.0, "end_s": 7.0}]
    else:
        evidence = [{"kind": "speech", "turn_id": 1, "quote": "hola"}]
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value or f"thing {claim_id}",
            "evidence": evidence,
            "status": status,
        }
    )


CLAIMS = ClaimSet(
    claims=[
        make_claim("a", "verified"),
        make_claim("b", "unsupported"),
        make_claim("c", "verified"),
        make_claim("d", "proposed", category="place"),
        make_claim("e", "verified", visual=False),
    ]
)


def make_probes(seed=7):
    return build_probes(
        CLAIMS, not_real=["c"], decoys=("horse", "piano"), duration_s=20.0, seed=seed
    )


def by_concept(probes):
    return {p.item.concept: p for p in probes}


def test_probes_cover_object_claims_with_visual_evidence_plus_decoys():
    probes = make_probes()
    assert sorted(p.item.concept for p in probes) == [
        "horse", "piano", "thing a", "thing b", "thing c",
    ]
    assert [p.item.item_id for p in probes] == ["q1", "q2", "q3", "q4", "q5"]


def test_probe_flags_come_from_the_audit_and_the_earlier_verdicts():
    probes = by_concept(make_probes())
    assert (probes["thing a"].real, probes["thing a"].sam_said_yes) == (True, True)
    assert (probes["thing b"].real, probes["thing b"].sam_said_yes) == (True, False)
    assert (probes["thing c"].real, probes["thing c"].sam_said_yes) == (False, True)
    horse = probes["horse"]
    assert (horse.source, horse.claim_id, horse.real, horse.sam_said_yes) == (
        "decoy", None, False, None,
    )


def test_decoy_windows_are_spread_across_the_clip():
    probes = by_concept(make_probes())
    assert (probes["horse"].item.start_s, probes["horse"].item.end_s) == (0.0, 4.0)
    assert (probes["piano"].item.start_s, probes["piano"].item.end_s) == (16.0, 20.0)


def test_same_seed_gives_the_same_order():
    first = [p.item.concept for p in make_probes(seed=7)]
    second = [p.item.concept for p in make_probes(seed=7)]
    assert first == second


def test_invalid_duration_is_rejected():
    with pytest.raises(ValueError):
        build_probes(CLAIMS, not_real=[], decoys=(), duration_s=0)


def answers_for(probes, table):
    return {
        probes[concept].item.item_id: ArbiterAnswer(
            probes[concept].item.item_id, visible, None, "r"
        )
        for concept, visible in table.items()
    }


def test_score_and_disagreements():
    probes = make_probes()
    index = by_concept(probes)
    answers = answers_for(
        index,
        {"thing a": True, "thing b": True, "thing c": False, "horse": True},
    )
    result = score(probes, answers)
    assert result["real_total"] == 2
    assert (result["real_answered"], result["real_visible"]) == (2, 2)
    assert (result["fake_claims_total"], result["fake_claims_visible"]) == (1, 0)
    assert (result["decoys_total"], result["decoys_answered"]) == (2, 1)
    assert result["decoys_visible"] == 1
    assert result["sam_rejected_real_total"] == 1
    assert result["sam_rejected_real_wonder_visible"] == 1
    assert result["sam_accepted_fake_total"] == 1
    assert result["sam_accepted_fake_wonder_visible"] == 0

    lines = disagreements(probes, answers)
    assert any(line.startswith("ACCEPTED FALSE | decoy | horse") for line in lines)
    assert any(line.startswith("NO ANSWER | decoy | piano") for line in lines)
    assert len(lines) == 2


def test_missed_real_claims_are_listed_with_the_other_verdict():
    probes = make_probes()
    answers = answers_for(by_concept(probes), {"thing a": False})
    lines = disagreements(probes, answers)
    assert any(
        line.startswith("MISSED REAL | claim | thing a") and "SAM yes" in line
        for line in lines
    )


def test_merge_scores_sums_every_key():
    probes = make_probes()
    answers = answers_for(by_concept(probes), {"thing a": True})
    one = score(probes, answers)
    merged = merge_scores([one, one])
    assert merged["real_total"] == 2 * one["real_total"]
    assert merged["real_visible"] == 2 * one["real_visible"]


def test_load_not_real(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps({"clips": {"x": {"not_real": ["c1"]}, "y": {}}}), encoding="utf-8"
    )
    assert load_not_real(path, "x") == ["c1"]
    assert load_not_real(path, "y") == []
    with pytest.raises(KeyError):
        load_not_real(path, "zzz")
    path.write_text(json.dumps({"clips": {"x": {"not_real": "c1"}}}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_not_real(path, "x")
