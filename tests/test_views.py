from datetime import UTC, datetime

from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.models import parse_transcript
from app.pipeline.review import Decision, ReviewStore, record_decision
from app.views import build_views, count_states, state_of

NOW = datetime(2026, 10, 30, 12, 0, tzinfo=UTC)

TRANSCRIPT = parse_transcript(
    {
        "transcript": "hola",
        "audioDurationMs": 6000,
        "turns": [{"turnId": 1, "startMs": 1500, "endMs": 3000, "transcript": "hola"}],
    }
)


def make_claim(claim_id, status, evidence, category="object", value="bicycle"):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value,
            "evidence": evidence,
            "status": status,
        }
    )


VISUAL = [{"kind": "visual", "start_s": 4.0, "end_s": 7.0, "description": "a bike"}]
SPEECH = [{"kind": "speech", "turn_id": 1, "quote": "hola"}]
CLAIMS = ClaimSet(
    claims=[
        make_claim("a", "verified", VISUAL),
        make_claim("b", "proposed", SPEECH, category="place", value="Santo Domingo"),
        make_claim("c", "unsupported", VISUAL, value="chair"),
        make_claim("d", "proposed", [{"kind": "speech", "turn_id": 9, "quote": "x"}]),
    ]
)


def test_states_follow_decisions_then_automatic_status():
    store = record_decision(ReviewStore(), CLAIMS, "c", Decision.REJECTED, now=NOW)
    states = {c.claim_id: state_of(c, store) for c in CLAIMS.claims}
    assert states == {"a": "verified", "b": "pending", "c": "rejected", "d": "pending"}


def test_views_are_ordered_needs_review_first_and_keep_positions():
    store = record_decision(ReviewStore(), CLAIMS, "c", Decision.REJECTED, now=NOW)
    store = record_decision(store, CLAIMS, "b", Decision.EDITED, value="Santo Domingo, DR", now=NOW)
    views = build_views(CLAIMS, TRANSCRIPT, store)
    assert [v.claim_id for v in views] == ["d", "a", "b", "c"]
    assert [v.position for v in views] == [3, 0, 1, 2]
    edited = views[2]
    assert edited.value == "Santo Domingo, DR"
    assert edited.original_value == "Santo Domingo"
    assert edited.decided_at == NOW.isoformat()


def test_evidence_views_carry_the_seek_time():
    views = {v.claim_id: v for v in build_views(CLAIMS, TRANSCRIPT, ReviewStore())}
    visual = views["a"].evidence[0]
    assert (visual.kind, visual.seek) == ("visual", 4.0)
    assert "Visual 4.0-7.0 s: a bike" == visual.text
    speech = views["b"].evidence[0]
    assert speech.seek == 1.5
    assert speech.text == 'Turn 1 (1.5-3.0 s): "hola"'
    missing = views["d"].evidence[0]
    assert missing.seek is None
    assert "turn not found" in missing.text


def test_count_states_includes_every_state():
    store = record_decision(ReviewStore(), CLAIMS, "c", Decision.ACCEPTED, now=NOW)
    counts = count_states(build_views(CLAIMS, TRANSCRIPT, store))
    assert counts == {"pending": 2, "verified": 1, "accepted": 1, "edited": 0, "rejected": 0}
