import pytest
from pydantic import ValidationError

from app.pipeline.claims import (
    Claim,
    ClaimSet,
    ClaimStatus,
    SpeechEvidence,
    VisualEvidence,
    check_speech_evidence,
    normalize_text,
)
from app.pipeline.models import parse_transcript

TRANSCRIPT = parse_transcript(
    {
        "transcript": "Estamos en el Malecón de Santo Domingo. ¡Qué día tan bonito!",
        "audioDurationMs": 6000,
        "turns": [
            {
                "turnId": 1,
                "startMs": 0,
                "endMs": 3000,
                "transcript": "Estamos en el Malecón de Santo Domingo.",
            },
            {
                "turnId": 2,
                "startMs": 3200,
                "endMs": 6000,
                "transcript": "¡Qué día tan bonito!",
            },
        ],
    }
)


def make_claim(**overrides):
    data = {
        "claim_id": "c1",
        "category": "place",
        "value": "Malecón de Santo Domingo",
        "evidence": [
            {"kind": "speech", "turn_id": 1, "quote": "el Malecón de Santo Domingo"}
        ],
    }
    data.update(overrides)
    return Claim.model_validate(data)


def test_valid_claim_defaults():
    claim = make_claim()
    assert claim.status == ClaimStatus.PROPOSED
    assert isinstance(claim.evidence[0], SpeechEvidence)


def test_claim_requires_evidence():
    with pytest.raises(ValidationError):
        make_claim(evidence=[])


def test_visual_end_before_start_rejected():
    with pytest.raises(ValidationError):
        VisualEvidence(start_s=5.0, end_s=1.0)


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        make_claim(model_confidence=1.5)


def test_duplicate_claim_ids_rejected():
    with pytest.raises(ValidationError):
        ClaimSet(claims=[make_claim(), make_claim()])


def test_normalize_text():
    assert normalize_text("¡Qué DÍA, tan bonito!") == "que dia tan bonito"


def test_speech_check_ignores_accents_case_and_punctuation():
    evidence = SpeechEvidence(turn_id=1, quote="el MALECON de santo domingo")
    assert check_speech_evidence(evidence, TRANSCRIPT) is True


def test_speech_check_rejects_invented_quote():
    evidence = SpeechEvidence(turn_id=1, quote="la Zona Colonial")
    assert check_speech_evidence(evidence, TRANSCRIPT) is False


def test_speech_check_rejects_unknown_turn():
    evidence = SpeechEvidence(turn_id=99, quote="Santo Domingo")
    assert check_speech_evidence(evidence, TRANSCRIPT) is False


def test_speech_check_rejects_quote_from_another_turn():
    evidence = SpeechEvidence(turn_id=1, quote="día tan bonito")
    assert check_speech_evidence(evidence, TRANSCRIPT) is False


def test_speech_check_rejects_punctuation_only_quote():
    evidence = SpeechEvidence(turn_id=1, quote="...")
    assert check_speech_evidence(evidence, TRANSCRIPT) is False
