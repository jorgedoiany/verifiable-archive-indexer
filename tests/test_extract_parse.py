import json

import pytest

from app.pipeline.claims import ClaimStatus
from app.pipeline.extract_parse import (
    ExtractParseError,
    apply_checks,
    build_extraction_prompt,
    parse_model_output,
)
from app.pipeline.models import parse_transcript

TRANSCRIPT = parse_transcript(
    {
        "transcript": "Estamos en el Malecón. Mira esa bicicleta.",
        "audioDurationMs": 10000,
        "turns": [
            {
                "turnId": 1,
                "startMs": 0,
                "endMs": 4000,
                "transcript": "Estamos en el Malecón.",
                "speaker": "A",
            },
            {
                "turnId": 2,
                "startMs": 4500,
                "endMs": 8000,
                "transcript": "Mira esa bicicleta.",
                "speaker": "B",
            },
        ],
    }
)

SPEECH_OK = {"kind": "speech", "turn_id": 1, "quote": "en el Malecón"}
SPEECH_BAD = {"kind": "speech", "turn_id": 1, "quote": "en la Zona Colonial"}
VISUAL = {"kind": "visual", "start_s": 4.0, "end_s": 7.0, "description": "bicycle"}


def claim(claim_id, evidence, **extra):
    data = {
        "claim_id": claim_id,
        "category": "object",
        "value": "valor",
        "evidence": evidence,
    }
    data.update(extra)
    return data


def model_text(*claims, fenced=False):
    body = json.dumps({"claims": list(claims)})
    return f"```json\n{body}\n```" if fenced else body


def test_prompt_lists_turns_and_rules():
    prompt = build_extraction_prompt(TRANSCRIPT)
    assert "[turn 1 | 0.0-4.0 s | speaker A] Estamos en el Malecón." in prompt
    assert "verbatim" in prompt
    assert "JSON only" in prompt


def test_prompt_without_speech():
    empty = parse_transcript(
        {"transcript": "", "audioDurationMs": 1000, "turns": []}
    )
    assert "(no speech detected)" in build_extraction_prompt(empty)


def test_parses_fenced_json():
    claim_set, errors = parse_model_output(
        model_text(claim("c1", [SPEECH_OK]), fenced=True)
    )
    assert errors == []
    assert [c.claim_id for c in claim_set.claims] == ["c1"]


def test_invalid_claims_are_skipped_with_reasons():
    text = model_text(
        claim("c1", [SPEECH_OK]),
        claim("c2", []),
        claim("c1", [VISUAL]),
    )
    claim_set, errors = parse_model_output(text)
    assert [c.claim_id for c in claim_set.claims] == ["c1"]
    assert len(errors) == 2
    assert "duplicate" in errors[1]


def test_unreadable_output_raises():
    with pytest.raises(ExtractParseError):
        parse_model_output("sorry, I cannot help")
    with pytest.raises(ExtractParseError):
        parse_model_output('{"claims": "nope"}')
    with pytest.raises(ExtractParseError):
        parse_model_output("{broken json}")


def test_checks_set_statuses():
    claim_set, _ = parse_model_output(
        model_text(
            claim("ok", [SPEECH_OK]),
            claim("invented", [SPEECH_BAD]),
            claim("late", [{"kind": "visual", "start_s": 9.0, "end_s": 12.0}]),
            claim("pending", [SPEECH_OK, VISUAL]),
        )
    )
    checked = apply_checks(claim_set, TRANSCRIPT, video_duration_s=10.0)
    status = {c.claim_id: c.status for c in checked.claims}
    assert status == {
        "ok": ClaimStatus.VERIFIED,
        "invented": ClaimStatus.UNSUPPORTED,
        "late": ClaimStatus.UNSUPPORTED,
        "pending": ClaimStatus.PROPOSED,
    }
    notes = {c.claim_id: c.notes for c in checked.claims}
    assert "quote" in notes["invented"]
