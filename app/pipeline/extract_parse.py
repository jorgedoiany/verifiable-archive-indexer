"""Prompt for the vision model, and strict parsing/checking of what it returns."""
from __future__ import annotations

import json

from pydantic import ValidationError

from app.pipeline.claims import (
    Claim,
    ClaimSet,
    ClaimStatus,
    SpeechEvidence,
    VisualEvidence,
    check_speech_evidence,
)
from app.pipeline.models import Transcript


class ExtractParseError(ValueError):
    pass


def build_extraction_prompt(transcript: Transcript) -> str:
    lines = []
    for turn in transcript.turns:
        speaker = f" | speaker {turn.speaker}" if turn.speaker else ""
        lines.append(
            f"[turn {turn.turn_id} | {turn.start_ms / 1000:.1f}-"
            f"{turn.end_ms / 1000:.1f} s{speaker}] {turn.transcript}"
        )
    turns_block = "\n".join(lines) if lines else "(no speech detected)"
    return (
        "You are cataloguing an audiovisual archive clip. The video has no audio "
        "track; its speech was transcribed separately, turn by turn:\n\n"
        + turns_block
        + "\n\nPropose metadata claims about people, places, topics, scenes and "
        "visible objects. Every claim MUST carry evidence:\n"
        '- speech evidence: {"kind": "speech", "turn_id": <int>, "quote": "<text '
        "copied verbatim from that turn>\"}\n"
        '- visual evidence: {"kind": "visual", "start_s": <seconds>, "end_s": '
        '<seconds>, "description": "<what is visible>"}\n\n'
        "Rules: never invent a quote; copy it exactly from the turn you cite. Do "
        "not claim anything you cannot support with evidence. Categories: person, "
        "place, topic, scene, object. For category object, value must be ONE short "
        "English noun phrase naming a single concrete visible thing (for example "
        "'red bicycle'), because a segmentation model will check it; never put "
        "several objects in one claim.\n\n"
        "Reply with JSON only, in this shape:\n"
        '{"claims": [{"claim_id": "c1", "category": "place", "value": "...", '
        '"evidence": [...], "model_confidence": 0.0, "notes": null}]}'
    )


def _load_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ExtractParseError("no JSON object found in the model output")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ExtractParseError(f"model output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("claims"), list):
        raise ExtractParseError('expected an object with a "claims" list')
    return data


def parse_model_output(text: str) -> tuple[ClaimSet, list[str]]:
    """Validate each claim on its own; skip the bad ones and report why."""
    data = _load_json(text)
    claims: list[Claim] = []
    errors: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(data["claims"]):
        try:
            claim = Claim.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            errors.append(f"claim #{index}: {where}: {first['msg']}")
            continue
        if claim.claim_id in seen:
            errors.append(f"claim #{index}: duplicate claim_id {claim.claim_id!r}")
            continue
        seen.add(claim.claim_id)
        claims.append(claim)
    return ClaimSet(claims=claims), errors


def apply_checks(
    claim_set: ClaimSet, transcript: Transcript, video_duration_s: float
) -> ClaimSet:
    """Set each claim's status from checks that need no model.

    - an invented quote, or a time range past the end of the clip -> unsupported
    - speech evidence only, all quotes found -> verified
    - any visual evidence -> stays proposed until the visual stage runs
    """
    checked: list[Claim] = []
    for claim in claim_set.claims:
        speech = [e for e in claim.evidence if isinstance(e, SpeechEvidence)]
        visual = [e for e in claim.evidence if isinstance(e, VisualEvidence)]
        bad_speech = [e for e in speech if not check_speech_evidence(e, transcript)]
        bad_visual = [e for e in visual if e.end_s > video_duration_s]
        if bad_speech:
            update = {
                "status": ClaimStatus.UNSUPPORTED,
                "notes": "quote not found in the cited turn",
            }
        elif bad_visual:
            update = {
                "status": ClaimStatus.UNSUPPORTED,
                "notes": "visual evidence time range is past the end of the clip",
            }
        elif not visual:
            update = {"status": ClaimStatus.VERIFIED}
        else:
            update = {"status": ClaimStatus.PROPOSED}
        checked.append(claim.model_copy(update=update))
    return ClaimSet(claims=checked)
