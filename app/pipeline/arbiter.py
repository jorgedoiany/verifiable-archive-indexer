"""Second, independent verifier: ask Muse Wonder whether each object is visible.

Muse Spark proposes the claims and SAM localizes them, but neither can be
trusted to reject a plausible object that is not there. This module asks a
different model, with the same silent video, to judge every item blind: it
never sees the other verifier's verdict or the proposing model's confidence.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field, StrictBool, ValidationError

from app.config import Settings
from app.pipeline.extract import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    RETRY_MAX_OUTPUT_TOKENS,
    ExtractIncompleteError,
    call_model,
)

MAX_ITEMS = 60
MAX_REASON_CHARS = 200


class ArbiterParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArbiterItem:
    item_id: str
    concept: str
    start_s: float
    end_s: float


@dataclass(frozen=True, slots=True)
class ArbiterAnswer:
    item_id: str
    visible: bool
    confidence: float | None
    reason: str | None


class _RawAnswer(BaseModel):
    id: str
    visible: StrictBool
    confidence: float | None = Field(default=None, ge=0, le=1)
    reason: str | None = None


def build_arbiter_prompt(items: Sequence[ArbiterItem]) -> str:
    if not items:
        raise ValueError("no items to check")
    if len(items) > MAX_ITEMS:
        raise ValueError(f"too many items: {len(items)} (the limit is {MAX_ITEMS})")
    lines = [
        f"{item.item_id} | {item.start_s:.1f}-{item.end_s:.1f} s | "
        f"{' '.join(item.concept.split())}"
        for item in items
    ]
    return (
        "You are fact-checking an automatic catalog of a silent video. Each item "
        "below names an object and a time range in seconds. For each item decide "
        "whether that object is clearly visible at some moment between its start "
        "and end time (a margin of one second is fine). Some items may describe "
        "things that are not in the video at all, so judge each one only on what "
        "you can actually see. If you cannot clearly see the object, answer "
        "false.\n\n"
        + "\n".join(lines)
        + "\n\nReply with JSON only, in this shape:\n"
        '{"answers": [{"id": "q1", "visible": true, "confidence": 0.9, '
        '"reason": "five words at most"}]}\n'
        "Answer every item exactly once, using the ids above."
    )


def parse_arbiter_output(
    text: str, items: Sequence[ArbiterItem]
) -> tuple[dict[str, ArbiterAnswer], list[str]]:
    """Validate each answer on its own; report problems instead of guessing."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ArbiterParseError("no JSON object found in the model output")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ArbiterParseError(f"model output is not valid JSON: {exc}") from exc
    raw = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise ArbiterParseError('expected an object with an "answers" list')

    known = {item.item_id for item in items}
    answers: dict[str, ArbiterAnswer] = {}
    errors: list[str] = []
    for index, entry in enumerate(raw):
        try:
            parsed = _RawAnswer.model_validate(entry)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            errors.append(f"answer #{index}: {where}: {first['msg']}")
            continue
        if parsed.id not in known:
            errors.append(f"answer #{index}: unknown id {parsed.id!r}")
            continue
        if parsed.id in answers:
            errors.append(f"answer #{index}: duplicate id {parsed.id!r}")
            continue
        reason = parsed.reason[:MAX_REASON_CHARS] if parsed.reason else None
        answers[parsed.id] = ArbiterAnswer(
            parsed.id, parsed.visible, parsed.confidence, reason
        )
    errors.extend(
        f"missing answer for {item.item_id}"
        for item in items
        if item.item_id not in answers
    )
    return answers, errors


def ask_arbiter(
    file_id: str,
    items: Sequence[ArbiterItem],
    settings: Settings,
    *,
    budget: int = DEFAULT_MAX_OUTPUT_TOKENS,
) -> tuple[dict[str, ArbiterAnswer], list[str]]:
    """One call to Muse Wonder with the uploaded silent video; one retry if truncated."""
    prompt = build_arbiter_prompt(items)
    model = settings.wonder_model
    try:
        text = call_model(
            file_id, prompt, settings, model=model, max_output_tokens=budget
        )
    except ExtractIncompleteError as exc:
        if exc.reason != "max_output_tokens" or budget >= RETRY_MAX_OUTPUT_TOKENS:
            raise
        text = call_model(
            file_id,
            prompt,
            settings,
            model=model,
            max_output_tokens=min(budget * 2, RETRY_MAX_OUTPUT_TOKENS),
        )
    return parse_arbiter_output(text, items)
