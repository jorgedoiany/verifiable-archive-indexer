"""Claims and their evidence: the unit the whole pipeline reasons about."""
from __future__ import annotations

import re
import unicodedata
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.pipeline.models import Transcript


class ClaimCategory(str, Enum):
    PERSON = "person"
    PLACE = "place"
    TOPIC = "topic"
    SCENE = "scene"
    OBJECT = "object"


class ClaimStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    UNSUPPORTED = "unsupported"


class SpeechEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["speech"] = "speech"
    turn_id: int
    quote: str = Field(min_length=1)


class VisualEvidence(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["visual"] = "visual"
    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    description: str | None = None

    @model_validator(mode="after")
    def _end_not_before_start(self) -> "VisualEvidence":
        if self.end_s < self.start_s:
            raise ValueError("end_s must not be earlier than start_s")
        return self


Evidence = Annotated[
    Union[SpeechEvidence, VisualEvidence], Field(discriminator="kind")
]


class Claim(BaseModel):
    claim_id: str = Field(min_length=1)
    category: ClaimCategory
    value: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    model_confidence: float | None = Field(default=None, ge=0, le=1)
    status: ClaimStatus = ClaimStatus.PROPOSED
    notes: str | None = None


class ClaimSet(BaseModel):
    claims: list[Claim] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "ClaimSet":
        ids = [claim.claim_id for claim in self.claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim_id values must be unique")
        return self


def normalize_text(text: str) -> str:
    """Lowercase, drop accents and punctuation, collapse whitespace."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[\W_]+", " ", stripped.casefold()).strip()


def check_speech_evidence(evidence: SpeechEvidence, transcript: Transcript) -> bool:
    """True only if the quote really occurs in the turn it points to."""
    turn = transcript.turn_by_id(evidence.turn_id)
    if turn is None:
        return False
    quote = normalize_text(evidence.quote)
    if not quote:
        return False
    return quote in normalize_text(turn.transcript)
