"""Data models for the transcription stage.

Shapes follow the documented response of POST /v1/asr/transcribe:
turn-level timestamps only, no word-level timestamps, no confidence scores.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Turn(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    turn_id: int = Field(alias="turnId")
    start_ms: int = Field(alias="startMs", ge=0)
    end_ms: int = Field(alias="endMs", ge=0)
    transcript: str
    speaker: str | None = None

    @model_validator(mode="after")
    def _end_not_before_start(self) -> Turn:
        if self.end_ms < self.start_ms:
            raise ValueError("end_ms must not be earlier than start_ms")
        return self


class Transcript(BaseModel):
    model_config = ConfigDict(populate_by_name=True, frozen=True)

    session_id: str | None = Field(default=None, alias="sessionId")
    transcript: str
    audio_duration_ms: int = Field(alias="audioDurationMs", ge=0)
    turns: list[Turn] = Field(default_factory=list)

    def turn_by_id(self, turn_id: int) -> Turn | None:
        for turn in self.turns:
            if turn.turn_id == turn_id:
                return turn
        return None


def parse_transcript(payload: dict) -> Transcript:
    """Validate a raw API response and return a Transcript."""
    return Transcript.model_validate(payload)
