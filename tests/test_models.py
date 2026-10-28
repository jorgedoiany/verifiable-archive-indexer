import pytest
from pydantic import ValidationError

from app.pipeline.models import parse_transcript

SAMPLE = {
    "sessionId": "9f1c",
    "transcript": "How is the weather? It is raining.",
    "audioDurationMs": 8240,
    "turns": [
        {
            "turnId": 1,
            "startMs": 1520,
            "endMs": 4640,
            "transcript": "How is the weather?",
            "speaker": "A",
        },
        {
            "turnId": 2,
            "startMs": 5900,
            "endMs": 8240,
            "transcript": "It is raining.",
            "speaker": "B",
        },
    ],
}


def test_parses_documented_response():
    result = parse_transcript(SAMPLE)
    assert result.audio_duration_ms == 8240
    assert [t.turn_id for t in result.turns] == [1, 2]
    assert result.turns[0].speaker == "A"


def test_turns_may_be_empty_and_speaker_optional():
    result = parse_transcript(
        {"transcript": "hola", "audioDurationMs": 1000, "turns": []}
    )
    assert result.turns == []
    result = parse_transcript(
        {
            "transcript": "hola",
            "audioDurationMs": 1000,
            "turns": [
                {"turnId": 1, "startMs": 0, "endMs": 900, "transcript": "hola"}
            ],
        }
    )
    assert result.turns[0].speaker is None


def test_turn_end_before_start_is_rejected():
    bad = {
        "transcript": "x",
        "audioDurationMs": 1000,
        "turns": [{"turnId": 1, "startMs": 900, "endMs": 100, "transcript": "x"}],
    }
    with pytest.raises(ValidationError):
        parse_transcript(bad)


def test_turn_lookup_by_id():
    result = parse_transcript(SAMPLE)
    assert result.turn_by_id(2).transcript == "It is raining."
    assert result.turn_by_id(99) is None
