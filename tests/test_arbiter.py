import json

import httpx
import pytest

from app.config import Settings
from app.pipeline.arbiter import (
    ArbiterItem,
    ArbiterParseError,
    ask_arbiter,
    build_arbiter_prompt,
    parse_arbiter_output,
)
from app.pipeline.extract import ExtractIncompleteError

URL = "https://api.meta.ai/v1/responses"
ITEMS = [ArbiterItem("q1", "bicycle", 4.0, 7.0), ArbiterItem("q2", "horse", 10.0, 14.0)]
ANSWERS = json.dumps(
    {
        "answers": [
            {"id": "q1", "visible": True, "confidence": 0.9, "reason": "a bike"},
            {"id": "q2", "visible": False, "confidence": 0.8, "reason": "no horse"},
        ]
    }
)
INCOMPLETE = {
    "status": "incomplete",
    "incomplete_details": {"reason": "max_output_tokens"},
    "output": [],
}


def reply(text):
    return {
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": text}]}
        ],
    }


@pytest.fixture
def settings():
    return Settings(
        api_key="test-key",
        api_base="https://api.meta.ai/v1",
        spark_model="muse-spark-1.3",
        wonder_model="muse-wonder-1.0",
        asr_model="muse-voice-transcribe-1.0",
        sam_model="sam-3.1",
    )


def test_prompt_lists_items_and_warns_that_some_may_be_absent():
    prompt = build_arbiter_prompt(ITEMS)
    assert "q1 | 4.0-7.0 s | bicycle" in prompt
    assert "q2 | 10.0-14.0 s | horse" in prompt
    assert "not in the video at all" in prompt
    assert "JSON only" in prompt


def test_prompt_rejects_empty_and_oversized_lists():
    with pytest.raises(ValueError):
        build_arbiter_prompt([])
    many = [ArbiterItem(f"q{i}", "x", 0.0, 1.0) for i in range(61)]
    with pytest.raises(ValueError):
        build_arbiter_prompt(many)


def test_parses_fenced_json():
    answers, errors = parse_arbiter_output(f"```json\n{ANSWERS}\n```", ITEMS)
    assert errors == []
    assert answers["q1"].visible is True
    assert answers["q1"].confidence == 0.9
    assert answers["q2"].visible is False


def test_bad_entries_are_reported_not_guessed():
    text = json.dumps(
        {
            "answers": [
                {"id": "q1", "visible": "yes"},
                {"id": "zz", "visible": True},
                {"id": "q2", "visible": False},
                {"id": "q2", "visible": True},
                {"id": "q1", "visible": True, "confidence": 2},
            ]
        }
    )
    answers, errors = parse_arbiter_output(text, ITEMS)
    assert set(answers) == {"q2"}
    assert answers["q2"].visible is False
    assert len(errors) == 5
    assert "missing answer for q1" in errors


def test_long_reasons_are_truncated():
    text = json.dumps({"answers": [{"id": "q1", "visible": True, "reason": "x" * 500}]})
    answers, _ = parse_arbiter_output(text, ITEMS[:1])
    assert len(answers["q1"].reason) == 200


@pytest.mark.parametrize("text", ["sorry", '{"answers": "nope"}', "{broken}"])
def test_unreadable_output_raises(text):
    with pytest.raises(ArbiterParseError):
        parse_arbiter_output(text, ITEMS)


def test_ask_uses_the_wonder_model_and_the_uploaded_video(settings, respx_mock):
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=reply(ANSWERS)))
    answers, errors = ask_arbiter("file-1", ITEMS, settings)
    payload = json.loads(route.calls.last.request.content)
    assert payload["model"] == "muse-wonder-1.0"
    assert payload["max_output_tokens"] == 16_000
    assert payload["input"][0]["content"][1] == {"type": "input_file", "file_id": "file-1"}
    assert errors == []
    assert answers["q1"].visible is True


def test_truncated_response_is_retried_once_with_a_bigger_budget(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        side_effect=[
            httpx.Response(200, json=INCOMPLETE),
            httpx.Response(200, json=reply(ANSWERS)),
        ]
    )
    answers, _ = ask_arbiter("file-1", ITEMS, settings)
    budgets = [json.loads(c.request.content)["max_output_tokens"] for c in route.calls]
    assert budgets == [16_000, 32_000]
    assert len(answers) == 2


def test_second_truncation_is_an_error(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        side_effect=[httpx.Response(200, json=INCOMPLETE)] * 2
    )
    with pytest.raises(ExtractIncompleteError):
        ask_arbiter("file-1", ITEMS, settings)
    assert route.call_count == 2
