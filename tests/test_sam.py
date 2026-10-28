import json

import httpx
import pytest

from app.config import Settings
from app.pipeline.sam import (
    SamCallError,
    clean_concept,
    estimate_cost,
    segment_concept,
)

URL = "https://api.meta.ai/v1/responses"

SAM_TEXT = (
    "<0f>0<|box;x1=10;y1=20;x2=30;y2=40;w=320;h=240;c=0.9|>"
    "<|mask;x=0;y=0;c=0.9;data=1,1,!a|>\n"
    "<2f>0<|box;x1=11;y1=21;x2=31;y2=41;w=320;h=240;c=0.8|>"
    "<|mask;x=0;y=0;c=0.8;data=1,1,!b|>"
)


def reply(text, frames=205, status="completed"):
    output = (
        [{"type": "message", "content": [{"type": "output_text", "text": text}]}]
        if text
        else []
    )
    return {
        "status": status,
        "output": output,
        "usage": {"video_frames_processed": frames},
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


def test_sends_documented_request(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply(SAM_TEXT))
    )
    result = segment_concept("file-1", "  red   bicycle ", settings)
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "sam-3.1"
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "red bicycle"}
    assert content[1] == {"type": "input_video", "file_id": "file-1"}
    assert payload["metadata"] == {
        "mask_encoding": "one_bit",
        "include_confidence": "true",
    }
    assert payload.get("stream") is not True
    assert result.concept == "red bicycle"
    assert [(d.frame, d.confidence) for d in result.detections] == [(0, 0.9), (2, 0.8)]
    assert result.frames_billed == 205


def test_zero_matches_is_not_an_error(settings, respx_mock):
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=reply("")))
    result = segment_concept("file-1", "unicorn", settings)
    assert result.detections == ()
    assert result.frames_billed == 205


def test_incomplete_response_is_an_error(settings, respx_mock):
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply("", status="incomplete"))
    )
    with pytest.raises(SamCallError):
        segment_concept("file-1", "chair", settings)


def test_unreadable_output_is_wrapped(settings, respx_mock):
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200, json=reply("0<|box;x1=1;y1=2;x2=3;y2=4;w=9;h=9|>")
        )
    )
    with pytest.raises(SamCallError):
        segment_concept("file-1", "chair", settings)


def test_retries_on_429_then_succeeds(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=reply(SAM_TEXT))]
    )
    waits = []
    result = segment_concept("file-1", "chair", settings, sleep=waits.append)
    assert len(result.detections) == 2
    assert route.call_count == 2
    assert waits == [1]


def test_client_error_is_not_retried(settings, respx_mock):
    route = respx_mock.post(URL).mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(SamCallError) as info:
        segment_concept("file-1", "chair", settings, sleep=lambda s: None)
    assert info.value.status_code == 400
    assert route.call_count == 1


def test_clean_concept():
    assert clean_concept("  red   bicycle ") == "red bicycle"
    for bad in ("", "   ", "bus, bicycle", "bus; bicycle", "bus+bicycle", "a\nb"):
        with pytest.raises(ValueError):
            clean_concept(bad)
    with pytest.raises(ValueError):
        clean_concept("a" * 61)


def test_estimate_cost():
    assert estimate_cost(205) == pytest.approx(0.041)
    assert estimate_cost(10) == pytest.approx(0.01)  # billed as 50 frames
    assert estimate_cost(205, requests=3) == pytest.approx(0.123)
    assert estimate_cost(205, requests=0) == 0
    with pytest.raises(ValueError):
        estimate_cost(-1)
