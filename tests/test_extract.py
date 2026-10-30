import json

import httpx
import pytest

from app.config import Settings
from app.pipeline import extract
from app.pipeline.claims import ClaimStatus
from app.pipeline.extract import (
    ExtractCallError,
    call_model,
    extract_claims,
    run_extract_stage,
)
from app.pipeline.models import parse_transcript

URL = "https://api.meta.ai/v1/responses"

TRANSCRIPT = parse_transcript(
    {
        "transcript": "Estamos en el Malecón.",
        "audioDurationMs": 10000,
        "turns": [
            {
                "turnId": 1,
                "startMs": 0,
                "endMs": 4000,
                "transcript": "Estamos en el Malecón.",
            }
        ],
    }
)

CLAIMS_TEXT = json.dumps(
    {
        "claims": [
            {
                "claim_id": "c1",
                "category": "place",
                "value": "Malecón",
                "evidence": [
                    {"kind": "speech", "turn_id": 1, "quote": "en el Malecón"}
                ],
            },
            {
                "claim_id": "c2",
                "category": "place",
                "value": "Zona Colonial",
                "evidence": [
                    {"kind": "speech", "turn_id": 1, "quote": "la Zona Colonial"}
                ],
            },
            {"claim_id": "bad", "category": "object", "value": "x", "evidence": []},
        ]
    }
)


def reply(text):
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
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


def test_call_model_sends_documented_request(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply("hola"))
    )
    assert call_model("file-1", "describe", settings) == "hola"
    request = route.calls.last.request
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "muse-spark-1.3"
    assert payload["max_output_tokens"] >= 4000
    content = payload["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "describe"}
    assert content[1] == {"type": "input_file", "file_id": "file-1"}


def test_top_level_output_text_is_accepted(settings, respx_mock):
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json={"output_text": "directo"})
    )
    assert call_model("file-1", "p", settings) == "directo"


def test_incomplete_response_is_an_error(settings, respx_mock):
    respx_mock.post(URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
            },
        )
    )
    with pytest.raises(ExtractCallError) as info:
        call_model("file-1", "p", settings)
    assert "max_output_tokens" in str(info.value)


def test_empty_output_is_an_error(settings, respx_mock):
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json={"status": "completed", "output": []})
    )
    with pytest.raises(ExtractCallError):
        call_model("file-1", "p", settings)


def test_retries_on_429(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        side_effect=[httpx.Response(429), httpx.Response(200, json=reply("ok"))]
    )
    waits = []
    assert call_model("file-1", "p", settings, sleep=waits.append) == "ok"
    assert route.call_count == 2
    assert waits == [1]


def test_client_error_is_not_retried(settings, respx_mock):
    route = respx_mock.post(URL).mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(ExtractCallError) as info:
        call_model("file-1", "p", settings, sleep=lambda s: None)
    assert info.value.status_code == 400
    assert route.call_count == 1


def test_extract_claims_checks_quotes(settings, respx_mock):
    respx_mock.post(URL).mock(
        return_value=httpx.Response(200, json=reply(CLAIMS_TEXT))
    )
    claim_set, errors = extract_claims("file-1", TRANSCRIPT, settings)
    status = {c.claim_id: c.status for c in claim_set.claims}
    assert status == {"c1": ClaimStatus.VERIFIED, "c2": ClaimStatus.UNSUPPORTED}
    assert len(errors) == 1 and "claim #2" in errors[0]


def test_stage_writes_claims_and_keeps_file(tmp_path, settings, respx_mock, monkeypatch):
    deleted = []
    monkeypatch.setattr(extract, "upload_video", lambda path, s, **kw: "file-1")
    monkeypatch.setattr(extract, "delete_file", lambda fid, s, **kw: deleted.append(fid))
    respx_mock.post(URL).mock(return_value=httpx.Response(200, json=reply(CLAIMS_TEXT)))

    _claim_set, _errors, file_id, out = run_extract_stage(
        tmp_path / "clip.silent.mp4", TRANSCRIPT, tmp_path / "work", settings
    )
    assert file_id == "file-1"
    assert deleted == []
    assert out == tmp_path / "work" / "clip.claims.json"
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [c["claim_id"] for c in saved["claims"]] == ["c1", "c2"]
    assert saved["claims"][0]["status"] == "verified"
    assert len(saved["parse_errors"]) == 1


def test_stage_deletes_uploaded_file_when_call_fails(
    tmp_path, settings, respx_mock, monkeypatch
):
    deleted = []
    monkeypatch.setattr(extract, "upload_video", lambda path, s, **kw: "file-1")
    monkeypatch.setattr(extract, "delete_file", lambda fid, s, **kw: deleted.append(fid))
    respx_mock.post(URL).mock(return_value=httpx.Response(400, text="bad"))
    with pytest.raises(ExtractCallError):
        run_extract_stage(
            tmp_path / "clip.silent.mp4", TRANSCRIPT, tmp_path / "work", settings
        )
    assert deleted == ["file-1"]


INCOMPLETE = {
    "status": "incomplete",
    "incomplete_details": {"reason": "max_output_tokens"},
    "output": [],
}


def test_default_budget_leaves_room_for_an_inventory():
    assert extract.DEFAULT_MAX_OUTPUT_TOKENS == 16_000
    assert extract.RETRY_MAX_OUTPUT_TOKENS == 32_000


def test_truncated_response_is_retried_once_with_a_bigger_budget(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        side_effect=[
            httpx.Response(200, json=INCOMPLETE),
            httpx.Response(200, json=reply(CLAIMS_TEXT)),
        ]
    )
    claim_set, _ = extract_claims("file-1", TRANSCRIPT, settings)
    budgets = [json.loads(c.request.content)["max_output_tokens"] for c in route.calls]
    assert budgets == [16_000, 32_000]
    assert len(claim_set.claims) == 2


def test_second_truncation_is_an_error(settings, respx_mock):
    route = respx_mock.post(URL).mock(
        side_effect=[httpx.Response(200, json=INCOMPLETE)] * 2
    )
    with pytest.raises(extract.ExtractIncompleteError):
        extract_claims("file-1", TRANSCRIPT, settings)
    assert route.call_count == 2


def test_no_retry_when_already_at_the_maximum(settings, respx_mock):
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=INCOMPLETE))
    with pytest.raises(extract.ExtractIncompleteError):
        extract_claims("file-1", TRANSCRIPT, settings, max_output_tokens=32_000)
    assert route.call_count == 1


def test_other_incomplete_reasons_are_not_retried(settings, respx_mock):
    body = {
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
        "output": [],
    }
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, json=body))
    with pytest.raises(ExtractCallError):
        extract_claims("file-1", TRANSCRIPT, settings)
    assert route.call_count == 1
