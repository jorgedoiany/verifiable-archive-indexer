import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline.claims import Claim
from app.pipeline.review import load_store

VIDEO_BYTES = bytes(range(256)) * 4  # 1024 bytes

TRANSCRIPT = {
    "transcript": "Estamos en Santo Domingo hola",
    "audioDurationMs": 6000,
    "turns": [
        {
            "turnId": 1,
            "startMs": 0,
            "endMs": 3000,
            "transcript": "Estamos en Santo Domingo hola",
        }
    ],
}


def make_claim(claim_id, category, value, status, evidence):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value,
            "evidence": evidence,
            "status": status,
        }
    )


CLAIMS = [
    make_claim(
        "a", "object", "bicycle", "verified",
        [{"kind": "visual", "start_s": 4.0, "end_s": 7.0}],
    ),
    make_claim(
        "b", "place", "Santo Domingo", "proposed",
        [{"kind": "speech", "turn_id": 1, "quote": "santo domingo"}],
    ),
    make_claim(
        "c", "object", "chair", "unsupported",
        [{"kind": "visual", "start_s": 1.0, "end_s": 2.0}],
    ),
    make_claim(
        "d", "topic", "<script>alert(1)</script>", "proposed",
        [{"kind": "speech", "turn_id": 1, "quote": "hola"}],
    ),
]


@pytest.fixture
def env(tmp_path):
    work = tmp_path / "work"
    clips = tmp_path / "clips"
    work.mkdir()
    clips.mkdir()
    report = {"claims": [c.model_dump(mode="json") for c in CLAIMS], "sam": {}}
    (work / "clip.verified.json").write_text(json.dumps(report), encoding="utf-8")
    (work / "clip.transcript.json").write_text(json.dumps(TRANSCRIPT), encoding="utf-8")
    (clips / "clip.mp4").write_bytes(VIDEO_BYTES)
    app = create_app(work, clips)
    client = TestClient(app, base_url="http://localhost")
    return client, app.state.csrf_token, work, clips


def decide(client, token, claim_id, decision, **extra):
    data = {
        "csrf": token,
        "claim_id": claim_id,
        "decision": decision,
        "position": "1",
        **extra,
    }
    return client.post("/clips/clip/decide", data=data, follow_redirects=False)


def test_index_lists_only_complete_valid_runs(env):
    client, _, work, _ = env
    (work / "other.verified.json").write_text("{}", encoding="utf-8")  # no transcript
    (work / "bad name.verified.json").write_text("{}", encoding="utf-8")
    (work / "bad name.transcript.json").write_text("{}", encoding="utf-8")
    response = client.get("/")
    assert response.status_code == 200
    assert 'href="/clips/clip"' in response.text
    assert "other" not in response.text
    assert "bad name" not in response.text


def test_index_with_no_runs_explains_what_to_do(tmp_path):
    (tmp_path / "work").mkdir()
    client = TestClient(create_app(tmp_path / "work", tmp_path), base_url="http://localhost")
    assert "run_pipeline.py" in client.get("/").text


def test_review_page_orders_claims_escapes_html_and_has_the_token(env):
    client, token, _, _ = env
    text = client.get("/clips/clip").text
    assert text.index('id="claim-1"') < text.index('id="claim-0"')  # pending before verified
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "<script>alert(1)</script>" not in text
    assert token in text
    assert 'data-seek="4.0"' in text
    assert 'src="/media/clip"' in text


def test_review_page_without_the_video_file(env):
    client, _, _, clips = env
    (clips / "clip.mp4").unlink()
    text = client.get("/clips/clip").text
    assert "<video" not in text
    assert "was not found" in text


def test_unknown_or_invalid_clip_is_a_404_page(env):
    client, *_ = env
    assert client.get("/clips/nope").status_code == 404
    response = client.get("/clips/bad.name")
    assert response.status_code == 404
    assert "Clip not found" in response.text


def test_accept_is_saved_and_shows_an_undo_button(env):
    client, token, work, _ = env
    response = decide(client, token, "b", "accepted", note="checked")
    assert response.status_code == 303
    assert response.headers["location"] == "/clips/clip#claim-1"
    store = load_store(work / "clip.review.json")
    assert store.decisions["b"].decision.value == "accepted"
    assert store.decisions["b"].note == "checked"
    assert "/clips/clip/undo" in client.get("/clips/clip").text


def test_edit_and_undo(env):
    client, token, work, _ = env
    decide(client, token, "b", "edited", value="Santo Domingo, DR")
    assert load_store(work / "clip.review.json").decisions["b"].value == "Santo Domingo, DR"
    response = client.post(
        "/clips/clip/undo",
        data={"csrf": token, "claim_id": "b", "position": "1"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert load_store(work / "clip.review.json").decisions == {}


def test_bad_token_is_forbidden_and_saves_nothing(env):
    client, _, work, _ = env
    response = decide(client, "wrong-token", "b", "accepted")
    assert response.status_code == 403
    assert not (work / "clip.review.json").exists()
    assert decide(client, "", "b", "accepted").status_code == 403


def test_invalid_submissions_are_rejected(env):
    client, token, work, _ = env
    edit = decide(client, token, "b", "edited", value="  ")
    assert edit.status_code == 400
    assert "non-empty value" in edit.text
    assert decide(client, token, "b", "none").status_code == 400
    assert decide(client, token, "zzz", "accepted").status_code == 404
    assert not (work / "clip.review.json").exists()


def test_exports_include_only_trusted_claims_unless_asked(env):
    client, token, _, _ = env
    decide(client, token, "b", "accepted")
    decide(client, token, "d", "rejected")
    catalog = client.get("/clips/clip/export.json").json()
    assert [e["claim_id"] for e in catalog["entries"]] == ["a", "b"]
    assert catalog["rejected_claim_ids"] == ["d"]
    assert catalog["pending_claim_ids"] == ["c"]

    response = client.get("/clips/clip/export.csv", params={"include_pending": "true"})
    assert 'filename="clip.catalog.csv"' in response.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(response.text)))
    assert [r[0] for r in rows[1:]] == ["a", "b", "c"]
    assert rows[3][3] == "pending"


def test_media_serves_the_whole_file_and_byte_ranges(env):
    client, *_ = env
    full = client.get("/media/clip")
    assert full.status_code == 200
    assert full.headers["accept-ranges"] == "bytes"
    assert full.content == VIDEO_BYTES

    part = client.get("/media/clip", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.headers["content-range"] == "bytes 0-99/1024"
    assert part.content == VIDEO_BYTES[:100]

    tail = client.get("/media/clip", headers={"Range": "bytes=-50"})
    assert tail.status_code == 206
    assert tail.content == VIDEO_BYTES[-50:]

    bad = client.get("/media/clip", headers={"Range": "bytes=5000-"})
    assert bad.status_code == 416
    assert bad.headers["content-range"] == "bytes */1024"


def test_media_rejects_unknown_and_invalid_names(env):
    client, *_ = env
    assert client.get("/media/nope").status_code == 404
    assert client.get("/media/bad.name").status_code == 404


def test_only_localhost_names_are_accepted(env):
    _, _, work, clips = env
    other = TestClient(create_app(work, clips), base_url="http://evil.example")
    assert other.get("/").status_code == 400


def test_security_headers_and_static_assets(env):
    client, *_ = env
    headers = client.get("/").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/review.js").status_code == 200
