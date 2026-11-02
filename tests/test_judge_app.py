import json

import pytest
from fastapi.testclient import TestClient

from app.judge_app import create_judge_app
from app.judging import build_sheet, load_judgments, load_sheet
from app.pipeline.claims import Claim

VIDEO_BYTES = bytes(range(256)) * 4
SPEECH = {"kind": "speech", "turn_id": 1, "quote": "lamp"}
FORBIDDEN = (
    "SAM",
    "verif",
    "unsupported",
    "disputed",
    "proposed",
    "confidence",
    "agree",
    "found no match",
)


def make_claim(claim_id, category, value, status, notes, evidence):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value,
            "evidence": evidence,
            "status": status,
            "notes": notes,
        }
    )


CLAIMS = [
    make_claim(
        "a", "object", "bicycle", "verified", "SAM found 'bicycle' in 12 frames",
        [{"kind": "visual", "start_s": 4.0, "end_s": 7.0}],
    ),
    make_claim(
        "b", "object", "<b>chair</b>", "unsupported", "SAM found no match for chair",
        [{"kind": "visual", "start_s": 1.0, "end_s": 2.5}],
    ),
    make_claim("c", "object", "lamp", "proposed", None, [SPEECH]),
    make_claim("d", "place", "Santo Domingo", "proposed", None, [SPEECH]),
]


def write_claims(work, claims):
    report = {"claims": [claim.model_dump(mode="json") for claim in claims]}
    (work / "blind_01.verified.json").write_text(json.dumps(report), encoding="utf-8")


@pytest.fixture
def env(tmp_path):
    work = tmp_path / "work"
    clips = tmp_path / "clips"
    work.mkdir()
    clips.mkdir()
    write_claims(work, CLAIMS)
    (clips / "blind_01.mp4").write_bytes(VIDEO_BYTES)
    build_sheet(work, seed=1)
    app = create_judge_app(work, clips)
    return TestClient(app, base_url="http://localhost"), app.state.csrf_token, work


def post(client, token, item_id, verdict, **extra):
    data = {"csrf": token, "verdict": verdict, **extra}
    return client.post(f"/judge/{item_id}", data=data, follow_redirects=False)


def test_without_a_sheet_the_home_page_explains_what_to_do(tmp_path):
    (tmp_path / "work").mkdir()
    client = TestClient(create_judge_app(tmp_path / "work", tmp_path), base_url="http://localhost")
    response = client.get("/")
    assert response.status_code == 404
    assert "make_judge_sheet.py" in response.text


def test_home_shows_progress_and_a_link_to_the_next_item(env):
    client, token, work = env
    first, second = load_sheet(work).items[:2]
    text = client.get("/").text
    assert "0 of 3 judged" in text
    assert f'href="/judge/{first.item_id}"' in text
    post(client, token, first.item_id, "exists")
    text = client.get("/").text
    assert "1 of 3 judged" in text
    assert f'href="/judge/{second.item_id}"' in text


def test_an_item_page_shows_the_claim_but_nothing_from_any_verifier(env):
    client, _, work = env
    pages = {
        item.claim_id: client.get(f"/judge/{item.item_id}").text
        for item in load_sheet(work).items
    }
    assert "bicycle" in pages["a"]
    assert "4.0 to 7.0 s" in pages["a"]
    assert "&lt;b&gt;chair&lt;/b&gt;" in pages["b"]
    assert "1.0 to 2.5 s" in pages["b"]
    assert "no time range cited" in pages["c"]
    assert "Santo Domingo" not in "".join(pages.values())  # places are not judged here
    for text in pages.values():
        for word in FORBIDDEN:
            assert word not in text


def test_pressing_enter_in_the_note_cannot_submit_a_verdict(env):
    client, _, work = env
    item = load_sheet(work).items[0]
    text = client.get(f"/judge/{item.item_id}").text
    assert text.index('type="submit" disabled hidden') < text.index('name="verdict"')


def test_a_judgment_is_saved_and_the_next_item_opens(env):
    client, token, work = env
    first, second = load_sheet(work).items[:2]
    response = post(client, token, first.item_id, "exists", note=" looks right ")
    assert response.status_code == 303
    assert response.headers["location"] == f"/judge/{second.item_id}"
    saved = load_judgments(work).judgments[first.item_id]
    assert saved.verdict.value == "exists"
    assert saved.note == "looks right"


def test_the_last_judgment_returns_home(env):
    client, token, work = env
    locations = [
        post(client, token, item.item_id, "unsure").headers["location"]
        for item in load_sheet(work).items
    ]
    assert locations[-1] == "/"
    assert "3 of 3 judged" in client.get("/").text
    assert "Everything is judged" in client.get("/").text


def test_a_judgment_can_be_changed_and_is_shown_again(env):
    client, token, work = env
    item = load_sheet(work).items[0]
    post(client, token, item.item_id, "exists")
    post(client, token, item.item_id, "not_exists", note="only elsewhere")
    saved = load_judgments(work).judgments[item.item_id]
    assert (saved.verdict.value, saved.note) == ("not_exists", "only elsewhere")
    assert len(load_judgments(work).judgments) == 1
    text = client.get(f"/judge/{item.item_id}").text
    assert "Your current judgment: Not visible in that range" in text


def test_a_bad_token_is_forbidden_and_saves_nothing(env):
    client, _, work = env
    item = load_sheet(work).items[0]
    assert post(client, "wrong", item.item_id, "exists").status_code == 403
    assert post(client, "", item.item_id, "exists").status_code == 403
    assert not (work / "judging" / "judgments.json").exists()


def test_invalid_submissions_are_rejected(env):
    client, token, work = env
    item = load_sheet(work).items[0]
    assert post(client, token, item.item_id, "maybe").status_code == 400
    assert post(client, token, item.item_id, "exists", note="x" * 201).status_code == 400
    assert post(client, token, "j999", "exists").status_code == 404
    assert client.get("/judge/j999").status_code == 404
    assert not (work / "judging" / "judgments.json").exists()


def test_an_item_whose_claim_vanished_is_a_clear_error(env):
    client, _, work = env
    write_claims(work, [CLAIMS[3]])
    item = load_sheet(work).items[0]
    response = client.get(f"/judge/{item.item_id}")
    assert response.status_code == 500
    assert "could not be read" in response.text


def test_media_serves_ranges_and_rejects_bad_names(env):
    client, *_ = env
    part = client.get("/media/blind_01", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206
    assert part.content == VIDEO_BYTES[:100]
    assert client.get("/media/nope").status_code == 404
    assert client.get("/media/bad.name").status_code == 404


def test_only_localhost_names_are_accepted(env):
    _, _, work = env
    other = TestClient(create_judge_app(work, work), base_url="http://evil.example")
    assert other.get("/").status_code == 400


def test_security_headers_and_static_assets(env):
    client, *_ = env
    headers = client.get("/").headers
    assert "default-src 'self'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"
    for name in ("style.css", "judge.css", "judge.js"):
        assert client.get(f"/static/{name}").status_code == 200
