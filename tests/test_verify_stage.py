import json

import pytest

from app.config import Settings
from app.pipeline import verify_stage as stage
from app.pipeline.claims import Claim, ClaimSet, ClaimStatus
from app.pipeline.files_api import FilesError
from app.pipeline.sam import SamCallError, SamResult
from app.pipeline.sam_parse import Detection
from app.pipeline.verify_stage import VerifyBudgetError, run_verify_stage


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


def make_claim(claim_id, value="bicycle", category="object"):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": category,
            "value": value,
            "evidence": [{"kind": "visual", "start_s": 4.0, "end_s": 7.0}],
        }
    )


def make_detection(frame):
    return Detection(
        frame=frame,
        object_id=0,
        x1=0,
        y1=0,
        x2=1,
        y2=1,
        width=10,
        height=10,
        confidence=0.9,
    )


@pytest.fixture
def wiring(monkeypatch):
    calls = {"downsample": [], "upload": [], "delete": [], "segment": []}

    def fake_downsample(video, out, fps=5):
        calls["downsample"].append(out)
        return out

    def fake_upload(path, settings, **kwargs):
        calls["upload"].append(path)
        return "file-9"

    def fake_delete(file_id, settings, **kwargs):
        calls["delete"].append(file_id)

    def fake_segment(file_id, concept, settings, **kwargs):
        calls["segment"].append(concept)
        return SamResult(
            concept=concept,
            detections=(make_detection(22), make_detection(23)),
            frames_billed=205,
        )

    monkeypatch.setattr(stage, "probe_duration", lambda video: 41.0)
    monkeypatch.setattr(stage, "downsample_video", fake_downsample)
    monkeypatch.setattr(stage, "upload_video", fake_upload)
    monkeypatch.setattr(stage, "delete_file", fake_delete)
    monkeypatch.setattr(stage, "segment_concept", fake_segment)
    return calls


def test_happy_path_verifies_writes_report_and_deletes_file(
    tmp_path, settings, wiring
):
    claims = ClaimSet(claims=[make_claim("a")])
    outcome = run_verify_stage(tmp_path / "clip.mp4", claims, tmp_path / "work", settings)

    assert outcome.claim_set.claims[0].status == ClaimStatus.VERIFIED
    assert wiring["segment"] == ["bicycle"]
    assert wiring["delete"] == ["file-9"]
    assert outcome.frames_billed == 205
    assert outcome.actual_cost == pytest.approx(0.041)
    assert outcome.output == tmp_path / "work" / "clip.verified.json"

    saved = json.loads(outcome.output.read_text(encoding="utf-8"))
    assert saved["claims"][0]["status"] == "verified"
    assert saved["sam"]["estimated_cost"] == pytest.approx(0.041)
    assert saved["sam"]["concepts"] == ["bicycle"]


def test_nothing_for_sam_means_no_upload_and_no_cost(tmp_path, settings, wiring):
    claims = ClaimSet(claims=[make_claim("p", "kitchen", category="place")])
    outcome = run_verify_stage(tmp_path / "clip.mp4", claims, tmp_path / "work", settings)
    assert wiring["upload"] == []
    assert wiring["downsample"] == []
    assert outcome.frames_billed == 0
    assert outcome.claim_set.claims[0].status == ClaimStatus.PROPOSED
    assert outcome.output.exists()


def test_budget_cap_aborts_before_any_spending(tmp_path, settings, wiring):
    claims = ClaimSet(claims=[make_claim("a")])
    with pytest.raises(VerifyBudgetError):
        run_verify_stage(
            tmp_path / "clip.mp4", claims, tmp_path / "work", settings, max_cost=0.001
        )
    assert wiring["upload"] == []


def test_one_failing_concept_does_not_stop_the_others(
    tmp_path, settings, wiring, monkeypatch
):
    def flaky(file_id, concept, settings, **kwargs):
        if concept == "chair":
            raise SamCallError("HTTP 500")
        return SamResult(
            concept=concept,
            detections=(make_detection(22), make_detection(23)),
            frames_billed=205,
        )

    monkeypatch.setattr(stage, "segment_concept", flaky)
    claims = ClaimSet(claims=[make_claim("a", "bicycle"), make_claim("b", "chair")])
    outcome = run_verify_stage(tmp_path / "clip.mp4", claims, tmp_path / "work", settings)

    status = {c.claim_id: c.status for c in outcome.claim_set.claims}
    assert status == {"a": ClaimStatus.VERIFIED, "b": ClaimStatus.PROPOSED}
    assert "chair" in outcome.errors
    assert wiring["delete"] == ["file-9"]


def test_delete_failure_does_not_hide_results(tmp_path, settings, wiring, monkeypatch):
    def boom(file_id, settings, **kwargs):
        raise FilesError("cannot delete")

    monkeypatch.setattr(stage, "delete_file", boom)
    claims = ClaimSet(claims=[make_claim("a")])
    outcome = run_verify_stage(tmp_path / "clip.mp4", claims, tmp_path / "work", settings)
    assert outcome.claim_set.claims[0].status == ClaimStatus.VERIFIED


def test_upload_failure_propagates_and_deletes_nothing(
    tmp_path, settings, wiring, monkeypatch
):
    def boom(path, settings, **kwargs):
        raise FilesError("upload failed")

    monkeypatch.setattr(stage, "upload_video", boom)
    claims = ClaimSet(claims=[make_claim("a")])
    with pytest.raises(FilesError):
        run_verify_stage(tmp_path / "clip.mp4", claims, tmp_path / "work", settings)
    assert wiring["delete"] == []
