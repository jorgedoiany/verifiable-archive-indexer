import json

import pytest

from app.config import Settings
from app.pipeline import run as run_module
from app.pipeline.claims import Claim, ClaimSet, ClaimStatus
from app.pipeline.files_api import FilesError
from app.pipeline.models import parse_transcript
from app.pipeline.verify import VerifyPlan
from app.pipeline.verify_stage import VerifyBudgetError, VerifyOutcome

TRANSCRIPT = parse_transcript(
    {
        "transcript": "Hola.",
        "audioDurationMs": 4000,
        "turns": [{"turnId": 1, "startMs": 0, "endMs": 900, "transcript": "Hola."}],
    }
)


def make_claim(claim_id, status):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": "object",
            "value": "bicycle",
            "evidence": [{"kind": "visual", "start_s": 1.0, "end_s": 2.0}],
            "status": status,
        }
    )


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


@pytest.fixture
def stages(monkeypatch, tmp_path):
    calls = []
    final = ClaimSet(
        claims=[make_claim("a", "verified"), make_claim("b", "unsupported")]
    )

    def fake_transcript(video, work_dir, settings, **kwargs):
        calls.append(("transcript", video))
        return TRANSCRIPT, work_dir / "clip.transcript.json"

    def fake_strip(video, out):
        calls.append(("strip", out))
        return out

    def fake_extract(silent, transcript, work_dir, settings, **kwargs):
        calls.append(("extract", silent))
        return ClaimSet(), ["claim #2: bad"], "file-5", work_dir / "clip.claims.json"

    def fake_delete(file_id, settings, **kwargs):
        calls.append(("delete", file_id))

    def fake_verify(video, claim_set, work_dir, settings, **kwargs):
        calls.append(("verify", video))
        plan = VerifyPlan(
            concepts=("bicycle",), capped_claim_ids=(), estimated_cost=0.04
        )
        return VerifyOutcome(
            final, plan, 205, 0.041, {}, work_dir / "clip.verified.json"
        )

    monkeypatch.setattr(run_module, "run_transcript_stage", fake_transcript)
    monkeypatch.setattr(run_module, "strip_audio", fake_strip)
    monkeypatch.setattr(run_module, "run_extract_stage", fake_extract)
    monkeypatch.setattr(run_module, "delete_file", fake_delete)
    monkeypatch.setattr(run_module, "run_verify_stage", fake_verify)
    return calls


def test_runs_stages_in_order_and_writes_summary(tmp_path, settings, stages):
    video = tmp_path / "clip.mp4"
    work = tmp_path / "work"
    result = run_module.run_pipeline(video, work, settings)

    assert [name for name, _ in stages] == [
        "transcript",
        "strip",
        "extract",
        "delete",
        "verify",
    ]
    assert dict(stages)["extract"] == work / "clip.silent.mp4"
    assert dict(stages)["verify"] == video
    assert dict(stages)["delete"] == "file-5"
    assert result.status_counts == {
        ClaimStatus.VERIFIED.value: 1,
        ClaimStatus.UNSUPPORTED.value: 1,
    }

    saved = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert saved["clip"] == "clip.mp4"
    assert saved["claims_total"] == 2
    assert saved["parse_errors"] == 1
    assert saved["sam"]["frames_billed"] == 205
    assert saved["files"]["verified"].endswith("clip.verified.json")


def test_extract_upload_is_deleted_even_if_verification_fails(
    tmp_path, settings, stages, monkeypatch
):
    def boom(*args, **kwargs):
        raise VerifyBudgetError("too expensive")

    monkeypatch.setattr(run_module, "run_verify_stage", boom)
    with pytest.raises(VerifyBudgetError):
        run_module.run_pipeline(tmp_path / "clip.mp4", tmp_path / "work", settings)
    assert ("delete", "file-5") in stages


def test_delete_failure_is_ignored(tmp_path, settings, stages, monkeypatch):
    def boom(file_id, settings, **kwargs):
        raise FilesError("cannot delete")

    monkeypatch.setattr(run_module, "delete_file", boom)
    result = run_module.run_pipeline(
        tmp_path / "clip.mp4", tmp_path / "work", settings
    )
    assert result.status_counts[ClaimStatus.VERIFIED.value] == 1


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [({}, "basic"), ({"prompt_variant": "inventory"}, "inventory")],
)
def test_prompt_variant_is_passed_on_and_recorded(
    tmp_path, settings, stages, monkeypatch, kwargs, expected
):
    seen = {}

    def capture(silent, transcript, work_dir, settings, **options):
        seen.update(options)
        return ClaimSet(), [], "file-5", work_dir / "clip.claims.json"

    monkeypatch.setattr(run_module, "run_extract_stage", capture)
    result = run_module.run_pipeline(
        tmp_path / "clip.mp4", tmp_path / "work", settings, **kwargs
    )
    assert seen == {"prompt_variant": expected}
    saved = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert saved["prompt_variant"] == expected
