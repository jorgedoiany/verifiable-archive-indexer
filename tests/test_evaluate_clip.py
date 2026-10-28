import json

import pytest

from app.pipeline.evaluate import GroundTruth
from app.pipeline.evaluate_clip import EvaluationInputError, evaluate_clip
from app.pipeline.models import parse_transcript

CLAIM = {
    "claim_id": "c1",
    "category": "object",
    "value": "bicycle",
    "evidence": [{"kind": "visual", "start_s": 4.0, "end_s": 7.0}],
    "status": "verified",
}


def write_run(tmp_path, claims=None):
    work = tmp_path / "work"
    work.mkdir()
    report = {"claims": [CLAIM] if claims is None else claims, "sam": {}}
    (work / "clip.verified.json").write_text(json.dumps(report), encoding="utf-8")
    transcript = parse_transcript(
        {
            "transcript": "Estamos en el Malecón",
            "audioDurationMs": 4000,
            "turns": [
                {"turnId": 1, "startMs": 0, "endMs": 900, "transcript": "Estamos"}
            ],
        }
    )
    (work / "clip.transcript.json").write_text(
        transcript.model_dump_json(by_alias=True), encoding="utf-8"
    )
    truth = GroundTruth.model_validate(
        {
            "clip": "clip.mp4",
            "transcript": "Estamos en el Malecón",
            "items": [
                {"category": "object", "names": ["bicycle"], "start_s": 3, "end_s": 8}
            ],
        }
    )
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(truth.model_dump_json(), encoding="utf-8")
    return work, truth_path


def test_evaluates_a_saved_run_and_writes_metrics(tmp_path):
    work, truth_path = write_run(tmp_path)
    metrics = evaluate_clip("clip", work, truth_path)
    assert metrics.wer == 0
    assert metrics.precision_verified == 1.0
    assert metrics.recall_verified == 1.0
    saved = json.loads((work / "clip.metrics.json").read_text(encoding="utf-8"))
    assert saved["verified_total"] == 1


def test_missing_input_file_is_reported(tmp_path):
    work, truth_path = write_run(tmp_path)
    truth_path.unlink()
    with pytest.raises(EvaluationInputError):
        evaluate_clip("clip", work, truth_path)


def test_report_without_claims_is_rejected(tmp_path):
    work, truth_path = write_run(tmp_path)
    (work / "clip.verified.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        evaluate_clip("clip", work, truth_path)
