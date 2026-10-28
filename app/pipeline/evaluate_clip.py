"""Load a pipeline run from disk and evaluate it against a ground truth file."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.evaluate import Metrics, evaluate, load_ground_truth, metrics_to_dict
from app.pipeline.models import parse_transcript


class EvaluationInputError(FileNotFoundError):
    pass


class _VerifiedReport(BaseModel):
    claims: list[Claim]


def evaluate_clip(stem: str, work_dir: Path, truth_path: Path) -> Metrics:
    """Evaluate <stem>.verified.json and <stem>.transcript.json; write <stem>.metrics.json."""
    work_dir = Path(work_dir)
    truth_path = Path(truth_path)
    verified_path = work_dir / f"{stem}.verified.json"
    transcript_path = work_dir / f"{stem}.transcript.json"
    for path in (verified_path, transcript_path, truth_path):
        if not path.exists():
            raise EvaluationInputError(f"missing file: {path}")

    report = _VerifiedReport.model_validate_json(
        verified_path.read_text(encoding="utf-8")
    )
    claim_set = ClaimSet(claims=report.claims)
    transcript = parse_transcript(
        json.loads(transcript_path.read_text(encoding="utf-8"))
    )
    metrics = evaluate(claim_set, transcript.transcript, load_ground_truth(truth_path))

    out = work_dir / f"{stem}.metrics.json"
    out.write_text(json.dumps(metrics_to_dict(metrics), indent=2), encoding="utf-8")
    return metrics
