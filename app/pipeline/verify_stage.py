"""Stage 3 runner: SAM checks for object claims, then statuses and a JSON report."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.pipeline.claims import ClaimSet
from app.pipeline.files_api import FilesError, delete_file, upload_video
from app.pipeline.media import downsample_video, probe_duration
from app.pipeline.sam import PRICE_PER_1000_FRAMES, SamCallError, segment_concept
from app.pipeline.verify import (
    DEFAULT_MAX_CONCEPTS,
    VerifyPlan,
    concept_key,
    plan_verification,
    verify_claims,
)

DEFAULT_FPS = 5
DEFAULT_MAX_COST = 2.0  # dollars; the stage aborts before spending more


class VerifyBudgetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class VerifyOutcome:
    claim_set: ClaimSet
    plan: VerifyPlan
    frames_billed: int
    actual_cost: float
    errors: dict[str, str]
    output: Path


def _write_report(
    path: Path,
    claim_set: ClaimSet,
    plan: VerifyPlan,
    frames_billed: int,
    actual_cost: float,
    errors: dict[str, str],
) -> None:
    report = {
        "claims": claim_set.model_dump(mode="json")["claims"],
        "sam": {
            "concepts": list(plan.concepts),
            "capped_claim_ids": list(plan.capped_claim_ids),
            "estimated_cost": plan.estimated_cost,
            "frames_billed": frames_billed,
            "actual_cost": actual_cost,
            "errors": errors,
        },
    }
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run_verify_stage(
    video: Path,
    claim_set: ClaimSet,
    work_dir: Path,
    settings: Settings,
    *,
    fps: int = DEFAULT_FPS,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    max_cost: float = DEFAULT_MAX_COST,
    expires_seconds: int = 7_200,
) -> VerifyOutcome:
    """Verify object claims with SAM and write <stem>.verified.json."""
    video = Path(video)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{video.stem}.verified.json"

    plan = plan_verification(
        claim_set,
        duration_s=probe_duration(video),
        fps=fps,
        max_concepts=max_concepts,
    )
    results = {}
    errors: dict[str, str] = {}
    frames_billed = 0

    if plan.concepts:
        if plan.estimated_cost > max_cost:
            raise VerifyBudgetError(
                f"estimated SAM cost ${plan.estimated_cost:.2f} exceeds the "
                f"${max_cost:.2f} cap; lower the fps or the number of concepts"
            )
        low_fps = downsample_video(video, work_dir / f"{video.stem}.sam.mp4", fps=fps)
        file_id = upload_video(low_fps, settings, expires_seconds=expires_seconds)
        try:
            for concept in plan.concepts:
                key = concept_key(concept)
                try:
                    result = segment_concept(file_id, concept, settings)
                except SamCallError as exc:
                    errors[key] = str(exc)[:160]
                    continue
                results[key] = result
                frames_billed += result.frames_billed or 0
        finally:
            try:
                delete_file(file_id, settings)
            except FilesError:
                pass  # the file expires on its own; do not hide the results

    verified = verify_claims(
        claim_set,
        results,
        fps=fps,
        errors=errors,
        capped_ids=plan.capped_claim_ids,
    )
    actual_cost = frames_billed * PRICE_PER_1000_FRAMES / 1000
    _write_report(out, verified, plan, frames_billed, actual_cost, errors)
    return VerifyOutcome(verified, plan, frames_billed, actual_cost, errors, out)
