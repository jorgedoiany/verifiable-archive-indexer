"""Run the pipeline over many clips with spending caps and a per-clip report.

A clip that already has a result in the work folder is skipped, so one variant
runs once per clip. A failing clip does not stop the others. Blind clips are
only run if the frozen manifest still matches.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from app.config import Settings
from app.freeze import FreezeError, verify_manifest
from app.pipeline.run import PipelineResult, run_pipeline
from app.pipeline.verify import DEFAULT_MAX_CONCEPTS
from app.pipeline.verify_stage import DEFAULT_FPS, DEFAULT_MAX_COST

GROUPS = ("tune", "blind", "all")
REPORT_NAME = "batch_report.json"
DEFAULT_MAX_TOTAL_COST = 10.0


class BatchReportError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BatchEntry:
    stem: str
    status: str  # done, failed, missing, skipped_existing, not_run_budget
    claims_total: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    parse_errors: int = 0
    frames_billed: int = 0
    actual_cost: float = 0.0
    error: str | None = None


def select_stems(clips_dir: Path, group: str) -> list[str]:
    if group not in GROUPS:
        raise ValueError(f"unknown group: {group!r}")
    prefixes = {
        "tune": ("tune_",),
        "blind": ("blind_",),
        "all": ("tune_", "blind_"),
    }[group]
    return sorted(
        p.stem for p in Path(clips_dir).glob("*.mp4") if p.stem.startswith(prefixes)
    )


def check_blind_frozen(
    stems: Sequence[str], manifest_path: Path, clips_dir: Path, truth_dir: Path
) -> None:
    """Refuse to run blind clips unless the frozen manifest still matches."""
    if not any(stem.startswith("blind_") for stem in stems):
        return
    try:
        problems = verify_manifest(manifest_path, clips_dir, truth_dir)
    except FreezeError as exc:
        raise FreezeError(f"cannot run blind clips: {exc}") from exc
    if problems:
        raise FreezeError(
            "the blind set changed since it was frozen: " + "; ".join(problems)
        )


def load_report(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {
            "prompt_variant": data["prompt_variant"],
            "entries": dict(data["entries"]),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise BatchReportError(f"cannot read {path}: {exc}") from exc


def _save_report(
    path: Path,
    prompt_variant: str,
    fps: int,
    max_total_cost: float,
    previous: dict | None,
    entries: dict[str, BatchEntry],
) -> None:
    merged = dict(previous["entries"]) if previous else {}
    merged.update({stem: asdict(entry) for stem, entry in entries.items()})
    report = {
        "prompt_variant": prompt_variant,
        "fps": fps,
        "max_total_cost": max_total_cost,
        "entries": dict(sorted(merged.items())),
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _done_entry(stem: str, result: PipelineResult) -> BatchEntry:
    outcome = result.verified
    return BatchEntry(
        stem=stem,
        status="done",
        claims_total=len(outcome.claim_set.claims),
        status_counts=dict(result.status_counts),
        parse_errors=len(result.parse_errors),
        frames_billed=outcome.frames_billed,
        actual_cost=outcome.actual_cost,
    )


def run_batch(
    stems: Sequence[str],
    clips_dir: Path,
    work_dir: Path,
    settings: Settings,
    *,
    prompt_variant: str,
    fps: int = DEFAULT_FPS,
    max_concepts: int = DEFAULT_MAX_CONCEPTS,
    max_cost_per_clip: float = DEFAULT_MAX_COST,
    max_total_cost: float = DEFAULT_MAX_TOTAL_COST,
    runner: Callable[..., PipelineResult] = run_pipeline,
) -> list[BatchEntry]:
    """Run the clips in order; return one entry per stem."""
    clips_dir = Path(clips_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / REPORT_NAME

    previous = load_report(report_path)
    if previous is not None and previous["prompt_variant"] != prompt_variant:
        raise BatchReportError(
            f"{work_dir} was used with the '{previous['prompt_variant']}' prompt; "
            "use a separate folder for each variant"
        )

    entries: list[BatchEntry] = []
    persisted: dict[str, BatchEntry] = {}
    spent = 0.0
    for stem in stems:
        clip = clips_dir / f"{stem}.mp4"
        if not clip.exists():
            entry = BatchEntry(stem, "missing", error=f"clip not found: {clip.name}")
        elif (work_dir / f"{stem}.result.json").exists():
            entries.append(BatchEntry(stem, "skipped_existing"))
            continue  # the earlier entry in the report stays as it was
        elif spent >= max_total_cost:
            entry = BatchEntry(stem, "not_run_budget")
        else:
            try:
                result = runner(
                    clip,
                    work_dir,
                    settings,
                    fps=fps,
                    max_concepts=max_concepts,
                    max_cost=max_cost_per_clip,
                    prompt_variant=prompt_variant,
                )
            except Exception as exc:  # noqa: BLE001 - one clip must not stop the batch
                entry = BatchEntry(
                    stem, "failed", error=f"{type(exc).__name__}: {str(exc)[:160]}"
                )
            else:
                entry = _done_entry(stem, result)
                spent += entry.actual_cost
        entries.append(entry)
        persisted[stem] = entry
        _save_report(
            report_path, prompt_variant, fps, max_total_cost, previous, persisted
        )
    return entries
