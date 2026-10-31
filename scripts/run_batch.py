"""Run the pipeline over a group of clips with a spending cap.

Usage:
  python scripts/run_batch.py --group tune --prompt basic --work-dir data/work_tune_basic
  python scripts/run_batch.py --group blind --prompt basic --work-dir data/work_blind_basic --dry-run

Use a separate work folder for every prompt variant. A clip that already has a
result there is skipped. Blind clips are refused if the frozen manifest no
longer matches. --dry-run lists what would run and calls nothing. Output has
counts and costs only, never the content of the clips.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.batch import (
    DEFAULT_MAX_TOTAL_COST,
    GROUPS,
    BatchReportError,
    check_blind_frozen,
    run_batch,
    select_stems,
)
from app.config import load_settings
from app.freeze import FreezeError
from app.pipeline.extract_parse import PROMPT_VARIANTS
from app.pipeline.verify import DEFAULT_MAX_CONCEPTS
from app.pipeline.verify_stage import DEFAULT_FPS, DEFAULT_MAX_COST


def describe(entry) -> str:
    text = f"{entry.stem}: {entry.status}"
    if entry.status == "done":
        text += (
            f", {entry.claims_total} claims, {entry.frames_billed} SAM frames, "
            f"${entry.actual_cost:.3f}"
        )
    elif entry.error:
        text += f", {entry.error}"
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--prompt", choices=PROMPT_VARIANTS, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips"))
    parser.add_argument("--truth-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/blind-manifest.json")
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-concepts", type=int, default=DEFAULT_MAX_CONCEPTS)
    parser.add_argument("--max-cost-per-clip", type=float, default=DEFAULT_MAX_COST)
    parser.add_argument("--max-total-cost", type=float, default=DEFAULT_MAX_TOTAL_COST)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stems = select_stems(args.clips_dir, args.group)
    if not stems:
        print("refused: no clips found for that group")
        sys.exit(1)
    try:
        check_blind_frozen(stems, args.manifest, args.clips_dir, args.truth_dir)
    except FreezeError as exc:
        print(f"refused: {exc}")
        sys.exit(1)

    if args.dry_run:
        done = [s for s in stems if (args.work_dir / f"{s}.result.json").exists()]
        print(f"would run {len(stems) - len(done)} of {len(stems)} clips: " + ", ".join(stems))
        print(f"already have a result: {len(done)}")
        print(
            f"prompt: {args.prompt}; caps: ${args.max_cost_per_clip:.2f} per clip, "
            f"${args.max_total_cost:.2f} total"
        )
        print("dry run: no API calls were made")
        return

    try:
        entries = run_batch(
            stems,
            args.clips_dir,
            args.work_dir,
            load_settings(),
            prompt_variant=args.prompt,
            fps=args.fps,
            max_concepts=args.max_concepts,
            max_cost_per_clip=args.max_cost_per_clip,
            max_total_cost=args.max_total_cost,
        )
    except BatchReportError as exc:
        print(f"refused: {exc}")
        sys.exit(1)

    for entry in entries:
        print(describe(entry))
    print(f"total SAM cost this run: ${sum(e.actual_cost for e in entries):.3f}")
    print(f"report: {args.work_dir / 'batch_report.json'}")


if __name__ == "__main__":
    main()
