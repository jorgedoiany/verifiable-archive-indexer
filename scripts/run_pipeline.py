"""Run the whole pipeline on one clip.

Usage: python scripts/run_pipeline.py data/clips/my_clip.mp4

Makes live API calls and costs a few cents; keep the clip under a minute.
Prints counts only, never the content of the clip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_settings
from app.pipeline.run import run_pipeline
from app.pipeline.verify import DEFAULT_MAX_CONCEPTS
from app.pipeline.verify_stage import DEFAULT_FPS, DEFAULT_MAX_COST


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-concepts", type=int, default=DEFAULT_MAX_CONCEPTS)
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST)
    args = parser.parse_args()

    result = run_pipeline(
        args.video,
        args.work_dir,
        load_settings(),
        fps=args.fps,
        max_concepts=args.max_concepts,
        max_cost=args.max_cost,
    )
    sam = result.verified
    print(f"turns: {len(result.transcript.turns)}")
    print(f"claims by status: {result.status_counts}")
    print(f"claims skipped by the parser: {len(result.parse_errors)}")
    print(
        f"SAM: {len(sam.plan.concepts)} concepts, {sam.frames_billed} frames, "
        f"${sam.actual_cost:.3f}, {len(sam.errors)} errors"
    )
    print(f"summary: {result.summary_path}")


if __name__ == "__main__":
    main()
