"""Evaluate one pipeline run against a hand-labeled ground truth.

Usage: python scripts/evaluate_clip.py data/clips/my_clip.mp4

Reads data/work/<stem>.verified.json and <stem>.transcript.json plus the truth
file data/ground_truth/<stem>.json. Prints percentages and counts only, never
the content of the clip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.evaluate_clip import evaluate_clip


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--truth", type=Path, default=None)
    args = parser.parse_args()

    stem = args.video.stem
    truth = args.truth or Path("data/ground_truth") / f"{stem}.json"
    metrics = evaluate_clip(stem, args.work_dir, truth)

    print(f"transcription word error rate: {pct(metrics.wer)}")
    print(f"claims: {metrics.claims_total} by status {metrics.by_status}")
    print(f"precision, all claims (model alone): {pct(metrics.precision_all)}")
    print(
        f"precision, verified claims: {pct(metrics.precision_verified)} "
        f"({metrics.verified_total} verified)"
    )
    print(
        f"recall, verified claims: {pct(metrics.recall_verified)} "
        f"(of {metrics.truth_total} truth items)"
    )
    print(
        f"rejected but actually true: {metrics.rejected_but_true} "
        f"of {metrics.rejected_total} rejected"
    )
    print(f"left for human review: {metrics.review_queue}")
    print(f"metrics file: {args.work_dir / (stem + '.metrics.json')}")


if __name__ == "__main__":
    main()
