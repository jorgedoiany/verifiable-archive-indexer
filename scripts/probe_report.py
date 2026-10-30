"""Analyse a saved SAM probe offline (no API calls).

Usage: python scripts/probe_report.py [--probe data/work/sam_probe.json] [--fps 5]

Prints concept names from your clips; keep the output private.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.evaluate import load_ground_truth
from app.pipeline.probe_report import describe_problems, load_probe, sweep


def pct(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{part / whole * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=Path("data/work/sam_probe.json"))
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()

    probe = load_probe(args.probe)
    truths = {
        clip["clip"]: load_ground_truth(
            Path("data/ground_truth") / f"{Path(clip['clip']).stem}.json"
        )
        for clip in probe["clips"]
    }
    for row in sweep(probe, truths, args.fps):
        print(
            f"threshold {row['threshold']}: sensitivity "
            f"{pct(row['found'], row['positives'])} ({row['found']} of {row['positives']}), "
            f"false alarms {pct(row['false_alarms'], row['negatives'])} "
            f"({row['false_alarms']} of {row['negatives']})"
        )
    print("\n".join(describe_problems(probe, truths, args.fps)))


if __name__ == "__main__":
    main()
