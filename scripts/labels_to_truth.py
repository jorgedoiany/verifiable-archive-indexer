"""Convert the text label files in data/labels to ground-truth JSON.

Usage: python scripts/labels_to_truth.py

Checks every file and reports problems by line number. A ground-truth file
that already exists is frozen and skipped.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.labels import convert_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips"))
    parser.add_argument("--truth-dir", type=Path, default=Path("data/ground_truth"))
    args = parser.parse_args()

    report = convert_labels(args.labels_dir, args.clips_dir, args.truth_dir)
    for stem in report.written:
        print(f"{stem}: written")
    for stem in report.skipped:
        print(f"{stem}: already frozen, skipped")
    for stem, problems in report.failed.items():
        print(f"{stem}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem}")
    print(
        f"written {len(report.written)}, skipped {len(report.skipped)}, "
        f"with problems {len(report.failed)}"
    )


if __name__ == "__main__":
    main()
