"""Create one blank label file per blind clip in data/labels (never overwrites).

Usage: python scripts/label_template.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.labels import write_templates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips"))
    parser.add_argument("--labels-dir", type=Path, default=Path("data/labels"))
    parser.add_argument("--prefix", default="blind_")
    args = parser.parse_args()

    created = write_templates(args.clips_dir, args.labels_dir, args.prefix)
    print(f"created {len(created)} label files in {args.labels_dir}")
    for stem in created:
        print(f"  {stem}.txt")


if __name__ == "__main__":
    main()
