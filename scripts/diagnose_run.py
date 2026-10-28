"""Explain a run claim by claim against the ground truth.

Usage: python scripts/diagnose_run.py data/clips/my_clip.mp4

Prints the content of the clip (claim values); keep the output private.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.claims import ClaimSet
from app.pipeline.diagnose import describe_run
from app.pipeline.evaluate import load_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    args = parser.parse_args()

    stem = args.video.stem
    report = json.loads(
        (args.work_dir / f"{stem}.verified.json").read_text(encoding="utf-8")
    )
    claim_set = ClaimSet.model_validate({"claims": report["claims"]})
    truth = load_ground_truth(Path("data/ground_truth") / f"{stem}.json")
    print("\n".join(describe_run(claim_set, truth)))


if __name__ == "__main__":
    main()
