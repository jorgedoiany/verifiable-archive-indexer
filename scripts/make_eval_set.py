"""Draw the tuning/blind split once and copy the clips into data/clips.

Usage: python scripts/make_eval_set.py ~/Desktop/pool_clean --seed 20261031

Prints counts only. The private record of which source became which clip is
written to data/work_eval/split.json (ignored by Git).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.eval_set import build_eval_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", type=Path)
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips"))
    parser.add_argument(
        "--record", type=Path, default=Path("data/work_eval/split.json")
    )
    parser.add_argument("--tune", type=int, default=5)
    parser.add_argument("--blind", type=int, default=10)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    split = build_eval_set(
        args.pool,
        args.clips_dir,
        args.record,
        tune=args.tune,
        blind=args.blind,
        seed=args.seed,
    )
    print(f"tune clips: {len(split.tune)}")
    print(f"blind clips: {len(split.blind)}")
    print(f"seed: {args.seed}")
    print(f"private record: {args.record}")


if __name__ == "__main__":
    main()
