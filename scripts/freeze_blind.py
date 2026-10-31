"""Freeze (or verify) the blind evaluation set.

Usage: python scripts/freeze_blind.py           # write docs/blind-manifest.json
       python scripts/freeze_blind.py --verify  # check that nothing changed

Prints counts only. Freezing is refused if any blind clip already has pipeline
outputs, if a label file is missing, or if the manifest already exists.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.freeze import FreezeError, freeze_blind, verify_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips"))
    parser.add_argument("--truth-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/blind-manifest.json")
    )
    args = parser.parse_args()

    try:
        if args.verify:
            problems = verify_manifest(args.manifest, args.clips_dir, args.truth_dir)
            print("ok" if not problems else f"{len(problems)} problem(s)")
            for problem in problems:
                print(f"  {problem}")
            if problems:
                sys.exit(1)
            return
        manifest = freeze_blind(
            args.clips_dir, args.truth_dir, args.data_dir, args.manifest
        )
    except FreezeError as exc:
        print(f"refused: {exc}")
        sys.exit(1)
    print(f"frozen {len(manifest['clips'])} blind clips")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
