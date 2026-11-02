"""Create the blind judging sheet for one pipeline run (once).

Usage: python scripts/make_judge_sheet.py --work-dir data/work_tune_basic

Lists the object claims of every run in the folder in shuffled order under
anonymous ids. It refuses to overwrite an existing sheet. Prints counts only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.judging import build_sheet, sheet_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    try:
        sheet = build_sheet(args.work_dir, seed=args.seed)
    except (FileExistsError, ValueError) as exc:
        print(f"refused: {exc}")
        sys.exit(1)
    print(f"items: {len(sheet.items)}")
    print(f"clips: {len({item.stem for item in sheet.items})}")
    print(f"sheet: {sheet_path(args.work_dir)}")


if __name__ == "__main__":
    main()
