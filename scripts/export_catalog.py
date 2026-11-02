"""Export the final catalog of one clip as JSON and CSV.

Usage: python scripts/export_catalog.py data/clips/my_clip.mp4 [--include-pending]

Reads data/work/<stem>.verified.json and, if present, <stem>.review.json.
Prints counts only, never the content of the clip.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.catalog import export_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--include-pending", action="store_true")
    args = parser.parse_args()

    catalog, json_path, csv_path = export_catalog(
        args.video.stem, args.work_dir, include_pending=args.include_pending
    )
    counts = Counter(entry.final_status for entry in catalog.entries)
    print(f"entries by status: {dict(counts)}")
    print(
        f"rejected: {len(catalog.rejected_claim_ids)}; "
        f"pending: {len(catalog.pending_claim_ids)}"
    )
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
