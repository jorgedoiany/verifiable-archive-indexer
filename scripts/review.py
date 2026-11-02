"""Open the archivist review app (review, search and export) in the browser.

Usage: python scripts/review.py --work-dir data/work_inventory

Then open http://localhost:8000 and stop with Ctrl+C. The app only listens on
this computer. Decisions are saved after every click.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips"))
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    app = create_app(args.work_dir, args.clips_dir)
    print(f"Open http://localhost:{args.port} in your browser. Stop with Ctrl+C.")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
