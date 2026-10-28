"""Run the transcription stage on one clip.

Usage: python scripts/transcribe_clip.py data/clips/my_clip.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_settings  # noqa: E402
from app.pipeline.transcript_stage import run_transcript_stage  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--keyword", action="append", default=[])
    args = parser.parse_args()

    transcript, out = run_transcript_stage(
        args.video, args.work_dir, load_settings(), keywords=args.keyword
    )
    print(f"{len(transcript.turns)} turns, {transcript.audio_duration_ms} ms -> {out}")


if __name__ == "__main__":
    main()
