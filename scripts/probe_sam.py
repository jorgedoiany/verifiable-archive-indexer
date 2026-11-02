"""Measure SAM as a verifier on labeled clips.

Usage: python scripts/probe_sam.py data/clips/clip_a.mp4 data/clips/clip_b.mp4

Uses data/ground_truth/<stem>.json for each clip. Makes live API calls (about
$1 for two short clips). Prints concept names of the clips; keep it private.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_settings
from app.pipeline.evaluate import load_ground_truth
from app.pipeline.sam_probe import DEFAULT_PROBE_MAX_COST, probe_clip, summarize


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--work-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--max-cost", type=float, default=DEFAULT_PROBE_MAX_COST)
    args = parser.parse_args()

    settings = load_settings()
    probes = []
    for video in args.videos:
        truth = load_ground_truth(Path("data/ground_truth") / f"{video.stem}.json")
        probe = probe_clip(
            video, truth, args.work_dir, settings, max_cost=args.max_cost
        )
        probes.append(probe)
        ok = [i for i in probe.items if i.error is None]
        positives = [i for i in ok if i.expected]
        negatives = [i for i in ok if not i.expected]
        print(
            f"{video.name}: found {sum(i.found for i in positives)} of "
            f"{len(positives)} true objects; false alarms "
            f"{sum(i.found for i in negatives)} of {len(negatives)} absent concepts"
        )
        for item in positives:
            if not item.found:
                print(f"  missed: {item.concept} ({item.counted_frames} frames in range)")
        for item in negatives:
            if item.found:
                print(f"  false alarm: {item.concept} ({item.counted_frames} frames)")
        for item in probe.items:
            if item.error:
                print(f"  error on {item.concept}: {item.error}")

    summary = summarize(probes)
    out = args.work_dir / "sam_probe.json"
    out.write_text(
        json.dumps(
            {"summary": summary, "clips": [asdict(p) for p in probes]}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"sensitivity: {pct(summary['sensitivity'])} ({summary['found']} of {summary['positives']})")
    print(
        f"false alarm rate: {pct(summary['false_alarm_rate'])} "
        f"({summary['false_alarms']} of {summary['negatives']})"
    )
    print(f"errors: {summary['errors']}; frames billed: {summary['frames_billed']}")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
