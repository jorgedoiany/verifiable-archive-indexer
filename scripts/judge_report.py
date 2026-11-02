"""Summarize the judgments of one run.

Usage: python scripts/judge_report.py --work-dir data/work_tune_basic

Prints counts and percentages only, never the content of the clips.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.judge_report import GroupStats, summarize_judgments, summary_to_dict
from app.judging import JudgingDataError, judging_dir

LABELS = {
    "all": "all object claims",
    "checks_agree": "automatic checks agree",
    "automatic_miss": "automatic miss (no match, or not in the cited range)",
    "unchecked": "not checked automatically",
}


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def describe(label: str, stats: GroupStats) -> str:
    interval = stats.interval()
    spread = (
        ""
        if interval is None
        else f", 95% interval {pct(interval[0])} to {pct(interval[1])}"
    )
    return (
        f"{label}: {stats.judged} judged ({stats.exists} visible in the range, "
        f"{stats.not_exists} not visible, {stats.unsure} unsure); share visible "
        f"{pct(stats.share_existing)} of {stats.decided} decided{spread}; "
        f"counting every unsure as not visible: "
        f"{pct(stats.share_counting_unsure_as_missing)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        summary = summarize_judgments(args.work_dir)
    except (FileNotFoundError, JudgingDataError, ValueError) as exc:
        print(f"refused: {exc}")
        sys.exit(1)

    print(f"items judged: {summary.items_judged} of {summary.items_total}")
    if summary.items_judged < summary.items_total:
        print("WARNING: not everything is judged yet, so these numbers are partial")
    for key, label in LABELS.items():
        print(describe(label, summary.groups[key]))
    out = judging_dir(args.work_dir) / "report.json"
    out.write_text(json.dumps(summary_to_dict(summary), indent=2), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
