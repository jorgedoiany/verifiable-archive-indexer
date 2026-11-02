"""Pooled measurements of a batch run against the frozen ground truth.

Usage: python scripts/blind_eval.py --work-dir data/work_blind_basic

For the blind group it first checks that the frozen manifest still matches.
Prints counts and percentages only, never the content of the clips.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.batch import check_blind_frozen, select_stems
from app.blind_eval import measure_clip, pool
from app.freeze import FreezeError


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def span(interval: list[float] | None) -> str:
    return "" if interval is None else f", 95% interval {pct(interval[0])} to {pct(interval[1])}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--group", choices=("tune", "blind"), default="blind")
    parser.add_argument("--clips-dir", type=Path, default=Path("data/clips"))
    parser.add_argument("--truth-dir", type=Path, default=Path("data/ground_truth"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("docs/blind-manifest.json")
    )
    args = parser.parse_args()

    stems = select_stems(args.clips_dir, args.group)
    try:
        check_blind_frozen(stems, args.manifest, args.clips_dir, args.truth_dir)
    except FreezeError as exc:
        print(f"refused: {exc}")
        sys.exit(1)

    measures = []
    not_run = []
    for stem in stems:
        if not (args.work_dir / f"{stem}.verified.json").exists():
            not_run.append(stem)
            continue
        measures.append(measure_clip(stem, args.work_dir, args.truth_dir))
    if not measures:
        print("refused: no completed clips found in that folder")
        sys.exit(1)

    print("clip | reference words | word edits | objects labeled | matched by any claim | matched where checks agree")
    for m in measures:
        print(
            f"{m.stem} | {m.reference_words} | {m.edits} | {m.truth_objects} | "
            f"{m.matched_by_any_claim} | {m.matched_by_checks_agree}"
        )
    if not_run:
        print(f"not run: {', '.join(not_run)}")

    total = pool(measures)
    print(
        f"POOLED word error rate: {pct(total['word_error_rate'])} "
        f"({total['edits']} edits over {total['reference_words']} reference words)"
    )
    print(
        f"POOLED invented words on clips without speech: "
        f"{total['invented_words_on_silent_clips']} ({total['silent_clips']} such clips)"
    )
    print(
        f"POOLED coverage, any claim: {pct(total['coverage_any_claim'])} "
        f"({total['matched_by_any_claim']} of {total['truth_objects']} labeled objects"
        f"{span(total['coverage_any_claim_interval_95'])})"
    )
    print(
        f"POOLED coverage, checks agree: {pct(total['coverage_checks_agree'])} "
        f"({total['matched_by_checks_agree']} of {total['truth_objects']}"
        f"{span(total['coverage_checks_agree_interval_95'])})"
    )
    print(
        f"object claims: {total['object_claims']}; SAM frames: {total['frames_billed']}; "
        f"SAM cost: ${total['cost']:.3f}"
    )
    out = args.work_dir / "blind_eval.json"
    out.write_text(json.dumps({"pooled": total}, indent=2), encoding="utf-8")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
