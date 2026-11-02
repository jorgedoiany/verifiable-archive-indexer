"""Test Muse Wonder as a second, blind verifier on the inventory runs.

Usage: python scripts/arbiter_experiment.py data/clips/test_clip.mp4 data/clips/clip2.mp4

Needs data/work_inventory/<stem>.verified.json (from a pipeline run) and the
audit file data/ground_truth/audit_inventory.json. Makes one live Wonder call
per clip (a few tens of cents in total). Prints concept names; keep it private.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import load_settings
from app.pipeline.arbiter import ask_arbiter
from app.pipeline.arbiter_experiment import (
    DEFAULT_DECOYS,
    build_probes,
    disagreements,
    load_not_real,
    merge_scores,
    score,
)
from app.pipeline.claims import ClaimSet
from app.pipeline.files_api import FilesError, delete_file, upload_video
from app.pipeline.media import probe_duration, strip_audio


def pct(part: int, whole: int) -> str:
    return "n/a" if whole == 0 else f"{part / whole * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="+")
    parser.add_argument("--work-dir", type=Path, default=Path("data/work_inventory"))
    parser.add_argument(
        "--audit", type=Path, default=Path("data/ground_truth/audit_inventory.json")
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    settings = load_settings()
    scores = []
    for video in args.videos:
        stem = video.stem
        report = json.loads(
            (args.work_dir / f"{stem}.verified.json").read_text(encoding="utf-8")
        )
        claim_set = ClaimSet.model_validate({"claims": report["claims"]})
        probes = build_probes(
            claim_set,
            not_real=load_not_real(args.audit, stem),
            decoys=DEFAULT_DECOYS,
            duration_s=probe_duration(video),
            seed=args.seed,
        )
        silent = strip_audio(video, args.work_dir / f"{stem}.arbiter.mp4")
        file_id = upload_video(silent, settings, expires_seconds=7_200)
        try:
            answers, errors = ask_arbiter(file_id, [p.item for p in probes], settings)
        finally:
            try:
                delete_file(file_id, settings)
            except FilesError:
                pass  # the file expires on its own

        result = score(probes, answers)
        scores.append(result)
        out = args.work_dir / f"{stem}.arbiter.json"
        out.write_text(
            json.dumps(
                {
                    "probes": [asdict(p) for p in probes],
                    "answers": {k: asdict(v) for k, v in answers.items()},
                    "parse_errors": errors,
                    "score": result,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(
            f"{stem}: real claims judged visible {result['real_visible']} of "
            f"{result['real_answered']}; invented claims judged visible "
            f"{result['fake_claims_visible']} of {result['fake_claims_answered']}; "
            f"absent decoys judged visible {result['decoys_visible']} of "
            f"{result['decoys_answered']}; unreadable answers {len(errors)}"
        )
        print(
            f"  SAM-rejected real claims that Wonder accepted: "
            f"{result['sam_rejected_real_wonder_visible']} of "
            f"{result['sam_rejected_real_total']}"
        )
        print(
            f"  SAM-accepted invented claims that Wonder accepted: "
            f"{result['sam_accepted_fake_wonder_visible']} of "
            f"{result['sam_accepted_fake_total']}"
        )
        for line in disagreements(probes, answers):
            print("  " + line)

    total = merge_scores(scores)
    fake_total = total["fake_claims_answered"] + total["decoys_answered"]
    fake_visible = total["fake_claims_visible"] + total["decoys_visible"]
    print(
        f"POOLED real accepted: {pct(total['real_visible'], total['real_answered'])} "
        f"({total['real_visible']} of {total['real_answered']})"
    )
    print(
        f"POOLED false rejected: {pct(fake_total - fake_visible, fake_total)} "
        f"({fake_total - fake_visible} of {fake_total})"
    )
    print(
        f"POOLED invented claims rejected: "
        f"{total['fake_claims_answered'] - total['fake_claims_visible']} of "
        f"{total['fake_claims_answered']}"
    )


if __name__ == "__main__":
    main()
