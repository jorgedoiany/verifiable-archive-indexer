"""Experiment: does a second model catch objects that SAM lets through?

Probes mix the object claims of a run (with an audit of which ones are real)
and decoys, concepts known to be absent, shuffled and anonymized so the judge
cannot tell them apart.
"""
from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.pipeline.arbiter import ArbiterAnswer, ArbiterItem
from app.pipeline.claims import ClaimCategory, ClaimSet, ClaimStatus, VisualEvidence

DEFAULT_DECOYS = ("horse", "piano", "umbrella", "airplane", "dog", "bicycle")
DECOY_WINDOW_S = 4.0


@dataclass(frozen=True, slots=True)
class Probe:
    item: ArbiterItem
    source: str  # "claim" or "decoy"
    claim_id: str | None
    real: bool
    sam_said_yes: bool | None  # SAM's earlier verdict; None for decoys and undecided


def _sam_said_yes(status: ClaimStatus) -> bool | None:
    if status == ClaimStatus.VERIFIED:
        return True
    if status in (ClaimStatus.UNSUPPORTED, ClaimStatus.DISPUTED):
        return False
    return None


def load_not_real(path: Path, stem: str) -> list[str]:
    """Claim ids the author audited as NOT in the clip; everything else is real."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    clips = data.get("clips") if isinstance(data, dict) else None
    if not isinstance(clips, dict) or stem not in clips:
        raise KeyError(f"no audit verdicts for {stem}")
    ids = clips[stem].get("not_real", [])
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"'not_real' for {stem} must be a list of claim ids")
    return ids


def build_probes(
    claim_set: ClaimSet,
    *,
    not_real: Sequence[str],
    decoys: Sequence[str],
    duration_s: float,
    seed: int = 7,
) -> list[Probe]:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    entries: list[tuple[str, float, float, str, str | None, bool, bool | None]] = []
    for claim in claim_set.claims:
        visual = [e for e in claim.evidence if isinstance(e, VisualEvidence)]
        if claim.category != ClaimCategory.OBJECT or not visual:
            continue
        entries.append(
            (
                claim.value,
                min(e.start_s for e in visual),
                max(e.end_s for e in visual),
                "claim",
                claim.claim_id,
                claim.claim_id not in not_real,
                _sam_said_yes(claim.status),
            )
        )
    window = min(DECOY_WINDOW_S, duration_s)
    step = (duration_s - window) / (len(decoys) - 1) if len(decoys) > 1 else 0.0
    for index, name in enumerate(decoys):
        start = index * step
        entries.append((name, start, start + window, "decoy", None, False, None))

    random.Random(seed).shuffle(entries)
    return [
        Probe(ArbiterItem(f"q{number}", concept, start, end), source, claim_id, real, sam)
        for number, (concept, start, end, source, claim_id, real, sam) in enumerate(
            entries, start=1
        )
    ]


SCORE_KEYS = (
    "real_total",
    "real_answered",
    "real_visible",
    "fake_claims_total",
    "fake_claims_answered",
    "fake_claims_visible",
    "decoys_total",
    "decoys_answered",
    "decoys_visible",
    "sam_rejected_real_total",
    "sam_rejected_real_wonder_visible",
    "sam_accepted_fake_total",
    "sam_accepted_fake_wonder_visible",
)


def score(probes: Sequence[Probe], answers: dict[str, ArbiterAnswer]) -> dict[str, int]:
    out = dict.fromkeys(SCORE_KEYS, 0)
    for probe in probes:
        answer = answers.get(probe.item.item_id)
        verdict = None if answer is None else answer.visible
        if probe.source == "decoy":
            group = "decoys"
        elif probe.real:
            group = "real"
        else:
            group = "fake_claims"
        out[f"{group}_total"] += 1
        if verdict is not None:
            out[f"{group}_answered"] += 1
            if verdict:
                out[f"{group}_visible"] += 1
        if probe.source != "claim":
            continue
        if probe.real and probe.sam_said_yes is False:
            out["sam_rejected_real_total"] += 1
            if verdict:
                out["sam_rejected_real_wonder_visible"] += 1
        if not probe.real and probe.sam_said_yes is True:
            out["sam_accepted_fake_total"] += 1
            if verdict:
                out["sam_accepted_fake_wonder_visible"] += 1
    return out


def merge_scores(scores: Sequence[dict[str, int]]) -> dict[str, int]:
    return {key: sum(s[key] for s in scores) for key in SCORE_KEYS}


def disagreements(probes: Sequence[Probe], answers: dict[str, ArbiterAnswer]) -> list[str]:
    """One line per wrong or missing judgment, to inspect them by hand."""
    lines: list[str] = []
    for probe in probes:
        answer = answers.get(probe.item.item_id)
        item = probe.item
        where = f"{probe.source} | {item.concept} | {item.start_s:.1f}-{item.end_s:.1f}s"
        if answer is None:
            lines.append(f"NO ANSWER | {where}")
        elif probe.real and not answer.visible:
            sam = {True: "yes", False: "no", None: "n/a"}[probe.sam_said_yes]
            lines.append(f"MISSED REAL | {where} | SAM {sam} | {answer.reason or ''}")
        elif not probe.real and answer.visible:
            sam = {True: "yes", False: "no", None: "n/a"}[probe.sam_said_yes]
            lines.append(f"ACCEPTED FALSE | {where} | SAM {sam} | {answer.reason or ''}")
    return lines
