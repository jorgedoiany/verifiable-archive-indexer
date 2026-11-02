"""Summarize human judgments against the automatic statuses of the claims."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from app.judging import (
    JudgingDataError,
    Verdict,
    load_claims,
    load_judgments,
    load_sheet,
)
from app.pipeline.claims import ClaimStatus

GROUP_NAMES = ("checks_agree", "automatic_miss", "unchecked")
_GROUP_OF_STATUS = {
    ClaimStatus.VERIFIED: "checks_agree",
    ClaimStatus.UNSUPPORTED: "automatic_miss",
    ClaimStatus.DISPUTED: "automatic_miss",
    ClaimStatus.PROPOSED: "unchecked",
}


def wilson_interval(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float] | None:
    """Wilson score interval for a proportion; None when there is nothing to count."""
    if total <= 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


ATTRIBUTE_PREFIX = "attr:"


def is_attribute_note(note: str | None) -> bool:
    """True if a judgment note starts with the attribute marker."""
    return bool(note) and note.strip().casefold().startswith(ATTRIBUTE_PREFIX)


@dataclass(frozen=True, slots=True)
class GroupStats:
    judged: int = 0
    exists: int = 0
    not_exists: int = 0
    unsure: int = 0
    attribute_notes: int = 0  # visible, but a detail such as the color was wrong

    @property
    def decided(self) -> int:
        return self.exists + self.not_exists

    @property
    def share_existing(self) -> float | None:
        return None if self.decided == 0 else self.exists / self.decided

    @property
    def share_counting_unsure_as_missing(self) -> float | None:
        return None if self.judged == 0 else self.exists / self.judged

    def interval(self) -> tuple[float, float] | None:
        return wilson_interval(self.exists, self.decided)


@dataclass(frozen=True, slots=True)
class JudgingSummary:
    items_total: int
    items_judged: int
    groups: dict[str, GroupStats]
    attribute_notes_elsewhere: int = 0


def _stats(verdicts: Iterable[Verdict], attribute_notes: int = 0) -> GroupStats:
    counts = Counter(verdicts)
    return GroupStats(
        judged=sum(counts.values()),
        exists=counts[Verdict.EXISTS],
        not_exists=counts[Verdict.NOT_EXISTS],
        unsure=counts[Verdict.UNSURE],
        attribute_notes=attribute_notes,
    )


def summarize_judgments(work_dir: Path) -> JudgingSummary:
    sheet = load_sheet(work_dir)
    store = load_judgments(work_dir)
    cache = {}
    verdicts = {name: [] for name in GROUP_NAMES}
    judged_items = 0
    attribute_notes = dict.fromkeys(GROUP_NAMES, 0)
    elsewhere = 0
    for item in sheet.items:
        judgment = store.judgments.get(item.item_id)
        if judgment is None:
            continue
        if item.stem not in cache:
            cache[item.stem] = {
                claim.claim_id: claim for claim in load_claims(work_dir, item.stem).claims
            }
        claim = cache[item.stem].get(item.claim_id)
        if claim is None:
            raise JudgingDataError(
                f"claim {item.claim_id} is no longer in the run of {item.stem}"
            )
        group_name = _GROUP_OF_STATUS[claim.status]
        verdicts[group_name].append(judgment.verdict)
        if is_attribute_note(judgment.note):
            if judgment.verdict == Verdict.EXISTS:
                attribute_notes[group_name] += 1
            else:
                elsewhere += 1
        judged_items += 1
    everything = [verdict for group in verdicts.values() for verdict in group]
    groups = {"all": _stats(everything, sum(attribute_notes.values()))}
    groups.update(
        {name: _stats(group, attribute_notes[name]) for name, group in verdicts.items()}
    )
    return JudgingSummary(len(sheet.items), judged_items, groups, elsewhere)


def summary_to_dict(summary: JudgingSummary) -> dict:
    groups = {}
    for name, stats in summary.groups.items():
        interval = stats.interval()
        groups[name] = {
            "judged": stats.judged,
            "exists": stats.exists,
            "not_exists": stats.not_exists,
            "unsure": stats.unsure,
            "attribute_notes": stats.attribute_notes,
            "share_existing": stats.share_existing,
            "interval_95": None if interval is None else [interval[0], interval[1]],
            "share_counting_unsure_as_missing": stats.share_counting_unsure_as_missing,
        }
    return {
        "items_total": summary.items_total,
        "items_judged": summary.items_judged,
        "attribute_notes_elsewhere": summary.attribute_notes_elsewhere,
        "groups": groups,
    }
