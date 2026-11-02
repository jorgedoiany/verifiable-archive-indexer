"""Blind judging sheet: the author judges object claims without seeing any verdict.

The sheet lists the object claims of a run, shuffled and anonymized (j001, j002...).
It keeps only the private key needed to find each claim again. The page shows the
clip, the object and the time range the model cited, nothing else. Judgments are
stored apart from the pipeline output.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from app.pipeline.claims import Claim, ClaimCategory, ClaimSet, VisualEvidence
from app.run_data import is_valid_stem

JUDGING_DIR = "judging"
SHEET_NAME = "sheet.json"
JUDGMENTS_NAME = "judgments.json"
NOTE_MAX_CHARS = 200


class JudgingDataError(RuntimeError):
    pass


class Verdict(str, Enum):
    EXISTS = "exists"  # clearly visible at some moment inside the cited range
    NOT_EXISTS = "not_exists"  # not visible in that range (absent, or only elsewhere)
    UNSURE = "unsure"


class SheetItem(BaseModel):
    item_id: str
    stem: str
    claim_id: str


class Sheet(BaseModel):
    seed: int
    items: list[SheetItem]


class Judgment(BaseModel):
    verdict: Verdict
    note: str | None = Field(default=None, max_length=NOTE_MAX_CHARS)
    judged_at: str


class JudgmentStore(BaseModel):
    judgments: dict[str, Judgment] = Field(default_factory=dict)


class _Verified(BaseModel):
    claims: list[Claim]


@dataclass(frozen=True, slots=True)
class ItemView:
    item_id: str
    position: int
    total: int
    stem: str
    value: str
    start_s: float | None
    end_s: float | None
    verdict: str | None
    note: str | None
    prev_id: str | None
    next_id: str | None


def judging_dir(work_dir: Path) -> Path:
    return Path(work_dir) / JUDGING_DIR


def sheet_path(work_dir: Path) -> Path:
    return judging_dir(work_dir) / SHEET_NAME


def judgments_path(work_dir: Path) -> Path:
    return judging_dir(work_dir) / JUDGMENTS_NAME


def _atomic_write(path: Path, text: str) -> None:
    """Write atomically, and keep Git away from the folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    guard = path.parent / ".gitignore"
    if not guard.exists():
        guard.write_text("*\n", encoding="utf-8")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def load_claims(work_dir: Path, stem: str) -> ClaimSet:
    if not is_valid_stem(stem):
        raise FileNotFoundError(f"invalid clip name: {stem!r}")
    path = Path(work_dir) / f"{stem}.verified.json"
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    report = _Verified.model_validate_json(path.read_text(encoding="utf-8"))
    return ClaimSet(claims=report.claims)


def cited_range(claim: Claim) -> tuple[float, float] | None:
    """The span of the visual evidence the model cited, or None without any."""
    visual = [e for e in claim.evidence if isinstance(e, VisualEvidence)]
    if not visual:
        return None
    return min(e.start_s for e in visual), max(e.end_s for e in visual)


def build_sheet(work_dir: Path, *, seed: int) -> Sheet:
    """Shuffle every object claim of every run in the folder into one anonymous sheet."""
    work_dir = Path(work_dir)
    path = sheet_path(work_dir)
    if path.exists():
        raise FileExistsError(
            f"the sheet already exists ({path}); delete it by hand only if nothing "
            "has been judged yet"
        )
    entries: list[tuple[str, str]] = []
    for verified in sorted(work_dir.glob("*.verified.json")):
        stem = verified.name.removesuffix(".verified.json")
        if not is_valid_stem(stem):
            continue
        entries.extend(
            (stem, claim.claim_id)
            for claim in load_claims(work_dir, stem).claims
            if claim.category == ClaimCategory.OBJECT
        )
    if not entries:
        raise ValueError(f"no object claims found in {work_dir}")
    random.Random(seed).shuffle(entries)
    items = [
        SheetItem(item_id=f"j{number:03d}", stem=stem, claim_id=claim_id)
        for number, (stem, claim_id) in enumerate(entries, start=1)
    ]
    sheet = Sheet(seed=seed, items=items)
    _atomic_write(path, sheet.model_dump_json(indent=2))
    return sheet


def load_sheet(work_dir: Path) -> Sheet:
    path = sheet_path(work_dir)
    if not path.exists():
        raise FileNotFoundError(f"no judging sheet yet: {path}")
    return Sheet.model_validate_json(path.read_text(encoding="utf-8"))


def load_judgments(work_dir: Path) -> JudgmentStore:
    path = judgments_path(work_dir)
    if not path.exists():
        return JudgmentStore()
    return JudgmentStore.model_validate_json(path.read_text(encoding="utf-8"))


def save_judgments(work_dir: Path, store: JudgmentStore) -> None:
    _atomic_write(judgments_path(work_dir), store.model_dump_json(indent=2))


def record_judgment(
    store: JudgmentStore,
    sheet: Sheet,
    item_id: str,
    verdict: Verdict,
    *,
    note: str | None = None,
    now: datetime | None = None,
) -> JudgmentStore:
    """Return a new store with the judgment recorded; the old one is unchanged."""
    if item_id not in {item.item_id for item in sheet.items}:
        raise KeyError(f"unknown item: {item_id}")
    stamp = (now or datetime.now(UTC)).isoformat()
    entry = Judgment(
        verdict=verdict, note=(note or "").strip() or None, judged_at=stamp
    )
    return JudgmentStore(judgments={**store.judgments, item_id: entry})


def next_unjudged(
    sheet: Sheet, store: JudgmentStore, after: str | None = None
) -> SheetItem | None:
    """The first unjudged item after `after` in sheet order, wrapping around."""
    items = sheet.items
    start = 0
    if after is not None:
        for index, item in enumerate(items):
            if item.item_id == after:
                start = index + 1
                break
    for item in [*items[start:], *items[:start]]:
        if item.item_id not in store.judgments:
            return item
    return None


def progress(sheet: Sheet, store: JudgmentStore) -> tuple[int, int]:
    ids = {item.item_id for item in sheet.items}
    return len(ids & set(store.judgments)), len(ids)


def build_item_view(
    work_dir: Path, sheet: Sheet, store: JudgmentStore, item_id: str
) -> ItemView:
    items = sheet.items
    index = next((i for i, item in enumerate(items) if item.item_id == item_id), None)
    if index is None:
        raise KeyError(f"unknown item: {item_id}")
    item = items[index]
    claim = next(
        (c for c in load_claims(work_dir, item.stem).claims if c.claim_id == item.claim_id),
        None,
    )
    if claim is None:
        raise JudgingDataError(
            f"claim {item.claim_id} is no longer in the run of {item.stem}"
        )
    span = cited_range(claim)
    judgment = store.judgments.get(item_id)
    return ItemView(
        item_id=item_id,
        position=index + 1,
        total=len(items),
        stem=item.stem,
        value=claim.value,
        start_s=None if span is None else span[0],
        end_s=None if span is None else span[1],
        verdict=None if judgment is None else judgment.verdict.value,
        note=None if judgment is None else judgment.note,
        prev_id=items[index - 1].item_id if index > 0 else None,
        next_id=items[index + 1].item_id if index + 1 < len(items) else None,
    )
