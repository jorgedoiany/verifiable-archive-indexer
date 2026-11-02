"""Search the catalogs of every processed clip: what is in the archive, and where.

It only reads files the pipeline and the review app already wrote, so it makes
no API calls and costs nothing. Claims the archivist rejected never appear;
claims that still need review appear only when asked for, and are labelled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.pipeline.claims import ClaimCategory, normalize_text
from app.pipeline.review import load_store
from app.run_data import list_stems, load_run
from app.views import ClaimView, build_views

MAX_QUERY_CHARS = 100
MAX_RESULTS = 200
TRUSTED_STATES = ("verified", "accepted", "edited")
CATEGORY_CHOICES = tuple(category.value for category in ClaimCategory)


@dataclass(frozen=True, slots=True)
class Hit:
    stem: str
    view: ClaimView


@dataclass(frozen=True, slots=True)
class SearchResult:
    hits: tuple[Hit, ...]
    total: int  # matches before the display cap
    clips_searched: int
    clips_unreadable: int
    clips_with_matches: int  # distinct clips among all matches


def matches(value: str, query: str) -> bool:
    """Every word of the query must start some word of the value (case and accents ignored)."""
    tokens = normalize_text(query).split()
    if not tokens:
        return True
    words = normalize_text(value).split()
    return all(any(word.startswith(token) for word in words) for token in tokens)


def search_catalog(
    work_dir: Path,
    query: str = "",
    *,
    category: str | None = None,
    include_pending: bool = False,
) -> SearchResult:
    if category is not None and category not in CATEGORY_CHOICES:
        raise ValueError(f"unknown category: {category!r}")
    query = query.strip()[:MAX_QUERY_CHARS]
    work_dir = Path(work_dir)
    states = (*TRUSTED_STATES, "pending") if include_pending else TRUSTED_STATES

    hits: list[Hit] = []
    searched = 0
    unreadable = 0
    for stem in list_stems(work_dir):
        try:
            run = load_run(stem, work_dir)
            store = load_store(work_dir / f"{stem}.review.json")
        except (OSError, ValueError):
            unreadable += 1
            continue
        searched += 1
        views = build_views(run.claim_set, run.transcript, store)
        for view in sorted(views, key=lambda v: v.position):
            if view.state not in states:
                continue
            if category is not None and view.category != category:
                continue
            if matches(view.value, query):
                hits.append(Hit(stem, view))
    return SearchResult(
        tuple(hits[:MAX_RESULTS]),
        len(hits),
        searched,
        unreadable,
        len({hit.stem for hit in hits}),
    )
