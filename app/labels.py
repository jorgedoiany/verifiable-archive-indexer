"""Plain-text labels for clips, converted to the ground-truth JSON.

Format: one line per item; blank lines and lines starting with # are ignored.

    transcript: what is said, word for word
    place: Santo Domingo
    scene: bedroom, habitacion
    person: man, hombre | 0-20
    object: books, libros | 3-12, 15-18
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.pipeline.claims import ClaimCategory
from app.pipeline.evaluate import GroundTruth, TruthItem
from app.pipeline.media import MediaError, probe_duration
from app.run_data import is_valid_stem

RANGE_MARGIN_S = 1.0
NO_SPEECH = frozenset(
    {"none", "no speech", "sin voz", "sin dialogo", "sin diálogo"}
)
_NUMBER = r"\d+(?:\.\d+)?"
_RANGE = re.compile(rf"({_NUMBER})\s*-\s*({_NUMBER})")

TEMPLATE = """# Labels for {stem}.mp4  (lines starting with # are ignored)
#
# Fill this in by watching the clip, BEFORE running anything on it.
#
# transcript: write what is said, word for word, as you hear it.
#   Keep colloquial words. Write numbers as words. Punctuation does not matter.
#   If nobody speaks in the clip, write exactly: transcript: none
# place / scene / topic / person: names separated by commas.
# object: names in English AND Spanish, then a bar and the seconds when it is
#   visible on screen, one range per appearance (give 1 second of margin).
#
# Example (made up):
#   transcript: aqui estoy en mi sala mostrando mi guitarra
#   place: Santo Domingo
#   scene: living room, sala
#   person: man, hombre | 0-20
#   object: guitar, guitarra | 6-14, 17-19
#
# Delete this example and write your lines below.

transcript:
"""


class LabelError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True, slots=True)
class ConvertReport:
    written: tuple[str, ...]
    skipped: tuple[str, ...]
    failed: dict[str, list[str]]


def _parse_ranges(
    text: str, number: int, duration: float | None, problems: list[str]
) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = _RANGE.fullmatch(part)
        if match is None:
            problems.append(f"line {number}: '{part}' is not a time range like 3-12")
            continue
        start, end = float(match.group(1)), float(match.group(2))
        if end < start:
            problems.append(f"line {number}: range {part} ends before it starts")
        elif duration is not None and end > duration + RANGE_MARGIN_S:
            problems.append(
                f"line {number}: range {part} goes past the end of the clip "
                f"({duration:.1f} s)"
            )
        else:
            ranges.append((start, end))
    return ranges


def parse_labels(text: str, stem: str, duration_s: float | None = None) -> GroundTruth:
    """Parse a label file; raise LabelError listing every problem found."""
    problems: list[str] = []
    transcript: str | None = None
    no_speech = False
    items: list[TruthItem] = []
    categories = {category.value for category in ClaimCategory}

    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, rest = line.partition(":")
        key = key.strip().lower()
        if not separator:
            problems.append(f"line {number}: expected 'category: names | times'")
            continue
        if key == "transcript":
            if transcript is not None:
                problems.append(f"line {number}: more than one transcript line")
            value = rest.strip()
            if value.casefold() in NO_SPEECH:
                no_speech = True
                transcript = ""
            else:
                transcript = value
            continue
        if key not in categories:
            problems.append(
                f"line {number}: unknown category '{key}' "
                "(use person, place, topic, scene, object or transcript)"
            )
            continue

        before = len(problems)
        names_part, bar, ranges_part = rest.partition("|")
        names = list(dict.fromkeys(n.strip() for n in names_part.split(",") if n.strip()))
        if not names:
            problems.append(f"line {number}: no names given")
            continue
        ranges = _parse_ranges(ranges_part, number, duration_s, problems) if bar else []
        if key == "object" and not ranges_part.strip():
            problems.append(f"line {number}: an object needs the seconds it is visible")
        if len(problems) == before:
            items.append(TruthItem(category=ClaimCategory(key), names=names, ranges=ranges))

    if transcript is None or (not transcript.strip() and not no_speech):
        problems.append("transcript is missing or empty")
    if not items and not problems:
        problems.append("no labeled items")
    if problems:
        raise LabelError(problems)
    return GroundTruth(clip=f"{stem}.mp4", transcript=transcript or "", items=items)


def write_templates(clips_dir: Path, labels_dir: Path, prefix: str = "blind_") -> tuple[str, ...]:
    """Create a blank label file for each matching clip; never overwrite one."""
    clips_dir = Path(clips_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    guard = labels_dir / ".gitignore"
    if not guard.exists():
        guard.write_text("*\n", encoding="utf-8")
    created: list[str] = []
    for clip in sorted(clips_dir.glob(f"{prefix}*.mp4")):
        target = labels_dir / f"{clip.stem}.txt"
        if target.exists():
            continue
        target.write_text(TEMPLATE.format(stem=clip.stem), encoding="utf-8")
        created.append(clip.stem)
    return tuple(created)


def convert_labels(
    labels_dir: Path,
    clips_dir: Path,
    truth_dir: Path,
    *,
    duration_of: Callable[[Path], float] = probe_duration,
) -> ConvertReport:
    """Turn every label file into a ground-truth JSON; never overwrite one.

    An existing ground-truth file counts as frozen: it is skipped. To change one
    before anything has been run on the clip, delete it by hand first.
    """
    labels_dir = Path(labels_dir)
    clips_dir = Path(clips_dir)
    truth_dir = Path(truth_dir)
    written: list[str] = []
    skipped: list[str] = []
    failed: dict[str, list[str]] = {}

    for path in sorted(labels_dir.glob("*.txt")):
        stem = path.stem
        if not is_valid_stem(stem):
            failed[stem] = ["file name may only use letters, digits, - and _"]
            continue
        clip = clips_dir / f"{stem}.mp4"
        if not clip.exists():
            failed[stem] = [f"clip not found: {clip.name}"]
            continue
        target = truth_dir / f"{stem}.json"
        if target.exists():
            skipped.append(stem)
            continue
        try:
            truth = parse_labels(path.read_text(encoding="utf-8"), stem, duration_of(clip))
        except LabelError as exc:
            failed[stem] = exc.problems
            continue
        except MediaError as exc:
            failed[stem] = [str(exc)]
            continue
        truth_dir.mkdir(parents=True, exist_ok=True)
        target.write_text(truth.model_dump_json(indent=2), encoding="utf-8")
        written.append(stem)
    return ConvertReport(tuple(written), tuple(skipped), failed)
