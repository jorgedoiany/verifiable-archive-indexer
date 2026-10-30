"""Measure SAM 3.1 as a verifier on clips with a hand-labeled ground truth.

Sensitivity: does SAM find the objects that really are in the clip, inside
their labeled time ranges? False alarms: does SAM "find" concepts that are
absent? Per-concept frames and confidences are kept so other thresholds can be
analysed offline without new API calls.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.pipeline.claims import ClaimCategory, VisualEvidence
from app.pipeline.evaluate import GroundTruth, name_matches
from app.pipeline.files_api import FilesError, delete_file, upload_video
from app.pipeline.media import downsample_video, probe_duration
from app.pipeline.sam import SamCallError, estimate_cost, segment_concept
from app.pipeline.sam_parse import Detection, frames_present
from app.pipeline.verify import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_FRAMES,
    frames_in_ranges,
)

DEFAULT_NEGATIVES = (
    "horse",
    "piano",
    "bicycle",
    "umbrella",
    "guitar",
    "airplane",
    "dog",
    "giraffe",
)
DEFAULT_FPS = 5
DEFAULT_PROBE_MAX_COST = 1.5  # dollars per clip


class ProbeBudgetError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProbeItem:
    concept: str
    expected: bool  # True: the object is in the clip; False: it is absent
    found: bool  # SAM's decision under the production rule
    counted_frames: int
    frames: tuple[tuple[int, float | None], ...]  # (frame, best confidence)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ClipProbe:
    clip: str
    items: tuple[ProbeItem, ...]
    frames_billed: int
    estimated_cost: float


def _frame_summary(
    detections: Sequence[Detection],
) -> tuple[tuple[int, float | None], ...]:
    best: dict[int, float | None] = {}
    for detection in detections:
        if detection.frame not in best:
            best[detection.frame] = detection.confidence
            continue
        current = best[detection.frame]
        if detection.confidence is not None and (
            current is None or detection.confidence > current
        ):
            best[detection.frame] = detection.confidence
    return tuple(sorted(best.items()))


def decide(
    detections: Sequence[Detection],
    ranges: Sequence[tuple[float, float]],
    fps: float,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> tuple[bool, int]:
    """Apply the production rule: enough confident frames, inside the ranges if any."""
    present = frames_present(detections, min_confidence=min_confidence)
    if ranges:
        evidence = [VisualEvidence(start_s=s, end_s=e) for s, e in ranges]
        counted = frames_in_ranges(present, evidence, fps)
    else:
        counted = present
    return len(counted) >= min_frames, len(counted)


def probe_clip(
    video: Path,
    truth: GroundTruth,
    work_dir: Path,
    settings: Settings,
    *,
    negatives: Sequence[str] = DEFAULT_NEGATIVES,
    fps: int = DEFAULT_FPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_cost: float = DEFAULT_PROBE_MAX_COST,
) -> ClipProbe:
    video = Path(video)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    objects = [i for i in truth.items if i.category == ClaimCategory.OBJECT]
    truth_names = [name for item in truth.items for name in item.names]
    absent = [n for n in negatives if not name_matches(n, truth_names)]
    concepts: list[tuple[str, bool, list[tuple[float, float]]]] = [
        (item.names[0], True, item.time_ranges()) for item in objects
    ] + [(name, False, []) for name in absent]

    frames = math.ceil(probe_duration(video) * fps)
    estimate = estimate_cost(frames, requests=len(concepts))
    if estimate > max_cost:
        raise ProbeBudgetError(
            f"estimated SAM cost ${estimate:.2f} exceeds the ${max_cost:.2f} cap"
        )

    low_fps = downsample_video(video, work_dir / f"{video.stem}.probe.mp4", fps=fps)
    file_id = upload_video(low_fps, settings, expires_seconds=7_200)
    items: list[ProbeItem] = []
    billed = 0
    try:
        for concept, expected, ranges in concepts:
            try:
                result = segment_concept(file_id, concept, settings)
            except SamCallError as exc:
                items.append(ProbeItem(concept, expected, False, 0, (), str(exc)[:160]))
                continue
            billed += result.frames_billed or 0
            found, counted = decide(
                result.detections,
                ranges,
                fps,
                min_confidence=min_confidence,
                min_frames=min_frames,
            )
            items.append(
                ProbeItem(
                    concept,
                    expected,
                    found,
                    counted,
                    _frame_summary(result.detections),
                )
            )
    finally:
        try:
            delete_file(file_id, settings)
        except FilesError:
            pass  # the file expires on its own
    return ClipProbe(video.name, tuple(items), billed, estimate)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize(probes: Sequence[ClipProbe]) -> dict:
    """Aggregate over clips; concepts that errored are excluded from the rates."""
    valid = [i for p in probes for i in p.items if i.error is None]
    positives = [i for i in valid if i.expected]
    negatives = [i for i in valid if not i.expected]
    found = sum(i.found for i in positives)
    alarms = sum(i.found for i in negatives)
    return {
        "positives": len(positives),
        "found": found,
        "sensitivity": _ratio(found, len(positives)),
        "negatives": len(negatives),
        "false_alarms": alarms,
        "false_alarm_rate": _ratio(alarms, len(negatives)),
        "errors": sum(1 for p in probes for i in p.items if i.error is not None),
        "frames_billed": sum(p.frames_billed for p in probes),
    }
