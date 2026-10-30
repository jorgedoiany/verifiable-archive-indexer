"""Offline analysis of a saved SAM probe: no API calls and no cost."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from app.pipeline.claims import ClaimCategory
from app.pipeline.evaluate import GroundTruth
from app.pipeline.sam_parse import Detection
from app.pipeline.sam_probe import decide
from app.pipeline.verify import DEFAULT_MIN_CONFIDENCE, DEFAULT_MIN_FRAMES

DEFAULT_THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def load_probe(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frames_to_spans(frames: Sequence[int], fps: float) -> list[tuple[float, float]]:
    """Group consecutive frame indices into (start, end) spans in seconds."""
    spans: list[tuple[float, float]] = []
    start: int | None = None
    previous = 0
    for frame in sorted(set(frames)):
        if start is None:
            start = previous = frame
        elif frame - previous <= 1:
            previous = frame
        else:
            spans.append((start / fps, previous / fps))
            start = previous = frame
    if start is not None:
        spans.append((start / fps, previous / fps))
    return spans


def _rebuild(frames: Sequence[Sequence[float | None]]) -> list[Detection]:
    return [
        Detection(
            frame=int(frame),
            object_id=0,
            x1=0,
            y1=0,
            x2=1,
            y2=1,
            width=1,
            height=1,
            confidence=None if conf is None else float(conf),
        )
        for frame, conf in frames
    ]


def _ranges_for(item: dict, truth: GroundTruth) -> list[tuple[float, float]]:
    if not item["expected"]:
        return []
    for entry in truth.items:
        if entry.category == ClaimCategory.OBJECT and entry.names[0] == item["concept"]:
            return entry.time_ranges()
    return []


def sweep(
    probe: dict,
    truths: dict[str, GroundTruth],
    fps: float,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> list[dict]:
    """Sensitivity and false-alarm rate at several confidence thresholds."""
    rows = []
    for threshold in thresholds:
        positives = found = negatives = alarms = 0
        for clip in probe["clips"]:
            truth = truths[clip["clip"]]
            for item in clip["items"]:
                if item.get("error"):
                    continue
                ok, _ = decide(
                    _rebuild(item["frames"]),
                    _ranges_for(item, truth),
                    fps,
                    min_confidence=threshold,
                    min_frames=min_frames,
                )
                if item["expected"]:
                    positives += 1
                    found += ok
                else:
                    negatives += 1
                    alarms += ok
        rows.append(
            {
                "threshold": threshold,
                "positives": positives,
                "found": found,
                "negatives": negatives,
                "false_alarms": alarms,
            }
        )
    return rows


def describe_problems(
    probe: dict,
    truths: dict[str, GroundTruth],
    fps: float,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> list[str]:
    """One line per miss and per false alarm, with where SAM saw anything."""
    lines = ["clip | concept | problem | frames | max confidence | spans | counted in range"]
    for clip in probe["clips"]:
        truth = truths[clip["clip"]]
        for item in clip["items"]:
            if item.get("error"):
                lines.append(f"{clip['clip']} | {item['concept']} | ERROR | | | |")
                continue
            if item["expected"] == item["found"]:
                continue
            frames = [f for f, _ in item["frames"]]
            confs = [c for _, c in item["frames"] if c is not None]
            _, counted = decide(
                _rebuild(item["frames"]),
                _ranges_for(item, truth),
                fps,
                min_confidence=min_confidence,
            )
            spans = "; ".join(f"{a:.1f}-{b:.1f}s" for a, b in frames_to_spans(frames, fps))
            lines.append(
                f"{clip['clip']} | {item['concept']} | "
                f"{'MISSED' if item['expected'] else 'FALSE ALARM'} | {len(frames)} | "
                f"{max(confs):.2f}" if confs else
                f"{clip['clip']} | {item['concept']} | "
                f"{'MISSED' if item['expected'] else 'FALSE ALARM'} | {len(frames)} | n/a"
            )
            lines[-1] += f" | {spans or '-'} | {counted}"
    return lines
