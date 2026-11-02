"""Parse SAM 3.1 special-token output into boxes per frame.

Grammar (docs, "Read segmentation output"): one line per frame; a line is a
frame marker <Nf> followed by comma-separated records; a record is an object
id, one box token, then one mask token. Frames can skip or arrive out of
order, and object ids are sparse. Masks are not needed to verify that a
concept appears, so they are removed before parsing.
"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


class SamParseError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Detection:
    frame: int
    object_id: int
    x1: int
    y1: int
    x2: int
    y2: int
    width: int
    height: int
    confidence: float | None = None


_MASK = re.compile(r"<\|mask;.*?\|>", re.DOTALL)
_TOKEN = re.compile(
    r"<(?P<frame>\d+)f>"
    r"|(?P<oid>\d+)<\|box;x1=(?P<x1>-?\d+);y1=(?P<y1>-?\d+);"
    r"x2=(?P<x2>-?\d+);y2=(?P<y2>-?\d+);w=(?P<w>\d+);h=(?P<h>\d+)"
    r"(?:;c=(?P<c>\d+(?:\.\d+)?))?\|>"
)


def parse_sam_text(text: str) -> list[Detection]:
    """Return one Detection per box, in output order."""
    cleaned = _MASK.sub("", text)
    detections: list[Detection] = []
    frame: int | None = None
    for match in _TOKEN.finditer(cleaned):
        if match.group("frame") is not None:
            frame = int(match.group("frame"))
            continue
        if frame is None:
            raise SamParseError("box found before any frame marker")
        confidence = match.group("c")
        detections.append(
            Detection(
                frame=frame,
                object_id=int(match.group("oid")),
                x1=int(match.group("x1")),
                y1=int(match.group("y1")),
                x2=int(match.group("x2")),
                y2=int(match.group("y2")),
                width=int(match.group("w")),
                height=int(match.group("h")),
                confidence=None if confidence is None else float(confidence),
            )
        )
    return detections


def by_frame(detections: Iterable[Detection]) -> dict[int, list[Detection]]:
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.frame].append(detection)
    return dict(grouped)


def frames_present(
    detections: Iterable[Detection], *, min_confidence: float = 0.0
) -> list[int]:
    """Sorted frames with a detection; a missing confidence counts as present."""
    return sorted(
        {
            d.frame
            for d in detections
            if d.confidence is None or d.confidence >= min_confidence
        }
    )


def frame_to_seconds(frame: int, fps: float) -> float:
    if fps <= 0:
        raise ValueError("fps must be positive")
    return frame / fps
