import pytest

from app.pipeline.evaluate import GroundTruth
from app.pipeline.probe_report import describe_problems, frames_to_spans, sweep

TRUTHS = {
    "clip.mp4": GroundTruth.model_validate(
        {
            "clip": "clip.mp4",
            "transcript": "t",
            "items": [
                {"category": "object", "names": ["books"], "ranges": [[10, 20]]}
            ],
        }
    )
}

PROBE = {
    "clips": [
        {
            "clip": "clip.mp4",
            "items": [
                {
                    "concept": "books",
                    "expected": True,
                    "found": False,
                    "counted_frames": 1,
                    "frames": [[60, 0.9], [61, 0.4]],
                    "error": None,
                },
                {
                    "concept": "horse",
                    "expected": False,
                    "found": True,
                    "counted_frames": 2,
                    "frames": [[5, 0.95], [6, 0.95], [20, 0.95]],
                    "error": None,
                },
                {
                    "concept": "piano",
                    "expected": False,
                    "found": False,
                    "counted_frames": 0,
                    "frames": [],
                    "error": "HTTP 500",
                },
            ],
        }
    ]
}


def test_frames_to_spans_groups_consecutive_frames():
    assert frames_to_spans([1, 2, 3, 10, 11], 5) == [(0.2, 0.6), (2.0, 2.2)]
    assert frames_to_spans([], 5) == []
    assert frames_to_spans([7], 5) == [(1.4, 1.4)]


def test_sweep_changes_with_the_threshold_and_skips_errors():
    rows = sweep(PROBE, TRUTHS, fps=5, thresholds=(0.3, 0.5))
    low, high = rows
    assert (low["positives"], low["found"]) == (1, 1)  # both frames count at 0.3
    assert (high["positives"], high["found"]) == (1, 0)  # only one frame at 0.5
    assert low["negatives"] == high["negatives"] == 1  # the errored concept is skipped
    assert low["false_alarms"] == high["false_alarms"] == 1


def test_describe_problems_lists_only_wrong_decisions():
    lines = describe_problems(PROBE, TRUTHS, fps=5)
    text = "\n".join(lines)
    assert "books | MISSED" in text
    assert "horse | FALSE ALARM" in text
    assert "1.0-1.2s" in text  # frames 5 and 6 at 5 fps
    assert "piano | ERROR" in text
    assert pytest.approx(0.9) == 0.9
