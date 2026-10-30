import pytest

from app.config import Settings
from app.pipeline import sam_probe as module
from app.pipeline.evaluate import GroundTruth
from app.pipeline.files_api import FilesError
from app.pipeline.sam import SamCallError, SamResult
from app.pipeline.sam_parse import Detection
from app.pipeline.sam_probe import (
    ClipProbe,
    ProbeBudgetError,
    ProbeItem,
    decide,
    probe_clip,
    summarize,
)

FPS = 5

TRUTH = GroundTruth.model_validate(
    {
        "clip": "clip.mp4",
        "transcript": "t",
        "items": [
            {"category": "object", "names": ["books", "libros"], "ranges": [[10, 20]]},
            {"category": "object", "names": ["chair", "silla"], "start_s": 22, "end_s": 30},
            {"category": "place", "names": ["Santo Domingo"]},
        ],
    }
)


def make_detection(frame, confidence=0.9):
    return Detection(
        frame=frame,
        object_id=0,
        x1=0,
        y1=0,
        x2=1,
        y2=1,
        width=10,
        height=10,
        confidence=confidence,
    )


@pytest.fixture
def settings():
    return Settings(
        api_key="test-key",
        api_base="https://api.meta.ai/v1",
        spark_model="muse-spark-1.3",
        wonder_model="muse-wonder-1.0",
        asr_model="muse-voice-transcribe-1.0",
        sam_model="sam-3.1",
    )


@pytest.fixture
def wiring(monkeypatch):
    calls = {"upload": [], "delete": []}

    def fake_upload(path, settings, **kwargs):
        calls["upload"].append(path)
        return "file-7"

    def fake_delete(file_id, settings, **kwargs):
        calls["delete"].append(file_id)

    monkeypatch.setattr(module, "probe_duration", lambda video: 40.0)
    monkeypatch.setattr(module, "downsample_video", lambda video, out, fps=5: out)
    monkeypatch.setattr(module, "upload_video", fake_upload)
    monkeypatch.setattr(module, "delete_file", fake_delete)
    return calls


def fake_segment_factory(table, failing=()):
    def fake_segment(file_id, concept, settings, **kwargs):
        if concept in failing:
            raise SamCallError("HTTP 500")
        frames = table.get(concept, [])
        return SamResult(
            concept=concept,
            detections=tuple(make_detection(f) for f in frames),
            frames_billed=200,
        )

    return fake_segment


def test_decide_uses_ranges_confidence_and_min_frames():
    two = [make_detection(60), make_detection(61)]  # 12.0 s and 12.2 s
    assert decide(two, [(10, 20)], FPS) == (True, 2)
    assert decide(two, [(30, 40)], FPS)[0] is False
    assert decide(two, [], FPS) == (True, 2)  # no ranges: anywhere in the clip
    assert decide([make_detection(60)], [(10, 20)], FPS)[0] is False
    low = [make_detection(60, 0.2), make_detection(61, 0.2)]
    assert decide(low, [(10, 20)], FPS) == (False, 0)


def test_probe_clip_labels_hits_misses_and_false_alarms(
    tmp_path, settings, wiring, monkeypatch
):
    monkeypatch.setattr(
        module,
        "segment_concept",
        fake_segment_factory({"books": [60, 61], "horse": [5, 6]}),
    )
    # "chair" collides with a truth name, so it is not used as a negative
    result = probe_clip(
        tmp_path / "clip.mp4",
        TRUTH,
        tmp_path / "work",
        settings,
        negatives=("horse", "piano", "chair"),
    )
    labels = [(i.concept, i.expected, i.found) for i in result.items]
    assert labels == [
        ("books", True, True),
        ("chair", True, False),
        ("horse", False, True),
        ("piano", False, False),
    ]
    assert result.frames_billed == 800
    assert len(wiring["upload"]) == 1
    assert wiring["delete"] == ["file-7"]
    assert result.items[0].frames == ((60, 0.9), (61, 0.9))


def test_one_failing_concept_is_recorded_and_the_rest_continue(
    tmp_path, settings, wiring, monkeypatch
):
    monkeypatch.setattr(
        module,
        "segment_concept",
        fake_segment_factory({"books": [60, 61]}, failing=("chair",)),
    )
    result = probe_clip(
        tmp_path / "clip.mp4", TRUTH, tmp_path / "work", settings, negatives=()
    )
    by_concept = {i.concept: i for i in result.items}
    assert by_concept["chair"].error is not None
    assert by_concept["books"].found is True
    assert wiring["delete"] == ["file-7"]


def test_delete_failure_does_not_hide_results(tmp_path, settings, wiring, monkeypatch):
    def boom(file_id, settings, **kwargs):
        raise FilesError("cannot delete")

    monkeypatch.setattr(module, "delete_file", boom)
    monkeypatch.setattr(module, "segment_concept", fake_segment_factory({}))
    result = probe_clip(
        tmp_path / "clip.mp4", TRUTH, tmp_path / "work", settings, negatives=()
    )
    assert len(result.items) == 2


def test_budget_cap_aborts_before_any_upload(tmp_path, settings, wiring):
    with pytest.raises(ProbeBudgetError):
        probe_clip(
            tmp_path / "clip.mp4",
            TRUTH,
            tmp_path / "work",
            settings,
            max_cost=0.0001,
        )
    assert wiring["upload"] == []


def test_summarize_counts_rates_and_excludes_errors():
    probe_a = ClipProbe(
        clip="a.mp4",
        items=(
            ProbeItem("a", True, True, 3, ()),
            ProbeItem("b", True, False, 0, ()),
            ProbeItem("c", False, True, 4, ()),
            ProbeItem("d", False, False, 0, ()),
            ProbeItem("e", True, False, 0, (), "HTTP 500"),
        ),
        frames_billed=500,
        estimated_cost=0.1,
    )
    probe_b = ClipProbe(clip="b.mp4", items=(), frames_billed=250, estimated_cost=0.05)
    summary = summarize([probe_a, probe_b])
    assert summary["positives"] == 2
    assert summary["found"] == 1
    assert summary["sensitivity"] == pytest.approx(0.5)
    assert summary["negatives"] == 2
    assert summary["false_alarms"] == 1
    assert summary["false_alarm_rate"] == pytest.approx(0.5)
    assert summary["errors"] == 1
    assert summary["frames_billed"] == 750
