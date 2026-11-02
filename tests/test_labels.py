import pytest

from app.labels import (
    LabelError,
    convert_labels,
    parse_labels,
    write_templates,
)
from app.pipeline.claims import ClaimCategory
from app.pipeline.evaluate import load_ground_truth

GOOD = """
# a comment
transcript: hola mundo
place: Santo Domingo
scene: bedroom, habitacion
person: man, hombre | 0-20
Object: books, libros, books | 3-12, 15-18
"""

BAD = """
transcript:
colour: red
object: books
object: lamp | 5-2
object: shelf | 1-100
object: chair | abc
speech
"""


def test_parses_every_kind_of_line():
    truth = parse_labels(GOOD, "blind_01", duration_s=20.0)
    assert truth.clip == "blind_01.mp4"
    assert truth.transcript == "hola mundo"
    assert [item.category for item in truth.items] == [
        ClaimCategory.PLACE,
        ClaimCategory.SCENE,
        ClaimCategory.PERSON,
        ClaimCategory.OBJECT,
    ]
    books = truth.items[3]
    assert books.names == ["books", "libros"]
    assert books.ranges == [(3.0, 12.0), (15.0, 18.0)]
    assert truth.items[0].ranges == []


def test_every_problem_is_reported_with_its_line():
    with pytest.raises(LabelError) as info:
        parse_labels(BAD, "blind_01", duration_s=20.0)
    problems = info.value.problems
    assert len(problems) == 7
    text = "\n".join(problems)
    assert "unknown category 'colour'" in text
    assert "line 4: an object needs the seconds it is visible" in text
    assert "ends before it starts" in text
    assert "goes past the end of the clip" in text
    assert "'abc' is not a time range" in text
    assert "expected 'category: names | times'" in text
    assert "transcript is missing or empty" in text


def test_a_range_may_overshoot_the_clip_by_the_margin():
    parse_labels("transcript: x\nobject: a | 0-20.9", "c", duration_s=20.0)
    with pytest.raises(LabelError):
        parse_labels("transcript: x\nobject: a | 0-21.5", "c", duration_s=20.0)


def test_only_one_transcript_and_at_least_one_item():
    with pytest.raises(LabelError, match="more than one transcript"):
        parse_labels("transcript: a\ntranscript: b\nplace: x", "c")
    with pytest.raises(LabelError, match="no labeled items"):
        parse_labels("transcript: hola", "c")


def test_places_and_scenes_need_no_times():
    truth = parse_labels("transcript: x\nplace: Santiago\nscene: kitchen", "c")
    assert [item.ranges for item in truth.items] == [[], []]


def test_templates_are_created_once_and_never_overwritten(tmp_path):
    clips = tmp_path / "clips"
    labels = tmp_path / "labels"
    clips.mkdir()
    for name in ("blind_01.mp4", "blind_02.mp4", "tune_01.mp4"):
        (clips / name).write_bytes(b"x")

    assert write_templates(clips, labels) == ("blind_01", "blind_02")
    assert (labels / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert not (labels / "tune_01.txt").exists()
    assert "transcript:" in (labels / "blind_01.txt").read_text(encoding="utf-8")

    (labels / "blind_01.txt").write_text("mine", encoding="utf-8")
    assert write_templates(clips, labels) == ()
    assert (labels / "blind_01.txt").read_text(encoding="utf-8") == "mine"


def test_an_unfilled_template_is_rejected(tmp_path):
    clips = tmp_path / "clips"
    labels = tmp_path / "labels"
    clips.mkdir()
    (clips / "blind_01.mp4").write_bytes(b"x")
    write_templates(clips, labels)
    report = convert_labels(labels, clips, tmp_path / "truth", duration_of=lambda c: 20.0)
    assert report.written == ()
    assert "transcript is missing or empty" in report.failed["blind_01"]


def test_convert_writes_skips_and_reports(tmp_path):
    clips = tmp_path / "clips"
    labels = tmp_path / "labels"
    truth_dir = tmp_path / "truth"
    clips.mkdir()
    labels.mkdir()
    (clips / "blind_01.mp4").write_bytes(b"x")
    (labels / "blind_01.txt").write_text(GOOD, encoding="utf-8")
    (labels / "blind_02.txt").write_text(GOOD, encoding="utf-8")  # no clip
    (labels / "bad name.txt").write_text(GOOD, encoding="utf-8")

    report = convert_labels(labels, clips, truth_dir, duration_of=lambda c: 20.0)
    assert report.written == ("blind_01",)
    assert "clip not found" in report.failed["blind_02"][0]
    assert "letters, digits" in report.failed["bad name"][0]
    assert load_ground_truth(truth_dir / "blind_01.json") == parse_labels(
        GOOD, "blind_01", 20.0
    )

    (labels / "blind_01.txt").write_text("transcript: changed\nplace: x", encoding="utf-8")
    again = convert_labels(labels, clips, truth_dir, duration_of=lambda c: 20.0)
    assert again.skipped == ("blind_01",)
    assert load_ground_truth(truth_dir / "blind_01.json").transcript == "hola mundo"


def test_labels_that_break_the_rules_write_nothing(tmp_path):
    clips = tmp_path / "clips"
    labels = tmp_path / "labels"
    clips.mkdir()
    labels.mkdir()
    (clips / "blind_01.mp4").write_bytes(b"x")
    (labels / "blind_01.txt").write_text(BAD, encoding="utf-8")
    report = convert_labels(labels, clips, tmp_path / "truth", duration_of=lambda c: 20.0)
    assert len(report.failed["blind_01"]) == 7
    assert not (tmp_path / "truth").exists()


@pytest.mark.parametrize(
    "marker", ["none", "None", "sin voz", "Sin diálogo", "no speech"]
)
def test_a_clip_with_no_speech_is_declared_explicitly(marker):
    truth = parse_labels(
        f"transcript: {marker}\nobject: lamp, lampara | 1-5", "c", duration_s=10.0
    )
    assert truth.transcript == ""


def test_a_blank_transcript_is_still_an_error():
    with pytest.raises(LabelError, match="transcript is missing or empty"):
        parse_labels("transcript:\nplace: x", "c")
    with pytest.raises(LabelError, match="more than one transcript"):
        parse_labels("transcript: none\ntranscript: hola\nplace: x", "c")
