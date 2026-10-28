import pytest

from app.pipeline.sam_parse import (
    Detection,
    SamParseError,
    by_frame,
    frame_to_seconds,
    frames_present,
    parse_sam_text,
)

# Frame 0 of the video example in the SAM documentation (two objects).
VIDEO_FRAME = r"""<0f>0<|box;x1=211;y1=228;x2=270;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,60,~!!!!M!0c[0o91w?q1!pIH4pPRVp2B3'7`e.ioeAf6-k/#Xd8%dX9x(|>,1<|box;x1=155;y1=228;x2=202;y2=254;w=320;h=334|><|mask;x=0;y=0;data=27,48,~!!!!J!0c[q=Pj_zs=*4C4(/./x#:/`S_GnD`=o3?X{emCgO$y@|>"""

WITH_CONFIDENCE = (
    "<0f>1<|box;x1=444;y1=625;x2=679;y2=850;w=1800;h=1200;c=0.9609375|>"
    "<|mask;x=0;y=0;c=0.9609375;data=226,236,!...|>"
)


def box(frame_marker, object_id, x1=1, y1=2, x2=3, y2=4, w=320, h=240):
    return (
        f"{frame_marker}{object_id}"
        f"<|box;x1={x1};y1={y1};x2={x2};y2={y2};w={w};h={h}|>"
        "<|mask;x=0;y=0;data=1,1,!a|>"
    )


def make_detection(frame, object_id=0, confidence=None):
    return Detection(
        frame=frame,
        object_id=object_id,
        x1=0,
        y1=0,
        x2=1,
        y2=1,
        width=10,
        height=10,
        confidence=confidence,
    )


def test_parses_documented_video_frame():
    detections = parse_sam_text(VIDEO_FRAME)
    assert [(d.frame, d.object_id) for d in detections] == [(0, 0), (0, 1)]
    first, second = detections
    assert (first.x1, first.y1, first.x2, first.y2) == (211, 228, 270, 254)
    assert (first.width, first.height) == (320, 334)
    assert first.confidence is None
    assert (second.x1, second.x2) == (155, 202)


def test_reads_confidence():
    (detection,) = parse_sam_text(WITH_CONFIDENCE)
    assert detection.object_id == 1
    assert detection.confidence == 0.9609375


def test_empty_output_means_no_matches():
    assert parse_sam_text("") == []
    assert parse_sam_text("  \n ") == []


def test_sparse_ids_multi_digit_and_out_of_order_frames():
    text = (
        box("<30f>", 0)
        + ","
        + box("", 12, x1=5, y1=6, x2=7, y2=8)
        + "\n"
        + box("<29f>", 0)
    )
    detections = parse_sam_text(text)
    assert [(d.frame, d.object_id) for d in detections] == [(30, 0), (30, 12), (29, 0)]
    assert detections[1].x1 == 5


def test_mask_payload_is_not_mistaken_for_markers():
    text = (
        "<0f>0<|box;x1=1;y1=2;x2=3;y2=4;w=9;h=9|>"
        "<|mask;x=0;y=0;data=1,1,!<5f>7|>"
    )
    detections = parse_sam_text(text)
    assert [(d.frame, d.object_id) for d in detections] == [(0, 0)]


def test_box_without_frame_marker_is_an_error():
    with pytest.raises(SamParseError):
        parse_sam_text(box("", 0))


def test_groups_by_frame_and_filters_by_confidence():
    detections = [
        make_detection(3, 0, 0.9),
        make_detection(1, 0, 0.3),
        make_detection(3, 1, None),
    ]
    grouped = by_frame(detections)
    assert sorted(grouped) == [1, 3]
    assert len(grouped[3]) == 2
    assert frames_present(detections) == [1, 3]
    assert frames_present(detections, min_confidence=0.5) == [3]


def test_frame_to_seconds():
    assert frame_to_seconds(30, 5.0) == 6.0
    with pytest.raises(ValueError):
        frame_to_seconds(1, 0)
