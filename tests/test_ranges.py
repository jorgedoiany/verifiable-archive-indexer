import pytest

from app.ranges import RangeNotSatisfiableError, iter_file, parse_range


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (None, None),
        ("", None),
        ("bytes=0-99", (0, 99)),
        ("bytes=900-", (900, 999)),
        ("bytes=-50", (950, 999)),
        ("bytes=-5000", (0, 999)),
        ("bytes=500-5000", (500, 999)),
        ("bytes=0-", (0, 999)),
        ("bytes=5-2", None),
        ("bytes=-", None),
        ("bytes=0-1,5-6", None),
        ("items=0-1", None),
        ("garbage", None),
    ],
)
def test_parse_range(header, expected):
    assert parse_range(header, 1000) == expected


@pytest.mark.parametrize("header", ["bytes=1000-", "bytes=2000-3000", "bytes=-0"])
def test_unsatisfiable_ranges(header):
    with pytest.raises(RangeNotSatisfiableError):
        parse_range(header, 1000)


def test_empty_file_has_no_satisfiable_suffix():
    with pytest.raises(RangeNotSatisfiableError):
        parse_range("bytes=-10", 0)


def test_iter_file_returns_the_requested_slice(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(200)))
    assert b"".join(iter_file(path, 10, 19)) == bytes(range(10, 20))
    assert b"".join(iter_file(path, 190, 500)) == bytes(range(190, 200))
