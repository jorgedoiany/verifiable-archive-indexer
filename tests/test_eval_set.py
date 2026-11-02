import json

import pytest

from app.eval_set import build_eval_set, split_pool, target_name

NAMES = [f"p_{i:03d}.mp4" for i in range(1, 16)]


def make_pool(tmp_path, count=15):
    pool = tmp_path / "pool"
    pool.mkdir()
    for number in range(1, count + 1):
        (pool / f"p_{number:03d}.mp4").write_bytes(f"clip {number}".encode())
    (pool / "_map.csv").write_text("private", encoding="utf-8")
    (pool / "notes.txt").write_text("x", encoding="utf-8")
    return pool


def test_split_has_the_right_sizes_and_is_disjoint_and_complete():
    split = split_pool(NAMES, 5, 10, seed=1)
    assert len(split.tune) == 5
    assert len(split.blind) == 10
    assert set(split.tune).isdisjoint(split.blind)
    assert set(split.tune) | set(split.blind) == set(NAMES)


def test_split_is_reproducible_and_independent_of_input_order():
    first = split_pool(NAMES, 5, 10, seed=7)
    assert split_pool(list(reversed(NAMES)), 5, 10, seed=7) == first
    assert split_pool(NAMES, 5, 10, seed=8) != first


def test_split_needs_enough_clips_and_sane_counts():
    with pytest.raises(ValueError):
        split_pool(NAMES[:3], 2, 2, seed=1)
    with pytest.raises(ValueError):
        split_pool(NAMES, -1, 5, seed=1)


def test_target_name():
    assert target_name("tune", 1) == "tune_01.mp4"
    assert target_name("blind", 10) == "blind_10.mp4"


def test_build_copies_only_pool_clips_and_records_the_split(tmp_path):
    pool = make_pool(tmp_path)
    clips = tmp_path / "clips"
    record = tmp_path / "work_eval" / "split.json"
    build_eval_set(pool, clips, record, tune=5, blind=10, seed=20261031)

    expected = sorted(
        [target_name("tune", i) for i in range(1, 6)]
        + [target_name("blind", i) for i in range(1, 11)]
    )
    assert sorted(p.name for p in clips.iterdir()) == expected

    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["seed"] == 20261031
    assert len(data["tune"]) == 5
    assert len(data["blind"]) == 10
    source = data["blind"]["blind_01.mp4"]
    assert (clips / "blind_01.mp4").read_bytes() == (pool / source).read_bytes()


def test_the_split_is_never_redrawn(tmp_path):
    pool = make_pool(tmp_path)
    clips = tmp_path / "clips"
    record = tmp_path / "work_eval" / "split.json"
    build_eval_set(pool, clips, record, tune=5, blind=10, seed=1)
    before = record.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_eval_set(pool, clips, record, tune=5, blind=10, seed=2)
    assert record.read_text(encoding="utf-8") == before


def test_existing_clips_are_never_overwritten(tmp_path):
    pool = make_pool(tmp_path)
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "tune_01.mp4").write_bytes(b"mine")
    record = tmp_path / "work_eval" / "split.json"
    with pytest.raises(FileExistsError):
        build_eval_set(pool, clips, record, tune=5, blind=10, seed=1)
    assert (clips / "tune_01.mp4").read_bytes() == b"mine"
    assert [p.name for p in clips.iterdir()] == ["tune_01.mp4"]
    assert not record.exists()


def test_too_few_clips_copies_nothing(tmp_path):
    pool = make_pool(tmp_path, count=3)
    clips = tmp_path / "clips"
    record = tmp_path / "work_eval" / "split.json"
    with pytest.raises(ValueError):
        build_eval_set(pool, clips, record, tune=5, blind=10, seed=1)
    assert not clips.exists()
    assert not record.exists()
