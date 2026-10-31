import json
from datetime import UTC, datetime

import pytest

from app.freeze import (
    FreezeError,
    blind_stems,
    find_outputs,
    freeze_blind,
    sha256_file,
    verify_manifest,
)

NOW = datetime(2026, 10, 31, 20, 0, tzinfo=UTC)


def make_data(tmp_path, stems=("blind_01", "blind_02")):
    data = tmp_path / "data"
    clips = data / "clips"
    truth = data / "ground_truth"
    clips.mkdir(parents=True)
    truth.mkdir(parents=True)
    for stem in stems:
        (clips / f"{stem}.mp4").write_bytes(f"clip {stem}".encode())
        (truth / f"{stem}.json").write_text(
            json.dumps({"clip": f"{stem}.mp4"}), encoding="utf-8"
        )
    (clips / "tune_01.mp4").write_bytes(b"tune")
    return data, clips, truth, tmp_path / "docs" / "blind-manifest.json"


def test_sha256_matches_the_known_value(tmp_path):
    path = tmp_path / "abc.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_only_blind_clips_are_listed(tmp_path):
    _, clips, _, _ = make_data(tmp_path)
    assert blind_stems(clips) == ["blind_01", "blind_02"]


def test_freeze_writes_a_manifest_that_verifies(tmp_path):
    data, clips, truth, manifest_path = make_data(tmp_path)
    manifest = freeze_blind(clips, truth, data, manifest_path, now=NOW)
    assert manifest["created_utc"] == "2026-10-31T20:00:00+00:00"
    assert sorted(manifest["clips"]) == ["blind_01.mp4", "blind_02.mp4"]
    assert sorted(manifest["ground_truth"]) == ["blind_01.json", "blind_02.json"]
    assert manifest["clips"]["blind_01.mp4"] == sha256_file(clips / "blind_01.mp4")
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert verify_manifest(manifest_path, clips, truth) == []


def test_find_outputs_only_reports_files_of_the_given_clips(tmp_path):
    data, _, _, _ = make_data(tmp_path)
    work = data / "work_inventory"
    work.mkdir()
    (work / "blind_01.result.json").write_text("{}", encoding="utf-8")
    (work / "tune_01.result.json").write_text("{}", encoding="utf-8")
    assert find_outputs(["blind_01", "blind_02"], data) == [
        "work_inventory/blind_01.result.json"
    ]


def test_freeze_is_refused_if_a_blind_clip_already_ran(tmp_path):
    data, clips, truth, manifest_path = make_data(tmp_path)
    work = data / "work"
    work.mkdir()
    (work / "blind_02.transcript.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FreezeError, match="were not blind"):
        freeze_blind(clips, truth, data, manifest_path, now=NOW)
    assert not manifest_path.exists()


def test_outputs_of_tuning_clips_and_the_split_record_do_not_block(tmp_path):
    data, clips, truth, manifest_path = make_data(tmp_path)
    (data / "work").mkdir()
    (data / "work" / "tune_01.result.json").write_text("{}", encoding="utf-8")
    (data / "work_eval").mkdir()
    (data / "work_eval" / "split.json").write_text("{}", encoding="utf-8")
    freeze_blind(clips, truth, data, manifest_path, now=NOW)
    assert manifest_path.exists()


def test_freeze_is_refused_when_labels_are_missing(tmp_path):
    data, clips, truth, manifest_path = make_data(tmp_path)
    (truth / "blind_02.json").unlink()
    with pytest.raises(FreezeError, match="labels missing for: blind_02"):
        freeze_blind(clips, truth, data, manifest_path, now=NOW)
    assert not manifest_path.exists()


def test_freeze_is_refused_without_blind_clips(tmp_path):
    data, clips, truth, manifest_path = make_data(tmp_path, stems=())
    with pytest.raises(FreezeError, match="no blind_"):
        freeze_blind(clips, truth, data, manifest_path, now=NOW)


def test_a_manifest_is_never_overwritten(tmp_path):
    data, clips, truth, manifest_path = make_data(tmp_path)
    freeze_blind(clips, truth, data, manifest_path, now=NOW)
    before = manifest_path.read_text(encoding="utf-8")
    with pytest.raises(FreezeError, match="already exists"):
        freeze_blind(clips, truth, data, manifest_path, now=NOW)
    assert manifest_path.read_text(encoding="utf-8") == before


def test_verify_detects_changes_missing_files_and_new_clips(tmp_path):
    data, clips, truth, manifest_path = make_data(tmp_path)
    freeze_blind(clips, truth, data, manifest_path, now=NOW)
    (clips / "blind_01.mp4").write_bytes(b"changed")
    (truth / "blind_02.json").unlink()
    (clips / "blind_03.mp4").write_bytes(b"new")
    assert verify_manifest(manifest_path, clips, truth) == [
        "changed: clips/blind_01.mp4",
        "missing: ground_truth/blind_02.json",
        "not in manifest: clips/blind_03.mp4",
    ]


def test_an_unreadable_manifest_is_reported(tmp_path):
    _, clips, truth, manifest_path = make_data(tmp_path)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")
    with pytest.raises(FreezeError, match="cannot read the manifest"):
        verify_manifest(manifest_path, clips, truth)
