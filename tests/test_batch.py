import json

import pytest

from app.batch import (
    BatchReportError,
    check_blind_frozen,
    load_report,
    run_batch,
    select_stems,
)
from app.config import Settings
from app.freeze import FreezeError, freeze_blind
from app.pipeline.claims import Claim, ClaimSet
from app.pipeline.models import parse_transcript
from app.pipeline.run import PipelineResult
from app.pipeline.verify import VerifyPlan
from app.pipeline.verify_stage import VerifyOutcome

TRANSCRIPT = parse_transcript(
    {"transcript": "", "audioDurationMs": 1000, "turns": []}
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


def make_claim(claim_id):
    return Claim.model_validate(
        {
            "claim_id": claim_id,
            "category": "object",
            "value": "bicycle",
            "evidence": [{"kind": "visual", "start_s": 1.0, "end_s": 2.0}],
        }
    )


def make_result(work_dir, stem, cost):
    claim_set = ClaimSet(claims=[make_claim("a"), make_claim("b")])
    plan = VerifyPlan(concepts=("bicycle",), capped_claim_ids=(), estimated_cost=cost)
    outcome = VerifyOutcome(
        claim_set, plan, 200, cost, {}, work_dir / f"{stem}.verified.json"
    )
    return PipelineResult(
        TRANSCRIPT, ["skipped"], outcome, {"verified": 2}, work_dir / f"{stem}.result.json"
    )


def make_runner(calls, costs=None, fail=()):
    def runner(clip, work_dir, settings, **kwargs):
        calls.append((clip.stem, kwargs))
        if clip.stem in fail:
            raise RuntimeError("boom")
        result = make_result(work_dir, clip.stem, (costs or {}).get(clip.stem, 0.3))
        result.summary_path.write_text("{}", encoding="utf-8")
        return result

    return runner


@pytest.fixture
def clips(tmp_path):
    folder = tmp_path / "clips"
    folder.mkdir()
    for name in ("tune_01", "tune_02", "tune_03", "blind_01"):
        (folder / f"{name}.mp4").write_bytes(name.encode())
    return folder


def test_select_stems_by_group(clips):
    assert select_stems(clips, "tune") == ["tune_01", "tune_02", "tune_03"]
    assert select_stems(clips, "blind") == ["blind_01"]
    assert select_stems(clips, "all") == ["blind_01", "tune_01", "tune_02", "tune_03"]
    with pytest.raises(ValueError):
        select_stems(clips, "everything")


def test_runs_in_order_passes_the_options_and_writes_a_report(tmp_path, settings, clips):
    calls = []
    work = tmp_path / "work"
    entries = run_batch(
        ["tune_01", "tune_02"], clips, work, settings,
        prompt_variant="basic", runner=make_runner(calls),
    )
    assert [(e.stem, e.status) for e in entries] == [
        ("tune_01", "done"), ("tune_02", "done"),
    ]
    assert [stem for stem, _ in calls] == ["tune_01", "tune_02"]
    assert calls[0][1] == {
        "fps": 5, "max_concepts": 25, "max_cost": 2.0, "prompt_variant": "basic",
    }
    first = entries[0]
    assert (first.claims_total, first.parse_errors, first.frames_billed) == (2, 1, 200)
    assert first.actual_cost == pytest.approx(0.3)

    report = json.loads((work / "batch_report.json").read_text(encoding="utf-8"))
    assert report["prompt_variant"] == "basic"
    assert sorted(report["entries"]) == ["tune_01", "tune_02"]
    assert report["entries"]["tune_01"]["status_counts"] == {"verified": 2}


def test_a_clip_with_a_result_is_not_run_again_and_keeps_its_report_entry(
    tmp_path, settings, clips
):
    work = tmp_path / "work"
    first_calls = []
    run_batch(["tune_01"], clips, work, settings, prompt_variant="basic",
              runner=make_runner(first_calls))
    again_calls = []
    entries = run_batch(["tune_01", "tune_02"], clips, work, settings,
                        prompt_variant="basic", runner=make_runner(again_calls))
    assert [(e.stem, e.status) for e in entries] == [
        ("tune_01", "skipped_existing"), ("tune_02", "done"),
    ]
    assert [stem for stem, _ in again_calls] == ["tune_02"]
    report = json.loads((work / "batch_report.json").read_text(encoding="utf-8"))
    assert report["entries"]["tune_01"]["status"] == "done"
    assert report["entries"]["tune_02"]["status"] == "done"


def test_a_failing_clip_does_not_stop_the_batch(tmp_path, settings, clips):
    calls = []
    entries = run_batch(
        ["tune_01", "tune_02", "tune_03"], clips, tmp_path / "work", settings,
        prompt_variant="basic", runner=make_runner(calls, fail=("tune_02",)),
    )
    assert [e.status for e in entries] == ["done", "failed", "done"]
    assert entries[1].error == "RuntimeError: boom"
    assert len(calls) == 3


def test_a_failed_clip_can_be_retried_because_it_has_no_result(tmp_path, settings, clips):
    work = tmp_path / "work"
    run_batch(["tune_01"], clips, work, settings, prompt_variant="basic",
              runner=make_runner([], fail=("tune_01",)))
    calls = []
    entries = run_batch(["tune_01"], clips, work, settings, prompt_variant="basic",
                        runner=make_runner(calls))
    assert entries[0].status == "done"
    assert len(calls) == 1
    report = json.loads((work / "batch_report.json").read_text(encoding="utf-8"))
    assert report["entries"]["tune_01"]["status"] == "done"


def test_the_total_cap_stops_later_clips(tmp_path, settings, clips):
    calls = []
    entries = run_batch(
        ["tune_01", "tune_02", "tune_03"], clips, tmp_path / "work", settings,
        prompt_variant="basic", max_total_cost=1.0,
        runner=make_runner(calls, costs={"tune_01": 0.6, "tune_02": 0.6}),
    )
    assert [e.status for e in entries] == ["done", "done", "not_run_budget"]
    assert len(calls) == 2


def test_a_missing_clip_is_reported_and_skipped(tmp_path, settings, clips):
    calls = []
    entries = run_batch(
        ["nope_01", "tune_01"], clips, tmp_path / "work", settings,
        prompt_variant="basic", runner=make_runner(calls),
    )
    assert [e.status for e in entries] == ["missing", "done"]
    assert "clip not found" in entries[0].error


def test_a_work_folder_is_never_shared_between_prompt_variants(tmp_path, settings, clips):
    work = tmp_path / "work"
    run_batch(["tune_01"], clips, work, settings, prompt_variant="basic",
              runner=make_runner([]))
    calls = []
    with pytest.raises(BatchReportError, match="separate folder"):
        run_batch(["tune_02"], clips, work, settings, prompt_variant="inventory",
                  runner=make_runner(calls))
    assert calls == []


def test_an_unreadable_report_is_an_error(tmp_path):
    path = tmp_path / "batch_report.json"
    assert load_report(path) is None
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(BatchReportError):
        load_report(path)


def make_frozen_blind_set(tmp_path):
    data = tmp_path / "data"
    clips_dir = data / "clips"
    truth = data / "ground_truth"
    clips_dir.mkdir(parents=True)
    truth.mkdir(parents=True)
    (clips_dir / "blind_01.mp4").write_bytes(b"blind")
    (truth / "blind_01.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    freeze_blind(clips_dir, truth, data, manifest)
    return clips_dir, truth, manifest


def test_blind_clips_need_an_intact_frozen_manifest(tmp_path):
    clips_dir, truth, manifest = make_frozen_blind_set(tmp_path)
    check_blind_frozen(["blind_01"], manifest, clips_dir, truth)  # intact: no error
    (clips_dir / "blind_01.mp4").write_bytes(b"changed")
    with pytest.raises(FreezeError, match="changed since it was frozen"):
        check_blind_frozen(["blind_01"], manifest, clips_dir, truth)


def test_blind_clips_are_refused_without_a_manifest_but_tuning_clips_are_not(tmp_path):
    clips_dir, truth, _ = make_frozen_blind_set(tmp_path)
    missing = tmp_path / "nope.json"
    with pytest.raises(FreezeError, match="cannot run blind clips"):
        check_blind_frozen(["blind_01"], missing, clips_dir, truth)
    check_blind_frozen(["tune_01"], missing, clips_dir, truth)  # no check needed
