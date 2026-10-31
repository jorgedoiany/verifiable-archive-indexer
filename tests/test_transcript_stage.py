import json

from app.config import Settings
from app.pipeline import transcript_stage as stage
from app.pipeline.models import parse_transcript

SAMPLE = {
    "sessionId": "abc",
    "transcript": "Hola mundo.",
    "audioDurationMs": 1000,
    "turns": [
        {
            "turnId": 1,
            "startMs": 0,
            "endMs": 900,
            "transcript": "Hola mundo.",
            "speaker": "A",
        }
    ],
}


def make_settings():
    return Settings(
        api_key="k",
        api_base="https://api.meta.ai/v1",
        spark_model="muse-spark-1.3",
        wonder_model="muse-wonder-1.0",
        asr_model="muse-voice-transcribe-1.0",
        sam_model="sam-3.1",
    )


def test_stage_writes_round_trippable_json(tmp_path, monkeypatch):
    seen = {}

    def fake_extract(video, out_wav):
        seen["wav"] = out_wav
        return out_wav

    def fake_transcribe(wav, settings, keywords=()):
        seen["keywords"] = tuple(keywords)
        return parse_transcript(SAMPLE)

    monkeypatch.setattr(stage, "has_audio", lambda video: True)
    monkeypatch.setattr(stage, "extract_audio", fake_extract)
    monkeypatch.setattr(stage, "transcribe_file", fake_transcribe)

    work = tmp_path / "work"
    transcript, out = stage.run_transcript_stage(
        tmp_path / "clip.mp4", work, make_settings(), keywords=["Santo Domingo"]
    )

    assert out == work / "clip.transcript.json"
    assert seen["wav"] == work / "clip.wav"
    assert seen["keywords"] == ("Santo Domingo",)
    assert parse_transcript(json.loads(out.read_text(encoding="utf-8"))) == transcript


def test_clip_without_an_audio_track_skips_the_api(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the API must not be called")

    monkeypatch.setattr(stage, "has_audio", lambda video: False)
    monkeypatch.setattr(stage, "probe_duration", lambda video: 12.5)
    monkeypatch.setattr(stage, "extract_audio", explode)
    monkeypatch.setattr(stage, "transcribe_file", explode)

    transcript, out = stage.run_transcript_stage(
        tmp_path / "clip.mp4", tmp_path / "work", make_settings()
    )
    assert transcript.turns == []
    assert transcript.transcript == ""
    assert transcript.audio_duration_ms == 12_500
    assert parse_transcript(json.loads(out.read_text(encoding="utf-8"))) == transcript
