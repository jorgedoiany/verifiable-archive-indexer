import subprocess
from pathlib import Path

import pytest

from app.pipeline import media
from app.pipeline.media import MediaError, extract_audio, strip_audio


@pytest.fixture
def calls(monkeypatch):
    recorded = []
    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(args, **kwargs):
        recorded.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    return recorded


def test_extract_audio_command(calls, tmp_path):
    out = extract_audio(Path("in.mp4"), tmp_path / "work" / "a.wav")
    args = calls[0]
    assert args[0] == "/usr/bin/ffmpeg"
    assert "-vn" in args
    assert args[args.index("-ar") + 1] == "24000"
    assert args[args.index("-c:a") + 1] == "pcm_s16le"
    assert args[-1] == str(out)
    assert out.parent.exists()


def test_strip_audio_command(calls, tmp_path):
    out = strip_audio(Path("in.mov"), tmp_path / "work" / "v.mp4")
    args = calls[0]
    assert "-an" in args
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[-1] == str(out)


def test_missing_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(MediaError):
        extract_audio(Path("in.mp4"), tmp_path / "a.wav")


def test_ffmpeg_failure_is_wrapped(monkeypatch, tmp_path):
    monkeypatch.setattr(media.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def boom(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="boom")

    monkeypatch.setattr(media.subprocess, "run", boom)
    with pytest.raises(MediaError):
        strip_audio(Path("in.mp4"), tmp_path / "v.mp4")
