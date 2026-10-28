import subprocess
from pathlib import Path

import pytest

from app.pipeline import media
from app.pipeline.media import MediaError, downsample_video, probe_duration


@pytest.fixture
def tools_present(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: f"/usr/bin/{name}")


def stub_run(monkeypatch, stdout=""):
    recorded = []

    def fake_run(args, **kwargs):
        recorded.append(args)
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(media.subprocess, "run", fake_run)
    return recorded


def test_downsample_command(tools_present, monkeypatch, tmp_path):
    recorded = stub_run(monkeypatch)
    out = downsample_video(Path("in.mp4"), tmp_path / "work" / "v.mp4", fps=5)
    args = recorded[0]
    assert args[0] == "/usr/bin/ffmpeg"
    assert "-an" in args
    assert args[args.index("-vf") + 1] == "fps=5"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[-1] == str(out)
    assert out.parent.exists()


def test_downsample_rejects_bad_fps(tools_present, tmp_path):
    with pytest.raises(ValueError):
        downsample_video(Path("in.mp4"), tmp_path / "v.mp4", fps=0)


def test_probe_duration_parses_seconds(tools_present, monkeypatch):
    recorded = stub_run(monkeypatch, stdout="41.234000\n")
    assert probe_duration(Path("in.mp4")) == pytest.approx(41.234)
    assert recorded[0][0] == "/usr/bin/ffprobe"


@pytest.mark.parametrize("raw", ["N/A\n", "", "nan\n", "-3\n", "0\n"])
def test_probe_duration_rejects_unusable_output(tools_present, monkeypatch, raw):
    stub_run(monkeypatch, stdout=raw)
    with pytest.raises(MediaError):
        probe_duration(Path("in.mp4"))


def test_probe_failure_is_wrapped(tools_present, monkeypatch):
    def boom(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="boom")

    monkeypatch.setattr(media.subprocess, "run", boom)
    with pytest.raises(MediaError):
        probe_duration(Path("in.mp4"))


def test_missing_ffprobe(monkeypatch):
    monkeypatch.setattr(media.shutil, "which", lambda name: None)
    with pytest.raises(MediaError):
        probe_duration(Path("in.mp4"))
