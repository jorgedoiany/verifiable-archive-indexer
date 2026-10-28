"""ffmpeg helpers: extract speech audio and make an audio-free copy of a video."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def _ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise MediaError("ffmpeg not found; install it (for example: brew install ffmpeg)")
    return path


def _run(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"ffmpeg failed: {(exc.stderr or '')[-300:]}") from exc


def extract_audio(video: Path, out_wav: Path, rate: int = 24_000) -> Path:
    """Write a mono, 16-bit PCM WAV, the format the transcription endpoint accepts."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg(), "-y", "-i", str(video),
            "-vn", "-ac", "1", "-ar", str(rate),
            "-c:a", "pcm_s16le", "-map_metadata", "-1",
            str(out_wav),
        ]
    )
    return out_wav


def strip_audio(video: Path, out_mp4: Path) -> Path:
    """Write an H.264 mp4 without audio.

    Meta documents degraded responses when audio is included for Muse Spark 1.3,
    and speech is handled by the transcription model instead.
    """
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg(), "-y", "-i", str(video),
            "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_mp4),
        ]
    )
    return out_mp4


def downsample_video(video: Path, out_mp4: Path, fps: int = 5) -> Path:
    """Write an audio-free, low-fps mp4 for SAM, which bills per frame."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _ffmpeg(), "-y", "-i", str(video),
            "-an", "-vf", f"fps={fps}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out_mp4),
        ]
    )
    return out_mp4


def _ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path is None:
        raise MediaError("ffprobe not found; it ships with ffmpeg")
    return path


def probe_duration(video: Path) -> float:
    """Return the container duration in seconds."""
    args = [
        _ffprobe(), "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video),
    ]
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise MediaError(f"ffprobe failed: {(exc.stderr or '')[-300:]}") from exc
    raw = result.stdout.strip()
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise MediaError(f"could not read duration: {raw!r}") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise MediaError(f"invalid duration: {raw!r}")
    return seconds
