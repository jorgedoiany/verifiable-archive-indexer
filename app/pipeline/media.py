"""ffmpeg helpers: extract speech audio and make an audio-free copy of a video."""
from __future__ import annotations

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
