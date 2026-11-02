"""Freeze the blind evaluation set: record SHA-256 hashes before anything runs on it."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

CHUNK_BYTES = 1024 * 1024


class FreezeError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def blind_stems(clips_dir: Path, prefix: str = "blind_") -> list[str]:
    return sorted(p.stem for p in Path(clips_dir).glob(f"{prefix}*.mp4"))


def find_outputs(stems: list[str], data_dir: Path) -> list[str]:
    """Pipeline files that already exist for these clips in any data/work* folder."""
    found: list[str] = []
    for work_dir in sorted(Path(data_dir).glob("work*")):
        if not work_dir.is_dir():
            continue
        for stem in stems:
            found.extend(f"{work_dir.name}/{p.name}" for p in sorted(work_dir.glob(f"{stem}.*")))
    return found


def freeze_blind(
    clips_dir: Path,
    truth_dir: Path,
    data_dir: Path,
    manifest_path: Path,
    *,
    prefix: str = "blind_",
    now: datetime | None = None,
) -> dict:
    """Write the manifest; refuse if it exists, labels are missing or clips already ran."""
    clips_dir = Path(clips_dir)
    truth_dir = Path(truth_dir)
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        raise FreezeError(f"{manifest_path} already exists; a frozen set is never refrozen")
    stems = blind_stems(clips_dir, prefix)
    if not stems:
        raise FreezeError(f"no {prefix}*.mp4 clips found in {clips_dir}")
    missing = [s for s in stems if not (truth_dir / f"{s}.json").exists()]
    if missing:
        raise FreezeError(f"labels missing for: {', '.join(missing)}")
    outputs = find_outputs(stems, data_dir)
    if outputs:
        raise FreezeError(
            "pipeline outputs already exist for blind clips, so they were not blind: "
            + ", ".join(outputs)
        )

    stamp = (now or datetime.now(UTC)).isoformat(timespec="seconds")
    manifest = {
        "created_utc": stamp,
        "algorithm": "sha256",
        "clips": {f"{s}.mp4": sha256_file(clips_dir / f"{s}.mp4") for s in stems},
        "ground_truth": {
            f"{s}.json": sha256_file(truth_dir / f"{s}.json") for s in stems
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_manifest(
    manifest_path: Path, clips_dir: Path, truth_dir: Path, prefix: str = "blind_"
) -> list[str]:
    """Return the problems found; an empty list means nothing changed."""
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        clips = dict(manifest["clips"])
        truth = dict(manifest["ground_truth"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise FreezeError(f"cannot read the manifest: {exc}") from exc

    problems: list[str] = []
    for section, base, recorded in (
        ("clips", clips_dir, clips),
        ("ground_truth", truth_dir, truth),
    ):
        for name, expected in recorded.items():
            path = Path(base) / name
            if not path.exists():
                problems.append(f"missing: {section}/{name}")
            elif sha256_file(path) != expected:
                problems.append(f"changed: {section}/{name}")
    current = {f"{stem}.mp4" for stem in blind_stems(clips_dir, prefix)}
    problems.extend(f"not in manifest: clips/{n}" for n in sorted(current - set(clips)))
    return problems
