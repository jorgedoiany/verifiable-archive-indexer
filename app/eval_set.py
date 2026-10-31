"""Split a pool of prepared clips into a tuning group and a blind group.

The split is drawn once with a fixed seed and then recorded; it is never
redrawn, so nobody can keep drawing until the split looks convenient.
"""
from __future__ import annotations

import json
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Split:
    tune: tuple[str, ...]
    blind: tuple[str, ...]


def target_name(prefix: str, number: int) -> str:
    return f"{prefix}_{number:02d}.mp4"


def split_pool(names: Sequence[str], tune: int, blind: int, seed: int) -> Split:
    """Shuffle the sorted names with the seed; the first `tune` are for tuning."""
    if tune < 0 or blind < 0:
        raise ValueError("counts must not be negative")
    ordered = sorted(set(names))
    if len(ordered) < tune + blind:
        raise ValueError(f"need {tune + blind} clips, found {len(ordered)}")
    shuffled = ordered[:]
    random.Random(seed).shuffle(shuffled)
    return Split(tuple(shuffled[:tune]), tuple(shuffled[tune : tune + blind]))


def build_eval_set(
    pool_dir: Path,
    clips_dir: Path,
    record_path: Path,
    *,
    tune: int,
    blind: int,
    seed: int,
) -> Split:
    """Copy the split clips under neutral names and record which source is which."""
    pool_dir = Path(pool_dir)
    clips_dir = Path(clips_dir)
    record_path = Path(record_path)
    if record_path.exists():
        raise FileExistsError(
            f"the split was already made ({record_path}); it is never redrawn"
        )

    names = sorted(p.name for p in pool_dir.glob("p_*.mp4") if p.is_file())
    split = split_pool(names, tune, blind, seed)
    tune_map = {
        target_name("tune", i): source for i, source in enumerate(split.tune, start=1)
    }
    blind_map = {
        target_name("blind", i): source for i, source in enumerate(split.blind, start=1)
    }

    clashes = [n for n in (*tune_map, *blind_map) if (clips_dir / n).exists()]
    if clashes:
        raise FileExistsError(f"refusing to overwrite existing clips: {clashes}")

    clips_dir.mkdir(parents=True, exist_ok=True)
    for target, source in {**tune_map, **blind_map}.items():
        shutil.copyfile(pool_dir / source, clips_dir / target)

    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(
        json.dumps(
            {"seed": seed, "tune": tune_map, "blind": blind_map}, indent=2
        ),
        encoding="utf-8",
    )
    return split
