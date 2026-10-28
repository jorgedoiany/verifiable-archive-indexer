"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_API_BASE = "https://api.meta.ai/v1"


@dataclass(frozen=True)
class Settings:
    api_key: str
    api_base: str
    spark_model: str
    wonder_model: str
    asr_model: str
    sam_model: str


def load_settings() -> Settings:
    load_dotenv()
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MODEL_API_KEY is not set; copy .env.example to .env and add your key."
        )
    return Settings(
        api_key=api_key,
        api_base=os.environ.get("META_API_BASE", DEFAULT_API_BASE).rstrip("/"),
        spark_model=os.environ.get("SPARK_MODEL", "muse-spark-1.3"),
        wonder_model=os.environ.get("WONDER_MODEL", "muse-wonder-1.0"),
        asr_model=os.environ.get("ASR_MODEL", "muse-voice-transcribe-1.0"),
        sam_model=os.environ.get("SAM_MODEL", "sam-3.1"),
    )
