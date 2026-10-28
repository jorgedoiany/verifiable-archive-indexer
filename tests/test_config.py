import pytest

from app.config import DEFAULT_API_BASE, load_settings

OPTIONAL = ("META_API_BASE", "SPARK_MODEL", "WONDER_MODEL", "ASR_MODEL", "SAM_MODEL")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
    for name in OPTIONAL:
        monkeypatch.delenv(name, raising=False)


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_settings()


def test_defaults(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    settings = load_settings()
    assert settings.api_base == DEFAULT_API_BASE
    assert settings.spark_model == "muse-spark-1.3"
    assert settings.sam_model == "sam-3.1"


def test_trailing_slash_is_stripped(monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    monkeypatch.setenv("META_API_BASE", "https://api.meta.ai/v1/")
    assert load_settings().api_base == "https://api.meta.ai/v1"
