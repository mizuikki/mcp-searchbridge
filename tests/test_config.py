import pytest
from pydantic import ValidationError

from mcp_searchbridge.config import DEFAULT_SYSTEM_PROMPT, Settings


def test_settings_parse_required_and_optional_values() -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        OPENAI_BASE_URL="https://api.example.com/v1",
        OPENAI_MODEL="search-model",
        OPENAI_TIMEOUT_SECONDS=30,
        OPENAI_MAX_RETRIES=1,
        SEARCHBRIDGE_DEFAULT_MAX_SOURCES=7,
        SEARCHBRIDGE_LOG_LEVEL="debug",
    )

    assert settings.openai_model == "search-model"
    assert settings.openai_timeout_seconds == 30
    assert settings.searchbridge_default_max_sources == 7
    assert settings.searchbridge_log_level == "DEBUG"


def test_settings_require_mandatory_env_values() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://api.example.com/v1",
        )


def test_settings_empty_system_prompt_falls_back_to_default() -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        OPENAI_BASE_URL="https://api.example.com/v1",
        OPENAI_MODEL="search-model",
        SEARCHBRIDGE_SYSTEM_PROMPT="",
    )

    assert settings.searchbridge_system_prompt == DEFAULT_SYSTEM_PROMPT
