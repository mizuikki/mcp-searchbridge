import pytest
from pydantic import ValidationError

from mcp_searchbridge.config import DEFAULT_SYSTEM_PROMPT, Settings
from tests.helpers import make_settings


def test_settings_parse_required_and_optional_values() -> None:
    settings = make_settings(
        OPENAI_MODEL="search-model",
        OPENAI_TIMEOUT_SECONDS=30,
        OPENAI_MAX_RETRIES=1,
        SEARCHBRIDGE_DEFAULT_MAX_SOURCES=7,
        SEARCHBRIDGE_LOG_LEVEL="debug",
    )

    assert settings.openai_model == "search-model"
    assert settings.resolved_openai_models == ["search-model"]
    assert settings.openai_timeout_seconds == 30
    assert settings.searchbridge_default_max_sources == 7
    assert settings.searchbridge_log_level == "DEBUG"


def test_settings_parse_openai_model_chain() -> None:
    settings = make_settings(OPENAI_MODEL=" primary , fallback-a, fallback-b ")

    assert settings.openai_model == "primary,fallback-a,fallback-b"
    assert settings.resolved_openai_model == "primary"
    assert settings.resolved_openai_models == [
        "primary",
        "fallback-a",
        "fallback-b",
    ]


def test_settings_ignore_empty_segments_in_openai_model_chain() -> None:
    settings = make_settings(OPENAI_MODEL="primary,,fallback-a,")

    assert settings.resolved_openai_models == ["primary", "fallback-a"]


def test_settings_reject_duplicate_models_in_openai_model_chain() -> None:
    with pytest.raises(
        ValidationError,
        match="OPENAI_MODEL must not contain duplicate model names",
    ):
        make_settings(OPENAI_MODEL="primary,fallback-a,primary")


def test_settings_require_mandatory_env_values() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,  # pyright: ignore[reportCallIssue]
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL="https://api.example.com/v1",
        )


def test_settings_empty_system_prompt_falls_back_to_default() -> None:
    settings = make_settings(
        OPENAI_MODEL="search-model",
        SEARCHBRIDGE_SYSTEM_PROMPT="",
    )

    assert settings.searchbridge_system_prompt == DEFAULT_SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("raw_level", "expected"),
    [
        ("debug", "DEBUG"),
        ("info", "INFO"),
        ("warning", "WARNING"),
        ("error", "ERROR"),
        ("critical", "CRITICAL"),
    ],
)
def test_settings_normalize_all_log_levels(raw_level: str, expected: str) -> None:
    settings = make_settings(SEARCHBRIDGE_LOG_LEVEL=raw_level)

    assert settings.searchbridge_log_level == expected


def test_settings_reject_invalid_log_level() -> None:
    with pytest.raises(ValidationError, match="SEARCHBRIDGE_LOG_LEVEL must be one of"):
        make_settings(SEARCHBRIDGE_LOG_LEVEL="verbose")


def test_settings_default_backend_kind_is_openai() -> None:
    settings = make_settings()

    assert settings.searchbridge_backend_kind == "openai"
    assert settings.searchbridge_private_backend_fallback_to_openai is False


def test_settings_require_private_backend_url_for_private_http_mode() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "SEARCHBRIDGE_PRIVATE_BACKEND_URL is required when "
            "SEARCHBRIDGE_BACKEND_KIND=private_http"
        ),
    ):
        make_settings(
            SEARCHBRIDGE_BACKEND_KIND="private_http",
            OPENAI_API_KEY=None,
            OPENAI_BASE_URL=None,
            OPENAI_MODEL=None,
        )


def test_settings_allow_private_http_without_openai_when_fallback_disabled() -> None:
    settings = make_settings(
        SEARCHBRIDGE_BACKEND_KIND="private_http",
        SEARCHBRIDGE_PRIVATE_BACKEND_URL="https://private.example.com",
        OPENAI_API_KEY=None,
        OPENAI_BASE_URL=None,
        OPENAI_MODEL=None,
    )

    assert settings.searchbridge_backend_kind == "private_http"
    assert (
        str(settings.searchbridge_private_backend_url) == "https://private.example.com/"
    )


def test_settings_require_openai_config_when_private_fallback_enabled() -> None:
    with pytest.raises(
        ValidationError,
        match=(
            "OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL must be set when "
            "SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=true"
        ),
    ):
        make_settings(
            SEARCHBRIDGE_BACKEND_KIND="private_http",
            SEARCHBRIDGE_PRIVATE_BACKEND_URL="https://private.example.com",
            SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=True,
            OPENAI_API_KEY=None,
            OPENAI_BASE_URL=None,
            OPENAI_MODEL=None,
        )
