"""Environment-backed configuration for the MCP server."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYSTEM_PROMPT = """You are a web search tool behind an MCP server.
Use the upstream model's available internet or search capability if the
provider supports it.
Return factual, current information.
Prefer authoritative sources.
Always include source URLs when available.
If the provider does not actually support live web access, say so explicitly
instead of fabricating freshness."""


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    openai_api_key: str = Field(alias="OPENAI_API_KEY", min_length=1)
    openai_base_url: HttpUrl = Field(alias="OPENAI_BASE_URL")
    openai_model: str = Field(alias="OPENAI_MODEL", min_length=1)
    openai_timeout_seconds: float = Field(
        default=60.0,
        alias="OPENAI_TIMEOUT_SECONDS",
        gt=0,
    )
    openai_max_retries: int = Field(
        default=2,
        alias="OPENAI_MAX_RETRIES",
        ge=0,
    )
    openai_organization: str | None = Field(default=None, alias="OPENAI_ORGANIZATION")
    openai_project: str | None = Field(default=None, alias="OPENAI_PROJECT")
    searchbridge_system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT,
        alias="SEARCHBRIDGE_SYSTEM_PROMPT",
    )
    searchbridge_default_max_sources: int = Field(
        default=5,
        alias="SEARCHBRIDGE_DEFAULT_MAX_SOURCES",
        ge=1,
        le=20,
    )
    searchbridge_log_level: str = Field(default="INFO", alias="SEARCHBRIDGE_LOG_LEVEL")

    @field_validator("searchbridge_log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            msg = (
                "SEARCHBRIDGE_LOG_LEVEL must be one of "
                f"{', '.join(sorted(valid_levels))}"
            )
            raise ValueError(msg)
        return normalized

    @field_validator("searchbridge_system_prompt", mode="before")
    @classmethod
    def normalize_system_prompt(cls, value: Any) -> str:
        if value is None:
            return DEFAULT_SYSTEM_PROMPT

        text = str(value).strip()
        return text or DEFAULT_SYSTEM_PROMPT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings for the current process."""

    return Settings()
