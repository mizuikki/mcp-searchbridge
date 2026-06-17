"""Environment-backed configuration for the MCP server."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, cast

from pydantic import Field, HttpUrl, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .type_utils import LogLevel

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

    openai_api_key: str | None = Field(
        default=None,
        alias="OPENAI_API_KEY",
        min_length=1,
    )
    openai_base_url: HttpUrl | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_model: str | None = Field(
        default=None,
        alias="OPENAI_MODEL",
        min_length=1,
    )
    openai_timeout_seconds: float = Field(
        default=180.0,
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
    searchbridge_backend_kind: Literal["openai", "private_http"] = Field(
        default="openai",
        alias="SEARCHBRIDGE_BACKEND_KIND",
    )
    searchbridge_private_backend_url: HttpUrl | None = Field(
        default=None,
        alias="SEARCHBRIDGE_PRIVATE_BACKEND_URL",
    )
    searchbridge_private_backend_api_key: str | None = Field(
        default=None,
        alias="SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY",
    )
    searchbridge_private_backend_timeout_seconds: float = Field(
        default=30.0,
        alias="SEARCHBRIDGE_PRIVATE_BACKEND_TIMEOUT_SECONDS",
        gt=0,
    )
    searchbridge_private_backend_fallback_to_openai: bool = Field(
        default=False,
        alias="SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI",
    )
    searchbridge_log_level: LogLevel = Field(
        default="INFO",
        alias="SEARCHBRIDGE_LOG_LEVEL",
    )

    @field_validator(
        "openai_api_key",
        "openai_model",
        "openai_organization",
        "openai_project",
        "searchbridge_private_backend_api_key",
        mode="before",
    )
    @classmethod
    def normalize_optional_text(cls, value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("searchbridge_backend_kind", mode="before")
    @classmethod
    def normalize_backend_kind(cls, value: object) -> str:
        return str(value).strip().lower()

    @field_validator("searchbridge_log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> LogLevel:
        normalized = str(value).upper()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in valid_levels:
            msg = (
                "SEARCHBRIDGE_LOG_LEVEL must be one of "
                f"{', '.join(sorted(valid_levels))}"
            )
            raise ValueError(msg)
        return cast(LogLevel, normalized)

    @field_validator("searchbridge_system_prompt", mode="before")
    @classmethod
    def normalize_system_prompt(cls, value: Any) -> str:
        if value is None:
            return DEFAULT_SYSTEM_PROMPT

        text = str(value).strip()
        return text or DEFAULT_SYSTEM_PROMPT

    @model_validator(mode="after")
    def validate_backend_configuration(self) -> Settings:
        if (
            self.searchbridge_backend_kind == "private_http"
            and self.searchbridge_private_backend_url is None
        ):
            raise ValueError(
                "SEARCHBRIDGE_PRIVATE_BACKEND_URL is required when "
                "SEARCHBRIDGE_BACKEND_KIND=private_http"
            )

        if self.searchbridge_backend_kind == "openai":
            self._require_openai_configuration(
                reason="SEARCHBRIDGE_BACKEND_KIND=openai",
            )
        elif self.searchbridge_private_backend_fallback_to_openai:
            self._require_openai_configuration(
                reason="SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=true",
            )

        return self

    def _require_openai_configuration(self, *, reason: str) -> None:
        missing: list[str] = []
        if not self.openai_api_key:
            missing.append("OPENAI_API_KEY")
        if self.openai_base_url is None:
            missing.append("OPENAI_BASE_URL")
        if not self.openai_model:
            missing.append("OPENAI_MODEL")
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"{missing_text} must be set when {reason}")

    @property
    def resolved_openai_api_key(self) -> str:
        if self.openai_api_key is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")
        return self.openai_api_key

    @property
    def resolved_openai_base_url(self) -> HttpUrl:
        if self.openai_base_url is None:
            raise RuntimeError("OPENAI_BASE_URL is not configured")
        return self.openai_base_url

    @property
    def resolved_openai_model(self) -> str:
        if self.openai_model is None:
            raise RuntimeError("OPENAI_MODEL is not configured")
        return self.openai_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache settings for the current process."""

    return Settings()  # pyright: ignore[reportCallIssue]
