"""Data models for search requests and results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SearchRequest(BaseModel):
    """Normalized search tool input."""

    query: str = Field(min_length=1, description="The user search query.")
    recency: str | None = Field(
        default=None,
        description="Optional freshness hint such as day, week, month, or latest.",
    )
    max_sources: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Desired maximum number of sources in the response.",
    )
    domain_allowlist: list[str] = Field(
        default_factory=list,
        description="Preferred or restricted source domains.",
    )
    return_mode: Literal["concise", "standard"] = Field(
        default="standard",
        description="Controls answer brevity.",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be empty")
        return stripped

    @field_validator("recency")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("domain_allowlist")
    @classmethod
    def normalize_domain_allowlist(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized


class SearchSource(BaseModel):
    """Structured source metadata returned to MCP clients."""

    title: str = Field(min_length=1)
    url: HttpUrl
    snippet: str = ""


class SearchResult(BaseModel):
    """Structured MCP tool result."""

    answer: str
    sources: list[SearchSource] = Field(default_factory=list)
    provider: str
    model: str
    raw_text: str = ""
    warnings: list[str] = Field(default_factory=list)
