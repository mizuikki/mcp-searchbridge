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


class QueryEcho(BaseModel):
    """Echo of the normalized tool input."""

    text: str
    recency: str | None = None
    max_sources: int
    domain_allowlist: list[str] = Field(default_factory=list)
    return_mode: Literal["concise", "standard"]


class Citation(BaseModel):
    """Reference from the summary into a source evidence chunk."""

    source_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class Summary(BaseModel):
    """Top-level synthesized answer with source references."""

    text: str = ""
    citations: list[Citation] = Field(default_factory=list)


class EvidenceChunk(BaseModel):
    """Supporting evidence extracted for a source."""

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SearchSource(BaseModel):
    """Structured source metadata returned to MCP clients."""

    source_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    domain: str = Field(min_length=1)
    published_at: str | None = None
    domain_allowed: bool = True
    evidence: list[EvidenceChunk] = Field(default_factory=list)


class WarningInfo(BaseModel):
    """Structured warning returned to callers."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ErrorInfo(BaseModel):
    """Structured error details returned in diagnostics."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class ProviderInfo(BaseModel):
    """Provider metadata for diagnostics."""

    name: str = Field(min_length=1)
    model: str = Field(min_length=1)


class NormalizationInfo(BaseModel):
    """How the response was normalized locally."""

    response_format_requested: Literal["json_object", "none"] = "json_object"
    response_format_accepted: bool = True
    parse_mode: Literal[
        "structured_v2",
        "structured_legacy",
        "text_fallback",
        "url_fallback",
        "error",
    ]


class Coverage(BaseModel):
    """Coverage summary for returned evidence."""

    sources_requested: int = Field(ge=0)
    sources_returned: int = Field(ge=0)
    sources_with_evidence: int = Field(ge=0)
    evidence_chunks_returned: int = Field(ge=0)


class Diagnostics(BaseModel):
    """Diagnostics to help downstream LLMs assess result quality."""

    status: Literal["ok", "partial", "empty", "error"]
    provider: ProviderInfo
    normalization: NormalizationInfo
    coverage: Coverage
    warnings: list[WarningInfo] = Field(default_factory=list)
    error: ErrorInfo | None = None


class SearchResult(BaseModel):
    """Structured MCP tool result."""

    query: QueryEcho
    summary: Summary
    sources: list[SearchSource] = Field(default_factory=list)
    diagnostics: Diagnostics
