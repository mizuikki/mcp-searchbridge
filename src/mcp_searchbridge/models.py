"""Data models for MCP tool inputs and outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .type_utils import (
    ContentFormat,
    DocsAnswerMode,
    DocSourceType,
    ExtractMode,
    OutlineDepth,
    ParseMode,
    ReturnMode,
    ToolStatus,
)


class BaseToolRequest(BaseModel):
    """Base request model with shared text normalization."""

    @staticmethod
    def _strip_optional_text(value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @staticmethod
    def _strip_required_text(value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value must not be empty")
        return stripped


class SearchRequest(BaseToolRequest):
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
    return_mode: ReturnMode = Field(
        default="standard",
        description="Controls answer brevity.",
    )

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return cls._strip_required_text(value)

    @field_validator("recency")
    @classmethod
    def strip_optional_text(cls, value: str | None) -> str | None:
        return cls._strip_optional_text(value)

    @field_validator("domain_allowlist")
    @classmethod
    def normalize_domain_allowlist(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized


class ExtractUrlRequest(BaseToolRequest):
    """Request for URL extraction."""

    url: HttpUrl
    mode: ExtractMode = "best_effort"
    max_chars: int = Field(default=12000, ge=200, le=100000)


class OutlineUrlRequest(BaseToolRequest):
    """Request for URL outline generation."""

    url: HttpUrl
    depth: OutlineDepth = "standard"


class DocsQARequest(BaseToolRequest):
    """Request for documentation question answering."""

    question: str = Field(min_length=1)
    url: HttpUrl | None = None
    domain_allowlist: list[str] = Field(default_factory=list)
    answer_mode: DocsAnswerMode = "standard"

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        return cls._strip_required_text(value)

    @field_validator("domain_allowlist")
    @classmethod
    def normalize_domain_allowlist(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for item in value:
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized


class FindOfficialDocsRequest(BaseToolRequest):
    """Request for official docs discovery."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)

    @field_validator("query")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return cls._strip_required_text(value)


class DocSourceResolutionRequest(BaseToolRequest):
    """Request for document source resolution."""

    query_or_url: str = Field(min_length=1)

    @field_validator("query_or_url")
    @classmethod
    def strip_query_or_url(cls, value: str) -> str:
        return cls._strip_required_text(value)


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


class ToolDiagnostics(BaseModel):
    """Common diagnostics for all tools."""

    status: ToolStatus
    provider: ProviderInfo
    warnings: list[WarningInfo] = Field(default_factory=list)
    error: ErrorInfo | None = None


class SearchNormalizationInfo(BaseModel):
    """How a search result was normalized."""

    response_format_requested: Literal["json_object", "none"] = "json_object"
    response_format_accepted: bool = True
    parse_mode: ParseMode


class SearchCoverage(BaseModel):
    """Coverage summary for returned evidence."""

    sources_requested: int = Field(ge=0)
    sources_returned: int = Field(ge=0)
    sources_with_evidence: int = Field(ge=0)
    evidence_chunks_returned: int = Field(ge=0)


class SearchDiagnostics(ToolDiagnostics):
    """Search-specific diagnostics."""

    normalization: SearchNormalizationInfo
    coverage: SearchCoverage


# Backward-compatible aliases for in-flight refactors.
NormalizationInfo = SearchNormalizationInfo
Coverage = SearchCoverage
Diagnostics = SearchDiagnostics
ResolveDocSourceRequest = DocSourceResolutionRequest


class QueryEcho(BaseModel):
    """Echo of the normalized search input."""

    text: str
    recency: str | None = None
    max_sources: int
    domain_allowlist: list[str] = Field(default_factory=list)
    return_mode: ReturnMode


class Citation(BaseModel):
    """Reference from text into a source evidence chunk."""

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
    """Structured source metadata returned to clients."""

    source_id: str = Field(min_length=1)
    rank: int = Field(ge=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    domain: str = Field(min_length=1)
    published_at: str | None = None
    domain_allowed: bool = True
    evidence: list[EvidenceChunk] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Search/discovery tool result."""

    query: QueryEcho
    summary: Summary
    sources: list[SearchSource] = Field(default_factory=list)
    diagnostics: SearchDiagnostics


class ExtractRequestEcho(BaseModel):
    """Echo of extract_url input."""

    url: HttpUrl
    mode: ExtractMode
    max_chars: int


class ExtractResult(BaseModel):
    """URL extraction tool result."""

    request: ExtractRequestEcho
    title: str = ""
    url: HttpUrl
    content: str = ""
    content_format: ContentFormat = "text"
    truncated: bool = False
    likely_rewritten: bool = True
    diagnostics: ToolDiagnostics


class OutlineRequestEcho(BaseModel):
    """Echo of outline_url input."""

    url: HttpUrl
    depth: OutlineDepth


class OutlineSection(BaseModel):
    """One outline section."""

    title: str = Field(min_length=1)
    summary: str = ""


class OutlineResult(BaseModel):
    """URL outline tool result."""

    request: OutlineRequestEcho
    title: str = ""
    sections: list[OutlineSection] = Field(default_factory=list)
    diagnostics: ToolDiagnostics


class DocsQARequestEcho(BaseModel):
    """Echo of docs_qa input."""

    question: str
    url: HttpUrl | None = None
    domain_allowlist: list[str] = Field(default_factory=list)
    answer_mode: DocsAnswerMode


class DocsQAResult(BaseModel):
    """Documentation QA result."""

    request: DocsQARequestEcho
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    sources: list[SearchSource] = Field(default_factory=list)
    diagnostics: ToolDiagnostics


class OfficialDocMatch(BaseModel):
    """One official documentation match."""

    title: str = Field(min_length=1)
    url: HttpUrl
    domain: str = Field(min_length=1)
    rationale: str = ""


class OfficialDocsRequestEcho(BaseModel):
    """Echo of find_official_docs input."""

    query: str
    max_results: int


class OfficialDocsResult(BaseModel):
    """Official documentation discovery result."""

    request: OfficialDocsRequestEcho
    matches: list[OfficialDocMatch] = Field(default_factory=list)
    diagnostics: ToolDiagnostics


class DocSourceResolutionRequestEcho(BaseModel):
    """Echo of resolve_doc_source input."""

    query_or_url: str


class DocSourceResolutionResult(BaseModel):
    """Document source classification result."""

    request: DocSourceResolutionRequestEcho
    source_type: DocSourceType
    resolved_url: HttpUrl | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    diagnostics: ToolDiagnostics
