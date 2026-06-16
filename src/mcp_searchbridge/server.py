"""FastMCP server entry point."""

from __future__ import annotations

import logging
from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from .config import Settings, get_settings
from .errors import SearchBridgeError, UpstreamSearchError
from .models import (
    DocSourceResolutionRequest,
    DocSourceResolutionRequestEcho,
    DocSourceResolutionResult,
    DocsQARequest,
    DocsQARequestEcho,
    DocsQAResult,
    ErrorInfo,
    ExtractRequestEcho,
    ExtractResult,
    ExtractUrlRequest,
    FindOfficialDocsRequest,
    OfficialDocsRequestEcho,
    OfficialDocsResult,
    OutlineRequestEcho,
    OutlineResult,
    OutlineUrlRequest,
    ProviderInfo,
    SearchCoverage,
    SearchDiagnostics,
    SearchNormalizationInfo,
    SearchRequest,
    SearchResult,
    ToolDiagnostics,
)
from .openai_backend import OpenAIAggregationBackend

LOGGER = logging.getLogger(__name__)


def create_server(
    settings: Settings | None = None,
    backend: OpenAIAggregationBackend | None = None,
) -> FastMCP:
    """Create the FastMCP application and register tools."""

    active_settings = settings or get_settings()
    _configure_logging(active_settings.searchbridge_log_level)
    aggregation_backend = backend or OpenAIAggregationBackend(active_settings)

    mcp = FastMCP(
        name="mcp-searchbridge",
        instructions=(
            "Use the available tools to search the web, extract online documents, "
            "outline URL structures, answer documentation questions, discover "
            "official docs, and classify documentation sources."
        ),
        log_level=active_settings.searchbridge_log_level,
    )

    @mcp.tool(
        name="search_web",
        description=(
            "Search the web through an upstream OpenAI-compatible chat provider."
        ),
    )
    def search_web(
        query: str,
        recency: str | None = None,
        max_sources: int | None = None,
        domain_allowlist: list[str] | None = None,
        return_mode: Literal["concise", "standard"] = "standard",
    ) -> SearchResult:
        try:
            request = SearchRequest(
                query=query,
                recency=recency,
                max_sources=(
                    active_settings.searchbridge_default_max_sources
                    if max_sources is None
                    else max_sources
                ),
                domain_allowlist=domain_allowlist or [],
                return_mode=return_mode,
            )
            return aggregation_backend.search_web(request)
        except ValidationError as exc:
            LOGGER.error("search_web validation failed: %s", exc)
            return _search_error_result(
                query=query,
                recency=recency,
                max_sources=(
                    active_settings.searchbridge_default_max_sources
                    if max_sources is None
                    else max_sources
                ),
                domain_allowlist=domain_allowlist or [],
                return_mode=return_mode,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="invalid_request",
                message=f"Invalid search request: {exc}",
                retryable=False,
            )
        except UpstreamSearchError as exc:
            _log_upstream_failure("search_web", exc)
            return _search_error_result(
                query=query,
                recency=recency,
                max_sources=(
                    active_settings.searchbridge_default_max_sources
                    if max_sources is None
                    else max_sources
                ),
                domain_allowlist=domain_allowlist or [],
                return_mode=return_mode,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="upstream_request_failed",
                message=exc.client_message,
                retryable=exc.retryable,
            )
        except SearchBridgeError as exc:
            LOGGER.error("search_web failed: %s", exc)
            return _search_error_result(
                query=query,
                recency=recency,
                max_sources=(
                    active_settings.searchbridge_default_max_sources
                    if max_sources is None
                    else max_sources
                ),
                domain_allowlist=domain_allowlist or [],
                return_mode=return_mode,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="search_request_failed",
                message=f"Search request failed: {exc}",
                retryable=False,
            )

    @mcp.tool(
        name="extract_url",
        description="Extract the main content from a URL through the upstream model.",
    )
    def extract_url(
        url: str,
        mode: Literal["body", "markdown", "best_effort"] = "best_effort",
        max_chars: int = 12000,
    ) -> ExtractResult:
        try:
            request = ExtractUrlRequest(url=url, mode=mode, max_chars=max_chars)
            return aggregation_backend.extract_url(request)
        except ValidationError as exc:
            LOGGER.error("extract_url validation failed: %s", exc)
            return _extract_error_result(
                url=url,
                mode=mode,
                max_chars=max_chars,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="invalid_request",
                message=f"Invalid extract request: {exc}",
                retryable=False,
            )
        except UpstreamSearchError as exc:
            _log_upstream_failure("extract_url", exc)
            return _extract_error_result(
                url=url,
                mode=mode,
                max_chars=max_chars,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="upstream_request_failed",
                message=exc.client_message,
                retryable=exc.retryable,
            )

    @mcp.tool(
        name="outline_url",
        description="Return a structured outline of a URL or llms.txt-like index.",
    )
    def outline_url(
        url: str,
        depth: Literal["shallow", "standard", "deep"] = "standard",
    ) -> OutlineResult:
        try:
            request = OutlineUrlRequest(url=url, depth=depth)
            return aggregation_backend.outline_url(request)
        except ValidationError as exc:
            LOGGER.error("outline_url validation failed: %s", exc)
            return _outline_error_result(
                url=url,
                depth=depth,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="invalid_request",
                message=f"Invalid outline request: {exc}",
                retryable=False,
            )
        except UpstreamSearchError as exc:
            _log_upstream_failure("outline_url", exc)
            return _outline_error_result(
                url=url,
                depth=depth,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="upstream_request_failed",
                message=exc.client_message,
                retryable=exc.retryable,
            )

    @mcp.tool(
        name="docs_qa",
        description="Answer a documentation question using official online docs.",
    )
    def docs_qa(
        question: str,
        url: str | None = None,
        domain_allowlist: list[str] | None = None,
        answer_mode: Literal["concise", "standard"] = "standard",
    ) -> DocsQAResult:
        try:
            request = DocsQARequest(
                question=question,
                url=url,
                domain_allowlist=domain_allowlist or [],
                answer_mode=answer_mode,
            )
            return aggregation_backend.docs_qa(request)
        except ValidationError as exc:
            LOGGER.error("docs_qa validation failed: %s", exc)
            return _docs_qa_error_result(
                question=question,
                url=url,
                domain_allowlist=domain_allowlist or [],
                answer_mode=answer_mode,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="invalid_request",
                message=f"Invalid docs QA request: {exc}",
                retryable=False,
            )
        except UpstreamSearchError as exc:
            _log_upstream_failure("docs_qa", exc)
            return _docs_qa_error_result(
                question=question,
                url=url,
                domain_allowlist=domain_allowlist or [],
                answer_mode=answer_mode,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="upstream_request_failed",
                message=exc.client_message,
                retryable=exc.retryable,
            )

    @mcp.tool(
        name="find_official_docs",
        description="Find official documentation entry points for a topic or library.",
    )
    def find_official_docs(query: str, max_results: int = 5) -> OfficialDocsResult:
        try:
            request = FindOfficialDocsRequest(query=query, max_results=max_results)
            return aggregation_backend.find_official_docs(request)
        except ValidationError as exc:
            LOGGER.error("find_official_docs validation failed: %s", exc)
            return _official_docs_error_result(
                query=query,
                max_results=max_results,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="invalid_request",
                message=f"Invalid official docs request: {exc}",
                retryable=False,
            )
        except UpstreamSearchError as exc:
            _log_upstream_failure("find_official_docs", exc)
            return _official_docs_error_result(
                query=query,
                max_results=max_results,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="upstream_request_failed",
                message=exc.client_message,
                retryable=exc.retryable,
            )

    @mcp.tool(
        name="resolve_doc_source",
        description=(
            "Classify whether input is a page URL, llms.txt, docs query, "
            "or web search query."
        ),
    )
    def resolve_doc_source(query_or_url: str) -> DocSourceResolutionResult:
        try:
            request = DocSourceResolutionRequest(query_or_url=query_or_url)
            return aggregation_backend.resolve_doc_source(request)
        except ValidationError as exc:
            LOGGER.error("resolve_doc_source validation failed: %s", exc)
            return _resolve_source_error_result(
                query_or_url=query_or_url,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="invalid_request",
                message=f"Invalid source resolution request: {exc}",
                retryable=False,
            )
        except UpstreamSearchError as exc:
            _log_upstream_failure("resolve_doc_source", exc)
            return _resolve_source_error_result(
                query_or_url=query_or_url,
                provider=aggregation_backend.provider_name,
                model=active_settings.openai_model,
                code="upstream_request_failed",
                message=exc.client_message,
                retryable=exc.retryable,
            )

    return mcp


def main() -> None:
    """Run the FastMCP server over stdio."""

    server = create_server()
    server.run(transport="stdio")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _provider_info(provider: str, model: str) -> ProviderInfo:
    return ProviderInfo(name=provider, model=model)


def _log_upstream_failure(tool_name: str, exc: UpstreamSearchError) -> None:
    context = exc.log_context
    LOGGER.warning(
        "%s upstream failure [error_type=%s status_code=%s request_id=%s retryable=%s]",
        tool_name,
        context.error_type,
        context.status_code,
        context.request_id,
        exc.retryable,
    )


def _search_error_result(
    *,
    query: str,
    recency: str | None,
    max_sources: int | None,
    domain_allowlist: list[str],
    return_mode: Literal["concise", "standard"],
    provider: str,
    model: str,
    code: str,
    message: str,
    retryable: bool,
) -> SearchResult:
    from .models import QueryEcho, Summary

    return SearchResult(
        query=QueryEcho(
            text=query,
            recency=recency,
            max_sources=max_sources or 0,
            domain_allowlist=domain_allowlist,
            return_mode=return_mode,
        ),
        summary=Summary(text="", citations=[]),
        sources=[],
        diagnostics=SearchDiagnostics(
            status="error",
            provider=_provider_info(provider, model),
            normalization=SearchNormalizationInfo(
                response_format_requested="json_object",
                response_format_accepted=False,
                parse_mode="error",
            ),
            coverage=SearchCoverage(
                sources_requested=max_sources or 0,
                sources_returned=0,
                sources_with_evidence=0,
                evidence_chunks_returned=0,
            ),
            warnings=[],
            error=ErrorInfo(code=code, message=message, retryable=retryable),
        ),
    )


def _extract_error_result(
    *,
    url: str,
    mode: Literal["body", "markdown", "best_effort"],
    max_chars: int,
    provider: str,
    model: str,
    code: str,
    message: str,
    retryable: bool,
) -> ExtractResult:
    return ExtractResult(
        request=ExtractRequestEcho(url=url, mode=mode, max_chars=max_chars),
        title="",
        url=url,
        content="",
        content_format="text",
        truncated=False,
        likely_rewritten=True,
        diagnostics=ToolDiagnostics(
            status="error",
            provider=_provider_info(provider, model),
            warnings=[],
            error=ErrorInfo(code=code, message=message, retryable=retryable),
        ),
    )


def _outline_error_result(
    *,
    url: str,
    depth: Literal["shallow", "standard", "deep"],
    provider: str,
    model: str,
    code: str,
    message: str,
    retryable: bool,
) -> OutlineResult:
    return OutlineResult(
        request=OutlineRequestEcho(url=url, depth=depth),
        title="",
        sections=[],
        diagnostics=ToolDiagnostics(
            status="error",
            provider=_provider_info(provider, model),
            warnings=[],
            error=ErrorInfo(code=code, message=message, retryable=retryable),
        ),
    )


def _docs_qa_error_result(
    *,
    question: str,
    url: str | None,
    domain_allowlist: list[str],
    answer_mode: Literal["concise", "standard"],
    provider: str,
    model: str,
    code: str,
    message: str,
    retryable: bool,
) -> DocsQAResult:
    return DocsQAResult(
        request=DocsQARequestEcho(
            question=question,
            url=url,
            domain_allowlist=domain_allowlist,
            answer_mode=answer_mode,
        ),
        answer="",
        citations=[],
        sources=[],
        diagnostics=ToolDiagnostics(
            status="error",
            provider=_provider_info(provider, model),
            warnings=[],
            error=ErrorInfo(code=code, message=message, retryable=retryable),
        ),
    )


def _official_docs_error_result(
    *,
    query: str,
    max_results: int,
    provider: str,
    model: str,
    code: str,
    message: str,
    retryable: bool,
) -> OfficialDocsResult:
    return OfficialDocsResult(
        request=OfficialDocsRequestEcho(query=query, max_results=max_results),
        matches=[],
        diagnostics=ToolDiagnostics(
            status="error",
            provider=_provider_info(provider, model),
            warnings=[],
            error=ErrorInfo(code=code, message=message, retryable=retryable),
        ),
    )


def _resolve_source_error_result(
    *,
    query_or_url: str,
    provider: str,
    model: str,
    code: str,
    message: str,
    retryable: bool,
) -> DocSourceResolutionResult:
    return DocSourceResolutionResult(
        request=DocSourceResolutionRequestEcho(query_or_url=query_or_url),
        source_type="web_search_query",
        resolved_url=None,
        confidence=0.0,
        rationale="",
        diagnostics=ToolDiagnostics(
            status="error",
            provider=_provider_info(provider, model),
            warnings=[],
            error=ErrorInfo(code=code, message=message, retryable=retryable),
        ),
    )


if __name__ == "__main__":
    main()
