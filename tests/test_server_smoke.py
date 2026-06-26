from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from mcp_searchbridge.errors import UpstreamLogContext, UpstreamSearchError
from mcp_searchbridge.models import (
    Citation,
    DocSourceResolutionRequest,
    DocSourceResolutionRequestEcho,
    DocSourceResolutionResult,
    DocsQARequest,
    DocsQARequestEcho,
    DocsQAResult,
    EvidenceChunk,
    ExtractRequestEcho,
    ExtractResult,
    ExtractUrlRequest,
    OfficialDocMatch,
    OfficialDocsRequestEcho,
    OfficialDocsResult,
    OutlineRequestEcho,
    OutlineResult,
    OutlineSection,
    OutlineUrlRequest,
    ProviderInfo,
    QueryEcho,
    SearchCoverage,
    SearchDiagnostics,
    SearchNormalizationInfo,
    SearchRequest,
    SearchResult,
    SearchSource,
    Summary,
    ToolDiagnostics,
)
from mcp_searchbridge.server import create_server
from tests.helpers import make_settings, optional_url, url


class FakeBackend:
    provider_name = "fake-provider"
    provider_model = "fake-model"
    backend_kind = "fake"

    def __init__(self) -> None:
        self.search_requests: list[SearchRequest] = []
        self.extract_requests: list[ExtractUrlRequest] = []
        self.outline_requests: list[OutlineUrlRequest] = []
        self.docs_qa_requests: list[DocsQARequest] = []
        self.source_requests: list[DocSourceResolutionRequest] = []

    def search_web(self, request: SearchRequest) -> SearchResult:
        self.search_requests.append(request)
        return SearchResult(
            query=QueryEcho(
                text=request.query,
                recency=request.recency,
                max_sources=request.max_sources,
                domain_allowlist=request.domain_allowlist,
                return_mode=request.return_mode,
            ),
            summary=Summary(
                text=f"Echo: {request.query}",
                citations=[Citation(source_id="source_1", chunk_id="source_1_chunk_1")],
            ),
            sources=[
                SearchSource(
                    source_id="source_1",
                    rank=1,
                    title="Example",
                    url=url("https://example.com/search"),
                    domain="example.com",
                    published_at=None,
                    domain_allowed=True,
                    evidence=[
                        EvidenceChunk(
                            chunk_id="source_1_chunk_1",
                            text="Snippet",
                        )
                    ],
                )
            ],
            diagnostics=SearchDiagnostics(
                status="ok",
                provider=ProviderInfo(
                    name=self.provider_name,
                    model=self.provider_model,
                ),
                backend_kind=self.backend_kind,
                normalization=SearchNormalizationInfo(
                    response_format_requested="json_object",
                    response_format_accepted=True,
                    parse_mode="structured_v2",
                ),
                coverage=SearchCoverage(
                    sources_requested=request.max_sources,
                    sources_returned=1,
                    sources_with_evidence=1,
                    evidence_chunks_returned=1,
                ),
                warnings=[],
            ),
        )

    def extract_url(self, request: ExtractUrlRequest) -> ExtractResult:
        self.extract_requests.append(request)
        return ExtractResult(
            request=ExtractRequestEcho(
                url=request.url,
                mode=request.mode,
                max_chars=request.max_chars,
            ),
            title="Example Page",
            url=request.url,
            content="Example content",
            content_format="text",
            truncated=False,
            likely_rewritten=True,
            diagnostics=ToolDiagnostics(
                status="ok",
                provider=ProviderInfo(
                    name=self.provider_name,
                    model=self.provider_model,
                ),
                backend_kind=self.backend_kind,
                warnings=[],
            ),
        )

    def outline_url(self, request: OutlineUrlRequest) -> OutlineResult:
        self.outline_requests.append(request)
        return OutlineResult(
            request=OutlineRequestEcho(url=request.url, depth=request.depth),
            title="Example Outline",
            sections=[OutlineSection(title="Section A", summary="Summary A")],
            diagnostics=ToolDiagnostics(
                status="ok",
                provider=ProviderInfo(
                    name=self.provider_name,
                    model=self.provider_model,
                ),
                backend_kind=self.backend_kind,
                warnings=[],
            ),
        )

    def docs_qa(self, request: DocsQARequest) -> DocsQAResult:
        self.docs_qa_requests.append(request)
        return DocsQAResult(
            request=DocsQARequestEcho(
                question=request.question,
                url=request.url,
                domain_allowlist=request.domain_allowlist,
                answer_mode=request.answer_mode,
            ),
            answer="Docs answer",
            citations=[Citation(source_id="source_1", chunk_id="source_1_chunk_1")],
            sources=[
                SearchSource(
                    source_id="source_1",
                    rank=1,
                    title="Docs Source",
                    url=url("https://example.com/docs"),
                    domain="example.com",
                    published_at=None,
                    domain_allowed=True,
                    evidence=[
                        EvidenceChunk(
                            chunk_id="source_1_chunk_1",
                            text="Docs snippet",
                        )
                    ],
                )
            ],
            diagnostics=ToolDiagnostics(
                status="ok",
                provider=ProviderInfo(
                    name=self.provider_name,
                    model=self.provider_model,
                ),
                backend_kind=self.backend_kind,
                warnings=[],
            ),
        )

    def find_official_docs(self, request: Any) -> OfficialDocsResult:
        return OfficialDocsResult(
            request=OfficialDocsRequestEcho(
                query=request.query,
                max_results=request.max_results,
            ),
            matches=[
                OfficialDocMatch(
                    title="Official Docs",
                    url=url("https://example.com/docs"),
                    domain="example.com",
                    rationale="Canonical site",
                )
            ],
            diagnostics=ToolDiagnostics(
                status="ok",
                provider=ProviderInfo(
                    name=self.provider_name,
                    model=self.provider_model,
                ),
                backend_kind=self.backend_kind,
                warnings=[],
            ),
        )

    def resolve_doc_source(
        self,
        request: DocSourceResolutionRequest,
    ) -> DocSourceResolutionResult:
        self.source_requests.append(request)
        return DocSourceResolutionResult(
            request=DocSourceResolutionRequestEcho(query_or_url=request.query_or_url),
            source_type="llms_txt",
            resolved_url=optional_url("https://example.com/llms.txt"),
            confidence=0.9,
            rationale="Looks like llms.txt",
            diagnostics=ToolDiagnostics(
                status="ok",
                provider=ProviderInfo(
                    name=self.provider_name,
                    model=self.provider_model,
                ),
                backend_kind=self.backend_kind,
                warnings=[],
            ),
        )


class AsyncCoordinatedBackend:
    provider_name = FakeBackend.provider_name
    provider_model = FakeBackend.provider_model
    backend_kind = FakeBackend.backend_kind

    def __init__(self) -> None:
        self.backend = FakeBackend()
        self.entered_calls = 0
        self.both_calls_entered = asyncio.Event()
        self.release_calls = asyncio.Event()

    async def search_web(self, request: SearchRequest) -> SearchResult:
        self.entered_calls += 1
        if self.entered_calls == 2:
            self.both_calls_entered.set()
        await self.release_calls.wait()
        return self.backend.search_web(request)

    def extract_url(self, request: ExtractUrlRequest) -> ExtractResult:
        return self.backend.extract_url(request)

    def outline_url(self, request: OutlineUrlRequest) -> OutlineResult:
        return self.backend.outline_url(request)

    def docs_qa(self, request: DocsQARequest) -> DocsQAResult:
        return self.backend.docs_qa(request)

    def find_official_docs(self, request: Any) -> OfficialDocsResult:
        return self.backend.find_official_docs(request)

    def resolve_doc_source(
        self,
        request: DocSourceResolutionRequest,
    ) -> DocSourceResolutionResult:
        return self.backend.resolve_doc_source(request)


class FailingBackend(FakeBackend):
    def __init__(self, exc: UpstreamSearchError) -> None:
        super().__init__()
        self.exc = exc

    def search_web(self, request: SearchRequest) -> SearchResult:
        raise self.exc

    def extract_url(self, request: ExtractUrlRequest) -> ExtractResult:
        raise self.exc

    def outline_url(self, request: OutlineUrlRequest) -> OutlineResult:
        raise self.exc

    def docs_qa(self, request: DocsQARequest) -> DocsQAResult:
        raise self.exc

    def find_official_docs(self, request: Any) -> OfficialDocsResult:
        raise self.exc

    def resolve_doc_source(
        self,
        request: DocSourceResolutionRequest,
    ) -> DocSourceResolutionResult:
        raise self.exc


def test_create_server_registers_tools() -> None:
    settings = make_settings(OPENAI_MODEL="fake-model")
    backend = FakeBackend()

    server = create_server(settings=settings, backend=backend)

    tools = server._tool_manager.list_tools()
    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "search_web",
        "extract_url",
        "outline_url",
        "docs_qa",
        "find_official_docs",
        "resolve_doc_source",
    }


def test_search_tool_calls_can_overlap() -> None:
    settings = make_settings(OPENAI_MODEL="fake-model")
    backend = AsyncCoordinatedBackend()
    server = create_server(settings=settings, backend=backend)

    async def run_calls() -> None:
        calls = [
            asyncio.create_task(
                server._tool_manager.call_tool(
                    "search_web",
                    {"query": f"parallel request {index}"},
                )
            )
            for index in range(2)
        ]
        await asyncio.wait_for(backend.both_calls_entered.wait(), timeout=1)
        assert backend.entered_calls == 2
        backend.release_calls.set()
        results = await asyncio.gather(*calls)
        assert [result.query.text for result in results] == [
            "parallel request 0",
            "parallel request 1",
        ]

    asyncio.run(run_calls())


def test_server_returns_sanitized_upstream_errors_for_all_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = make_settings(OPENAI_MODEL="fake-model")
    exc = UpstreamSearchError(
        "Could not connect to the upstream provider.",
        retryable=True,
        log_context=UpstreamLogContext(
            error_type="APIConnectionError",
            status_code=None,
            request_id="req_123",
        ),
    )
    backend = FailingBackend(exc)
    server = create_server(settings=settings, backend=backend)
    caplog.set_level(logging.WARNING)

    async def run_calls() -> None:
        search_result = await server._tool_manager.call_tool(
            "search_web",
            {
                "query": "latest status",
                "max_sources": 3,
                "domain_allowlist": ["example.com"],
                "return_mode": "standard",
            },
        )
        assert search_result.diagnostics.error is not None
        assert (
            search_result.diagnostics.error.message
            == "Could not connect to the upstream provider."
        )
        assert search_result.diagnostics.error.retryable is True

        extract_result = await server._tool_manager.call_tool(
            "extract_url",
            {
                "url": "https://example.com/search",
                "mode": "best_effort",
                "max_chars": 1200,
            },
        )
        assert extract_result.diagnostics.error is not None
        assert (
            extract_result.diagnostics.error.message
            == "Could not connect to the upstream provider."
        )

        outline_result = await server._tool_manager.call_tool(
            "outline_url",
            {
                "url": "https://example.com/docs",
                "depth": "standard",
            },
        )
        assert outline_result.diagnostics.error is not None
        assert (
            outline_result.diagnostics.error.message
            == "Could not connect to the upstream provider."
        )

        docs_result = await server._tool_manager.call_tool(
            "docs_qa",
            {
                "question": "How does this work?",
                "url": "https://example.com/docs",
                "domain_allowlist": ["example.com"],
                "answer_mode": "standard",
            },
        )
        assert docs_result.diagnostics.error is not None
        assert docs_result.diagnostics.error.message == (
            "Could not connect to the upstream provider."
        )

        official_docs_result = await server._tool_manager.call_tool(
            "find_official_docs",
            {
                "query": "example docs",
                "max_results": 3,
            },
        )
        assert official_docs_result.diagnostics.error is not None
        assert official_docs_result.diagnostics.error.message == (
            "Could not connect to the upstream provider."
        )

        source_result = await server._tool_manager.call_tool(
            "resolve_doc_source",
            {
                "query_or_url": "https://example.com/llms.txt",
            },
        )
        assert source_result.diagnostics.error is not None
        assert source_result.diagnostics.error.message == (
            "Could not connect to the upstream provider."
        )

    asyncio.run(run_calls())

    log_text = caplog.text
    assert "192.168.5.1" not in log_text
    assert "http://" not in log_text
    assert "Could not connect to the upstream provider." not in log_text
    assert "APIConnectionError" in log_text


def test_server_preserves_structured_upstream_error_code() -> None:
    settings = make_settings(OPENAI_MODEL="fake-model")
    exc = UpstreamSearchError(
        "Private backend auth failed.",
        retryable=False,
        log_context=UpstreamLogContext(
            error_type="PrivateBackendErrorResponse",
            status_code=401,
            request_id="req_private",
        ),
        error_code="backend_auth_failed",
    )
    backend = FailingBackend(exc)
    server = create_server(settings=settings, backend=backend)

    async def run_call() -> None:
        result = await server._tool_manager.call_tool(
            "search_web",
            {
                "query": "private auth failure",
            },
        )

        assert result.diagnostics.error is not None
        assert result.diagnostics.error.code == "backend_auth_failed"
        assert result.diagnostics.error.message == "Private backend auth failed."
        assert result.diagnostics.error.retryable is False

    asyncio.run(run_call())


def test_server_includes_fallback_attempt_metadata_in_error_results() -> None:
    settings = make_settings(OPENAI_MODEL="fake-model,fallback-model")
    exc = UpstreamSearchError(
        "The upstream provider rate-limited the request.",
        retryable=True,
        log_context=UpstreamLogContext(
            error_type="RateLimitError",
            status_code=429,
            request_id="req_rate_limit",
        ),
        attempted_models=["fake-model", "fallback-model"],
        final_model="fallback-model",
        fallback_trigger="rate_limited",
    )
    backend = FailingBackend(exc)
    server = create_server(settings=settings, backend=backend)

    async def run_call() -> None:
        result = await server._tool_manager.call_tool(
            "search_web",
            {
                "query": "latest status",
            },
        )

        assert result.diagnostics.provider.model == "fallback-model"
        assert result.diagnostics.attempted_models == [
            "fake-model",
            "fallback-model",
        ]
        assert result.diagnostics.fallback_count == 1
        assert result.diagnostics.fallback_trigger == "rate_limited"

    asyncio.run(run_call())
