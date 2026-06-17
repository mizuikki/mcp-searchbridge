from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from mcp_searchbridge.errors import UpstreamSearchError
from mcp_searchbridge.models import (
    Citation,
    EvidenceChunk,
    ProviderInfo,
    QueryEcho,
    SearchCoverage,
    SearchDiagnostics,
    SearchNormalizationInfo,
    SearchRequest,
    SearchResult,
    SearchSource,
    Summary,
)
from mcp_searchbridge.private_backend import PrivateHttpAggregationBackend
from tests.helpers import host_port, make_settings, url


class _PrivateSearchHandler(BaseHTTPRequestHandler):
    last_path = ""
    last_auth = None
    last_body: dict[str, object] | None = None

    def do_POST(self) -> None:  # noqa: N802
        type(self).last_path = self.path
        type(self).last_auth = self.headers.get("Authorization")
        length = int(self.headers["Content-Length"])
        type(self).last_body = json.loads(self.rfile.read(length).decode("utf-8"))

        if self.path != "/v1/search_web":
            self.send_response(404)
            self.end_headers()
            return

        response = {
            "query": {
                "text": "private search",
                "recency": "latest",
                "max_sources": 2,
                "domain_allowlist": ["example.com"],
                "return_mode": "standard",
            },
            "summary": {
                "text": "Private backend answer",
                "citations": [
                    {"source_id": "source_1", "chunk_id": "source_1_chunk_1"}
                ],
            },
            "sources": [
                {
                    "source_id": "source_1",
                    "rank": 1,
                    "title": "Example Source",
                    "url": "https://example.com/private",
                    "domain": "example.com",
                    "published_at": "2026-06-16",
                    "domain_allowed": True,
                    "evidence": [
                        {
                            "chunk_id": "source_1_chunk_1",
                            "text": "Private evidence",
                        }
                    ],
                }
            ],
            "diagnostics": {
                "status": "ok",
                "provider": {
                    "name": "private-http",
                    "model": "private-backend",
                },
                "normalization": {
                    "response_format_requested": "json_object",
                    "response_format_accepted": True,
                    "parse_mode": "structured_v2",
                },
                "coverage": {
                    "sources_requested": 2,
                    "sources_returned": 1,
                    "sources_with_evidence": 1,
                    "evidence_chunks_returned": 1,
                },
                "warnings": [],
                "capabilities_used": ["private_http", "ranking"],
            },
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _StructuredPrivateErrorHandler(BaseHTTPRequestHandler):
    status_code = 502
    error_body = {
        "error": {
            "code": "backend_request_failed",
            "message": "Private backend upstream failed.",
            "retryable": True,
        }
    }

    def do_POST(self) -> None:  # noqa: N802
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(type(self).error_body).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _InvalidJsonHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{not-json")

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _InvalidResponseDataHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "query": {
                        "text": "private search",
                        "recency": "latest",
                        "max_sources": 2,
                        "domain_allowlist": ["example.com"],
                        "return_mode": "standard",
                    },
                    "summary": {
                        "text": "Missing diagnostics should fail validation",
                        "citations": [],
                    },
                    "sources": [],
                }
            ).encode("utf-8")
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _FallbackBackend:
    provider_name = "openai-compatible"
    provider_model = "fallback-model"
    backend_kind = "openai"

    def __init__(self) -> None:
        self.calls: list[SearchRequest] = []

    def search_web(self, request: SearchRequest) -> SearchResult:
        self.calls.append(request)
        return SearchResult(
            query=QueryEcho(
                text=request.query,
                recency=request.recency,
                max_sources=request.max_sources,
                domain_allowlist=request.domain_allowlist,
                return_mode=request.return_mode,
            ),
            summary=Summary(
                text="Fallback answer",
                citations=[Citation(source_id="source_1", chunk_id="source_1_chunk_1")],
            ),
            sources=[
                SearchSource(
                    source_id="source_1",
                    rank=1,
                    title="Fallback Source",
                    url=url("https://example.com/fallback"),
                    domain="example.com",
                    evidence=[
                        EvidenceChunk(
                            chunk_id="source_1_chunk_1",
                            text="Fallback evidence",
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


def _start_server(
    handler: type[BaseHTTPRequestHandler],
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _make_private_settings(port: int, **overrides: object):
    return make_settings(
        SEARCHBRIDGE_BACKEND_KIND="private_http",
        SEARCHBRIDGE_PRIVATE_BACKEND_URL=f"http://127.0.0.1:{port}",
        **overrides,
    )


def test_private_backend_posts_to_expected_endpoint_and_maps_response() -> None:
    server, _thread = _start_server(_PrivateSearchHandler)

    try:
        _host, port = host_port(server.server_address)
        settings = _make_private_settings(
            port,
            SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY="private-key",
            SEARCHBRIDGE_PRIVATE_BACKEND_TIMEOUT_SECONDS=12,
        )
        backend = PrivateHttpAggregationBackend(settings)

        assert isinstance(backend.client.timeout, httpx.Timeout)
        assert backend.client.timeout.read == 12

        result = backend.search_web(
            SearchRequest(
                query="private search",
                recency="latest",
                max_sources=2,
                domain_allowlist=["example.com"],
            )
        )

        assert _PrivateSearchHandler.last_path == "/v1/search_web"
        assert _PrivateSearchHandler.last_auth == "Bearer private-key"
        assert _PrivateSearchHandler.last_body == {
            "query": "private search",
            "recency": "latest",
            "max_sources": 2,
            "domain_allowlist": ["example.com"],
            "return_mode": "standard",
        }
        assert result.summary.text == "Private backend answer"
        assert result.diagnostics.provider.name == "private-http"
        assert result.diagnostics.provider.model == "private-backend"
        assert result.diagnostics.backend_kind == "private_http"
        assert result.diagnostics.capabilities_used == ["private_http", "ranking"]
    finally:
        backend.client.close()
        server.shutdown()
        server.server_close()


def test_private_backend_preserves_structured_error_body() -> None:
    server, _thread = _start_server(_StructuredPrivateErrorHandler)

    try:
        _host, port = host_port(server.server_address)
        backend = PrivateHttpAggregationBackend(_make_private_settings(port))

        with pytest.raises(UpstreamSearchError) as exc_info:
            backend.search_web(SearchRequest(query="private search"))

        exc = exc_info.value
        assert exc.client_message == "Private backend upstream failed."
        assert exc.error_code == "backend_request_failed"
        assert exc.retryable is True
        assert exc.allow_fallback is False
        assert exc.log_context.status_code == 502
        assert exc.log_context.error_type == "PrivateBackendErrorResponse"
    finally:
        backend.client.close()
        server.shutdown()
        server.server_close()


def test_private_backend_does_not_fallback_on_auth_error() -> None:
    class _AuthErrorHandler(_StructuredPrivateErrorHandler):
        status_code = 401
        error_body = {
            "error": {
                "code": "backend_auth_failed",
                "message": "Private backend auth failed.",
                "retryable": False,
            }
        }

    server, _thread = _start_server(_AuthErrorHandler)
    backend: PrivateHttpAggregationBackend | None = None

    try:
        _host, port = host_port(server.server_address)
        fallback_backend = _FallbackBackend()
        settings = _make_private_settings(
            port,
            SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=True,
        )
        backend = PrivateHttpAggregationBackend(
            settings,
            fallback_backend=fallback_backend,
        )

        with pytest.raises(UpstreamSearchError, match="Private backend auth failed."):
            backend.search_web(SearchRequest(query="private search"))

        assert fallback_backend.calls == []
    finally:
        if backend is not None:
            backend.client.close()
        server.shutdown()
        server.server_close()


def test_private_backend_does_not_fallback_on_invalid_json() -> None:
    server, _thread = _start_server(_InvalidJsonHandler)
    backend: PrivateHttpAggregationBackend | None = None

    try:
        _host, port = host_port(server.server_address)
        fallback_backend = _FallbackBackend()
        settings = _make_private_settings(
            port,
            SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=True,
        )
        backend = PrivateHttpAggregationBackend(
            settings,
            fallback_backend=fallback_backend,
        )

        with pytest.raises(
            UpstreamSearchError,
            match="Private backend returned invalid JSON.",
        ):
            backend.search_web(SearchRequest(query="private search"))

        assert fallback_backend.calls == []
    finally:
        if backend is not None:
            backend.client.close()
        server.shutdown()
        server.server_close()


def test_private_backend_does_not_fallback_on_response_model_validation_failure() -> None:
    server, _thread = _start_server(_InvalidResponseDataHandler)
    backend: PrivateHttpAggregationBackend | None = None

    try:
        _host, port = host_port(server.server_address)
        fallback_backend = _FallbackBackend()
        settings = _make_private_settings(
            port,
            SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=True,
        )
        backend = PrivateHttpAggregationBackend(
            settings,
            fallback_backend=fallback_backend,
        )

        with pytest.raises(
            UpstreamSearchError,
            match="Private backend returned invalid response data.",
        ) as exc_info:
            backend.search_web(SearchRequest(query="private search"))

        exc = exc_info.value
        assert exc.error_code == "invalid_private_backend_response_data"
        assert exc.retryable is False
        assert exc.allow_fallback is False
        assert fallback_backend.calls == []
    finally:
        if backend is not None:
            backend.client.close()
        server.shutdown()
        server.server_close()


def test_private_backend_incomplete_structured_error_body_falls_back_to_generic_http_error() -> None:
    class _IncompleteStructuredErrorHandler(_StructuredPrivateErrorHandler):
        status_code = 502
        error_body = {
            "error": {
                "code": "backend_request_failed",
            }
        }

    server, _thread = _start_server(_IncompleteStructuredErrorHandler)

    try:
        _host, port = host_port(server.server_address)
        backend = PrivateHttpAggregationBackend(_make_private_settings(port))

        with pytest.raises(
            UpstreamSearchError,
            match="Private backend returned HTTP 502.",
        ) as exc_info:
            backend.search_web(SearchRequest(query="private search"))

        exc = exc_info.value
        assert exc.error_code is None
        assert exc.retryable is True
        assert exc.allow_fallback is False
        assert exc.log_context.error_type == "HTTPStatusError"
    finally:
        backend.client.close()
        server.shutdown()
        server.server_close()


def test_private_backend_falls_back_on_retryable_5xx_transport_error() -> None:
    server, _thread = _start_server(_StructuredPrivateErrorHandler)
    backend: PrivateHttpAggregationBackend | None = None

    try:
        _host, port = host_port(server.server_address)
        fallback_backend = _FallbackBackend()
        settings = _make_private_settings(
            port,
            SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=True,
        )
        backend = PrivateHttpAggregationBackend(
            settings,
            fallback_backend=fallback_backend,
        )

        result = backend.search_web(SearchRequest(query="private search"))

        assert result.summary.text == "Fallback answer"
        assert len(fallback_backend.calls) == 1
        assert result.diagnostics.provider.model == "fallback-model"
        assert result.diagnostics.backend_kind == "openai"
    finally:
        if backend is not None:
            backend.client.close()
        server.shutdown()
        server.server_close()


def test_private_backend_falls_back_on_not_implemented_contract() -> None:
    class _NotImplementedHandler(_StructuredPrivateErrorHandler):
        status_code = 400
        error_body = {
            "error": {
                "code": "not_implemented",
                "message": "Endpoint not implemented.",
                "retryable": False,
            }
        }

    server, _thread = _start_server(_NotImplementedHandler)
    backend: PrivateHttpAggregationBackend | None = None

    try:
        _host, port = host_port(server.server_address)
        fallback_backend = _FallbackBackend()
        settings = _make_private_settings(
            port,
            SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=True,
        )
        backend = PrivateHttpAggregationBackend(
            settings,
            fallback_backend=fallback_backend,
        )

        result = backend.search_web(SearchRequest(query="private search"))

        assert result.summary.text == "Fallback answer"
        assert len(fallback_backend.calls) == 1
    finally:
        if backend is not None:
            backend.client.close()
        server.shutdown()
        server.server_close()
