from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mcp_searchbridge.config import Settings
from mcp_searchbridge.errors import UpstreamSearchError
from mcp_searchbridge.models import ExtractUrlRequest, SearchRequest
from mcp_searchbridge.openai_backend import OpenAIAggregationBackend


class _ChatCompletionsHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response_format = payload.get("response_format")
        user_prompt = payload["messages"][-1]["content"]

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if response_format == {"type": "json_object"}:
            if '"content": "page body or markdown"' in user_prompt:
                content = json.dumps(
                    {
                        "title": "Extracted Page",
                        "url": "https://example.com/page",
                        "content": "Page body",
                        "content_format": "text",
                        "truncated": False,
                        "likely_rewritten": True,
                        "warnings": [],
                    }
                )
            else:
                content = json.dumps(
                    {
                        "summary": {
                            "text": "Current status from fake endpoint",
                            "citations": [
                                {
                                    "source_id": "source_1",
                                    "chunk_id": "source_1_chunk_1",
                                }
                            ],
                        },
                        "sources": [
                            {
                                "source_id": "source_1",
                                "title": "Example Source",
                                "url": "https://example.com/status",
                                "published_at": "2026-06-15",
                                "evidence": [
                                    {
                                        "chunk_id": "source_1_chunk_1",
                                        "text": "Status snippet",
                                    }
                                ],
                            }
                        ],
                        "warnings": [],
                    }
                )
        else:
            content = (
                "Fallback answer\n\n"
                "Sources:\n"
                "- Example Source - https://example.com/fallback - Fallback snippet"
            )

        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _NotFoundExtractionHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response = {
            "id": "chatcmpl-404",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "404 - Page not found | Pydantic",
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _NotFoundWarningAliasHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response = {
            "id": "chatcmpl-404-alias",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "title": "404 - Page not found | Example",
                                "url": "https://example.com/missing",
                                "content": "404 - Page not found",
                                "content_format": "text",
                                "truncated": False,
                                "likely_rewritten": False,
                                "warnings": ["404_page", "page_not_found"],
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _FallbackChatCompletionsHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        type(self).request_count += 1
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        if payload.get("response_format") == {"type": "json_object"}:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "error": {
                            "message": "response_format not supported",
                            "type": "invalid_request_error",
                        }
                    }
                ).encode("utf-8")
            )
            return

        response = {
            "id": "chatcmpl-fallback",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Fallback answer\n\n"
                            "Sources:\n"
                            "- Example Source - https://example.com/fallback "
                            "- Fallback snippet"
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _GenericBadRequestHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        type(self).request_count += 1
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "error": {
                        "message": f"model '{payload['model']}' is invalid",
                        "type": "invalid_request_error",
                    }
                }
            ).encode("utf-8")
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_backend_search_against_fake_openai_endpoint() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = backend.search_web(
            SearchRequest(
                query="latest status",
                recency="latest",
                max_sources=3,
                domain_allowlist=["example.com"],
            )
        )

        assert result.summary.text == "Current status from fake endpoint"
        assert len(result.sources) == 1
        assert str(result.sources[0].url) == "https://example.com/status"
        assert result.sources[0].published_at == "2026-06-15"
        assert result.diagnostics.provider.name == "openai-compatible"
        assert result.diagnostics.provider.model == "fake-search-model"
        assert result.diagnostics.normalization.parse_mode == "structured_v2"
    finally:
        server.shutdown()
        server.server_close()


def test_backend_extract_against_fake_openai_endpoint() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = backend.extract_url(
            ExtractUrlRequest(
                url="https://example.com/page",
                mode="best_effort",
                max_chars=1000,
            )
        )

        assert result.title == "Extracted Page"
        assert str(result.url) == "https://example.com/page"
        assert result.content == "Page body"
        assert result.content_format == "text"
        assert result.diagnostics.status == "ok"
    finally:
        server.shutdown()
        server.server_close()


def test_backend_falls_back_when_structured_output_is_rejected() -> None:
    _FallbackChatCompletionsHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FallbackChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = backend.search_web(SearchRequest(query="fallback test"))

        assert result.summary.text == "Fallback answer"
        assert len(result.sources) == 1
        warning_codes = [warning.code for warning in result.diagnostics.warnings]
        assert "structured_output_not_supported" in warning_codes
        assert "text_fallback_used" in warning_codes
        assert result.diagnostics.normalization.response_format_accepted is False
        assert _FallbackChatCompletionsHandler.request_count == 2
    finally:
        server.shutdown()
        server.server_close()


def test_backend_skips_structured_output_after_capability_is_cached() -> None:
    _FallbackChatCompletionsHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FallbackChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        first = backend.search_web(SearchRequest(query="first fallback test"))
        second = backend.search_web(SearchRequest(query="second fallback test"))

        assert first.summary.text == "Fallback answer"
        assert second.summary.text == "Fallback answer"
        assert _FallbackChatCompletionsHandler.request_count == 3
        assert first.diagnostics.normalization.response_format_accepted is False
        assert second.diagnostics.normalization.response_format_accepted is False
    finally:
        server.shutdown()
        server.server_close()


def test_backend_does_not_fallback_for_unrelated_bad_request() -> None:
    _GenericBadRequestHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GenericBadRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        with pytest.raises(UpstreamSearchError, match="HTTP 400"):
            backend.search_web(SearchRequest(query="bad request test"))

        assert _GenericBadRequestHandler.request_count == 1
    finally:
        server.shutdown()
        server.server_close()


def test_backend_marks_404_like_pages_as_empty_or_partial() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundExtractionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        extract_result = backend.extract_url(
            ExtractUrlRequest(
                url="https://pydantic.dev/this-page-should-not-exist",
                mode="best_effort",
                max_chars=1200,
            )
        )

        assert extract_result.diagnostics.status == "empty"
        assert "404" in extract_result.content
        warning_codes = [
            warning.code for warning in extract_result.diagnostics.warnings
        ]
        assert "not_found_page" in warning_codes
    finally:
        server.shutdown()
        server.server_close()


def test_backend_normalizes_404_warning_aliases() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundWarningAliasHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
        settings = Settings(
            _env_file=None,
            OPENAI_API_KEY="test-key",
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = backend.extract_url(
            ExtractUrlRequest(
                url="https://example.com/missing",
                mode="best_effort",
                max_chars=500,
            )
        )

        assert result.diagnostics.status == "empty"
        warning_codes = [warning.code for warning in result.diagnostics.warnings]
        assert "not_found_page" in warning_codes
        assert "404_page" not in warning_codes
        assert "page_not_found" not in warning_codes
    finally:
        server.shutdown()
        server.server_close()
