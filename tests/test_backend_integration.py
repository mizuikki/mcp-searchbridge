from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import cast

import httpx
import openai
import pytest

import mcp_searchbridge.openai_backend as openai_backend_module
from mcp_searchbridge.errors import UpstreamSearchError
from mcp_searchbridge.models import (
    ConversationContinueRequest,
    ConversationGetRequest,
    ConversationStartRequest,
    ExtractUrlRequest,
    FindOfficialDocsRequest,
    SearchRequest,
)
from mcp_searchbridge.openai_backend import ChatNamespace, OpenAIAggregationBackend
from tests.helpers import host_port, make_settings, url

pytestmark = pytest.mark.asyncio


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


class _ConversationChatHandler(BaseHTTPRequestHandler):
    request_payloads: list[dict[str, object]] = []

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).request_payloads.append(payload)
        messages = payload["messages"]
        user_messages = [
            item["content"]
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        assistant_messages = [
            item["content"]
            for item in messages
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        first_user = user_messages[0]
        marker = first_user.split("Remember this token for the next round: ", 1)[1]
        marker = marker.split(".", 1)[0]

        if len(user_messages) == 1:
            content = marker
        else:
            if marker in assistant_messages[-1]:
                content = marker
            else:
                content = "No token was mentioned earlier."

        response = {
            "id": "chatcmpl-conversation",
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
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class _NoisyConversationChatHandler(BaseHTTPRequestHandler):
    request_payloads: list[dict[str, object]] = []

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).request_payloads.append(payload)
        messages = payload["messages"]
        user_messages = [
            item["content"]
            for item in messages
            if isinstance(item, dict) and item.get("role") == "user"
        ]
        assistant_messages = [
            item["content"]
            for item in messages
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        first_user = user_messages[0]
        marker = first_user.split("Remember this token for the next round: ", 1)[1]
        marker = marker.split(".", 1)[0]

        if len(user_messages) == 1:
            content = (
                f"{marker}\n\nThe query explicitly states to reply with the exact "
                "token."
            )
        else:
            content = marker if marker in assistant_messages[-1] else "missing"

        response = {
            "id": "chatcmpl-conversation-noisy",
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


class _ModelFallbackHandler(BaseHTTPRequestHandler):
    attempted_models: list[str] = []

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        model = str(payload["model"])
        type(self).attempted_models.append(model)

        if model == "primary-model":
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "error": {
                            "message": "rate limited",
                            "type": "rate_limit_error",
                        }
                    }
                ).encode("utf-8")
            )
            return

        response = {
            "id": "chatcmpl-model-fallback",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "summary": {
                                    "text": "Recovered via fallback model",
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
                                        "title": "Fallback Source",
                                        "url": "https://example.com/fallback-model",
                                        "published_at": "2026-06-18",
                                        "evidence": [
                                            {
                                                "chunk_id": "source_1_chunk_1",
                                                "text": "Recovered by fallback model",
                                            }
                                        ],
                                    }
                                ],
                                "warnings": [],
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


class _RetryingEmptyContentHandler(BaseHTTPRequestHandler):
    request_count = 0
    empty_attempts_before_success = 1
    empty_mode = "string"

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        type(self).request_count += 1
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))

        if type(self).request_count <= type(self).empty_attempts_before_success:
            message_content = "   " if type(self).empty_mode == "string" else None
        else:
            message_content = json.dumps(
                {
                    "summary": {
                        "text": "Recovered after retry",
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
                            "title": "Recovered Source",
                            "url": "https://example.com/recovered",
                            "published_at": "2026-06-18",
                            "evidence": [
                                {
                                    "chunk_id": "source_1_chunk_1",
                                    "text": "Recovered snippet",
                                }
                            ],
                        }
                    ],
                    "warnings": [],
                }
            )

        response = {
            "id": "chatcmpl-empty-retry",
            "object": "chat.completion",
            "created": 1,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message_content},
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


class _FencedJsonStringClient:
    def __init__(self, content: str) -> None:
        class _ChatCompletions:
            def __init__(self, value: str) -> None:
                self._value = value

            def create(self, **_: object) -> object:
                return self._value

        class _ChatNamespaceImpl:
            def __init__(self, value: str) -> None:
                self.completions = _ChatCompletions(value)

        self.chat = cast(ChatNamespace, _ChatNamespaceImpl(content))


async def _record_sleep(seconds: float, calls: list[float]) -> None:
    calls.append(seconds)


async def test_backend_search_against_fake_openai_endpoint() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = await backend.search_web(
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
        assert result.diagnostics.backend_kind == "openai"
        assert result.diagnostics.normalization.parse_mode == "structured_v2"
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_extract_against_fake_openai_endpoint() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = await backend.extract_url(
            ExtractUrlRequest(
                url=url("https://example.com/page"),
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


async def test_backend_falls_back_when_structured_output_is_rejected() -> None:
    _FallbackChatCompletionsHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FallbackChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = await backend.search_web(SearchRequest(query="fallback test"))

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


async def test_backend_skips_structured_output_after_capability_is_cached() -> None:
    _FallbackChatCompletionsHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FallbackChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        first = await backend.search_web(SearchRequest(query="first fallback test"))
        second = await backend.search_web(SearchRequest(query="second fallback test"))

        assert first.summary.text == "Fallback answer"
        assert second.summary.text == "Fallback answer"
        assert _FallbackChatCompletionsHandler.request_count == 3
        assert first.diagnostics.normalization.response_format_accepted is False
        assert second.diagnostics.normalization.response_format_accepted is False
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_does_not_fallback_for_unrelated_bad_request() -> None:
    _GenericBadRequestHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _GenericBadRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        with pytest.raises(UpstreamSearchError) as exc_info:
            await backend.search_web(SearchRequest(query="bad request test"))

        exc = exc_info.value
        assert exc.client_message == "The upstream provider rejected the request."
        assert exc.retryable is False
        assert exc.log_context.error_type == "BadRequestError"
        assert "http://" not in exc.client_message
        assert "127.0.0.1" not in exc.client_message

        assert _GenericBadRequestHandler.request_count == 1
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_falls_back_to_next_model_on_retryable_error() -> None:
    _ModelFallbackHandler.attempted_models = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ModelFallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="primary-model,fallback-model",
            OPENAI_MAX_RETRIES=0,
        )
        backend = OpenAIAggregationBackend(settings)

        result = await backend.search_web(SearchRequest(query="fallback model test"))

        assert result.summary.text == "Recovered via fallback model"
        assert result.diagnostics.provider.model == "fallback-model"
        assert result.diagnostics.attempted_models == [
            "primary-model",
            "fallback-model",
        ]
        assert result.diagnostics.fallback_count == 1
        assert result.diagnostics.fallback_trigger == "rate_limited"
        warning_codes = [warning.code for warning in result.diagnostics.warnings]
        assert "model_fallback_used" in warning_codes
        assert _ModelFallbackHandler.attempted_models == [
            "primary-model",
            "fallback-model",
        ]
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_connection_errors_are_sanitized() -> None:
    class _FailingChatCompletions:
        def create(self, **_: object) -> object:
            request = httpx.Request(
                "POST",
                "http://192.168.5.1:23000/v1/chat/completions",
            )
            raise openai.APIConnectionError(
                message="dial tcp 192.168.5.1:23000: connect: connection refused",
                request=request,
            )

    class _FailingClient:
        def __init__(self) -> None:
            class _ChatNamespaceImpl:
                completions = _FailingChatCompletions()

            self.chat = cast(ChatNamespace, _ChatNamespaceImpl())

    settings = make_settings(
        OPENAI_BASE_URL="http://192.168.5.1:23000/v1",
        OPENAI_MODEL="fake-search-model",
    )
    backend = OpenAIAggregationBackend(settings, client=_FailingClient())

    with pytest.raises(UpstreamSearchError) as exc_info:
        await backend.search_web(SearchRequest(query="connection test"))

    exc = exc_info.value
    assert exc.client_message == "Could not connect to the upstream provider."
    assert exc.retryable is True
    assert exc.log_context.error_type == "APIConnectionError"
    assert "192.168.5.1" not in exc.client_message
    assert "23000" not in exc.client_message
    assert "http://" not in exc.client_message


async def test_backend_retries_empty_string_responses(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RetryingEmptyContentHandler.request_count = 0
    _RetryingEmptyContentHandler.empty_attempts_before_success = 1
    _RetryingEmptyContentHandler.empty_mode = "string"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetryingEmptyContentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sleep_calls: list[float] = []

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
            OPENAI_MAX_RETRIES=1,
        )
        backend = OpenAIAggregationBackend(settings)
        monkeypatch.setattr(
            openai_backend_module.random,
            "random",
            lambda: 0.0,
        )
        monkeypatch.setattr(
            openai_backend_module.asyncio,
            "sleep",
            lambda seconds: _record_sleep(seconds, sleep_calls),
        )
        caplog.set_level(logging.WARNING)

        result = await backend.search_web(SearchRequest(query="retry empty string"))

        assert result.summary.text == "Recovered after retry"
        assert str(result.sources[0].url) == "https://example.com/recovered"
        assert _RetryingEmptyContentHandler.request_count == 2
        assert sleep_calls == [pytest.approx(0.5)]
        assert "delay_seconds=0.500000" in caplog.text
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_retries_empty_message_content_until_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RetryingEmptyContentHandler.request_count = 0
    _RetryingEmptyContentHandler.empty_attempts_before_success = 3
    _RetryingEmptyContentHandler.empty_mode = "message"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetryingEmptyContentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sleep_calls: list[float] = []

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
            OPENAI_MAX_RETRIES=1,
        )
        backend = OpenAIAggregationBackend(settings)
        monkeypatch.setattr(
            openai_backend_module.random,
            "random",
            lambda: 0.0,
        )
        monkeypatch.setattr(
            openai_backend_module.asyncio,
            "sleep",
            lambda seconds: _record_sleep(seconds, sleep_calls),
        )

        with pytest.raises(UpstreamSearchError) as exc_info:
            await backend.search_web(SearchRequest(query="retry empty message"))

        exc = exc_info.value
        assert exc.client_message == "Upstream response message content was empty."
        assert exc.retryable is True
        assert exc.error_code == "empty_upstream_response"
        assert exc.log_context.error_type == "EmptyMessageContent"
        assert _RetryingEmptyContentHandler.request_count == 2
        assert sleep_calls == [pytest.approx(0.5)]
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_uses_exponential_backoff_for_multiple_empty_response_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RetryingEmptyContentHandler.request_count = 0
    _RetryingEmptyContentHandler.empty_attempts_before_success = 2
    _RetryingEmptyContentHandler.empty_mode = "string"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetryingEmptyContentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sleep_calls: list[float] = []

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
            OPENAI_MAX_RETRIES=2,
        )
        backend = OpenAIAggregationBackend(settings)
        monkeypatch.setattr(
            openai_backend_module.random,
            "random",
            lambda: 0.0,
        )
        monkeypatch.setattr(
            openai_backend_module.asyncio,
            "sleep",
            lambda seconds: _record_sleep(seconds, sleep_calls),
        )

        result = await backend.search_web(
            SearchRequest(query="retry twice before success")
        )

        assert result.summary.text == "Recovered after retry"
        assert _RetryingEmptyContentHandler.request_count == 3
        assert sleep_calls == [pytest.approx(0.5), pytest.approx(1.0)]
    finally:
        server.shutdown()
        server.server_close()


async def test_find_official_docs_parses_fenced_json_from_sse_string_response() -> None:
    response = (
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,'
        '"model":"fake-search-model","choices":[{"index":0,"delta":{"role":'
        '"assistant","content":"**JSON output:**\\n\\n```json\\n{\\n  \\"matches\\": '
        '[\\n    {\\n      \\"title\\": \\"Relations & rollups - Notion Help Center\\",'
        '\\n      \\"url\\": \\"https://www.notion.com/help/relations-and-rollups\\",'
        '\\n      \\"rationale\\": \\"Canonical official Notion Help Center page\\"'
        '\\n    }\\n  ],\\n  \\"warnings\\": []\\n}\\n```\\n\\nThese are '
        'canonical."}}]}\n\n'
        "data: [DONE]\n\n"
    )
    settings = make_settings(
        OPENAI_BASE_URL="http://127.0.0.1:9999/v1",
        OPENAI_MODEL="fake-search-model",
    )
    backend = OpenAIAggregationBackend(
        settings,
        client=_FencedJsonStringClient(response),
    )

    result = await backend.find_official_docs(
        FindOfficialDocsRequest(
            query="Notion official docs databases relations rollups linked views",
            max_results=5,
        )
    )

    assert len(result.matches) == 1
    assert result.matches[0].title == "Relations & rollups - Notion Help Center"
    assert (
        str(result.matches[0].url)
        == "https://www.notion.com/help/relations-and-rollups"
    )
    assert result.diagnostics.status == "ok"


async def test_find_official_docs_logs_non_stream_request_receiving_sse(
    caplog: pytest.LogCaptureFixture,
) -> None:
    response = (
        'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,'
        '"model":"fake-search-model","choices":[{"index":0,"delta":{"role":'
        '"assistant","content":"```json\\n{\\n  \\"matches\\": [\\n    {\\n      '
        '\\"title\\": \\"Official Docs\\",\\n      \\"url\\": '
        '\\"https://example.com/docs\\",\\n      \\"rationale\\": \\"Canonical\\"'
        '\\n    }\\n  ],\\n  \\"warnings\\": []\\n}\\n```"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    settings = make_settings(
        OPENAI_BASE_URL="http://127.0.0.1:9999/v1",
        OPENAI_MODEL="fake-search-model",
    )
    backend = OpenAIAggregationBackend(
        settings,
        client=_FencedJsonStringClient(response),
    )
    caplog.set_level(logging.WARNING)

    result = await backend.find_official_docs(
        FindOfficialDocsRequest(query="official docs", max_results=1)
    )

    assert len(result.matches) == 1
    assert (
        "text/event-stream content for a non-stream chat.completions request"
        in caplog.text
    )


async def test_backend_marks_404_like_pages_as_empty_or_partial() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundExtractionHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        extract_result = await backend.extract_url(
            ExtractUrlRequest(
                url=url("https://pydantic.dev/this-page-should-not-exist"),
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


async def test_backend_normalizes_404_warning_aliases() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NotFoundWarningAliasHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="fake-search-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = await backend.extract_url(
            ExtractUrlRequest(
                url=url("https://example.com/missing"),
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


async def test_backend_conversation_replays_history_for_multi_round_chat() -> None:
    _ConversationChatHandler.request_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _ConversationChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="test-conversation-model",
        )
        backend = OpenAIAggregationBackend(settings)
        first_message = (
            "Remember this token for the next round: marker_backend_roundtrip. "
            "Reply only with the exact token."
        )

        started = await backend.conversation_start(
            ConversationStartRequest(message=first_message)
        )
        continued = await backend.conversation_continue(
            ConversationContinueRequest(
                conversation_id=started.conversation_id,
                message=(
                    "What token did I ask you to remember in the previous round? "
                    "Reply only with that token."
                ),
            )
        )
        current = await backend.conversation_get(
            ConversationGetRequest(conversation_id=started.conversation_id)
        )

        assert started.assistant_message == "marker_backend_roundtrip"
        assert continued.assistant_message == "marker_backend_roundtrip"
        assert len(_ConversationChatHandler.request_payloads) == 2
        second_messages = _ConversationChatHandler.request_payloads[1]["messages"]
        assert second_messages[1:] == [
            {"role": "user", "content": first_message},
            {"role": "assistant", "content": "marker_backend_roundtrip"},
            {
                "role": "user",
                "content": (
                    "What token did I ask you to remember in the previous round? "
                    "Reply only with that token."
                ),
            },
        ]
        assert [item.role for item in current.messages] == [
            "user",
            "assistant",
            "user",
            "assistant",
        ]
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_conversation_preserves_noisy_assistant_output() -> None:
    _NoisyConversationChatHandler.request_payloads = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _NoisyConversationChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="test-conversation-noisy-model",
        )
        backend = OpenAIAggregationBackend(settings)
        first_message = (
            "Remember this token for the next round: marker_grok_noise. "
            "Reply only with the exact token."
        )

        started = await backend.conversation_start(
            ConversationStartRequest(message=first_message)
        )
        continued = await backend.conversation_continue(
            ConversationContinueRequest(
                conversation_id=started.conversation_id,
                message=(
                    "What token did I ask you to remember in the previous round? "
                    "Reply only with that token."
                ),
            )
        )

        assert "marker_grok_noise" in started.assistant_message
        assert continued.assistant_message == "marker_grok_noise"
        second_messages = _NoisyConversationChatHandler.request_payloads[1]["messages"]
        assert (
            "The query explicitly states to reply with the exact token."
            in second_messages[2]["content"]
        )
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_conversation_retries_empty_string_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RetryingEmptyContentHandler.request_count = 0
    _RetryingEmptyContentHandler.empty_attempts_before_success = 1
    _RetryingEmptyContentHandler.empty_mode = "string"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetryingEmptyContentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sleep_calls: list[float] = []

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="test-conversation-model",
            OPENAI_MAX_RETRIES=1,
        )
        backend = OpenAIAggregationBackend(settings)
        monkeypatch.setattr(openai_backend_module.random, "random", lambda: 0.0)
        monkeypatch.setattr(
            openai_backend_module.asyncio,
            "sleep",
            lambda seconds: _record_sleep(seconds, sleep_calls),
        )

        result = await backend.conversation_start(
            ConversationStartRequest(message="retry conversation start")
        )

        assert "Recovered after retry" in result.assistant_message
        assert _RetryingEmptyContentHandler.request_count == 2
        assert sleep_calls == [pytest.approx(0.5)]
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_conversation_retries_empty_message_content_until_exhausted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _RetryingEmptyContentHandler.request_count = 0
    _RetryingEmptyContentHandler.empty_attempts_before_success = 3
    _RetryingEmptyContentHandler.empty_mode = "message"
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RetryingEmptyContentHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    sleep_calls: list[float] = []

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="test-conversation-model",
            OPENAI_MAX_RETRIES=1,
        )
        backend = OpenAIAggregationBackend(settings)
        monkeypatch.setattr(openai_backend_module.random, "random", lambda: 0.0)
        monkeypatch.setattr(
            openai_backend_module.asyncio,
            "sleep",
            lambda seconds: _record_sleep(seconds, sleep_calls),
        )

        with pytest.raises(UpstreamSearchError) as exc_info:
            await backend.conversation_start(
                ConversationStartRequest(message="retry conversation start")
            )

        exc = exc_info.value
        assert exc.client_message == "Upstream response message content was empty."
        assert exc.error_code == "empty_upstream_response"
        assert exc.log_context.error_type == "EmptyMessageContent"
        assert _RetryingEmptyContentHandler.request_count == 2
        assert sleep_calls == [pytest.approx(0.5)]
    finally:
        server.shutdown()
        server.server_close()


async def test_backend_conversation_preserves_structured_output_warning() -> None:
    _FallbackChatCompletionsHandler.request_count = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FallbackChatCompletionsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        settings = make_settings(
            OPENAI_BASE_URL=f"http://{host}:{port}/v1",
            OPENAI_MODEL="test-conversation-model",
        )
        backend = OpenAIAggregationBackend(settings)

        result = await backend.conversation_start(
            ConversationStartRequest(message="structured output fallback")
        )

        assert result.assistant_message.startswith("Fallback answer")
        warning_codes = [warning.code for warning in result.diagnostics.warnings]
        assert "structured_output_not_supported" in warning_codes
        assert _FallbackChatCompletionsHandler.request_count == 2
    finally:
        server.shutdown()
        server.server_close()
