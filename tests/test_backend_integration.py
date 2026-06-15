from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp_searchbridge.config import Settings
from mcp_searchbridge.models import SearchRequest
from mcp_searchbridge.openai_backend import OpenAIChatSearchBackend


class _ChatCompletionsHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        response_format = payload.get("response_format")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        if response_format == {"type": "json_object"}:
            content = json.dumps(
                {
                    "answer": "Current status from fake endpoint",
                    "sources": [
                        {
                            "title": "Example Source",
                            "url": "https://example.com/status",
                            "snippet": "Status snippet",
                        }
                    ],
                    "warnings": [],
                }
            )
        else:
            content = (
                "Current status from fake endpoint\n\n"
                "Sources:\n"
                "- Example Source - https://example.com/status - Status snippet"
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
        backend = OpenAIChatSearchBackend(settings)

        result = backend.search(
            SearchRequest(
                query="latest status",
                recency="latest",
                max_sources=3,
                domain_allowlist=["example.com"],
            )
        )

        assert result.answer == "Current status from fake endpoint"
        assert len(result.sources) == 1
        assert str(result.sources[0].url) == "https://example.com/status"
        assert result.provider == "openai-compatible"
        assert result.model == "fake-search-model"
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
        backend = OpenAIChatSearchBackend(settings)

        result = backend.search(SearchRequest(query="fallback test"))

        assert result.answer == "Fallback answer"
        assert len(result.sources) == 1
        assert "structured_output_not_supported" in result.warnings
        assert "text_fallback_used" in result.warnings
        assert _FallbackChatCompletionsHandler.request_count == 2
    finally:
        server.shutdown()
        server.server_close()
