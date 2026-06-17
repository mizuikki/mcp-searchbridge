from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from tests.helpers import host_port


class _MCPFakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        user_prompt = payload["messages"][-1]["content"]

        if '"content": "page body or markdown"' in user_prompt:
            content = json.dumps(
                {
                    "title": "Smoke Page",
                    "url": "https://example.com/smoke",
                    "content": "Smoke body",
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
                        "text": "Smoke test answer",
                        "citations": [
                            {"source_id": "source_1", "chunk_id": "source_1_chunk_1"}
                        ],
                    },
                    "sources": [
                        {
                            "source_id": "source_1",
                            "title": "Smoke Source",
                            "url": "https://example.com/smoke",
                            "published_at": "2026-06-15",
                            "evidence": [
                                {
                                    "chunk_id": "source_1_chunk_1",
                                    "text": "Smoke snippet",
                                }
                            ],
                        }
                    ],
                    "warnings": [],
                }
            )

        response = {
            "id": "chatcmpl-smoke",
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


class _MCPPrivateErrorHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
            json.dumps(
                {
                    "error": {
                        "code": "backend_auth_failed",
                        "message": "Private backend auth failed.",
                        "retryable": False,
                    }
                }
            ).encode("utf-8")
        )

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


def test_mcp_stdio_tools_list_and_call() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MCPFakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        env = os.environ.copy()
        env.update(
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": f"http://{host}:{port}/v1",
                "OPENAI_MODEL": "smoke-model",
            }
        )

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "mcp-searchbridge"],
            cwd=os.getcwd(),
            env=env,
        )

        async def run_smoke() -> None:
            async with (
                stdio_client(server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()

                tools = await session.list_tools()
                tool_names = {tool.name for tool in tools.tools}
                assert tool_names == {
                    "search_web",
                    "extract_url",
                    "outline_url",
                    "docs_qa",
                    "find_official_docs",
                    "resolve_doc_source",
                }

                search_result = await session.call_tool(
                    "search_web",
                    {
                        "query": "smoke test query",
                        "recency": "latest",
                        "max_sources": 2,
                        "domain_allowlist": ["example.com"],
                        "return_mode": "standard",
                    },
                )
                assert not search_result.isError
                assert search_result.structuredContent is not None
                structured = search_result.structuredContent
                assert structured["summary"]["text"] == "Smoke test answer"
                assert structured["sources"][0]["url"] == "https://example.com/smoke"
                assert structured["diagnostics"]["provider"]["model"] == "smoke-model"
                assert structured["diagnostics"]["backend_kind"] == "openai"

                extract_result = await session.call_tool(
                    "extract_url",
                    {
                        "url": "https://example.com/smoke",
                        "mode": "best_effort",
                        "max_chars": 1000,
                    },
                )
                assert not extract_result.isError
                assert extract_result.structuredContent is not None
                extracted = extract_result.structuredContent
                assert extracted["title"] == "Smoke Page"
                assert extracted["content"] == "Smoke body"

        asyncio.run(run_smoke())
    finally:
        server.shutdown()
        server.server_close()


def test_mcp_stdio_invalid_request_returns_structured_error() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MCPFakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        env = os.environ.copy()
        env.update(
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": f"http://{host}:{port}/v1",
                "OPENAI_MODEL": "smoke-model",
            }
        )

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "mcp-searchbridge"],
            cwd=os.getcwd(),
            env=env,
        )

        async def run_invalid() -> None:
            async with (
                stdio_client(server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "search_web",
                    {
                        "query": "invalid request",
                        "max_sources": 0,
                    },
                )

                assert not result.isError
                assert result.structuredContent is not None
                structured = result.structuredContent
                assert structured["diagnostics"]["status"] == "error"
                assert structured["diagnostics"]["error"]["code"] == "invalid_request"
                assert structured["diagnostics"]["error"]["message"].startswith(
                    "Invalid search request:"
                )

        asyncio.run(run_invalid())
    finally:
        server.shutdown()
        server.server_close()


def test_mcp_stdio_private_backend_preserves_structured_error_code() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MCPPrivateErrorHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = host_port(server.server_address)
        env = os.environ.copy()
        env.update(
            {
                "SEARCHBRIDGE_BACKEND_KIND": "private_http",
                "SEARCHBRIDGE_PRIVATE_BACKEND_URL": f"http://{host}:{port}",
            }
        )
        env.pop("OPENAI_API_KEY", None)
        env.pop("OPENAI_BASE_URL", None)
        env.pop("OPENAI_MODEL", None)

        server_params = StdioServerParameters(
            command="uv",
            args=["run", "mcp-searchbridge"],
            cwd=os.getcwd(),
            env=env,
        )

        async def run_private_error() -> None:
            async with (
                stdio_client(server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "search_web",
                    {
                        "query": "private auth failure",
                    },
                )

                assert not result.isError
                assert result.structuredContent is not None
                structured = result.structuredContent
                assert structured["diagnostics"]["status"] == "error"
                assert (
                    structured["diagnostics"]["error"]["code"] == "backend_auth_failed"
                )
                assert (
                    structured["diagnostics"]["error"]["message"]
                    == "Private backend auth failed."
                )
                assert structured["diagnostics"]["error"]["retryable"] is False
                assert structured["diagnostics"]["backend_kind"] == "private_http"

        asyncio.run(run_private_error())
    finally:
        server.shutdown()
        server.server_close()
