from __future__ import annotations

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


class _MCPFakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        content = json.dumps(
            {
                "answer": "Smoke test answer",
                "sources": [
                    {
                        "title": "Smoke Source",
                        "url": "https://example.com/smoke",
                        "snippet": "Smoke snippet",
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


def test_mcp_stdio_tools_list_and_call() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MCPFakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
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
                assert any(tool.name == "web_search" for tool in tools.tools)

                result = await session.call_tool(
                    "web_search",
                    {
                        "query": "smoke test query",
                        "recency": "latest",
                        "max_sources": 2,
                        "domain_allowlist": ["example.com"],
                        "return_mode": "standard",
                    },
                )

                assert not result.isError
                assert result.structuredContent is not None
                structured = result.structuredContent
                assert structured["answer"] == "Smoke test answer"
                assert structured["sources"][0]["url"] == "https://example.com/smoke"
                assert structured["model"] == "smoke-model"

        asyncio.run(run_smoke())
    finally:
        server.shutdown()
        server.server_close()


def test_mcp_stdio_invalid_request_returns_structured_error() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MCPFakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        host, port = server.server_address
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
                    "web_search",
                    {
                        "query": "invalid request",
                        "max_sources": 0,
                    },
                )

                assert not result.isError
                assert result.structuredContent is not None
                structured = result.structuredContent
                assert structured["warnings"] == ["invalid_request"]
                assert structured["answer"].startswith("Invalid search request:")

        asyncio.run(run_invalid())
    finally:
        server.shutdown()
        server.server_close()
