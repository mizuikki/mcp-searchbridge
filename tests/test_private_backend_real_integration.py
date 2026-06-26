from __future__ import annotations

import asyncio

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client

from tests.helpers import (
    local_mcp_server_params,
    run_local_searchbridge_core_api,
    searchbridge_core_workspace_available,
    searchbridge_core_workspace_unavailable_reason,
)

PRIVATE_TOKEN = "integration-secret-token"

pytestmark = pytest.mark.skipif(
    not searchbridge_core_workspace_available(),
    reason=searchbridge_core_workspace_unavailable_reason(),
)


def test_private_http_real_integration_happy_paths() -> None:
    with run_local_searchbridge_core_api(api_token=PRIVATE_TOKEN) as private_backend:
        assert private_backend["api_process"].cwd.name == "searchbridge-core"
        assert private_backend["api_process"].command[-1] == "searchbridge-core-api"

        server_params = local_mcp_server_params(
            env_overrides={
                "SEARCHBRIDGE_BACKEND_KIND": "private_http",
                "SEARCHBRIDGE_PRIVATE_BACKEND_URL": private_backend["base_url"],
                "SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY": PRIVATE_TOKEN,
            }
        )

        async def run_happy_paths() -> None:
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
                        "query": "latest OpenAI release notes",
                        "recency": "latest",
                        "max_sources": 3,
                        "domain_allowlist": ["openai.com"],
                        "return_mode": "standard",
                    },
                )
                assert not search_result.isError
                search = search_result.structuredContent
                assert search is not None
                assert search["diagnostics"]["backend_kind"] == "private_http"
                assert search["query"]["text"] == "latest OpenAI release notes"

                extract_result = await session.call_tool(
                    "extract_url",
                    {
                        "url": "https://example.com/docs/page",
                        "mode": "markdown",
                        "max_chars": 1200,
                    },
                )
                assert not extract_result.isError
                extract = extract_result.structuredContent
                assert extract is not None
                assert extract["diagnostics"]["backend_kind"] == "private_http"
                assert extract["content_format"] == "markdown"
                assert extract["diagnostics"]["source_id"] == "source_2"
                assert extract["diagnostics"]["retrieval_method"] == "registry_blob"

                outline_result = await session.call_tool(
                    "outline_url",
                    {
                        "url": "https://example.com/llms.txt",
                        "depth": "deep",
                    },
                )
                assert not outline_result.isError
                outline = outline_result.structuredContent
                assert outline is not None
                assert outline["diagnostics"]["backend_kind"] == "private_http"
                assert len(outline["sections"]) >= 1
                assert outline["diagnostics"]["source_id"] == "source_2"

                docs_result = await session.call_tool(
                    "docs_qa",
                    {
                        "question": "How does field_validator work in Pydantic v2?",
                        "url": "https://docs.pydantic.dev/latest/",
                        "domain_allowlist": ["docs.pydantic.dev"],
                        "answer_mode": "standard",
                    },
                )
                assert not docs_result.isError
                docs = docs_result.structuredContent
                assert docs is not None
                assert docs["diagnostics"]["backend_kind"] == "private_http"
                assert (
                    docs["request"]["question"]
                    == "How does field_validator work in Pydantic v2?"
                )
                assert docs["diagnostics"]["source_id"] == "source_1"
                assert docs["diagnostics"]["platform_kind"] == "docusaurus"
                assert docs["diagnostics"]["selected_document_ids"]
                assert docs["diagnostics"]["selected_chunk_ids"]

                official_docs_result = await session.call_tool(
                    "find_official_docs",
                    {"query": "Pydantic", "max_results": 2},
                )
                assert not official_docs_result.isError
                official_docs = official_docs_result.structuredContent
                assert official_docs is not None
                assert official_docs["diagnostics"]["backend_kind"] == "private_http"
                assert official_docs["matches"]
                assert len(official_docs["matches"]) == 2
                assert official_docs["diagnostics"]["source_id"] == "source_1"

                resolve_result = await session.call_tool(
                    "resolve_doc_source",
                    {"query_or_url": "https://example.com/docs/page"},
                )
                assert not resolve_result.isError
                resolved = resolve_result.structuredContent
                assert resolved is not None
                assert resolved["diagnostics"]["backend_kind"] == "private_http"
                assert resolved["source_type"] == "page_url"
                assert (
                    resolved["diagnostics"]["retrieval_method"] == "registry_url_lookup"
                )

        asyncio.run(run_happy_paths())


def test_private_http_real_integration_auth_failure_surfaces_structured_error() -> None:
    with run_local_searchbridge_core_api(api_token=PRIVATE_TOKEN) as private_backend:
        server_params = local_mcp_server_params(
            env_overrides={
                "SEARCHBRIDGE_BACKEND_KIND": "private_http",
                "SEARCHBRIDGE_PRIVATE_BACKEND_URL": private_backend["base_url"],
                "SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY": "wrong-token",
            }
        )

        async def run_auth_failure() -> None:
            async with (
                stdio_client(server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "search_web",
                    {"query": "auth failure"},
                )

                assert not result.isError
                structured = result.structuredContent
                assert structured is not None
                assert structured["diagnostics"]["backend_kind"] == "private_http"
                assert structured["diagnostics"]["status"] == "error"
                assert (
                    structured["diagnostics"]["error"]["code"] == "backend_auth_failed"
                )
                assert (
                    structured["diagnostics"]["error"]["message"]
                    == "Private backend auth failed."
                )

        asyncio.run(run_auth_failure())


def test_private_http_real_integration_not_implemented_surfaces_structured_error() -> (
    None
):
    with run_local_searchbridge_core_api(api_token=PRIVATE_TOKEN) as private_backend:
        bad_base_url = f"{private_backend['base_url']}/missing-prefix"
        server_params = local_mcp_server_params(
            env_overrides={
                "SEARCHBRIDGE_BACKEND_KIND": "private_http",
                "SEARCHBRIDGE_PRIVATE_BACKEND_URL": bad_base_url,
                "SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY": PRIVATE_TOKEN,
            }
        )

        async def run_not_implemented() -> None:
            async with (
                stdio_client(server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "search_web",
                    {"query": "missing route"},
                )

                assert not result.isError
                structured = result.structuredContent
                assert structured is not None
                assert structured["diagnostics"]["backend_kind"] == "private_http"
                assert structured["diagnostics"]["status"] == "error"
                assert structured["diagnostics"]["error"]["code"] == "not_implemented"
                assert (
                    structured["diagnostics"]["error"]["message"]
                    == "Endpoint not implemented."
                )

        asyncio.run(run_not_implemented())


def test_private_backend_direct_http_request_id_is_echoed() -> None:
    with run_local_searchbridge_core_api(api_token=PRIVATE_TOKEN) as private_backend:
        request_id = "integration-request-id"
        response = httpx.get(
            f"{private_backend['base_url']}/v1/capabilities",
            headers={
                "Authorization": f"Bearer {PRIVATE_TOKEN}",
                "x-request-id": request_id,
            },
            timeout=5.0,
        )

        assert response.status_code == 200
        assert response.headers["x-request-id"] == request_id


def test_private_http_real_integration_docs_qa_accepts_path_like_domain_allowlist() -> (
    None
):
    with run_local_searchbridge_core_api(api_token=PRIVATE_TOKEN) as private_backend:
        server_params = local_mcp_server_params(
            env_overrides={
                "SEARCHBRIDGE_BACKEND_KIND": "private_http",
                "SEARCHBRIDGE_PRIVATE_BACKEND_URL": private_backend["base_url"],
                "SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY": PRIVATE_TOKEN,
            }
        )

        async def run_invalid_request() -> None:
            async with (
                stdio_client(server_params) as (read, write),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                result = await session.call_tool(
                    "docs_qa",
                    {
                        "question": "How does field_validator work in Pydantic v2?",
                        "url": "https://docs.pydantic.dev/latest/",
                        "domain_allowlist": ["docs.pydantic.dev/latest"],
                        "answer_mode": "standard",
                    },
                )

                assert not result.isError
                structured = result.structuredContent
                assert structured is not None
                assert structured["diagnostics"]["backend_kind"] == "private_http"
                assert structured["diagnostics"]["status"] == "ok"
                assert structured["request"]["domain_allowlist"] == [
                    "docs.pydantic.dev/latest"
                ]
                assert structured["diagnostics"]["error"] is None

        asyncio.run(run_invalid_request())
