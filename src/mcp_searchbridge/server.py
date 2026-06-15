"""FastMCP server entry point."""

from __future__ import annotations

import logging
from typing import Literal

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from .config import Settings, get_settings
from .errors import SearchBridgeError, UpstreamSearchError
from .models import SearchRequest, SearchResult
from .openai_backend import OpenAIChatSearchBackend


def create_server(
    settings: Settings | None = None,
    backend: OpenAIChatSearchBackend | None = None,
) -> FastMCP:
    """Create the FastMCP application and register tools."""

    active_settings = settings or get_settings()
    _configure_logging(active_settings.searchbridge_log_level)
    search_backend = backend or OpenAIChatSearchBackend(active_settings)

    mcp = FastMCP(
        name="mcp-searchbridge",
        instructions=(
            "Use the web_search tool to query an upstream OpenAI-compatible "
            "search-capable chat model."
        ),
        log_level=active_settings.searchbridge_log_level,
    )

    @mcp.tool(
        name="web_search",
        description=(
            "Search the web through an upstream OpenAI-compatible chat "
            "completion provider."
        ),
    )
    def web_search(
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
            return search_backend.search(request)
        except ValidationError as exc:
            logging.getLogger(__name__).error("web_search validation failed: %s", exc)
            return SearchResult(
                answer=f"Invalid search request: {exc}",
                sources=[],
                provider=search_backend.provider_name,
                model=active_settings.openai_model,
                raw_text="",
                warnings=["invalid_request"],
            )
        except UpstreamSearchError as exc:
            logging.getLogger(__name__).error("web_search upstream failure: %s", exc)
            return SearchResult(
                answer=f"Upstream search request failed: {exc}",
                sources=[],
                provider=search_backend.provider_name,
                model=active_settings.openai_model,
                raw_text="",
                warnings=["upstream_request_failed"],
            )
        except SearchBridgeError as exc:
            logging.getLogger(__name__).error("web_search failed: %s", exc)
            return SearchResult(
                answer=f"Search request failed: {exc}",
                sources=[],
                provider=search_backend.provider_name,
                model=active_settings.openai_model,
                raw_text="",
                warnings=["search_request_failed"],
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


if __name__ == "__main__":
    main()
