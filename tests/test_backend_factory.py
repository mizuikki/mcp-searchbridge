from __future__ import annotations

from mcp_searchbridge.backend_factory import build_backend
from mcp_searchbridge.openai_backend import OpenAIAggregationBackend
from mcp_searchbridge.private_backend import PrivateHttpAggregationBackend
from tests.helpers import make_settings


def test_build_backend_returns_openai_backend_by_default() -> None:
    backend = build_backend(make_settings())

    assert isinstance(backend, OpenAIAggregationBackend)


def test_build_backend_returns_private_backend_when_configured() -> None:
    backend = build_backend(
        make_settings(
            SEARCHBRIDGE_BACKEND_KIND="private_http",
            SEARCHBRIDGE_PRIVATE_BACKEND_URL="https://private.example.com",
            OPENAI_API_KEY=None,
            OPENAI_BASE_URL=None,
            OPENAI_MODEL=None,
        )
    )

    assert isinstance(backend, PrivateHttpAggregationBackend)
