from __future__ import annotations

from typing import Any, cast

from mcp_searchbridge.config import Settings
from mcp_searchbridge.type_utils import parse_http_url, parse_optional_http_url


def make_settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://api.example.com/v1",
        "OPENAI_MODEL": "test-model",
        **overrides,
    }
    # BaseSettings accepts these runtime kwargs even though static typing
    # doesn't expose them on the generated __init__.
    return cast(Settings, Settings(_env_file=None, **payload))  # pyright: ignore[reportCallIssue]


def url(value: str):
    return parse_http_url(value)


def optional_url(value: str | None):
    return parse_optional_http_url(value)


def host_port(server_address: object) -> tuple[str, int]:
    if (
        isinstance(server_address, tuple)
        and len(server_address) >= 2
        and isinstance(server_address[0], str)
        and isinstance(server_address[1], int)
    ):
        return server_address[0], server_address[1]
    msg = f"Unexpected server address shape: {server_address!r}"
    raise TypeError(msg)
