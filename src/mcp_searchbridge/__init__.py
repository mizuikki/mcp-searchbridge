"""Public package entry points for mcp-searchbridge."""

from __future__ import annotations

from typing import Any


def create_server(*args: Any, **kwargs: Any):
    """Create the FastMCP server lazily."""

    from .server import create_server as _create_server

    return _create_server(*args, **kwargs)


def main() -> None:
    """Run the server entry point lazily."""

    from .server import main as _main

    _main()


__all__ = ["create_server", "main"]
