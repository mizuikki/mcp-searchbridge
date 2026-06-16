"""Shared type aliases and adapters for runtime-validated values."""

from __future__ import annotations

from typing import Literal, Protocol, cast

from pydantic import HttpUrl, TypeAdapter

type LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
type ReturnMode = Literal["concise", "standard"]
type ExtractMode = Literal["body", "markdown", "best_effort"]
type OutlineDepth = Literal["shallow", "standard", "deep"]
type DocsAnswerMode = Literal["concise", "standard"]
type ToolStatus = Literal["ok", "partial", "empty", "error"]
type ParseMode = Literal[
    "structured_v2",
    "structured_legacy",
    "text_fallback",
    "url_fallback",
    "error",
]
type DocSourceType = Literal[
    "llms_txt",
    "page_url",
    "library_docs_query",
    "web_search_query",
]
type ContentFormat = Literal["text", "markdown"]

_HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)
_URL_SET = {"url", "resolved_url"}


class SupportsModelDump(Protocol):
    def model_dump(self) -> dict[str, object]: ...


def parse_http_url(value: str) -> HttpUrl:
    """Validate and return an HttpUrl."""

    return _HTTP_URL_ADAPTER.validate_python(value)


def parse_optional_http_url(value: str | None) -> HttpUrl | None:
    """Validate an optional URL string."""

    if value is None:
        return None
    return parse_http_url(value)


def coerce_literal(value: str, *, allowed: set[str]) -> str:
    """Return a validated literal string."""

    if value not in allowed:
        msg = f"Expected one of {sorted(allowed)}, got {value!r}"
        raise ValueError(msg)
    return value


def cast_literal(value: str, *, allowed: set[str]) -> str:
    """Cast a value after explicit membership validation."""

    return coerce_literal(value, allowed=allowed)


def model_dump_strings(model: SupportsModelDump) -> dict[str, object]:
    """Dump Pydantic models while converting HttpUrl fields back to strings."""

    dumped = cast(dict[str, object], model.model_dump())
    return {
        key: str(value) if key in _URL_SET and value is not None else value
        for key, value in dumped.items()
    }
