"""Parse structured or semi-structured upstream search responses."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import HttpUrl, TypeAdapter, ValidationError

from .models import SearchResult, SearchSource

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
URL_RE = re.compile(r"https?://[^\s<>\"]+")
SOURCE_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+\.)\s*(?P<title>.+?)\s*-\s*(?P<url>https?://\S+)(?:\s*-\s*(?P<snippet>.+))?$"
)
HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)


def parse_search_response(
    *,
    content: str,
    provider: str,
    model: str,
    max_sources: int,
) -> SearchResult:
    """Parse model content into a normalized search result."""

    warnings: list[str] = []
    structured_payload = _extract_json_payload(content)
    if structured_payload is not None:
        try:
            return _result_from_payload(
                payload=structured_payload,
                provider=provider,
                model=model,
                raw_text=content,
                max_sources=max_sources,
            )
        except ValidationError:
            warnings.append("structured_response_validation_failed")
        except ValueError:
            warnings.append("structured_response_invalid")

    fallback_result = _parse_text_fallback(
        content=content,
        provider=provider,
        model=model,
        max_sources=max_sources,
    )
    fallback_result.warnings = _merge_warnings(
        fallback_result.warnings,
        warnings + ["text_fallback_used"],
    )
    return fallback_result


def _extract_json_payload(content: str) -> dict[str, Any] | None:
    text = content.strip()
    candidates = [text, _strip_common_wrappers(text)]
    fenced = JSON_BLOCK_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _strip_common_wrappers(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("**") and stripped.endswith("**"):
        return stripped[2:-2].strip()
    return stripped


def _result_from_payload(
    *,
    payload: dict[str, Any],
    provider: str,
    model: str,
    raw_text: str,
    max_sources: int,
) -> SearchResult:
    answer = str(payload.get("answer", "")).strip()
    warnings = [
        str(item).strip() for item in payload.get("warnings", []) if str(item).strip()
    ]

    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")

    sources: list[SearchSource] = []
    for source_payload in raw_sources[:max_sources]:
        if not isinstance(source_payload, dict):
            continue
        title = str(source_payload.get("title", "")).strip()
        url = str(source_payload.get("url", "")).strip()
        snippet = str(source_payload.get("snippet", "")).strip()
        if not title or not url:
            continue
        sources.append(SearchSource(title=title, url=url, snippet=snippet))

    if not answer:
        answer = raw_text.strip()
        warnings.append("answer_missing_in_structured_response")
    if not sources:
        warnings.append("sources_missing_or_unverifiable")

    return SearchResult(
        answer=answer,
        sources=sources,
        provider=provider,
        model=model,
        raw_text=raw_text,
        warnings=_merge_warnings(warnings),
    )


def _parse_text_fallback(
    *,
    content: str,
    provider: str,
    model: str,
    max_sources: int,
) -> SearchResult:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    answer_lines: list[str] = []
    sources: list[SearchSource] = []
    warnings: list[str] = []

    in_sources_section = False
    for line in lines:
        normalized = line.lower().rstrip(":")
        if normalized in {"sources", "citations", "references"}:
            in_sources_section = True
            continue
        if in_sources_section:
            source = _parse_source_line(line)
            if source is not None and len(sources) < max_sources:
                sources.append(source)
                continue
        answer_lines.append(line)

    if not sources:
        for url in URL_RE.findall(content):
            normalized_url = _normalize_url(url)
            if normalized_url is None:
                continue
            title = _title_from_url(normalized_url)
            sources.append(
                SearchSource(
                    title=title,
                    url=normalized_url,
                    snippet="",
                )
            )
            if len(sources) >= max_sources:
                break

    answer = "\n".join(answer_lines).strip() or content.strip()
    if not sources:
        warnings.append("sources_missing_or_unverifiable")

    return SearchResult(
        answer=answer,
        sources=sources[:max_sources],
        provider=provider,
        model=model,
        raw_text=content,
        warnings=_merge_warnings(warnings),
    )


def _parse_source_line(line: str) -> SearchSource | None:
    match = SOURCE_LINE_RE.match(line)
    if match is None:
        return None

    normalized_url = _normalize_url(match.group("url"))
    if normalized_url is None:
        return None

    return SearchSource(
        title=match.group("title").strip(),
        url=normalized_url,
        snippet=(match.group("snippet") or "").strip(),
    )


def _normalize_url(value: str) -> str | None:
    candidate = value.rstrip(").,;")
    try:
        return str(HTTP_URL_ADAPTER.validate_python(candidate))
    except ValidationError:
        return None


def _title_from_url(url: str) -> str:
    trimmed = url.removeprefix("https://").removeprefix("http://")
    return trimmed.split("/", 1)[0]


def _merge_warnings(*warning_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in warning_groups:
        for item in group:
            if item and item not in seen:
                seen.add(item)
                merged.append(item)
    return merged
