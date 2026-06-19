"""OpenAI-compatible aggregation backend implementation."""

from __future__ import annotations

import json
import logging
import random
import re
import threading
import time
from typing import Any, Protocol, cast
from urllib.parse import urlparse

import openai
from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params.response_format_json_object import (
    ResponseFormatJSONObject,
)

from .config import Settings
from .errors import UpstreamLogContext, UpstreamSearchError
from .models import (
    Citation,
    DocSourceResolutionRequest,
    DocSourceResolutionRequestEcho,
    DocSourceResolutionResult,
    DocsQARequest,
    DocsQARequestEcho,
    DocsQAResult,
    ErrorInfo,
    ExtractRequestEcho,
    ExtractResult,
    ExtractUrlRequest,
    FindOfficialDocsRequest,
    OfficialDocMatch,
    OfficialDocsRequestEcho,
    OfficialDocsResult,
    OutlineRequestEcho,
    OutlineResult,
    OutlineSection,
    OutlineUrlRequest,
    ProviderInfo,
    SearchDiagnostics,
    SearchRequest,
    SearchResult,
    SearchSource,
    ToolDiagnostics,
    WarningInfo,
)
from .parser import WARNING_MESSAGES, parse_search_response
from .prompts import (
    build_docs_qa_user_prompt,
    build_extract_user_prompt,
    build_find_official_docs_user_prompt,
    build_outline_user_prompt,
    build_resolve_doc_source_user_prompt,
    build_search_user_prompt,
    build_system_prompt,
)
from .type_utils import (
    ContentFormat,
    DocSourceType,
    ToolStatus,
    parse_http_url,
    parse_optional_http_url,
)

LOGGER = logging.getLogger(__name__)

STRUCTURED_RESPONSE_FORMAT: ResponseFormatJSONObject = {"type": "json_object"}
_STRUCTURED_OUTPUT_CACHE_LOCK = threading.Lock()
_STRUCTURED_OUTPUT_UNSUPPORTED_CACHE: dict[tuple[str, str], bool] = {}
_EMPTY_UPSTREAM_RESPONSE_ERROR_CODE = "empty_upstream_response"
_RETRYABLE_EMPTY_RESPONSE_ERROR_TYPES = {
    "EmptyStringResponse",
    "EmptyMessageContent",
}
_EMPTY_RESPONSE_INITIAL_RETRY_DELAY = 0.5
_EMPTY_RESPONSE_MAX_RETRY_DELAY = 2.0
_JSON_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


class ChatCompletionsClient(Protocol):
    def create(
        self,
        *,
        model: str,
        messages: list[ChatCompletionMessageParam],
        response_format: ResponseFormatJSONObject | None = None,
    ) -> ChatCompletion: ...


class ChatNamespace(Protocol):
    completions: ChatCompletionsClient


class OpenAIClient(Protocol):
    chat: ChatNamespace


class OpenAIAggregationBackend:
    """Bridge multiple MCP retrieval tools into an OpenAI-compatible provider."""

    provider_name = "openai-compatible"
    backend_kind = "openai"

    def __init__(self, settings: Settings, client: OpenAIClient | None = None) -> None:
        self.settings = settings
        self.provider_model = settings.resolved_openai_model
        self.client = client or OpenAI(
            api_key=settings.resolved_openai_api_key,
            base_url=str(settings.resolved_openai_base_url),
            organization=settings.openai_organization,
            project=settings.openai_project,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )
        self._structured_output_cache_key = (
            str(settings.resolved_openai_base_url),
            settings.resolved_openai_model,
        )

    def search_web(self, request: SearchRequest) -> SearchResult:
        """Call the upstream provider for search/discovery."""

        content, response_format_accepted, warning_codes = self._call_json_tool(
            build_search_user_prompt(request)
        )
        result = parse_search_response(
            content=content,
            request=request,
            provider=self.provider_name,
            model=self.provider_model,
            response_format_requested="json_object",
            response_format_accepted=response_format_accepted,
            backend_kind=self.backend_kind,
        )
        _append_warning_codes(result.diagnostics, warning_codes)
        return result

    def extract_url(self, request: ExtractUrlRequest) -> ExtractResult:
        """Call the upstream provider for URL extraction."""

        content, _, warning_codes = self._call_json_tool(
            build_extract_user_prompt(request)
        )
        payload = _extract_json_payload(content)
        title = _safe_text(payload.get("title", "")) if payload else ""
        url = (
            _safe_text(payload.get("url", str(request.url)))
            if payload
            else str(request.url)
        )
        body = _safe_text(payload.get("content", "")) if payload else content
        content_format = _safe_literal(
            payload.get("content_format") if payload else None,
            {"text", "markdown"},
            "text",
        )
        truncated = bool(payload.get("truncated", False)) if payload else False
        likely_rewritten = (
            bool(payload.get("likely_rewritten", True)) if payload else True
        )
        warning_codes.extend(
            _collect_warning_codes(payload.get("warnings", [])) if payload else []
        )
        warning_codes.extend(_extract_warning_codes(body))
        return ExtractResult(
            request=ExtractRequestEcho(
                url=request.url,
                mode=request.mode,
                max_chars=request.max_chars,
            ),
            title=title,
            url=parse_http_url(url),
            content=body,
            content_format=cast(ContentFormat, content_format),
            truncated=truncated,
            likely_rewritten=likely_rewritten,
            diagnostics=_tool_diagnostics(
                provider=self._provider_info(),
                warning_codes=warning_codes,
                status=_extract_status(body),
                backend_kind=self.backend_kind,
            ),
        )

    def outline_url(self, request: OutlineUrlRequest) -> OutlineResult:
        """Call the upstream provider for URL outline generation."""

        content, _, warning_codes = self._call_json_tool(
            build_outline_user_prompt(request)
        )
        body = content
        payload = _extract_json_payload(content)
        sections_payload = payload.get("sections", []) if payload else []
        sections: list[OutlineSection] = []
        if isinstance(sections_payload, list):
            for item in sections_payload:
                if not isinstance(item, dict):
                    continue
                title = _safe_text(item.get("title", ""))
                if not title:
                    continue
                sections.append(
                    OutlineSection(
                        title=title,
                        summary=_safe_text(item.get("summary", "")),
                    )
                )
        warning_codes.extend(
            _collect_warning_codes(payload.get("warnings", [])) if payload else []
        )
        warning_codes.extend(_outline_warning_codes(body, sections))
        return OutlineResult(
            request=OutlineRequestEcho(url=request.url, depth=request.depth),
            title=_safe_text(payload.get("title", "")) if payload else "",
            sections=sections,
            diagnostics=_tool_diagnostics(
                provider=self._provider_info(),
                warning_codes=warning_codes,
                status=_outline_status(
                    title=_safe_text(payload.get("title", "")) if payload else "",
                    sections=sections,
                    body=content,
                ),
                backend_kind=self.backend_kind,
            ),
        )

    def docs_qa(self, request: DocsQARequest) -> DocsQAResult:
        """Call the upstream provider for docs question answering."""

        content, _, warning_codes = self._call_json_tool(
            build_docs_qa_user_prompt(request)
        )
        payload = _extract_json_payload(content)
        sources = _parse_sources_from_payload(
            payload=payload,
            domain_allowlist=request.domain_allowlist,
        )
        citations = _parse_citations_from_payload(payload=payload)
        warning_codes.extend(
            _collect_warning_codes(payload.get("warnings", [])) if payload else []
        )
        return DocsQAResult(
            request=DocsQARequestEcho(
                question=request.question,
                url=request.url,
                domain_allowlist=request.domain_allowlist,
                answer_mode=request.answer_mode,
            ),
            answer=_safe_text(payload.get("answer", "")) if payload else content,
            citations=citations,
            sources=sources,
            diagnostics=_tool_diagnostics(
                provider=self._provider_info(),
                warning_codes=warning_codes,
                status="ok" if sources else "partial",
                backend_kind=self.backend_kind,
            ),
        )

    def find_official_docs(
        self,
        request: FindOfficialDocsRequest,
    ) -> OfficialDocsResult:
        """Call the upstream provider for official docs discovery."""

        content, _, warning_codes = self._call_json_tool(
            build_find_official_docs_user_prompt(request)
        )
        payload = _extract_json_payload(content)
        matches_payload = payload.get("matches", []) if payload else []
        matches: list[OfficialDocMatch] = []
        if isinstance(matches_payload, list):
            for item in matches_payload[: request.max_results]:
                if not isinstance(item, dict):
                    continue
                url = _safe_text(item.get("url", ""))
                title = _safe_text(item.get("title", ""))
                if not title or not url:
                    continue
                matches.append(
                    OfficialDocMatch(
                        title=title,
                        url=parse_http_url(url),
                        domain=_domain_from_url(url),
                        rationale=_safe_text(item.get("rationale", "")),
                    )
                )
        warning_codes.extend(
            _collect_warning_codes(payload.get("warnings", [])) if payload else []
        )
        return OfficialDocsResult(
            request=OfficialDocsRequestEcho(
                query=request.query,
                max_results=request.max_results,
            ),
            matches=matches,
            diagnostics=_tool_diagnostics(
                provider=self._provider_info(),
                warning_codes=warning_codes,
                status="ok" if matches else "empty",
                backend_kind=self.backend_kind,
            ),
        )

    def resolve_doc_source(
        self,
        request: DocSourceResolutionRequest,
    ) -> DocSourceResolutionResult:
        """Call the upstream provider for source resolution."""

        content, _, warning_codes = self._call_json_tool(
            build_resolve_doc_source_user_prompt(request)
        )
        payload = _extract_json_payload(content)
        source_type = _safe_literal(
            payload.get("source_type") if payload else None,
            {"llms_txt", "page_url", "library_docs_query", "web_search_query"},
            "web_search_query",
        )
        resolved_url = (
            _safe_optional_url(payload.get("resolved_url")) if payload else None
        )
        confidence = _safe_confidence(payload.get("confidence")) if payload else 0.5
        rationale = _safe_text(payload.get("rationale", "")) if payload else content
        warning_codes.extend(
            _collect_warning_codes(payload.get("warnings", [])) if payload else []
        )
        return DocSourceResolutionResult(
            request=DocSourceResolutionRequestEcho(query_or_url=request.query_or_url),
            source_type=cast(DocSourceType, source_type),
            resolved_url=parse_optional_http_url(resolved_url),
            confidence=confidence,
            rationale=rationale,
            diagnostics=_tool_diagnostics(
                provider=self._provider_info(),
                warning_codes=warning_codes,
                status="ok",
                backend_kind=self.backend_kind,
            ),
        )

    def _call_json_tool(self, user_prompt: str) -> tuple[str, bool, list[str]]:
        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=build_system_prompt(self.settings.searchbridge_system_prompt),
            ),
            ChatCompletionUserMessageParam(role="user", content=user_prompt),
        ]

        warning_codes: list[str] = []
        response_format_accepted = True
        structured_output_supported = self._structured_output_supported()
        attempts_remaining = self.settings.openai_max_retries + 1

        while attempts_remaining > 0:
            try:
                if structured_output_supported:
                    response = self.client.chat.completions.create(
                        model=self.provider_model,
                        messages=messages,
                        response_format=STRUCTURED_RESPONSE_FORMAT,
                    )
                else:
                    response_format_accepted = False
                    warning_codes.append("structured_output_not_supported")
                    response = self._fallback_completion(messages)
                return (
                    _message_content(response),
                    response_format_accepted,
                    warning_codes,
                )
            except UpstreamSearchError as exc:
                if not _is_retryable_empty_response_error(exc):
                    raise
                attempts_remaining -= 1
                if attempts_remaining == 0:
                    raise
                retries_taken = self.settings.openai_max_retries - attempts_remaining
                delay_seconds = _calculate_empty_response_retry_delay(retries_taken)
                LOGGER.warning(
                    "Upstream provider returned empty response content; retrying "
                    "[error_type=%s remaining_attempts=%s delay_seconds=%.6f]",
                    exc.log_context.error_type,
                    attempts_remaining,
                    delay_seconds,
                )
                time.sleep(delay_seconds)
            except openai.BadRequestError as exc:
                if not _is_structured_output_unsupported_error(exc):
                    raise _build_upstream_error(exc) from exc

                LOGGER.warning(
                    "Structured response rejected by upstream provider; retrying "
                    "plain text."
                )
                warning_codes.append("structured_output_not_supported")
                response_format_accepted = False
                structured_output_supported = False
                _mark_structured_output_unsupported(self._structured_output_cache_key)
            except (
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.AuthenticationError,
                openai.RateLimitError,
                openai.APIStatusError,
            ) as exc:
                raise _build_upstream_error(exc) from exc
            except openai.OpenAIError as exc:
                raise _build_upstream_error(exc) from exc

        raise RuntimeError("OpenAI retry loop exhausted without returning or raising")

    def _fallback_completion(self, messages: list[ChatCompletionMessageParam]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.provider_model,
                messages=messages,
            )
            return _message_content(response)
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.AuthenticationError,
            openai.RateLimitError,
            openai.APIStatusError,
        ) as exc:
            raise _build_upstream_error(exc) from exc
        except openai.OpenAIError as exc:
            raise _build_upstream_error(exc) from exc

    def _provider_info(self) -> ProviderInfo:
        return ProviderInfo(name=self.provider_name, model=self.provider_model)

    def _structured_output_supported(self) -> bool:
        return not _structured_output_unsupported_cached(
            self._structured_output_cache_key
        )


def _message_content(response: Any) -> str:
    if isinstance(response, str):
        if response.lstrip().startswith("data:"):
            LOGGER.warning(
                "Upstream provider returned text/event-stream content for a "
                "non-stream chat.completions request."
            )
        content = _content_from_string_response(response)
        if content:
            return content
        raise UpstreamSearchError(
            "Upstream string response content was empty.",
            retryable=True,
            log_context=UpstreamLogContext(error_type="EmptyStringResponse"),
            error_code=_EMPTY_UPSTREAM_RESPONSE_ERROR_CODE,
        )

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise UpstreamSearchError(
            "Upstream response did not contain a chat message.",
            retryable=False,
            log_context=UpstreamLogContext(error_type=type(exc).__name__),
        ) from exc

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    raise UpstreamSearchError(
        "Upstream response message content was empty.",
        retryable=True,
        log_context=UpstreamLogContext(error_type="EmptyMessageContent"),
        error_code=_EMPTY_UPSTREAM_RESPONSE_ERROR_CODE,
    )


def _is_retryable_empty_response_error(exc: UpstreamSearchError) -> bool:
    return (
        exc.retryable
        and exc.error_code == _EMPTY_UPSTREAM_RESPONSE_ERROR_CODE
        and exc.log_context.error_type in _RETRYABLE_EMPTY_RESPONSE_ERROR_TYPES
    )


def _calculate_empty_response_retry_delay(retries_taken: int) -> float:
    base_delay = min(
        _EMPTY_RESPONSE_INITIAL_RETRY_DELAY * pow(2.0, retries_taken),
        _EMPTY_RESPONSE_MAX_RETRY_DELAY,
    )
    jitter = 1 - 0.25 * random.random()
    delay = base_delay * jitter
    return delay if delay >= 0 else 0


def _structured_output_unsupported_cached(cache_key: tuple[str, str]) -> bool:
    with _STRUCTURED_OUTPUT_CACHE_LOCK:
        return _STRUCTURED_OUTPUT_UNSUPPORTED_CACHE.get(cache_key, False)


def _mark_structured_output_unsupported(cache_key: tuple[str, str]) -> None:
    with _STRUCTURED_OUTPUT_CACHE_LOCK:
        _STRUCTURED_OUTPUT_UNSUPPORTED_CACHE[cache_key] = True


def _is_structured_output_unsupported_error(exc: openai.BadRequestError) -> bool:
    body = exc.body
    if not isinstance(body, dict):
        return False

    error_message = _safe_text(body.get("message", "")).lower()
    error_type = _safe_text(body.get("type", "")).lower()
    if "response_format" not in error_message and "structured" not in error_message:
        return False
    return error_type == "invalid_request_error"


def _content_from_string_response(response: str) -> str:
    text = response.strip()
    if not text:
        return ""

    if text.startswith("data:"):
        return _content_from_sse_response(text)

    return text


def _content_from_sse_response(response: str) -> str:
    content_parts: list[str] = []

    for raw_line in response.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue

        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue

        choice = choices[0]
        if not isinstance(choice, dict):
            continue

        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue

        content = delta.get("content")
        if isinstance(content, str) and content:
            content_parts.append(content)

    return "".join(content_parts).strip()


def _extract_json_payload(content: str) -> dict[str, object] | None:
    text = content.strip()
    if not text:
        return None
    for candidate in _iter_json_candidates(text):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _iter_json_candidates(text: str) -> list[str]:
    candidates: list[str] = [text]

    for match in _JSON_CODE_BLOCK_PATTERN.finditer(text):
        candidate = match.group(1).strip()
        if candidate:
            candidates.append(candidate)

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            _, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        candidates.append(text[start : start + end].strip())
        start = text.find("{", start + 1)

    return candidates


def _parse_sources_from_payload(
    *,
    payload: dict[str, object] | None,
    domain_allowlist: list[str],
) -> list[SearchSource]:
    if not payload:
        return []
    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        return []

    sources: list[SearchSource] = []
    for index, item in enumerate(raw_sources, start=1):
        if not isinstance(item, dict):
            continue
        title = _safe_text(item.get("title", ""))
        url = _safe_text(item.get("url", ""))
        if not title or not url:
            continue
        evidence = []
        raw_evidence = item.get("evidence", [])
        if isinstance(raw_evidence, list):
            for evidence_index, raw_chunk in enumerate(raw_evidence, start=1):
                if not isinstance(raw_chunk, dict):
                    continue
                text = _safe_text(raw_chunk.get("text", ""))
                if not text:
                    continue
                chunk_id = _safe_text(raw_chunk.get("chunk_id", "")) or (
                    f"source_{index}_chunk_{evidence_index}"
                )
                evidence.append({"chunk_id": chunk_id, "text": text})
        sources.append(
            SearchSource(
                source_id=_safe_text(item.get("source_id", "")) or f"source_{index}",
                rank=index,
                title=title,
                url=parse_http_url(url),
                domain=_domain_from_url(url),
                published_at=_safe_optional_text(item.get("published_at")),
                domain_allowed=_domain_allowed(
                    domain=_domain_from_url(url),
                    allowlist=domain_allowlist,
                ),
                evidence=evidence,
            )
        )
    return sources


def _parse_citations_from_payload(payload: dict[str, object] | None) -> list[Citation]:
    if not payload:
        return []
    raw_citations = payload.get("citations", [])
    if not isinstance(raw_citations, list):
        return []
    citations: list[Citation] = []
    for item in raw_citations:
        if not isinstance(item, dict):
            continue
        source_id = _safe_text(item.get("source_id", ""))
        chunk_id = _safe_text(item.get("chunk_id", ""))
        if source_id and chunk_id:
            citations.append(Citation(source_id=source_id, chunk_id=chunk_id))
    return citations


def _collect_warning_codes(raw_warnings: object) -> list[str]:
    if not isinstance(raw_warnings, list):
        return []
    codes: list[str] = []
    for item in raw_warnings:
        code = _normalize_warning_code(str(item).strip())
        if code:
            codes.append(code)
    return codes


def _append_warning_codes(
    diagnostics: ToolDiagnostics | SearchDiagnostics,
    warning_codes: list[str],
) -> None:
    existing = {item.code for item in diagnostics.warnings}
    for code in warning_codes:
        if code in existing:
            continue
        diagnostics.warnings.append(
            WarningInfo(code=code, message=WARNING_MESSAGES.get(code, code))
        )
        existing.add(code)


def _tool_diagnostics(
    *,
    provider: ProviderInfo,
    warning_codes: list[str],
    status: ToolStatus,
    backend_kind: str | None = None,
    error: ErrorInfo | None = None,
) -> ToolDiagnostics:
    warnings = [
        WarningInfo(code=code, message=WARNING_MESSAGES.get(code, code))
        for code in _dedupe_warning_codes(warning_codes)
    ]
    return ToolDiagnostics(
        status=status,
        provider=provider,
        backend_kind=backend_kind,
        warnings=warnings,
        error=error,
    )


def _dedupe_warning_codes(codes: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            merged.append(code)
    return merged


def _domain_from_url(url: str) -> str:
    return urlparse(url).netloc.lower()


def _domain_allowed(*, domain: str, allowlist: list[str]) -> bool:
    if not allowlist:
        return True
    normalized_allowlist = [item.lower() for item in allowlist]
    for allowed in normalized_allowlist:
        if domain == allowed or domain.endswith(f".{allowed}"):
            return True
    return False


def _status_from_body(body: str) -> ToolStatus:
    return "ok" if body.strip() else "empty"


def _extract_status(body: str) -> ToolStatus:
    text = body.strip()
    if not text:
        return "empty"
    if _looks_like_not_found_page(text):
        return "empty"
    if _looks_like_placeholder_page(text):
        return "partial"
    return "ok"


def _extract_warning_codes(body: str) -> list[str]:
    if not body.strip():
        return []
    if _looks_like_not_found_page(body):
        return ["not_found_page"]
    if _looks_like_placeholder_page(body):
        return ["placeholder_page"]
    return []


def _outline_warning_codes(body: str, sections: list[OutlineSection]) -> list[str]:
    if not body.strip():
        return []
    if _looks_like_not_found_page(body):
        return ["not_found_page"]
    if _looks_like_placeholder_page(body):
        return ["placeholder_page"]
    if sections and not _extract_title_is_confident(body):
        return ["partial_content"]
    return []


def _outline_status(
    *, title: str, sections: list[OutlineSection], body: str
) -> ToolStatus:
    if not sections:
        return "empty"
    if _looks_like_not_found_page(body) or _looks_like_placeholder_page(body):
        return "partial"
    if not title.strip():
        return "partial"
    return "ok"


def _looks_like_not_found_page(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "404 - page not found",
        "page not found",
        "sorry, this page cannot be found",
        "the page you are looking for cannot be found",
        "404 not found",
    )
    return any(marker in lowered for marker in markers)


def _looks_like_placeholder_page(text: str) -> bool:
    lowered = text.lower()
    markers = (
        "click to expand",
        "table of contents",
        "loading...",
        "please enable javascript",
        "you need to enable javascript",
        "subscribe",
    )
    return any(marker in lowered for marker in markers) and len(text) < 4000


def _extract_title_is_confident(body: str) -> bool:
    lowered = body.lower()
    return "404 - page not found" not in lowered and "page not found" not in lowered


def _safe_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _safe_optional_text(value: object) -> str | None:
    text = _safe_text(value)
    return text or None


def _safe_optional_url(value: object) -> str | None:
    text = _safe_text(value)
    return text or None


def _safe_literal(value: object, allowed: set[str], default: str) -> str:
    candidate = _safe_text(value)
    return candidate if candidate in allowed else default


def _safe_confidence(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float, str)):
        return 0.5
    try:
        number = float(value)
    except TypeError, ValueError:
        return 0.5
    return min(1.0, max(0.0, number))


def _format_api_error(exc: Exception) -> str:
    return _build_upstream_error(exc).client_message


def _build_upstream_error(exc: Exception) -> UpstreamSearchError:
    if isinstance(exc, openai.AuthenticationError):
        return UpstreamSearchError(
            "Authentication with the upstream provider failed.",
            retryable=False,
            log_context=_upstream_log_context(exc),
        )
    if isinstance(exc, openai.RateLimitError):
        return UpstreamSearchError(
            "The upstream provider rate-limited the request.",
            retryable=True,
            log_context=_upstream_log_context(exc),
        )
    if isinstance(exc, openai.APITimeoutError):
        return UpstreamSearchError(
            "The upstream provider request timed out.",
            retryable=True,
            log_context=_upstream_log_context(exc),
        )
    if isinstance(exc, openai.APIConnectionError):
        return UpstreamSearchError(
            "Could not connect to the upstream provider.",
            retryable=True,
            log_context=_upstream_log_context(exc),
        )
    if isinstance(exc, openai.BadRequestError):
        return UpstreamSearchError(
            "The upstream provider rejected the request.",
            retryable=False,
            log_context=_upstream_log_context(exc),
        )
    if isinstance(exc, openai.APIStatusError):
        return UpstreamSearchError(
            f"Upstream provider returned HTTP {exc.status_code}.",
            retryable=500 <= exc.status_code < 600,
            log_context=_upstream_log_context(exc),
        )
    if isinstance(exc, openai.OpenAIError):
        return UpstreamSearchError(
            "The upstream provider request failed.",
            retryable=False,
            log_context=_upstream_log_context(exc),
        )

    return UpstreamSearchError(
        "The upstream provider request failed.",
        retryable=False,
        log_context=UpstreamLogContext(error_type=type(exc).__name__),
    )


def _upstream_log_context(exc: Exception) -> UpstreamLogContext:
    status_code: int | None = None
    request_id: str | None = None
    if isinstance(exc, openai.APIStatusError):
        status_code = exc.status_code
        request_id = exc.request_id
    return UpstreamLogContext(
        error_type=type(exc).__name__,
        status_code=status_code,
        request_id=request_id,
    )


def _normalize_warning_code(code: str) -> str:
    aliases = {
        "no_results_found": "no_results",
        "404_page": "not_found_page",
        "404_page_not_found": "not_found_page",
        "page_not_found": "not_found_page",
    }
    return aliases.get(code, code)
