"""Parse structured or semi-structured upstream search responses."""

from __future__ import annotations

import json
import re
from urllib.parse import urlparse

from pydantic import HttpUrl, TypeAdapter, ValidationError

from .models import (
    Citation,
    EvidenceChunk,
    ProviderInfo,
    QueryEcho,
    SearchCoverage,
    SearchDiagnostics,
    SearchNormalizationInfo,
    SearchRequest,
    SearchResult,
    SearchSource,
    Summary,
    WarningInfo,
)
from .type_utils import ParseMode, ToolStatus, parse_http_url

JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
URL_RE = re.compile(r"https?://[^\s<>\"]+")
SOURCE_LINE_RE = re.compile(
    r"^\s*(?:[-*]|\d+\.)\s*(?P<title>.+?)\s*-\s*(?P<url>https?://\S+)(?:\s*-\s*(?P<snippet>.+))?$"
)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HTTP_URL_ADAPTER = TypeAdapter(HttpUrl)

WARNING_MESSAGES = {
    "structured_response_invalid": (
        "Structured upstream response could not be normalized."
    ),
    "structured_response_validation_failed": (
        "Structured upstream response failed validation."
    ),
    "structured_output_not_supported": (
        "Upstream provider rejected JSON response_format."
    ),
    "legacy_response_shape_used": (
        "Upstream response matched the legacy answer/sources shape."
    ),
    "text_fallback_used": "Structured parsing failed and text fallback was used.",
    "url_fallback_used": (
        "Only bare URLs could be extracted from the upstream response."
    ),
    "summary_citations_unavailable": (
        "Summary citations were missing or could not be validated."
    ),
    "sources_missing_or_unverifiable": (
        "No verifiable sources were extracted from the response."
    ),
    "provider_reported_no_live_access": (
        "Upstream provider reported no live web access."
    ),
    "no_results": "The search completed but returned no matching sources.",
    "no_relevant_results": "The search completed but returned no matching sources.",
    "partial_content": "Only partial page content could be extracted.",
    "not_found_page": "The target page appears to be a 404 or not-found page.",
    "placeholder_page": "The target page appears to be a placeholder or nav-only page.",
    "empty_content_retried": (
        "The upstream returned empty content and the request was retried once."
    ),
    "published_at_unparseable": (
        "One or more published_at values were invalid and were dropped."
    ),
    "answer_missing_in_structured_response": (
        "Legacy structured response was missing answer text."
    ),
}


def parse_search_response(
    *,
    content: str,
    request: SearchRequest,
    provider: str,
    model: str,
    response_format_requested: str = "json_object",
    response_format_accepted: bool = True,
) -> SearchResult:
    """Parse model content into a normalized search result."""

    warning_codes: list[str] = []
    structured_payload = _extract_json_payload(content)
    if structured_payload is not None:
        try:
            result = _parse_structured_payload(
                payload=structured_payload,
                request=request,
                provider=provider,
                model=model,
                response_format_requested=response_format_requested,
                response_format_accepted=response_format_accepted,
            )
        except ValidationError, ValueError:
            result = None
            warning_codes.append("structured_response_invalid")

        if result is not None:
            return result

    return _parse_text_fallback(
        content=content,
        request=request,
        provider=provider,
        model=model,
        response_format_requested=response_format_requested,
        response_format_accepted=response_format_accepted,
        warning_codes=warning_codes,
    )


def _extract_json_payload(content: str) -> dict[str, object] | None:
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


def _parse_structured_payload(
    *,
    payload: dict[str, object],
    request: SearchRequest,
    provider: str,
    model: str,
    response_format_requested: str,
    response_format_accepted: bool,
) -> SearchResult | None:
    if "summary" in payload:
        return _result_from_v2_payload(
            payload=payload,
            request=request,
            provider=provider,
            model=model,
            response_format_requested=response_format_requested,
            response_format_accepted=response_format_accepted,
        )

    if "answer" in payload or "sources" in payload:
        return _result_from_legacy_payload(
            payload=payload,
            request=request,
            provider=provider,
            model=model,
            response_format_requested=response_format_requested,
            response_format_accepted=response_format_accepted,
        )

    return None


def _result_from_v2_payload(
    *,
    payload: dict[str, object],
    request: SearchRequest,
    provider: str,
    model: str,
    response_format_requested: str,
    response_format_accepted: bool,
) -> SearchResult:
    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")

    warning_codes = _normalize_provider_warning_codes(
        payload=payload,
        warning_codes=_collect_warning_codes(payload.get("warnings", [])),
    )
    sources: list[SearchSource] = []
    summary_citation_pairs: set[tuple[str, str]] = set()

    raw_summary = payload.get("summary", {})
    if not isinstance(raw_summary, dict):
        raw_summary = {}

    summary_text = str(raw_summary.get("text", "")).strip()
    raw_citations = raw_summary.get("citations", [])
    if not isinstance(raw_citations, list):
        raw_citations = []

    for index, source_payload in enumerate(raw_sources[: request.max_sources], start=1):
        if not isinstance(source_payload, dict):
            continue
        source = _source_from_payload(
            source_payload=source_payload,
            index=index,
            request=request,
            warning_codes=warning_codes,
        )
        if source is not None:
            sources.append(source)
            for chunk in source.evidence:
                summary_citation_pairs.add((source.source_id, chunk.chunk_id))

    citations: list[Citation] = []
    for citation_payload in raw_citations:
        if not isinstance(citation_payload, dict):
            continue
        source_id = str(citation_payload.get("source_id", "")).strip()
        chunk_id = str(citation_payload.get("chunk_id", "")).strip()
        if not source_id or not chunk_id:
            continue
        if (source_id, chunk_id) in summary_citation_pairs:
            citations.append(Citation(source_id=source_id, chunk_id=chunk_id))

    if summary_text and not citations and sources:
        warning_codes.append("summary_citations_unavailable")

    return _build_result(
        request=request,
        provider=provider,
        model=model,
        summary=Summary(text=summary_text, citations=citations),
        sources=sources,
        parse_mode="structured_v2",
        response_format_requested=response_format_requested,
        response_format_accepted=response_format_accepted,
        warning_codes=warning_codes,
    )


def _result_from_legacy_payload(
    *,
    payload: dict[str, object],
    request: SearchRequest,
    provider: str,
    model: str,
    response_format_requested: str,
    response_format_accepted: bool,
) -> SearchResult:
    answer = str(payload.get("answer", "")).strip()
    warning_codes = _normalize_provider_warning_codes(
        payload=payload,
        warning_codes=_collect_warning_codes(payload.get("warnings", [])),
    )
    warning_codes.append("legacy_response_shape_used")

    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")

    sources: list[SearchSource] = []
    citations: list[Citation] = []
    for index, source_payload in enumerate(raw_sources[: request.max_sources], start=1):
        if not isinstance(source_payload, dict):
            continue
        source = _source_from_legacy_payload(
            source_payload=source_payload,
            index=index,
            request=request,
            warning_codes=warning_codes,
        )
        if source is None:
            continue
        sources.append(source)
        if source.evidence:
            citations.append(
                Citation(
                    source_id=source.source_id,
                    chunk_id=source.evidence[0].chunk_id,
                )
            )

    if not answer:
        answer = _fallback_summary_from_sources(sources)
        warning_codes.append("answer_missing_in_structured_response")
    if answer and sources and not citations:
        warning_codes.append("summary_citations_unavailable")

    return _build_result(
        request=request,
        provider=provider,
        model=model,
        summary=Summary(text=answer, citations=citations),
        sources=sources,
        parse_mode="structured_legacy",
        response_format_requested=response_format_requested,
        response_format_accepted=response_format_accepted,
        warning_codes=warning_codes,
    )


def _parse_text_fallback(
    *,
    content: str,
    request: SearchRequest,
    provider: str,
    model: str,
    response_format_requested: str,
    response_format_accepted: bool,
    warning_codes: list[str],
) -> SearchResult:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    answer_lines: list[str] = []
    sources: list[SearchSource] = []
    in_sources_section = False

    for line in lines:
        normalized = line.lower().rstrip(":")
        if normalized in {"sources", "citations", "references"}:
            in_sources_section = True
            continue
        if in_sources_section:
            source = _parse_source_line(
                line=line,
                index=len(sources) + 1,
                request=request,
            )
            if source is not None and len(sources) < request.max_sources:
                sources.append(source)
                continue
        answer_lines.append(line)

    summary_text = "\n".join(answer_lines).strip() or _fallback_summary_from_sources(
        sources
    )
    parse_mode = "text_fallback"

    if not sources:
        sources = _sources_from_urls(content=content, request=request)
        if sources:
            parse_mode = "url_fallback"
            warning_codes.append("url_fallback_used")

    warning_codes.append("text_fallback_used")
    citations = _first_chunk_citations(sources) if summary_text else []
    if summary_text and sources and not citations:
        warning_codes.append("summary_citations_unavailable")

    return _build_result(
        request=request,
        provider=provider,
        model=model,
        summary=Summary(text=summary_text, citations=citations),
        sources=sources,
        parse_mode=parse_mode,
        response_format_requested=response_format_requested,
        response_format_accepted=response_format_accepted,
        warning_codes=warning_codes,
    )


def _source_from_payload(
    *,
    source_payload: dict[str, object],
    index: int,
    request: SearchRequest,
    warning_codes: list[str],
) -> SearchSource | None:
    title = str(source_payload.get("title", "")).strip()
    url = _normalize_url(str(source_payload.get("url", "")).strip())
    if not title or url is None:
        return None

    source_id = _normalize_source_id(source_payload.get("source_id"), index)
    evidence_payload = source_payload.get("evidence", [])
    evidence = _build_evidence_chunks(
        source_id=source_id,
        evidence_payload=evidence_payload,
        fallback_text="",
        request=request,
    )

    published_at = _normalize_published_at(
        source_payload.get("published_at"),
        warning_codes,
    )
    return _build_source(
        source_id=source_id,
        rank=index,
        title=title,
        url=url,
        published_at=published_at,
        evidence=evidence,
        request=request,
    )


def _source_from_legacy_payload(
    *,
    source_payload: dict[str, object],
    index: int,
    request: SearchRequest,
    warning_codes: list[str],
) -> SearchSource | None:
    title = str(source_payload.get("title", "")).strip()
    url = _normalize_url(str(source_payload.get("url", "")).strip())
    snippet = str(source_payload.get("snippet", "")).strip()
    if not title or url is None:
        return None

    source_id = f"source_{index}"
    evidence = _build_evidence_chunks(
        source_id=source_id,
        evidence_payload=[],
        fallback_text=snippet,
        request=request,
    )

    return _build_source(
        source_id=source_id,
        rank=index,
        title=title,
        url=url,
        published_at=None,
        evidence=evidence,
        request=request,
    )


def _parse_source_line(
    line: str,
    index: int,
    request: SearchRequest,
) -> SearchSource | None:
    match = SOURCE_LINE_RE.match(line)
    if match is None:
        return None

    url = _normalize_url(match.group("url"))
    if url is None:
        return None

    source_id = f"source_{index}"
    snippet = (match.group("snippet") or "").strip()
    evidence = _build_evidence_chunks(
        source_id=source_id,
        evidence_payload=[],
        fallback_text=snippet,
        request=request,
    )

    return _build_source(
        source_id=source_id,
        rank=index,
        title=match.group("title").strip(),
        url=url,
        published_at=None,
        evidence=evidence,
        request=request,
    )


def _sources_from_urls(content: str, request: SearchRequest) -> list[SearchSource]:
    sources: list[SearchSource] = []
    seen_urls: set[str] = set()
    for _index, url_match in enumerate(URL_RE.findall(content), start=1):
        normalized_url = _normalize_url(url_match)
        if normalized_url is None or normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        sources.append(
            _build_source(
                source_id=f"source_{len(sources) + 1}",
                rank=len(sources) + 1,
                title=_title_from_url(normalized_url),
                url=normalized_url,
                published_at=None,
                evidence=[],
                request=request,
            )
        )
        if len(sources) >= request.max_sources:
            break
    return sources


def _build_source(
    *,
    source_id: str,
    rank: int,
    title: str,
    url: str,
    published_at: str | None,
    evidence: list[EvidenceChunk],
    request: SearchRequest,
) -> SearchSource:
    domain = _domain_from_url(url)
    return SearchSource(
        source_id=source_id,
        rank=rank,
        title=title,
        url=parse_http_url(url),
        domain=domain,
        published_at=published_at,
        domain_allowed=_domain_allowed(
            domain=domain,
            allowlist=request.domain_allowlist,
        ),
        evidence=evidence,
    )


def _build_evidence_chunks(
    *,
    source_id: str,
    evidence_payload: object,
    fallback_text: str,
    request: SearchRequest,
) -> list[EvidenceChunk]:
    max_chunks = 1 if request.return_mode == "concise" else 2
    chunks: list[EvidenceChunk] = []

    if isinstance(evidence_payload, list):
        for item in evidence_payload:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            chunk_id = str(item.get("chunk_id", "")).strip() or (
                f"{source_id}_chunk_{len(chunks) + 1}"
            )
            chunks.append(EvidenceChunk(chunk_id=chunk_id, text=text))
            if len(chunks) >= max_chunks:
                break

    if not chunks and fallback_text:
        chunks.append(
            EvidenceChunk(chunk_id=f"{source_id}_chunk_1", text=fallback_text)
        )

    return chunks


def _normalize_source_id(value: object, index: int) -> str:
    source_id = str(value or "").strip()
    return source_id or f"source_{index}"


def _normalize_published_at(value: object, warning_codes: list[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if DATE_RE.match(text):
        return text
    warning_codes.append("published_at_unparseable")
    return None


def _normalize_url(value: str) -> str | None:
    candidate = value.rstrip(").,;")
    try:
        return str(HTTP_URL_ADAPTER.validate_python(candidate))
    except ValidationError:
        return None


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


def _title_from_url(url: str) -> str:
    trimmed = url.removeprefix("https://").removeprefix("http://")
    return trimmed.split("/", 1)[0]


def _fallback_summary_from_sources(sources: list[SearchSource]) -> str:
    if not sources:
        return ""
    return f"Found {len(sources)} source(s)."


def _first_chunk_citations(sources: list[SearchSource]) -> list[Citation]:
    citations: list[Citation] = []
    for source in sources:
        if source.evidence:
            citations.append(
                Citation(
                    source_id=source.source_id,
                    chunk_id=source.evidence[0].chunk_id,
                )
            )
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


def _build_result(
    *,
    request: SearchRequest,
    provider: str,
    model: str,
    summary: Summary,
    sources: list[SearchSource],
    parse_mode: ParseMode,
    response_format_requested: str,
    response_format_accepted: bool,
    warning_codes: list[str],
) -> SearchResult:
    normalized_warning_codes = _dedupe_warning_codes(warning_codes)
    normalized_warning_codes = _finalize_warning_codes(
        warning_codes=normalized_warning_codes,
        summary=summary,
        sources=sources,
    )

    warnings = [
        WarningInfo(code=code, message=WARNING_MESSAGES.get(code, code))
        for code in normalized_warning_codes
    ]

    evidence_chunks_returned = sum(len(source.evidence) for source in sources)
    sources_with_evidence = sum(1 for source in sources if source.evidence)

    status: ToolStatus
    if not sources:
        status = "empty"
    elif parse_mode in {"structured_v2", "structured_legacy"} and sources_with_evidence:
        status = "ok"
    else:
        status = "partial"

    return SearchResult(
        query=QueryEcho(
            text=request.query,
            recency=request.recency,
            max_sources=request.max_sources,
            domain_allowlist=request.domain_allowlist,
            return_mode=request.return_mode,
        ),
        summary=summary,
        sources=sources,
        diagnostics=SearchDiagnostics(
            status=status,
            provider=ProviderInfo(name=provider, model=model),
            normalization=SearchNormalizationInfo(
                response_format_requested=(
                    "json_object"
                    if response_format_requested == "json_object"
                    else "none"
                ),
                response_format_accepted=response_format_accepted,
                parse_mode=parse_mode,
            ),
            coverage=SearchCoverage(
                sources_requested=request.max_sources,
                sources_returned=len(sources),
                sources_with_evidence=sources_with_evidence,
                evidence_chunks_returned=evidence_chunks_returned,
            ),
            warnings=warnings,
        ),
    )


def _dedupe_warning_codes(codes: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for code in codes:
        if code and code not in seen:
            seen.add(code)
            merged.append(code)
    return merged


def _normalize_warning_code(code: str) -> str:
    aliases = {
        "no_results_found": "no_results",
        "no_relevant_results": "no_results",
        "404": "not_found_page",
        "404_page_not_found": "not_found_page",
    }
    return aliases.get(code, code)


def _normalize_provider_warning_codes(
    *,
    payload: dict[str, object],
    warning_codes: list[str],
) -> list[str]:
    normalized = _dedupe_warning_codes(warning_codes)
    if "provider_reported_no_live_access" not in normalized:
        return normalized

    if _payload_explicitly_reports_no_live_access(payload):
        return normalized

    return [code for code in normalized if code != "provider_reported_no_live_access"]


def _payload_explicitly_reports_no_live_access(payload: dict[str, object]) -> bool:
    summary_payload = payload.get("summary", {})
    summary_text = ""
    if isinstance(summary_payload, dict):
        summary_text = str(summary_payload.get("text", "")).strip().lower()
    elif "answer" in payload:
        summary_text = str(payload.get("answer", "")).strip().lower()

    if not summary_text:
        return False

    phrases = (
        "do not have live web access",
        "do not have access to live web",
        "no live web access",
        "cannot browse the web",
        "can't browse the web",
        "cannot access the web",
        "can't access the web",
        "no internet access",
        "cannot search the web",
        "can't search the web",
    )
    return any(phrase in summary_text for phrase in phrases)


def _finalize_warning_codes(
    *,
    warning_codes: list[str],
    summary: Summary,
    sources: list[SearchSource],
) -> list[str]:
    normalized = _dedupe_warning_codes(warning_codes)
    if sources:
        return normalized

    has_live_access_warning = "provider_reported_no_live_access" in normalized
    if not has_live_access_warning and summary.text:
        normalized.append("no_results")

    normalized.append("sources_missing_or_unverifiable")
    return _dedupe_warning_codes(normalized)
