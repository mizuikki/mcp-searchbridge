"""Prompt builders for the upstream model-backed tools."""

from __future__ import annotations

import json

from .models import (
    DocSourceResolutionRequest,
    DocsQARequest,
    ExtractUrlRequest,
    FindOfficialDocsRequest,
    OutlineUrlRequest,
    SearchRequest,
)


def build_system_prompt(system_prompt: str) -> str:
    """Return the configured system prompt."""

    return system_prompt.strip()


def build_search_user_prompt(request: SearchRequest) -> str:
    """Build the prompt sent to the upstream model for search."""

    payload = {
        "query": request.query,
        "recency": request.recency or "unspecified",
        "max_sources": request.max_sources,
        "domain_allowlist": request.domain_allowlist,
        "return_mode": request.return_mode,
    }
    return (
        "Use the upstream provider's native web or search capability if it exists.\n"
        "Do not claim to have searched the web unless the provider actually "
        "supports it.\n"
        "Do not fabricate URLs, publication dates, citations, or evidence text.\n"
        "This tool is for search and discovery, not full-page extraction.\n"
        f"Search parameters:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        "Return a JSON object with this shape:\n"
        "{\n"
        '  "summary": {\n'
        '    "text": "short factual synthesis",\n'
        '    "citations": [\n'
        '      {"source_id": "source_1", "chunk_id": "source_1_chunk_1"}\n'
        "    ]\n"
        "  },\n"
        '  "sources": [\n'
        "    {\n"
        '      "source_id": "source_1",\n'
        '      "title": "source title",\n'
        '      "url": "https://example.com",\n'
        '      "published_at": "YYYY-MM-DD or null",\n'
        '      "evidence": [\n'
        '        {"chunk_id": "source_1_chunk_1", "text": "supporting evidence"}\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "warnings": ["optional warning codes"]\n'
        "}\n"
        "Prefer authoritative sources and include direct URLs whenever available."
    )


def build_extract_user_prompt(request: ExtractUrlRequest) -> str:
    """Build the prompt sent to the upstream model for URL extraction."""

    payload = {
        "url": str(request.url),
        "mode": request.mode,
        "max_chars": request.max_chars,
    }
    payload_json = json.dumps(payload, ensure_ascii=True, indent=2)
    return (
        "Fetch the URL content as directly as possible.\n"
        "Do not fabricate access or claim exact fidelity if you only have partial "
        "content.\n"
        "If the content is too long, return the leading portion and clearly mark "
        "that it was truncated.\n"
        f"Extraction parameters:\n{payload_json}\n\n"
        "Return a JSON object with this shape:\n"
        "{\n"
        '  "title": "page title",\n'
        '  "url": "https://example.com",\n'
        '  "content": "page body or markdown",\n'
        '  "content_format": "text or markdown",\n'
        '  "truncated": true,\n'
        '  "likely_rewritten": false,\n'
        '  "warnings": ["optional warning codes"]\n'
        "}"
    )


def build_outline_user_prompt(request: OutlineUrlRequest) -> str:
    """Build the prompt sent to the upstream model for outline extraction."""

    payload = {"url": str(request.url), "depth": request.depth}
    return (
        "Inspect the URL and return a structured outline of its main sections or "
        "linked document groups.\n"
        "Prefer preserving the source's hierarchy over summarizing loosely.\n"
        f"Outline parameters:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        "Return a JSON object with this shape:\n"
        "{\n"
        '  "title": "page title",\n'
        '  "sections": [\n'
        '    {"title": "section title", "summary": "short description"}\n'
        "  ],\n"
        '  "warnings": ["optional warning codes"]\n'
        "}"
    )


def build_docs_qa_user_prompt(request: DocsQARequest) -> str:
    """Build the prompt sent to the upstream model for docs QA."""

    payload = {
        "question": request.question,
        "url": str(request.url) if request.url else None,
        "domain_allowlist": request.domain_allowlist,
        "answer_mode": request.answer_mode,
    }
    return (
        "Answer the documentation question using official online sources when "
        "possible.\n"
        "If a URL is provided, prioritize that URL and closely related official "
        "documentation.\n"
        "Include direct citations to the supporting source chunks.\n"
        f"QA parameters:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        "Return a JSON object with this shape:\n"
        "{\n"
        '  "answer": "documentation-backed answer",\n'
        '  "sources": [\n'
        "    {\n"
        '      "source_id": "source_1",\n'
        '      "title": "source title",\n'
        '      "url": "https://example.com",\n'
        '      "published_at": "YYYY-MM-DD or null",\n'
        '      "evidence": [\n'
        '        {"chunk_id": "source_1_chunk_1", "text": "supporting evidence"}\n'
        "      ]\n"
        "    }\n"
        "  ],\n"
        '  "citations": [\n'
        '    {"source_id": "source_1", "chunk_id": "source_1_chunk_1"}\n'
        "  ],\n"
        '  "warnings": ["optional warning codes"]\n'
        "}"
    )


def build_find_official_docs_user_prompt(request: FindOfficialDocsRequest) -> str:
    """Build the prompt sent to the upstream model for official docs discovery."""

    payload = {"query": request.query, "max_results": request.max_results}
    return (
        "Find the official documentation entry points for the given topic, library, "
        "framework, or product.\n"
        "Prefer canonical vendor or project documentation over blogs or mirrors.\n"
        f"Discovery parameters:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        "Return a JSON object with this shape:\n"
        "{\n"
        '  "matches": [\n'
        "    {\n"
        '      "title": "official docs title",\n'
        '      "url": "https://example.com/docs",\n'
        '      "rationale": "why this is official"\n'
        "    }\n"
        "  ],\n"
        '  "warnings": ["optional warning codes"]\n'
        "}"
    )


def build_resolve_doc_source_user_prompt(request: DocSourceResolutionRequest) -> str:
    """Build the prompt sent to the upstream model for source resolution."""

    payload = {"query_or_url": request.query_or_url}
    payload_json = json.dumps(payload, ensure_ascii=True, indent=2)
    return (
        "Classify whether the input is best handled as an llms.txt index, a normal "
        "page URL, a library docs query, or a web search query.\n"
        "Return a confidence score and a short rationale.\n"
        f"Resolution parameters:\n{payload_json}\n\n"
        "Return a JSON object with this shape:\n"
        "{\n"
        '  "source_type": '
        '"llms_txt or page_url or library_docs_query or web_search_query",\n'
        '  "resolved_url": "https://example.com or null",\n'
        '  "confidence": 0.95,\n'
        '  "rationale": "short explanation",\n'
        '  "warnings": ["optional warning codes"]\n'
        "}"
    )
