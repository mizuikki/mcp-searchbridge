"""Prompt builders for the upstream search model."""

from __future__ import annotations

import json

from .models import SearchRequest


def build_system_prompt(system_prompt: str) -> str:
    """Return the configured system prompt."""

    return system_prompt.strip()


def build_user_prompt(request: SearchRequest) -> str:
    """Build the search instruction sent to the upstream model."""

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
        "If the provider does not support live web access, return an empty sources "
        "array and include the warning code provider_reported_no_live_access.\n"
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
