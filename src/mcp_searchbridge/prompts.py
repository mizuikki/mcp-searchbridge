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
        f"Search parameters:\n{json.dumps(payload, ensure_ascii=True, indent=2)}\n\n"
        "Return a JSON object with this shape:\n"
        "{\n"
        '  "answer": "concise factual answer",\n'
        '  "sources": [\n'
        '    {"title": "source title", "url": "https://example.com", '
        '"snippet": "short supporting snippet"}\n'
        "  ],\n"
        '  "warnings": ["optional warning strings"]\n'
        "}\n"
        "Prefer authoritative sources and include direct URLs whenever available."
    )
