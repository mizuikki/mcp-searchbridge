"""Domain-specific exceptions for search bridge operations."""

from __future__ import annotations


class SearchBridgeError(Exception):
    """Base exception for expected runtime failures."""


class UpstreamSearchError(SearchBridgeError):
    """Raised when the upstream OpenAI-compatible provider call fails."""
