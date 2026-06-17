"""Domain-specific exceptions for search bridge operations."""

from __future__ import annotations

from dataclasses import dataclass


class SearchBridgeError(Exception):
    """Base exception for expected runtime failures."""


@dataclass(slots=True, frozen=True)
class UpstreamLogContext:
    """Sanitized metadata for expected upstream failures."""

    error_type: str
    status_code: int | None = None
    request_id: str | None = None


class UpstreamSearchError(SearchBridgeError):
    """Raised when the upstream OpenAI-compatible provider call fails."""

    def __init__(
        self,
        client_message: str,
        *,
        retryable: bool,
        log_context: UpstreamLogContext,
        error_code: str | None = None,
        allow_fallback: bool = False,
    ) -> None:
        super().__init__(client_message)
        self.client_message = client_message
        self.retryable = retryable
        self.log_context = log_context
        self.error_code = error_code
        self.allow_fallback = allow_fallback

    def __str__(self) -> str:
        return self.client_message
