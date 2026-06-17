"""Backend construction helpers."""

from __future__ import annotations

from .config import Settings
from .openai_backend import OpenAIAggregationBackend
from .private_backend import PrivateHttpAggregationBackend


def build_backend(
    settings: Settings,
) -> OpenAIAggregationBackend | PrivateHttpAggregationBackend:
    """Construct the configured aggregation backend."""

    if settings.searchbridge_backend_kind == "private_http":
        return PrivateHttpAggregationBackend(settings)
    return OpenAIAggregationBackend(settings)
