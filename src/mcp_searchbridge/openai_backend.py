"""OpenAI-compatible backend implementation."""

from __future__ import annotations

import json
import logging
from typing import Any

import openai
from openai import OpenAI

from .config import Settings
from .errors import UpstreamSearchError
from .models import SearchRequest, SearchResult
from .parser import parse_search_response
from .prompts import build_system_prompt, build_user_prompt

LOGGER = logging.getLogger(__name__)

STRUCTURED_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}


class OpenAIChatSearchBackend:
    """Bridge search requests into an OpenAI-compatible chat completion call."""

    provider_name = "openai-compatible"

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=str(settings.openai_base_url),
            organization=settings.openai_organization,
            project=settings.openai_project,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    def search(self, request: SearchRequest) -> SearchResult:
        """Call the upstream provider and normalize the model response."""

        messages = [
            {
                "role": "system",
                "content": build_system_prompt(
                    self.settings.searchbridge_system_prompt
                ),
            },
            {"role": "user", "content": build_user_prompt(request)},
        ]

        warnings: list[str] = []
        content = ""

        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                response_format=STRUCTURED_RESPONSE_FORMAT,
            )
            content = _message_content(response)
        except openai.BadRequestError as exc:
            LOGGER.warning(
                (
                    "Structured response request rejected by upstream "
                    "provider; retrying plain text. status=%s"
                ),
                exc.status_code,
            )
            warnings.append("structured_output_not_supported")
            content = self._fallback_completion(messages)
        except (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.AuthenticationError,
            openai.RateLimitError,
            openai.APIStatusError,
        ) as exc:
            raise UpstreamSearchError(_format_api_error(exc)) from exc
        except openai.OpenAIError as exc:
            raise UpstreamSearchError(str(exc)) from exc

        result = parse_search_response(
            content=content,
            provider=self.provider_name,
            model=self.settings.openai_model,
            max_sources=request.max_sources,
        )
        result.warnings = _merge_warnings(result.warnings, warnings)
        return result

    def _fallback_completion(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.settings.openai_model,
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
            raise UpstreamSearchError(_format_api_error(exc)) from exc
        except openai.OpenAIError as exc:
            raise UpstreamSearchError(str(exc)) from exc


def _message_content(response: Any) -> str:
    if isinstance(response, str):
        content = _content_from_string_response(response)
        if content:
            return content
        raise UpstreamSearchError("Upstream string response content was empty.")

    try:
        message = response.choices[0].message
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        raise UpstreamSearchError(
            "Upstream response did not contain a chat message."
        ) from exc

    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    raise UpstreamSearchError("Upstream response message content was empty.")


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


def _format_api_error(exc: Exception) -> str:
    if isinstance(exc, openai.AuthenticationError):
        return "Authentication with the upstream provider failed."
    if isinstance(exc, openai.RateLimitError):
        return "The upstream provider rate-limited the request."
    if isinstance(exc, openai.APITimeoutError):
        return "The upstream provider request timed out."
    if isinstance(exc, openai.APIConnectionError):
        return f"Could not connect to the upstream provider: {exc}"
    if isinstance(exc, openai.APIStatusError):
        return f"Upstream provider returned HTTP {exc.status_code}."
    return str(exc)


def _merge_warnings(*warning_groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in warning_groups:
        for item in group:
            if item and item not in seen:
                merged.append(item)
                seen.add(item)
    return merged
