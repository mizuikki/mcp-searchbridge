"""Private HTTP aggregation backend adapter."""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import Settings
from .errors import UpstreamLogContext, UpstreamSearchError
from .models import (
    DocSourceResolutionRequest,
    DocSourceResolutionResult,
    DocsQARequest,
    DocsQAResult,
    ExtractResult,
    ExtractUrlRequest,
    FindOfficialDocsRequest,
    OfficialDocsResult,
    OutlineResult,
    OutlineUrlRequest,
    SearchRequest,
    SearchResult,
)
from .openai_backend import OpenAIAggregationBackend

LOGGER = logging.getLogger(__name__)

ResultT = TypeVar("ResultT", bound=BaseModel)
_FALLBACK_ERROR_CODES = {"endpoint_not_implemented", "not_implemented"}


class PrivateHttpAggregationBackend:
    """HTTP adapter for a private searchbridge backend service."""

    provider_name = "private-http"
    provider_model = "private-backend"
    backend_kind = "private_http"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        fallback_backend: OpenAIAggregationBackend | None = None,
    ) -> None:
        self.settings = settings
        self.base_url = (
            f"{str(settings.searchbridge_private_backend_url).rstrip('/')}/"
            if settings.searchbridge_private_backend_url is not None
            else ""
        )
        headers: dict[str, str] = {}
        if settings.searchbridge_private_backend_api_key:
            headers["Authorization"] = (
                f"Bearer {settings.searchbridge_private_backend_api_key}"
            )
        self.client = client or httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=settings.searchbridge_private_backend_timeout_seconds,
        )
        self.fallback_backend = fallback_backend or (
            OpenAIAggregationBackend(settings)
            if settings.searchbridge_private_backend_fallback_to_openai
            else None
        )

    def search_web(self, request: SearchRequest) -> SearchResult:
        return self._call_or_fallback(
            "search_web",
            "v1/search_web",
            request,
            SearchResult,
        )

    def extract_url(self, request: ExtractUrlRequest) -> ExtractResult:
        return self._call_or_fallback(
            "extract_url",
            "v1/extract_url",
            request,
            ExtractResult,
        )

    def outline_url(self, request: OutlineUrlRequest) -> OutlineResult:
        return self._call_or_fallback(
            "outline_url",
            "v1/outline_url",
            request,
            OutlineResult,
        )

    def docs_qa(self, request: DocsQARequest) -> DocsQAResult:
        return self._call_or_fallback(
            "docs_qa",
            "v1/docs_qa",
            request,
            DocsQAResult,
        )

    def find_official_docs(
        self,
        request: FindOfficialDocsRequest,
    ) -> OfficialDocsResult:
        return self._call_or_fallback(
            "find_official_docs",
            "v1/find_official_docs",
            request,
            OfficialDocsResult,
        )

    def resolve_doc_source(
        self,
        request: DocSourceResolutionRequest,
    ) -> DocSourceResolutionResult:
        return self._call_or_fallback(
            "resolve_doc_source",
            "v1/resolve_doc_source",
            request,
            DocSourceResolutionResult,
        )

    def _call_or_fallback(
        self,
        method_name: str,
        endpoint: str,
        request: BaseModel,
        result_type: type[ResultT],
    ) -> ResultT:
        try:
            return self._call_private(
                endpoint=endpoint,
                request=request,
                result_type=result_type,
            )
        except UpstreamSearchError as exc:
            if self.fallback_backend is None or not exc.allow_fallback:
                raise
            LOGGER.warning(
                "Private backend %s failed; falling back to OpenAI "
                "[error_type=%s status_code=%s request_id=%s retryable=%s "
                "error_code=%s]",
                method_name,
                exc.log_context.error_type,
                exc.log_context.status_code,
                exc.log_context.request_id,
                exc.retryable,
                exc.error_code,
            )
            fallback_method = getattr(self.fallback_backend, method_name)
            return fallback_method(request)

    def _call_private(
        self,
        *,
        endpoint: str,
        request: BaseModel,
        result_type: type[ResultT],
    ) -> ResultT:
        try:
            response = self.client.post(endpoint, json=request.model_dump(mode="json"))
        except httpx.TimeoutException as exc:
            raise UpstreamSearchError(
                "The private backend request timed out.",
                retryable=True,
                log_context=UpstreamLogContext(error_type=type(exc).__name__),
                error_code="private_backend_timeout",
                allow_fallback=self.fallback_backend is not None,
            ) from exc
        except httpx.RequestError as exc:
            raise UpstreamSearchError(
                "Could not connect to the private backend.",
                retryable=True,
                log_context=UpstreamLogContext(error_type=type(exc).__name__),
                error_code="private_backend_connection_failed",
                allow_fallback=self.fallback_backend is not None,
            ) from exc

        if response.is_error:
            raise self._build_http_error(response)

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise UpstreamSearchError(
                "Private backend returned invalid JSON.",
                retryable=False,
                log_context=UpstreamLogContext(
                    error_type=type(exc).__name__,
                    status_code=response.status_code,
                    request_id=_response_request_id(response),
                ),
                error_code="invalid_private_backend_json",
            ) from exc

        if not isinstance(payload, dict):
            raise UpstreamSearchError(
                "Private backend returned an invalid response body.",
                retryable=False,
                log_context=UpstreamLogContext(
                    error_type="InvalidPrivateBackendResponse",
                    status_code=response.status_code,
                    request_id=_response_request_id(response),
                ),
                error_code="invalid_private_backend_response",
            )

        normalized_payload = self._normalize_payload(payload)
        try:
            return result_type.model_validate(normalized_payload)
        except ValidationError as exc:
            raise UpstreamSearchError(
                "Private backend returned invalid response data.",
                retryable=False,
                log_context=UpstreamLogContext(
                    error_type=type(exc).__name__,
                    status_code=response.status_code,
                    request_id=_response_request_id(response),
                ),
                error_code="invalid_private_backend_response_data",
            ) from exc

    def _build_http_error(self, response: httpx.Response) -> UpstreamSearchError:
        error_payload = _extract_error_payload(response)
        request_id = _response_request_id(response)
        status_code = response.status_code
        error_code = _safe_text(error_payload.get("code")) or None
        error_message = _safe_text(error_payload.get("message"))
        retryable = _coerce_retryable(error_payload.get("retryable"))

        if error_code is not None and error_message:
            return UpstreamSearchError(
                error_message,
                retryable=retryable,
                log_context=UpstreamLogContext(
                    error_type="PrivateBackendErrorResponse",
                    status_code=status_code,
                    request_id=request_id,
                ),
                error_code=error_code,
                allow_fallback=self._should_fallback_for_http_error(
                    status_code=status_code,
                    error_code=error_code,
                ),
            )

        return UpstreamSearchError(
            f"Private backend returned HTTP {status_code}.",
            retryable=500 <= status_code < 600,
            log_context=UpstreamLogContext(
                error_type="HTTPStatusError",
                status_code=status_code,
                request_id=request_id,
            ),
            error_code=None,
            allow_fallback=self._should_fallback_for_http_error(
                status_code=status_code,
                error_code=None,
            ),
        )

    def _should_fallback_for_http_error(
        self,
        *,
        status_code: int,
        error_code: str | None,
    ) -> bool:
        if self.fallback_backend is None:
            return False
        if status_code == 404:
            return True
        if 500 <= status_code < 600:
            return True
        if error_code is None:
            return False
        return error_code in _FALLBACK_ERROR_CODES

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, dict):
            return payload

        normalized = dict(payload)
        normalized_diagnostics = dict(diagnostics)
        provider = normalized_diagnostics.get("provider")
        if not isinstance(provider, dict):
            normalized_diagnostics["provider"] = {
                "name": self.provider_name,
                "model": self.provider_model,
            }
        else:
            normalized_provider = dict(provider)
            if not str(normalized_provider.get("name", "")).strip():
                normalized_provider["name"] = self.provider_name
            if not str(normalized_provider.get("model", "")).strip():
                normalized_provider["model"] = self.provider_model
            normalized_diagnostics["provider"] = normalized_provider

        if normalized_diagnostics.get("backend_kind") in {None, ""}:
            normalized_diagnostics["backend_kind"] = self.backend_kind

        capabilities = normalized_diagnostics.get("capabilities_used")
        if not isinstance(capabilities, list) or not capabilities:
            normalized_diagnostics["capabilities_used"] = [self.backend_kind]

        normalized["diagnostics"] = normalized_diagnostics
        return normalized


def _response_request_id(response: httpx.Response) -> str | None:
    request_id = response.headers.get("x-request-id")
    if request_id:
        return request_id
    return response.headers.get("request-id")


def _extract_error_payload(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    error = payload.get("error")
    if not isinstance(error, dict):
        return {}
    return error


def _safe_text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _coerce_retryable(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return False
