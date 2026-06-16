from mcp_searchbridge.models import SearchRequest
from mcp_searchbridge.parser import parse_search_response


def _request(return_mode: str = "standard") -> SearchRequest:
    return SearchRequest(
        query="latest update",
        recency="week",
        max_sources=5,
        domain_allowlist=["example.com"],
        return_mode=return_mode,
    )


def _warning_codes(result) -> list[str]:
    return [warning.code for warning in result.diagnostics.warnings]


def test_parse_structured_v2_response() -> None:
    content = """
    {
      "summary": {
        "text": "Result summary",
        "citations": [
          {"source_id": "source_1", "chunk_id": "source_1_chunk_1"}
        ]
      },
      "sources": [
        {
          "source_id": "source_1",
          "title": "Example",
          "url": "https://example.com/article",
          "published_at": "2026-06-15",
          "evidence": [
            {
              "chunk_id": "source_1_chunk_1",
              "text": "Supporting detail"
            }
          ]
        }
      ],
      "warnings": []
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    assert result.summary.text == "Result summary"
    assert result.summary.citations[0].source_id == "source_1"
    assert len(result.sources) == 1
    assert str(result.sources[0].url) == "https://example.com/article"
    assert result.sources[0].published_at == "2026-06-15"
    assert result.diagnostics.normalization.parse_mode == "structured_v2"
    assert result.diagnostics.status == "ok"


def test_parse_legacy_structured_response() -> None:
    content = """
    {
      "answer": "Result summary",
      "sources": [
        {
          "title": "Example",
          "url": "https://example.com/article",
          "snippet": "Supporting detail"
        }
      ],
      "warnings": []
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    assert result.summary.text == "Result summary"
    assert result.sources[0].evidence[0].text == "Supporting detail"
    assert result.diagnostics.normalization.parse_mode == "structured_legacy"
    assert "legacy_response_shape_used" in _warning_codes(result)


def test_parse_text_fallback_response() -> None:
    content = """
    Latest update summary.

    Sources:
    - Example Source - https://example.com/news - Supporting snippet
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    assert "Latest update summary." in result.summary.text
    assert len(result.sources) == 1
    assert result.sources[0].title == "Example Source"
    assert result.sources[0].evidence[0].text == "Supporting snippet"
    assert result.diagnostics.normalization.parse_mode == "text_fallback"
    assert "text_fallback_used" in _warning_codes(result)


def test_parse_invalid_structured_response_falls_back_to_url_mode() -> None:
    content = """
    ```json
    {"summary": {"text": "Broken"}, "sources": "invalid"}
    ```

    https://example.com/fallback
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    assert result.sources
    assert result.sources[0].evidence == []
    assert result.diagnostics.normalization.parse_mode == "url_fallback"
    assert "structured_response_invalid" in _warning_codes(result)
    assert "url_fallback_used" in _warning_codes(result)


def test_invalid_published_at_is_dropped() -> None:
    content = """
    {
      "summary": {
        "text": "Result summary",
        "citations": []
      },
      "sources": [
        {
          "source_id": "source_1",
          "title": "Example",
          "url": "https://example.com/article",
          "published_at": "June 15, 2026",
          "evidence": [
            {
              "chunk_id": "source_1_chunk_1",
              "text": "Supporting detail"
            }
          ]
        }
      ],
      "warnings": []
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    assert result.sources[0].published_at is None
    assert "published_at_unparseable" in _warning_codes(result)


def test_concise_mode_limits_evidence_chunks() -> None:
    content = """
    {
      "summary": {
        "text": "Result summary",
        "citations": [
          {"source_id": "source_1", "chunk_id": "source_1_chunk_1"}
        ]
      },
      "sources": [
        {
          "source_id": "source_1",
          "title": "Example",
          "url": "https://example.com/article",
          "published_at": "2026-06-15",
          "evidence": [
            {"chunk_id": "source_1_chunk_1", "text": "First detail"},
            {"chunk_id": "source_1_chunk_2", "text": "Second detail"}
          ]
        }
      ],
      "warnings": []
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(return_mode="concise"),
        provider="openai-compatible",
        model="test-model",
    )

    assert len(result.sources[0].evidence) == 1


def test_empty_results_without_live_access_claim_use_no_results_warning() -> None:
    content = """
    {
      "summary": {
        "text": "No results found for this query."
      },
      "sources": [],
      "warnings": ["provider_reported_no_live_access"]
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    warning_codes = _warning_codes(result)
    assert result.diagnostics.status == "empty"
    assert "no_results" in warning_codes
    assert "sources_missing_or_unverifiable" in warning_codes
    assert "provider_reported_no_live_access" not in warning_codes


def test_empty_results_normalize_no_results_found_alias() -> None:
    content = """
    {
      "summary": {
        "text": "No information found for this query."
      },
      "sources": [],
      "warnings": ["no_results_found"]
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    warning_codes = _warning_codes(result)
    assert result.diagnostics.status == "empty"
    assert "no_results" in warning_codes
    assert "no_results_found" not in warning_codes
    assert "sources_missing_or_unverifiable" in warning_codes


def test_empty_results_normalize_no_relevant_results_alias() -> None:
    content = """
    {
      "summary": {
        "text": "No relevant results were found for this query."
      },
      "sources": [],
      "warnings": ["no_relevant_results"]
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    warning_codes = _warning_codes(result)
    assert result.diagnostics.status == "empty"
    assert "no_results" in warning_codes
    assert "no_relevant_results" not in warning_codes


def test_explicit_no_live_access_claim_preserves_warning() -> None:
    content = """
    {
      "summary": {
        "text": "I cannot browse the web because I do not have live web access."
      },
      "sources": [],
      "warnings": ["provider_reported_no_live_access"]
    }
    """

    result = parse_search_response(
        content=content,
        request=_request(),
        provider="openai-compatible",
        model="test-model",
    )

    warning_codes = _warning_codes(result)
    assert result.diagnostics.status == "empty"
    assert "provider_reported_no_live_access" in warning_codes
    assert "no_results" not in warning_codes
