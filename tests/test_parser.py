from mcp_searchbridge.parser import parse_search_response


def test_parse_structured_json_response() -> None:
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
        provider="openai-compatible",
        model="test-model",
        max_sources=5,
    )

    assert result.answer == "Result summary"
    assert len(result.sources) == 1
    assert str(result.sources[0].url) == "https://example.com/article"
    assert result.warnings == []


def test_parse_text_fallback_response() -> None:
    content = """
    Latest update summary.

    Sources:
    - Example Source - https://example.com/news - Supporting snippet
    """

    result = parse_search_response(
        content=content,
        provider="openai-compatible",
        model="test-model",
        max_sources=5,
    )

    assert "Latest update summary." in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].title == "Example Source"
    assert "text_fallback_used" in result.warnings


def test_parse_invalid_structured_response_falls_back() -> None:
    content = """
    ```json
    {"answer": "", "sources": "invalid"}
    ```

    https://example.com/fallback
    """

    result = parse_search_response(
        content=content,
        provider="openai-compatible",
        model="test-model",
        max_sources=5,
    )

    assert result.sources
    assert "structured_response_invalid" in result.warnings
    assert "text_fallback_used" in result.warnings
