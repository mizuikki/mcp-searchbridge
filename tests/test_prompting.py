from mcp_searchbridge.config import DEFAULT_SYSTEM_PROMPT
from mcp_searchbridge.models import SearchRequest
from mcp_searchbridge.prompts import build_system_prompt, build_user_prompt


def test_build_system_prompt_uses_configured_text() -> None:
    assert build_system_prompt(DEFAULT_SYSTEM_PROMPT) == DEFAULT_SYSTEM_PROMPT


def test_build_user_prompt_embeds_search_parameters() -> None:
    request = SearchRequest(
        query="recent model updates",
        recency="week",
        max_sources=3,
        domain_allowlist=["openai.com", "developers.openai.com"],
        return_mode="concise",
    )

    prompt = build_user_prompt(request)

    assert "recent model updates" in prompt
    assert '"recency": "week"' in prompt
    assert '"domain_allowlist": [' in prompt
    assert "Return a JSON object" in prompt
