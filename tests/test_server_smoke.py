from mcp_searchbridge.config import Settings
from mcp_searchbridge.models import SearchRequest, SearchResult, SearchSource
from mcp_searchbridge.server import create_server


class FakeBackend:
    provider_name = "fake-provider"

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResult:
        self.requests.append(request)
        return SearchResult(
            answer=f"Echo: {request.query}",
            sources=[
                SearchSource(
                    title="Example",
                    url="https://example.com/search",
                    snippet="Snippet",
                )
            ],
            provider=self.provider_name,
            model="fake-model",
            raw_text='{"answer":"Echo"}',
            warnings=[],
        )


def test_create_server_registers_web_search_tool() -> None:
    settings = Settings(
        _env_file=None,
        OPENAI_API_KEY="test-key",
        OPENAI_BASE_URL="https://api.example.com/v1",
        OPENAI_MODEL="fake-model",
    )
    backend = FakeBackend()

    server = create_server(settings=settings, backend=backend)

    tools = server._tool_manager.list_tools()
    tool_names = [tool.name for tool in tools]

    assert "web_search" in tool_names
