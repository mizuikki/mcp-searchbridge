from mcp_searchbridge.config import Settings
from mcp_searchbridge.models import (
    Citation,
    Coverage,
    Diagnostics,
    EvidenceChunk,
    NormalizationInfo,
    ProviderInfo,
    QueryEcho,
    SearchRequest,
    SearchResult,
    SearchSource,
    Summary,
)
from mcp_searchbridge.server import create_server


class FakeBackend:
    provider_name = "fake-provider"

    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResult:
        self.requests.append(request)
        return SearchResult(
            query=QueryEcho(
                text=request.query,
                recency=request.recency,
                max_sources=request.max_sources,
                domain_allowlist=request.domain_allowlist,
                return_mode=request.return_mode,
            ),
            summary=Summary(
                text=f"Echo: {request.query}",
                citations=[Citation(source_id="source_1", chunk_id="source_1_chunk_1")],
            ),
            sources=[
                SearchSource(
                    source_id="source_1",
                    rank=1,
                    title="Example",
                    url="https://example.com/search",
                    domain="example.com",
                    published_at=None,
                    domain_allowed=True,
                    evidence=[
                        EvidenceChunk(
                            chunk_id="source_1_chunk_1",
                            text="Snippet",
                        )
                    ],
                )
            ],
            diagnostics=Diagnostics(
                status="ok",
                provider=ProviderInfo(name=self.provider_name, model="fake-model"),
                normalization=NormalizationInfo(
                    response_format_requested="json_object",
                    response_format_accepted=True,
                    parse_mode="structured_v2",
                ),
                coverage=Coverage(
                    sources_requested=request.max_sources,
                    sources_returned=1,
                    sources_with_evidence=1,
                    evidence_chunks_returned=1,
                ),
                warnings=[],
            ),
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
