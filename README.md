# mcp-searchbridge

`mcp-searchbridge` is a lightweight FastMCP server that exposes a small,
LLM-oriented retrieval surface backed by a configurable retrieval backend.

It does not ship its own crawler, search engine, or docs index. Freshness, live
web access, extraction fidelity, and source quality still depend on the upstream
model or gateway you configure.

## Features

- FastMCP server over stdio
- Six retrieval-oriented tools instead of a single search tool
- Switchable `openai` and `private_http` backend modes
- OpenAI-compatible `base_url`, `api_key`, and `model` configuration
- Ordered model fallback chain via comma-separated `OPENAI_MODEL`
- Private HTTP backend adapter for enhanced internal retrieval services
- JSON-first parsing with plain-text fallback where needed
- Structured diagnostics, warnings, and normalized source objects
- `uv`-managed Python 3.14 project with Ruff and pytest

## Breaking Changes

- `web_search` was renamed to `search_web`
- The server now exposes:
  - `search_web`
  - `extract_url`
  - `outline_url`
  - `docs_qa`
  - `find_official_docs`
  - `resolve_doc_source`
- Search responses are normalized for LLM consumption and are not backward
  compatible with earlier raw answer formats

## Requirements

- `uv`
- Python `3.14.x`

## Installation

```bash
uv python install 3.14
uv sync --dev
```

## Configuration

Copy `.env.example` into `.env` or set environment variables directly.

Default mode is `openai`:

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your-primary-model,your-fallback-model
OPENAI_TIMEOUT_SECONDS=180
OPENAI_MAX_RETRIES=2
OPENAI_ORGANIZATION=
OPENAI_PROJECT=
SEARCHBRIDGE_SYSTEM_PROMPT=
SEARCHBRIDGE_DEFAULT_MAX_SOURCES=5
SEARCHBRIDGE_BACKEND_KIND=openai
SEARCHBRIDGE_PRIVATE_BACKEND_URL=
SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY=
SEARCHBRIDGE_PRIVATE_BACKEND_TIMEOUT_SECONDS=30
SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=false
SEARCHBRIDGE_LOG_LEVEL=INFO
```

Required variables:

- For `SEARCHBRIDGE_BACKEND_KIND=openai`:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL`
  - `OPENAI_MODEL`
- For `SEARCHBRIDGE_BACKEND_KIND=private_http`:
  - `SEARCHBRIDGE_PRIVATE_BACKEND_URL`

`OPENAI_MODEL` accepts either a single model or a comma-separated fallback
chain. The first model is primary; later models are attempted in order only
after the current model exhausts retryable failures.

Private backend example:

```env
SEARCHBRIDGE_BACKEND_KIND=private_http
SEARCHBRIDGE_PRIVATE_BACKEND_URL=https://private-searchbridge.internal
SEARCHBRIDGE_PRIVATE_BACKEND_API_KEY=private-token
SEARCHBRIDGE_PRIVATE_BACKEND_TIMEOUT_SECONDS=30
SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=true

OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your-primary-model,your-fallback-model
```

`private_http` is an enhancement hook for an internal JSON API. The public MCP
surface stays the same; the server just swaps the backend implementation behind
the existing six tools. If `SEARCHBRIDGE_PRIVATE_BACKEND_FALLBACK_TO_OPENAI=true`,
the OpenAI settings must still be configured so failed private calls can fall back.
Fallback is intentionally narrow: the server only falls back for recoverable
private-backend failures such as transport errors, 5xx responses, or explicit
`not_implemented`/`endpoint_not_implemented` errors. Auth failures, invalid JSON,
and response-contract mismatches are returned as structured MCP errors instead of
being silently downgraded.

`mcp-searchbridge` reads `.env` for local runs, but MCP clients such as Claude
Code and Cursor do not automatically inherit that file. Pass the same values in
the MCP `env` block when launching through a client.

## Run

```bash
uv run mcp-searchbridge
```

This starts the MCP server on stdio.

## Container

This repository also includes a top-level `Dockerfile` for packaging the
`mcp-searchbridge` server itself as a container image.

The image:

- installs project dependencies with `uv`
- copies this repository into the image
- starts the MCP server with `uv run mcp-searchbridge`

Build example:

```bash
docker build -t mcp-searchbridge .
```

## Tools

### `search_web`

Purpose: current web discovery with normalized summaries, citations, and source
evidence.

Input:

```json
{
  "query": "latest OpenAI release notes",
  "recency": "latest",
  "max_sources": 5,
  "domain_allowlist": ["openai.com", "developers.openai.com"],
  "return_mode": "standard"
}
```

Output shape:

```json
{
  "query": {
    "text": "latest OpenAI release notes",
    "recency": "latest",
    "max_sources": 5,
    "domain_allowlist": ["openai.com", "developers.openai.com"],
    "return_mode": "standard"
  },
  "summary": {
    "text": "Short factual synthesis",
    "citations": [
      {
        "source_id": "source_1",
        "chunk_id": "source_1_chunk_1"
      }
    ]
  },
  "sources": [
    {
      "source_id": "source_1",
      "rank": 1,
      "title": "OpenAI release notes",
      "url": "https://example.com/release-notes",
      "domain": "example.com",
      "published_at": "2026-06-15",
      "domain_allowed": true,
      "evidence": [
        {
          "chunk_id": "source_1_chunk_1",
          "text": "Relevant source snippet"
        }
      ]
    }
  ],
  "diagnostics": {
    "status": "ok",
    "provider": {
      "name": "openai-compatible",
      "model": "your-model"
    },
    "backend_kind": "openai",
    "normalization": {
      "response_format_requested": "json_object",
      "response_format_accepted": true,
      "parse_mode": "structured_v2"
    },
    "coverage": {
      "sources_requested": 5,
      "sources_returned": 1,
      "sources_with_evidence": 1,
      "evidence_chunks_returned": 1
    },
    "capabilities_used": [],
    "reproducible": null,
    "cache_hit": null,
    "version_locked": null,
    "resolved_version": null,
    "warnings": [],
    "error": null
  }
}
```

### `extract_url`

Purpose: fetch the main body of a page as text or markdown-like content.

Input:

```json
{
  "url": "https://example.com/docs/page",
  "mode": "best_effort",
  "max_chars": 12000
}
```

Output highlights:

- `title`
- `url`
- `content`
- `content_format`
- `truncated`
- `likely_rewritten`
- `diagnostics`

Behavior notes:

- obvious 404 / not-found pages are normalized to `diagnostics.status="empty"`
- placeholder or nav-heavy pages may be downgraded with warnings instead of being
  treated as fully valid extracts

### `outline_url`

Purpose: return a compact structural outline for a page or llms.txt-like index.

Input:

```json
{
  "url": "https://example.com/llms.txt",
  "depth": "standard"
}
```

Output highlights:

- `title`
- `sections[]`
- `diagnostics`

Behavior notes:

- normal document pages typically return `status="ok"`
- 404 / not-found pages may still produce a shallow outline, but are downgraded
  to `status="partial"` with `not_found_page`

### `docs_qa`

Purpose: answer a documentation question using a provided docs URL or official
docs discovered by the model.

Input:

```json
{
  "question": "How do I create a chat completion?",
  "url": "https://platform.openai.com/docs",
  "domain_allowlist": ["openai.com", "platform.openai.com"],
  "answer_mode": "standard"
}
```

Output highlights:

- `answer`
- `citations[]`
- `sources[]`
- `diagnostics`

### `find_official_docs`

Purpose: resolve a topic or library name to likely canonical documentation entry
points.

Input:

```json
{
  "query": "Pydantic",
  "max_results": 5
}
```

Output highlights:

- `matches[]`
- `diagnostics`

### `resolve_doc_source`

Purpose: classify whether an input is best handled as:

- `llms_txt`
- `page_url`
- `library_docs_query`
- `web_search_query`

Input:

```json
{
  "query_or_url": "https://example.com/llms.txt"
}
```

Output highlights:

- `source_type`
- `resolved_url`
- `confidence`
- `rationale`
- `diagnostics`

## Diagnostics Semantics

All tools return `diagnostics.status` as one of:

- `ok`
- `partial`
- `empty`
- `error`

Search warnings currently include:

- `structured_output_not_supported`
- `structured_response_invalid`
- `legacy_response_shape_used`
- `text_fallback_used`
- `url_fallback_used`
- `summary_citations_unavailable`
- `sources_missing_or_unverifiable`
- `provider_reported_no_live_access`
- `no_results`
- `published_at_unparseable`

Non-search tool warnings currently include:

- `not_found_page`
- `placeholder_page`
- `partial_content`

Warning normalization:

- upstream aliases such as `no_results_found` and `no_relevant_results` are
  normalized to `no_results`
- upstream 404-like aliases such as `404_page`, `404_page_not_found`, and
  `page_not_found` are normalized to `not_found_page`

For empty search results:

- `status=empty` with `no_results` means the request completed but nothing usable
  matched
- `status=empty` with `provider_reported_no_live_access` means the upstream
  explicitly claimed it could not browse

For page extraction and outline results:

- `extract_url` uses `status=empty` when the target clearly looks like a 404 or
  not-found page
- `outline_url` uses `status=partial` when it can still summarize the page shell
  of a 404 / placeholder page but should not present it as a healthy document

## Claude Code / Cursor MCP Config

For local development from this repository checkout:

```json
{
  "mcpServers": {
    "searchbridge": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/mcp-searchbridge",
        "mcp-searchbridge"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "your-primary-model,your-fallback-model"
      }
    }
  }
}
```

For running directly from GitHub without publishing to PyPI:

```json
{
  "mcpServers": {
    "searchbridge": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/mizuikki/mcp-searchbridge@main",
        "mcp-searchbridge"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_MODEL": "your-primary-model,your-fallback-model"
      }
    }
  }
}
```

Use a tag or commit SHA instead of `@main` if you want a reproducible setup.

## Development

Run checks:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Verification Layers

This repository currently uses two different verification layers for the
`private_http` path:

- `tests/test_private_backend_real_integration.py` is a local source-tree API
  contract smoke test. It starts the public MCP server from this repo and the
  private API from the sibling `searchbridge-core` repo, but it does not
  provision shared Postgres, Redis, or blob storage.
- `tests/test_compose_real_topology.py` is the canonical split-topology
  verification path. It validates the real `mcp-searchbridge` +
  `searchbridge-core-api` + `searchbridge-core-worker` + shared Postgres +
  shared Redis + shared blob storage stack from workspace sources.

Use the compose topology test when you need evidence that worker job
consumption, shared-state retrieval, sync, dedup, or other cross-process
behavior is actually working.
