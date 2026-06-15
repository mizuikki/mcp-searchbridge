# mcp-searchbridge

`mcp-searchbridge` is a lightweight FastMCP server that exposes a single `web_search`
tool backed by an OpenAI-compatible Chat Completions endpoint.

It does not implement crawling, extraction, or its own search engine. Freshness,
live web access, and citations all depend on the upstream model or gateway you
configure.

## Features

- FastMCP server over stdio
- Single `web_search` tool with structured output
- OpenAI-compatible `base_url`, `api_key`, and `model` configuration
- JSON-first parsing with text fallback
- Basic timeout, retry, logging, and upstream error handling
- `uv`-managed Python 3.14 project with lockfile

## Requirements

- `uv`
- Python `3.14.x`

## Installation

```bash
uv python install 3.14
uv sync
```

## Configuration

Copy `.env.example` into `.env` or set environment variables directly:

```env
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your-search-capable-model
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=2
OPENAI_ORGANIZATION=
OPENAI_PROJECT=
SEARCHBRIDGE_SYSTEM_PROMPT=
SEARCHBRIDGE_DEFAULT_MAX_SOURCES=5
SEARCHBRIDGE_LOG_LEVEL=INFO
```

Required variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`

`mcp-searchbridge` reads `.env` for local runs, but Cursor / Claude Code MCP
configurations do not automatically inherit that file. Pass the same values in
the MCP `env` block when launching through an editor or client.

## Run

```bash
uv run mcp-searchbridge
```

This starts the MCP server on stdio.

## Tool

### `web_search`

Input:

```json
{
  "query": "What happened in the latest OpenAI release?",
  "recency": "latest",
  "max_sources": 5,
  "domain_allowlist": ["openai.com", "developers.openai.com"],
  "return_mode": "standard"
}
```

Output:

```json
{
  "query": {
    "text": "What happened in the latest OpenAI release?",
    "recency": "latest",
    "max_sources": 5,
    "domain_allowlist": ["openai.com", "developers.openai.com"],
    "return_mode": "standard"
  },
  "summary": {
    "text": "Short synthesis",
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
    "warnings": [],
    "error": null
  }
}
```

This response schema is intentionally LLM-oriented and breaking relative to
earlier `answer` / `raw_text` / `sources[].snippet` outputs.

Warning semantics:

- `no_results`: the search completed, but no matching sources were returned.
- `provider_reported_no_live_access`: preserve this only when the upstream
  response explicitly states that it could not browse or access the live web.
- `sources_missing_or_unverifiable`: no verifiable sources were extracted from
  the final normalized result.

For empty results, prefer interpreting warnings this way:

- `status=empty` with `no_results`: the query ran, but nothing usable matched.
- `status=empty` with `provider_reported_no_live_access`: the upstream model
  explicitly claimed it could not use live web access for the request.

## Claude Code / Cursor MCP config

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
        "OPENAI_MODEL": "your-search-capable-model"
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
        "OPENAI_MODEL": "your-search-capable-model"
      }
    }
  }
}
```

Use a tag or commit SHA instead of `@main` if you want a reproducible setup.

## Development

Run lint and formatting checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run tests:

```bash
uv run pytest
```

Apply automatic lint fixes and formatting:

```bash
uv run ruff check --fix .
uv run ruff format .
```

Run the package entry point directly:

```bash
uv run python -m mcp_searchbridge.server
```

## Notes

- This project only guarantees OpenAI-style Chat Completions compatibility.
- Some providers reject structured `response_format`; the server falls back to
  plain text parsing in that case.
- If the upstream model cannot browse the web, the tool may return a warning or
  state that live access is unavailable.
