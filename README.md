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
  "answer": "Summary text",
  "sources": [
    {
      "title": "OpenAI release notes",
      "url": "https://example.com/release-notes",
      "snippet": "Relevant source snippet"
    }
  ],
  "provider": "openai-compatible",
  "model": "your-model",
  "raw_text": "Raw upstream content",
  "warnings": []
}
```

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
