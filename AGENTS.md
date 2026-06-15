# AGENTS.md

This file provides shared repository guidance for AI coding assistants and
automation agents working in this repository.

`CLAUDE.md` is a symlink to this file, so both paths refer to the same
instructions. Update `AGENTS.md` only; do not maintain a separate `CLAUDE.md`.

## Project Structure & Module Organization
This repository is a small Python MCP server managed with `uv`.

- `src/mcp_searchbridge/`: application code
- `tests/`: pytest test suite
- `.github/workflows/ci.yml`: CI for lint and tests
- `pyproject.toml`: dependencies, Ruff config, project metadata
- `tmp/`: local experiment output only; do not commit generated files

Keep new runtime code under `src/mcp_searchbridge/`. Add tests alongside the relevant behavior in `tests/`, using focused files such as `test_parser.py` or `test_server_smoke.py`.

## Build, Test, and Development Commands
- `uv sync --dev`: install runtime and dev dependencies
- `uv run mcp-searchbridge`: run the MCP server over stdio
- `uv run python -m mcp_searchbridge.server`: run the server module directly
- `uv run ruff check .`: run lint checks
- `uv run ruff format --check .`: verify formatting
- `uv run ruff check --fix . && uv run ruff format .`: apply auto-fixes
- `uv run pytest`: run the full test suite

CI runs the same Ruff and pytest commands on `push` and `pull_request`.

## Coding Style & Naming Conventions
Target Python is `3.14`. Use 4-space indentation, type hints, and concise docstrings where helpful. Follow existing naming:

- modules: `snake_case`
- functions/variables: `snake_case`
- classes: `PascalCase`
- constants: `UPPER_SNAKE_CASE`

Formatting and linting are enforced by Ruff. The repository uses a pragmatic rule set (`E`, `F`, `I`, `UP`, `B`, `SIM`) with line length `88`.

## Testing Guidelines
Use `pytest` and `pytest-asyncio`. Name test files `test_*.py` and test functions `test_*`. Prefer small, behavior-focused tests with local fake HTTP handlers instead of hitting real upstream services by default.

Before opening a PR, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Commit & Pull Request Guidelines
This repository currently has no established commit history, so use short imperative commit messages, for example: `Add Ruff CI workflow`.

PRs should include:
- a brief summary of the change
- any config or env impacts
- test evidence (`uv run pytest`, `uv run ruff check .`)

## Security & Configuration Tips
Never commit real secrets. Keep local credentials in `.env`; only `.env.example` belongs in version control. When testing upstream providers, prefer sanitized examples in docs and tests.
