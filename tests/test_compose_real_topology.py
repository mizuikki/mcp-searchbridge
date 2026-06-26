from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import httpx
import pytest

from tests.helpers import REPO_ROOT, SEARCHBRIDGE_CORE_ROOT, find_free_port, host_port

PRIVATE_TOKEN = "compose-secret-token"
COMPOSE_SERVICES = (
    "postgres",
    "redis",
    "rustfs",
    "searchbridge-core-api",
    "searchbridge-core-worker",
    "mcp-searchbridge",
)


class _FetchFixtureState:
    def __init__(self) -> None:
        self.lock = Lock()
        self.request_count = 0
        self.dedup_request_count = 0
        self.not_found_count = 0
        self.embedding_request_count = 0
        self.last_embedding_request: dict[str, object] | None = None
        self.embedding_mode = "ok"
        self.mode = "fresh"
        self.etag = '"compose-fetch-v1"'
        self.last_modified = "Wed, 21 Oct 2015 07:28:00 GMT"
        self.body = (
            "# Remote Compose Doc\n\n"
            "## Overview\n\n"
            "Remote fetch content served through the compose smoke fixture.\n"
        )
        self.dedup_body = (
            "# Dedup Compose Doc\n\n"
            "## Overview\n\n"
            "Concurrent compose fetches should collapse to a single upstream hit.\n"
        )


class _FetchFixtureHandler(BaseHTTPRequestHandler):
    state: _FetchFixtureState

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/cached-doc":
            with self.state.lock:
                self.state.request_count += 1
                mode = self.state.mode
                etag = self.state.etag
                last_modified = self.state.last_modified
                body = self.state.body

            if mode == "not_modified":
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.end_headers()
                    return
                if self.headers.get("If-Modified-Since") == last_modified:
                    self.send_response(304)
                    self.end_headers()
                    return

            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown")
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/missing-doc":
            with self.state.lock:
                self.state.not_found_count += 1
            self.send_response(404)
            self.end_headers()
            return

        if self.path == "/dedup-doc":
            with self.state.lock:
                self.state.dedup_request_count += 1
                body = self.state.dedup_body
            time.sleep(1.0)
            payload = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/markdown")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if self.path == "/embeddings":
            content_length = int(self.headers.get("Content-Length", "0"))
            request_body = json.loads(self.rfile.read(content_length).decode())
            inputs = request_body["input"]
            assert isinstance(inputs, list)

            with self.state.lock:
                self.state.embedding_request_count += 1
                self.state.last_embedding_request = request_body
                embedding_mode = self.state.embedding_mode

            if embedding_mode == "error":
                payload = json.dumps(
                    {
                        "error": {
                            "message": "fixture embedding failure",
                            "type": "server_error",
                        }
                    }
                ).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            data = []
            for index, text in enumerate(inputs):
                text_value = str(text).lower()
                signal = 0.95 if "worker jobs" in text_value else 0.10
                embedding = [0.0] * 16
                embedding[0] = signal
                embedding[1] = 1.0 - signal
                data.append(
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": embedding,
                    }
                )

            payload = json.dumps(
                {
                    "object": "list",
                    "data": data,
                    "model": request_body["model"],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/embeddings":
            content_length = int(self.headers.get("Content-Length", "0"))
            request_body = json.loads(self.rfile.read(content_length).decode())
            inputs = request_body["input"]
            assert isinstance(inputs, list)

            with self.state.lock:
                self.state.embedding_request_count += 1
                self.state.last_embedding_request = request_body

            data = []
            for index, text in enumerate(inputs):
                text_value = str(text).lower()
                signal = 0.95 if "worker jobs" in text_value else 0.10
                embedding = [0.0] * 16
                embedding[0] = signal
                embedding[1] = 1.0 - signal
                data.append(
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": embedding,
                    }
                )

            payload = json.dumps(
                {
                    "object": "list",
                    "data": data,
                    "model": request_body["model"],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


@contextmanager
def run_fetch_fixture_server() -> Iterator[tuple[str, _FetchFixtureState]]:
    state = _FetchFixtureState()

    class Handler(_FetchFixtureHandler):
        pass

    Handler.state = state
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        _host, port = host_port(server.server_address)
        yield (f"http://127.0.0.1:{port}", state)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _host_docker_url(base_url: str, path: str) -> str:
    port = base_url.rsplit(":", 1)[1]
    return f"http://host.docker.internal:{port}{path}"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False

    result = subprocess.run(
        ["docker", "compose", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def _run_compose(
    *args: str,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", *args]
    result = subprocess.run(
        command,
        cwd=SEARCHBRIDGE_CORE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        msg = (
            f"docker compose command failed: {command!r}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        raise RuntimeError(msg)
    return result


def _compose_exec(
    service: str,
    *args: str,
    env: dict[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", "exec", "-T", service, *args]
    result = subprocess.run(
        command,
        cwd=SEARCHBRIDGE_CORE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        msg = (
            f"docker compose exec failed: {command!r}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
        raise RuntimeError(msg)
    return result


def _wait_for_http_ready(
    *,
    base_url: str,
    api_token: str,
    timeout_seconds: float = 60.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    headers = {"Authorization": f"Bearer {api_token}"}
    capabilities_url = f"{base_url}/v1/capabilities"
    last_error: str | None = None

    while time.monotonic() < deadline:
        try:
            response = httpx.get(capabilities_url, headers=headers, timeout=2.0)
            if response.status_code == 200:
                return
            last_error = f"unexpected readiness status {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        time.sleep(0.5)

    msg = f"compose backend did not become ready at {capabilities_url}"
    if last_error:
        msg = f"{msg}; last_error={last_error}"
    raise RuntimeError(msg)


def _wait_for_job_completion(
    *,
    base_url: str,
    api_token: str,
    source_alias: str,
    job_kind: str,
    timeout_seconds: float = 30.0,
) -> dict[str, object]:
    deadline = time.monotonic() + timeout_seconds
    headers = {"Authorization": f"Bearer {api_token}"}
    url = f"{base_url}/internal/v1/job_runs"

    while time.monotonic() < deadline:
        response = httpx.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
        runs = payload["job_runs"]
        matching_runs = [
            run
            for run in runs
            if run["source_alias"] == source_alias and run["job_kind"] == job_kind
        ]
        if matching_runs:
            latest = matching_runs[-1]
            if latest["status"] == "completed":
                return latest
            if latest["status"] == "failed":
                msg = f"compose job failed: {latest}"
                raise RuntimeError(msg)
        time.sleep(0.5)

    msg = f"timed out waiting for {job_kind} job completion for {source_alias}"
    raise RuntimeError(msg)


@pytest.mark.skipif(not _docker_available(), reason="docker compose is unavailable")
def test_compose_topology_builds_from_workspace_sources_and_serves_db_backed_api() -> (
    None
):
    project_name = f"searchbridge-smoke-{uuid.uuid4().hex[:8]}"
    api_port = find_free_port()
    postgres_port = find_free_port()
    redis_port = find_free_port()
    rustfs_port = find_free_port()
    rustfs_console_port = find_free_port()

    env = os.environ.copy()
    env.update(
        {
            "COMPOSE_PROJECT_NAME": project_name,
            "SEARCHBRIDGE_CORE_API_TOKEN": PRIVATE_TOKEN,
            "SEARCHBRIDGE_CORE_API_PORT": str(api_port),
            "SEARCHBRIDGE_CORE_POSTGRES_PORT": str(postgres_port),
            "SEARCHBRIDGE_CORE_REDIS_PORT": str(redis_port),
            "SEARCHBRIDGE_CORE_RUSTFS_PORT": str(rustfs_port),
            "SEARCHBRIDGE_CORE_RUSTFS_CONSOLE_PORT": str(rustfs_console_port),
            "OPENAI_API_KEY": "compose-openai-key",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "OPENAI_MODEL": "gpt-5",
            "SEARCHBRIDGE_CORE_EMBEDDING_PROVIDER_URL": "http://host.docker.internal:1",
            "SEARCHBRIDGE_CORE_EMBEDDING_API_KEY": "compose-embedding-key",
            "SEARCHBRIDGE_CORE_EMBEDDING_MODEL": "text-embedding-3-small",
        }
    )

    config = _run_compose("config", env=env)
    assert f"context: {REPO_ROOT}" in config.stdout
    assert f"context: {SEARCHBRIDGE_CORE_ROOT}" in config.stdout

    with run_fetch_fixture_server() as (fixture_base_url, fixture_state):
        env["SEARCHBRIDGE_CORE_EMBEDDING_PROVIDER_URL"] = _host_docker_url(
            fixture_base_url, ""
        ).rstrip("/")
        try:
            _run_compose(
                "up",
                "-d",
                "--build",
                *COMPOSE_SERVICES,
                env=env,
            )

            base_url = f"http://127.0.0.1:{api_port}"
            headers = {"Authorization": f"Bearer {PRIVATE_TOKEN}"}
            _wait_for_http_ready(base_url=base_url, api_token=PRIVATE_TOKEN)

            running = _run_compose("ps", "--services", "--status", "running", env=env)
            running_services = {
                line.strip() for line in running.stdout.splitlines() if line.strip()
            }
            assert running_services == set(COMPOSE_SERVICES)

            source_alias = "compose-smoke"
            seed_response = httpx.post(
                f"{base_url}/internal/v1/sources",
                headers=headers,
                json={
                    "alias": source_alias,
                    "display_name": "Compose Smoke Docs",
                    "platform_kind": "static",
                    "canonical_url": "https://compose.test/docs/",
                    "description": "Compose smoke-test source.",
                    "version": "latest",
                    "version_aliases": ["latest"],
                    "documents": [
                        {
                            "title": "Compose Intro",
                            "url": "https://compose.test/docs/intro",
                            "content": (
                                "# Compose Intro\n\n"
                                "## Overview\n\n"
                                "Compose verification stores normalized content "
                                "in Postgres metadata and blob storage.\n\n"
                                "## Details\n\n"
                                "Worker jobs are executed through the Postgres "
                                "task table.\n"
                            ),
                            "content_format": "markdown",
                            "outline": [
                                {
                                    "title": "Overview",
                                    "summary": "Compose verification overview.",
                                },
                                {
                                    "title": "Details",
                                    "summary": "Compose worker processing details.",
                                },
                            ],
                        }
                    ],
                },
                timeout=10.0,
            )
            seed_response.raise_for_status()
            seeded = seed_response.json()["source"]
            assert seeded["alias"] == source_alias

            extract_response = httpx.post(
                f"{base_url}/v1/extract_url",
                headers=headers,
                json={
                    "url": "https://compose.test/docs/intro",
                    "mode": "markdown",
                    "max_chars": 1200,
                },
                timeout=10.0,
            )
            extract_response.raise_for_status()
            extract = extract_response.json()
            assert extract["diagnostics"]["retrieval_method"] == "registry_blob"
            assert extract["diagnostics"]["source_id"]

            docs_response = httpx.post(
                f"{base_url}/v1/docs_qa",
                headers=headers,
                json={
                    "question": "How are worker jobs executed?",
                    "url": "https://compose.test/docs/intro",
                    "domain_allowlist": ["compose.test"],
                    "answer_mode": "standard",
                },
                timeout=10.0,
            )
            docs_response.raise_for_status()
            docs = docs_response.json()
            assert docs["diagnostics"]["source_id"] == seeded["source_id"]
            assert docs["diagnostics"]["selected_document_ids"]
            assert docs["diagnostics"]["selected_chunk_ids"]
            assert docs["diagnostics"]["retrieval_method"] == "pg_fts_pgvector_hybrid"
            with fixture_state.lock:
                assert fixture_state.embedding_request_count >= 2
                assert fixture_state.last_embedding_request is not None
                assert (
                    fixture_state.last_embedding_request["model"]
                    == "text-embedding-3-small"
                )
                assert fixture_state.last_embedding_request["dimensions"] == 16
                fixture_state.embedding_mode = "error"

            facade_check = _compose_exec(
                "mcp-searchbridge",
                "uv",
                "run",
                "python",
                "-c",
                textwrap.dedent(
                    """\
                    import asyncio
                    from mcp_searchbridge.config import Settings
                    from mcp_searchbridge.private_backend import (
                        PrivateHttpAggregationBackend,
                    )
                    from mcp_searchbridge.models import DocsQARequest

                    async def main():
                        backend = PrivateHttpAggregationBackend(Settings())
                        try:
                            result = await backend.docs_qa(DocsQARequest(
                                question='How are worker jobs executed?',
                                url='https://compose.test/docs/intro',
                                domain_allowlist=['compose.test'],
                                answer_mode='standard',
                            ))
                            print(result.diagnostics.backend_kind)
                            print(result.diagnostics.source_id)
                            print(result.answer)
                        finally:
                            await backend.aclose()

                    asyncio.run(main())
                    """
                ),
                env=env,
            )
            facade_lines = [
                line.strip()
                for line in facade_check.stdout.splitlines()
                if line.strip()
            ]
            assert facade_lines[0] == "private_http"
            assert facade_lines[1] == seeded["source_id"]
            assert (
                "Worker jobs are executed through the Postgres task table."
                in facade_lines[-1]
            )

            trigger_response = httpx.post(
                f"{base_url}/internal/v1/index_rebuild",
                headers=headers,
                json={"source_alias": source_alias},
                timeout=10.0,
            )
            trigger_response.raise_for_status()
            trigger = trigger_response.json()
            assert trigger["accepted"] is True
            assert trigger["job_kind"] == "index_rebuild"

            completed_run = _wait_for_job_completion(
                base_url=base_url,
                api_token=PRIVATE_TOKEN,
                source_alias=source_alias,
                job_kind="index_rebuild",
            )
            assert completed_run["status"] == "completed"

            docs_fallback_response = httpx.post(
                f"{base_url}/v1/docs_qa",
                headers=headers,
                json={
                    "question": "How are worker jobs executed?",
                    "url": "https://compose.test/docs/intro",
                    "domain_allowlist": ["compose.test"],
                    "answer_mode": "standard",
                },
                timeout=10.0,
            )
            docs_fallback_response.raise_for_status()
            docs_fallback = docs_fallback_response.json()
            assert docs_fallback["diagnostics"]["source_id"] == seeded["source_id"]
            assert docs_fallback["diagnostics"]["selected_document_ids"]
            assert docs_fallback["diagnostics"]["selected_chunk_ids"]
            assert (
                docs_fallback["diagnostics"]["retrieval_method"]
                == "pg_fts_pgvector_hybrid"
            )
            assert (
                "Worker jobs are executed through the Postgres task table."
                in docs_fallback["answer"]
            )

            cached_doc_url = _host_docker_url(fixture_base_url, "/cached-doc")
            fresh_fetch = httpx.post(
                f"{base_url}/v1/extract_url",
                headers=headers,
                json={
                    "url": cached_doc_url,
                    "mode": "markdown",
                    "max_chars": 1200,
                },
                timeout=10.0,
            )
            fresh_fetch.raise_for_status()
            fresh_payload = fresh_fetch.json()
            assert fresh_payload["diagnostics"]["retrieval_method"] == "fetch_writeback"
            assert (
                "Remote fetch content served through the compose smoke fixture."
                in fresh_payload["content"]
            )
            with fixture_state.lock:
                first_request_count = fixture_state.request_count
                fixture_state.mode = "not_modified"

            sync_trigger = httpx.post(
                f"{base_url}/internal/v1/sync",
                headers=headers,
                json={"source_alias": "cached-doc"},
                timeout=10.0,
            )
            sync_trigger.raise_for_status()
            sync_run = _wait_for_job_completion(
                base_url=base_url,
                api_token=PRIVATE_TOKEN,
                source_alias="cached-doc",
                job_kind="sync",
            )
            assert sync_run["status"] == "completed"
            with fixture_state.lock:
                assert fixture_state.request_count == first_request_count + 1

            revalidated_fetch = httpx.post(
                f"{base_url}/v1/extract_url",
                headers=headers,
                json={
                    "url": cached_doc_url,
                    "mode": "markdown",
                    "max_chars": 1200,
                },
                timeout=10.0,
            )
            revalidated_fetch.raise_for_status()
            revalidated_payload = revalidated_fetch.json()
            assert revalidated_payload["content"] == fresh_payload["content"]
            assert (
                revalidated_payload["diagnostics"]["retrieval_method"]
                == "registry_blob"
            )

            dedup_doc_url = _host_docker_url(fixture_base_url, "/dedup-doc")

            def fetch_dedup_doc() -> httpx.Response:
                return httpx.post(
                    f"{base_url}/v1/extract_url",
                    headers=headers,
                    json={
                        "url": dedup_doc_url,
                        "mode": "markdown",
                        "max_chars": 1200,
                    },
                    timeout=15.0,
                )

            with ThreadPoolExecutor(max_workers=2) as executor:
                dedup_first, dedup_second = list(
                    executor.map(lambda _unused: fetch_dedup_doc(), range(2))
                )

            dedup_first.raise_for_status()
            dedup_second.raise_for_status()
            assert (
                "Concurrent compose fetches should collapse"
                in dedup_first.json()["content"]
            )
            assert (
                "Concurrent compose fetches should collapse"
                in dedup_second.json()["content"]
            )
            with fixture_state.lock:
                assert fixture_state.dedup_request_count == 1

            missing_doc_url = _host_docker_url(fixture_base_url, "/missing-doc")
            missing_first = httpx.post(
                f"{base_url}/v1/extract_url",
                headers=headers,
                json={
                    "url": missing_doc_url,
                    "mode": "markdown",
                    "max_chars": 1200,
                },
                timeout=10.0,
            )
            assert missing_first.status_code == 502
            with fixture_state.lock:
                first_not_found_count = fixture_state.not_found_count

            missing_second = httpx.post(
                f"{base_url}/v1/extract_url",
                headers=headers,
                json={
                    "url": missing_doc_url,
                    "mode": "markdown",
                    "max_chars": 1200,
                },
                timeout=10.0,
            )
            assert missing_second.status_code == 502
            with fixture_state.lock:
                assert fixture_state.not_found_count == first_not_found_count
        finally:
            _run_compose("down", "-v", env=env, check=False)
