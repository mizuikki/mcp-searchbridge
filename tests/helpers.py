from __future__ import annotations

import os
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
from mcp.client.stdio import StdioServerParameters

from mcp_searchbridge.config import Settings
from mcp_searchbridge.type_utils import parse_http_url, parse_optional_http_url

REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCHBRIDGE_CORE_ROOT = REPO_ROOT.parent / "searchbridge-core"
SEARCHBRIDGE_CORE_BOOTSTRAP_FIXTURE = (
    SEARCHBRIDGE_CORE_ROOT
    / "crates"
    / "searchbridge-core"
    / "tests"
    / "fixtures"
    / "bootstrap_sources.json"
)


def make_settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://api.example.com/v1",
        "OPENAI_MODEL": "test-model",
        **overrides,
    }
    # BaseSettings accepts these runtime kwargs even though static typing
    # doesn't expose them on the generated __init__.
    return cast(Settings, Settings(_env_file=None, **payload))  # pyright: ignore[reportCallIssue]


def url(value: str):
    return parse_http_url(value)


def optional_url(value: str | None):
    return parse_optional_http_url(value)


def host_port(server_address: object) -> tuple[str, int]:
    if (
        isinstance(server_address, tuple)
        and len(server_address) >= 2
        and isinstance(server_address[0], str)
        and isinstance(server_address[1], int)
    ):
        return server_address[0], server_address[1]
    msg = f"Unexpected server address shape: {server_address!r}"
    raise TypeError(msg)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class ManagedProcess:
    name: str
    command: tuple[str, ...]
    cwd: Path
    process: subprocess.Popen[str]
    _output: str | None = None

    def read_output(self) -> str:
        if self._output is not None:
            return self._output

        if self.process.stdout is None:
            self._output = ""
            return self._output

        self._output, _stderr = self.process.communicate(timeout=1)
        return self._output

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

        if self._output is None:
            self._output = self.read_output()


@contextmanager
def run_local_searchbridge_core_api(
    *,
    api_token: str,
    timeout_seconds: int = 5,
):
    """Start only the local private API for contract smoke tests.

    This helper intentionally does not provision shared Postgres/Redis/blob
    storage and therefore does not validate the split API/worker topology.
    The canonical shared-infra verification path lives in the compose smoke
    tests.
    """
    api_port = find_free_port()
    env = os.environ.copy()
    env.update(
        {
            "SEARCHBRIDGE_CORE_HOST": "127.0.0.1",
            "SEARCHBRIDGE_CORE_API_TOKEN": api_token,
            "SEARCHBRIDGE_CORE_TIMEOUT_SECONDS": str(timeout_seconds),
            "SEARCHBRIDGE_CORE_LOG_LEVEL": "info",
            "SEARCHBRIDGE_CORE_BOOTSTRAP_SEED_PATH": str(
                SEARCHBRIDGE_CORE_BOOTSTRAP_FIXTURE
            ),
        }
    )
    api_command = (
        "cargo",
        "run",
        "-p",
        "searchbridge-core",
        "--bin",
        "searchbridge-core-api",
    )
    api_env = env | {"SEARCHBRIDGE_CORE_PORT": str(api_port)}
    api_process = ManagedProcess(
        name="searchbridge-core-api",
        command=api_command,
        cwd=SEARCHBRIDGE_CORE_ROOT,
        process=subprocess.Popen(
            api_command,
            cwd=SEARCHBRIDGE_CORE_ROOT,
            env=api_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ),
    )
    try:
        wait_for_private_backend_ready(
            base_url=f"http://127.0.0.1:{api_port}",
            api_token=api_token,
            process=api_process,
        )
        yield {
            "base_url": f"http://127.0.0.1:{api_port}",
            "port": api_port,
            "api_process": api_process,
        }
    finally:
        api_process.stop()


def local_mcp_server_params(
    *,
    env_overrides: dict[str, str],
) -> StdioServerParameters:
    env = os.environ.copy()
    env.update(env_overrides)
    return StdioServerParameters(
        command="uv",
        args=["run", "mcp-searchbridge"],
        cwd=str(REPO_ROOT),
        env=env,
    )


def wait_for_private_backend_ready(
    *,
    base_url: str,
    api_token: str,
    process: ManagedProcess,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    capabilities_url = f"{base_url}/v1/capabilities"
    headers = {"Authorization": f"Bearer {api_token}"}

    while time.monotonic() < deadline:
        if process.process.poll() is not None:
            output = process.read_output()
            msg = (
                f"{process.name} exited before becoming ready "
                f"[cwd={process.cwd} command={process.command!r}]\n{output}"
            )
            raise RuntimeError(msg)

        try:
            response = httpx.get(capabilities_url, headers=headers, timeout=1.0)
            if response.status_code == 200:
                return
            last_error = f"unexpected readiness status {response.status_code}"
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        time.sleep(0.25)

    output = process.read_output() if process.process.poll() is not None else ""
    msg = (
        f"{process.name} did not become ready at {capabilities_url} within "
        f"{timeout_seconds:.1f}s [cwd={process.cwd} command={process.command!r}]"
    )
    if last_error:
        msg = f"{msg}; last_error={last_error}"
    if output:
        msg = f"{msg}\n{output}"
    raise RuntimeError(msg)
