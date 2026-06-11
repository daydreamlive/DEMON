"""Pytest wiring for the golden/latency suite.

These tests need a live streaming server. Resolution order:

    pytest tests/golden --pod-url ws://HOST:1318   # explicit pod
    DEMON_POD_URL=ws://HOST:1318 pytest tests/golden
    pytest tests/golden                            # local CUDA box:
                                                   # spawns the server

Without a pod URL and without CUDA every test SKIPS: the suite never
breaks a plain ``pytest tests/unit`` environment. Each scenario's live session runs
ONCE per pytest session; the golden comparison and the latency checks
both read from that single run.
"""

import os
from pathlib import Path

import pytest

from .runner import run_scenario
from .scenarios import SCENARIOS_BY_NAME


def pytest_addoption(parser):
    parser.addoption(
        "--pod-url", action="store", default=None,
        help="ws://HOST:PORT of a live DEMON streaming server "
             "(default: $DEMON_POD_URL)")


@pytest.fixture(scope="session")
def pod_url(request, tmp_path_factory) -> str:
    url = (request.config.getoption("--pod-url")
           or os.environ.get("DEMON_POD_URL"))
    if url:
        return url
    # Local GPU is a first-class target: spawn the server ourselves.
    from .local_server import LocalServer, cuda_available

    if not cuda_available():
        pytest.skip("golden suite needs a live server: pass --pod-url, "
                    "set DEMON_POD_URL, or run on a CUDA machine")
    log = tmp_path_factory.mktemp("local-server") / "server.log"
    server = LocalServer(log_path=log)
    request.addfinalizer(server.stop)
    return server.url


@pytest.fixture(scope="session")
def run_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("golden-runs")


@pytest.fixture(scope="session")
def scenario_runs(pod_url, run_root):
    """Lazy per-scenario run cache shared by every test in the session."""
    cache: dict = {}

    def get(name: str) -> tuple[dict, Path]:
        if name not in cache:
            out_dir = run_root / name
            result = run_scenario(pod_url, SCENARIOS_BY_NAME[name],
                                  out_dir, save_blobs=False)
            cache[name] = (result, out_dir)
        return cache[name]

    return get
