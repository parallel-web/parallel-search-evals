"""Public modules must import independently, without a warmed agent package."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from parallel_search_evals import runtime


@pytest.mark.parametrize(
    "module",
    [
        "parallel_search_evals.tools.parallel_client",
        "parallel_search_evals.tools.schemas",
        "parallel_search_evals.tools",
        "parallel_search_evals.models",
        "parallel_search_evals.benchmarks",
        "parallel_search_evals.run_eval",
    ],
)
def test_public_module_imports_in_fresh_network_denied_process(module, tmp_path):
    code = """
import importlib
import socket
import sys
def deny(*args, **kwargs):
    raise AssertionError("Imports must not dispatch network requests")
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
socket.getaddrinfo = deny
importlib.import_module(sys.argv[1])
if sys.argv[1] == "parallel_search_evals.tools.schemas":
    assert "parallel_search_evals.agent" not in sys.modules
"""
    env = {**os.environ, "PYTHONPATH": str(Path(runtime.__file__).parent.parent)}
    result = subprocess.run(
        [sys.executable, "-c", code, module], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
