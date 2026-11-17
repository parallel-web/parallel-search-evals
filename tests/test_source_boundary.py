"""Non-file source paths must fail before a potentially blocking read."""

import os
from pathlib import Path

import pytest

from parallel_search_evals.benchmarks.base import BenchmarkQuestions


@pytest.mark.parametrize("suite", ["dsqa", "hle"])
@pytest.mark.parametrize("kind", ["missing", "directory", "fifo"])
def test_source_must_be_regular_file(tmp_path, monkeypatch, suite, kind):
    source = tmp_path / "source"
    if kind == "directory":
        source.mkdir()
    elif kind == "fifo":
        os.mkfifo(source)

    def forbidden_read(path):
        pytest.fail(f"Attempted a potentially blocking read of {kind}")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(ValueError, match="regular file"):
        BenchmarkQuestions(source, suite, 1, source_sha256="0" * 64, synthetic=True)
