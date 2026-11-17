"""Apply the packaged known-source URL policy locally to native tool responses.

The guard recognizes enumerated URL prefixes, not question text or unknown
mirrors. Only matching result/error entries are removed; admitted nested values,
opaque entries and all other native metadata retain their original contents.
"""

import json
import posixpath
from collections.abc import Iterable
from importlib.resources import files
from typing import Any
from urllib.parse import unquote, urlsplit

from parallel_search_evals.runtime import NativeResponseError

BENCHMARK_SOURCES: tuple[str, ...] = tuple(
    json.loads(files("parallel_search_evals.benchmarks").joinpath("assets/benchmark-sources.json").read_text())
)


def _normalized_url(url: object) -> str:
    """Canonicalize source identity while preserving case-sensitive dataset paths.

    Decode before resolving dot segments; only GitHub repository aliases fold case.
    """
    if not isinstance(url, str):
        return ""
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"}:
            return ""
        _ = parsed.port
        host = (parsed.hostname or "").lower().rstrip(".").removeprefix("www.")
        path = posixpath.normpath("/" + unquote(parsed.path).lstrip("/"))
        if host in {"github.com", "raw.githubusercontent.com", "api.github.com"}:
            path = path.lower()
        return "https://" + host + path.rstrip("/") + "/"
    except ValueError:
        return ""


def is_benchmark_source(url: object) -> bool:
    """Match only the enumerated host/path prefixes at a path boundary."""
    return _normalized_url(url).startswith(BENCHMARK_SOURCES)


def validate_extract_urls(urls: Iterable[object]) -> None:
    """Reject a whole Extract batch before any known benchmark source is fetched."""
    if any(is_benchmark_source(url) for url in urls):
        raise NativeResponseError("Benchmark source URLs are blocked by the retrieval filter")


def _allowed_entry(entry: object) -> bool:
    if not isinstance(entry, dict):
        return True
    return not is_benchmark_source(entry.get("url"))


def filter_response(response: dict[str, Any]) -> dict[str, Any]:
    """Remove matching result and error entries without changing other data."""
    filtered = response.copy()
    for field in ("results", "errors"):
        if isinstance(response.get(field), list):
            filtered[field] = list(filter(_allowed_entry, response[field]))
    return filtered
