"""Known-source URL boundaries and lossless local response filtering."""

from copy import deepcopy
from typing import Any

import pytest

from parallel_search_evals.tools.source_filter import filter_response, is_benchmark_source


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/datasets/google/deepsearchqa",
        "https://huggingface.co/datasets/cais/hle/tree/main",
        "https://hf.co/datasets/google/deepsearchqa/resolve/main/data.json",
        "https://hf.co/datasets/cais/hle",
        "https://huggingface.co/api/datasets/google/deepsearchqa/parquet",
        "https://huggingface.co/api/datasets/cais/hle",
        "https://hf.co/api/datasets/google/deepsearchqa",
        "https://hf.co/api/datasets/cais/hle/tree/main",
        "https://github.com/centerforaisafety/hle",
        "https://github.com/centerforaisafety/hle/blob/main/data.json",
        "https://raw.githubusercontent.com/centerforaisafety/hle/main/data.json",
        "https://api.github.com/repos/centerforaisafety/hle/contents/data.json",
        "https://kaggle.com/code/andrewmingwang/deepsearchqa-starter-code",
        "https://www.kaggle.com/code/andrewmingwang/deepsearchqa-starter-code/output",
        "HTTP://WWW.HUGGINGFACE.CO.:8080/datasets/google/deepsearchqa/?download=1#data",
        "https://github.com/centerforaisafety/%68le/blob/main/data.json",
        "https://github.com/centerforaisafety/other/../hle/./data.json",
        "https://hf.co/datasets%2Fgoogle%2Fdeepsearchqa%2Fdata.json",
        "https://hf.co/datasets/google/other/%2e%2e/deepsearchqa",
        "https://hf.co//datasets/google/deepsearchqa",
        "https://reader@github.com/centerforaisafety/hle",
        "https://github.com/CenterForAISafety/HLE/blob/main/data.json",
        "https://raw.githubusercontent.com/CenterForAISafety/HLE/main/data.json",
        "https://api.github.com/repos/CenterForAISafety/HLE/contents/data.json",
    ],
)
def test_known_sources_and_url_aliases_are_blocked(url: str) -> None:
    assert is_benchmark_source(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/datasets/google/deepsearchqa-other",
        "https://huggingface.co/datasets/cais/hle2",
        "https://huggingface.co/datasets/google/other",
        "https://huggingface.co/google/deepsearchqa",
        "https://github.com/centerforaisafety/hle-tools",
        "https://github.com/centerforaisafety/other",
        "https://github.com/other/hle",
        "https://raw.githubusercontent.com/other/hle/main/data.json",
        "https://api.github.com/repos/centerforaisafety/hle-tools",
        "https://kaggle.com/code/andrewmingwang/deepsearchqa-starter-code-extra",
        "https://kaggle.com/code/other/deepsearchqa-starter-code",
        "https://github.com/centerforaisafety/hle/../other",
        "https://github.com.attacker.invalid/centerforaisafety/hle",
        "https://github.com@attacker.invalid/centerforaisafety/hle",
        "https://attacker.invalid/https://github.com/centerforaisafety/hle",
        "https://example.invalid/?url=https://github.com/centerforaisafety/hle",
        "https://example.invalid/#https://github.com/centerforaisafety/hle",
        "https://unrelated-huggingface.co/datasets/cais/hle",
        "https://huggingface.co/datasets/cais/hle%2Dother",
        "https://huggingface.co/datasets/CAIS/HLE",
        "https://hf.co/DATASETS/cais/hle",
        "https://github.com/CenterForAISafety/HLE-tools",
        "https://huggingface.co:invalid/datasets/cais/hle",
        "https://huggingface.co:99999/datasets/cais/hle",
        "ftp://github.com/centerforaisafety/hle",
        "//github.com/centerforaisafety/hle",
        "github.com/centerforaisafety/hle",
        "https://[invalid/datasets/cais/hle",
        "https://github.com\uff0f@attacker.invalid/centerforaisafety/hle",
        "https://",
        "",
        None,
        4,
        b"https://github.com/centerforaisafety/hle",
        {"url": "https://github.com/centerforaisafety/hle"},
        ["https://github.com/centerforaisafety/hle"],
    ],
)
def test_unrelated_or_malformed_urls_are_allowed(url: object) -> None:
    assert not is_benchmark_source(url)


def test_filter_preserves_allowed_payload_order_metadata_and_input() -> None:
    response: dict[str, Any] = {
        "results": [
            {"url": "https://example.invalid/a", "excerpts": ["first"], "future": {"x": [1]}},
            {"url": "https://hf.co/datasets/cais/hle", "excerpts": ["blocked"]},
            {"url": "https://github.com/centerforaisafety/other", "content": "second"},
            {"content": "opaque"},
            None,
            "future-entry",
            {"url": ["https://hf.co/datasets/cais/hle"], "content": "opaque-url"},
        ],
        "errors": [
            {"url": "https://github.com/centerforaisafety/hle", "message": "blocked content"},
            {"url": "https://example.invalid/error", "message": "allowed error", "extra": [2]},
            {"message": "opaque error"},
            17,
        ],
        "request_id": "synthetic-request",
        "warnings": ["native warning"],
        "future": {"url": "https://hf.co/datasets/cais/hle", "keep": True},
    }
    original = deepcopy(response)

    filtered = filter_response(response)

    assert filtered == {
        "results": [
            {"url": "https://example.invalid/a", "excerpts": ["first"], "future": {"x": [1]}},
            {"url": "https://github.com/centerforaisafety/other", "content": "second"},
            {"content": "opaque"},
            None,
            "future-entry",
            {"url": ["https://hf.co/datasets/cais/hle"], "content": "opaque-url"},
        ],
        "errors": [
            {"url": "https://example.invalid/error", "message": "allowed error", "extra": [2]},
            {"message": "opaque error"},
            17,
        ],
        "request_id": "synthetic-request",
        "warnings": ["native warning"],
        "future": {"url": "https://hf.co/datasets/cais/hle", "keep": True},
    }
    assert response == original
    assert filtered["results"][0] is response["results"][0]
    assert filtered["errors"][0] is response["errors"][1]
    assert filtered["future"] is response["future"]


@pytest.mark.parametrize("value", [None, {}, "opaque", 8, (), []])
def test_nonlist_result_fields_are_retained(value: Any) -> None:
    response = {"results": value, "errors": value, "future": "retained"}

    assert filter_response(response) == response


def test_missing_result_fields_remain_missing() -> None:
    assert filter_response({"future": "retained"}) == {"future": "retained"}


def test_all_blocked_entries_produce_empty_lists_without_fabricated_metadata() -> None:
    response = {
        "results": [{"url": "https://hf.co/datasets/cais/hle", "content": "blocked"}],
        "errors": [{"url": "https://github.com/centerforaisafety/hle", "message": "blocked"}],
        "request_id": "synthetic-request",
    }

    assert filter_response(response) == {"results": [], "errors": [], "request_id": "synthetic-request"}
