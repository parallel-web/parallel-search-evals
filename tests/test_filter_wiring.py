"""Known-source policy is local to the shared HTTP boundary, with an exact opt-out."""

import json

import httpx
import pytest

from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools.parallel_client import ParallelClient

BLOCKED = "https://huggingface.co/datasets/cais/hle/resolve/main/test.jsonl"
ALLOWED = "https://example.org/evidence"


@pytest.mark.parametrize("endpoint", ["search", "extract"])
@pytest.mark.parametrize("enabled", [None, True, False])
async def test_default_policy_filters_responses_without_changing_native_http_arguments(endpoint, enabled):
    raw = {
        "results": [
            {"url": BLOCKED, "excerpts": ["synthetic benchmark marker"]},
            {"url": ALLOWED, "future": [None, "é"]},
        ],
        "errors": [
            {"url": BLOCKED, "content": "synthetic blocked error marker"},
            {"url": ALLOWED, "content": "opaque error"},
        ],
        "future_metadata": {"retained": [None, True, "雪"]},
    }
    arguments = {"objective": "Evidence", "search_queries": ["public evidence"]}
    if endpoint == "extract":
        arguments["urls"] = [ALLOWED]
    seen = []

    def serve(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="https://api.parallel.ai") as client:
        options = {} if enabled is None else {"filter_benchmarks": enabled}
        engine = ParallelClient(mode="fast", model="fixture", client=client, **options)
        actual = await engine.request(endpoint, arguments)
    assert seen == [{**arguments, "client_model": "fixture", **({"mode": "fast"} if endpoint == "search" else {})}]
    if enabled is False:
        assert actual == raw
    else:
        assert actual == {**raw, "results": raw["results"][1:], "errors": raw["errors"][1:]}
    assert len(raw["results"]) == 2 and len(raw["errors"]) == 2


@pytest.mark.parametrize("enabled", [None, True])
async def test_extract_blocks_entire_mixed_batch_before_http(enabled):
    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="https://api.parallel.ai") as client:
        options = {} if enabled is None else {"filter_benchmarks": enabled}
        engine = ParallelClient(client=client, **options)
        with pytest.raises(NativeResponseError, match="^Benchmark source URLs are blocked by the retrieval filter$"):
            await engine.request("extract", {"objective": "Evidence", "urls": [ALLOWED, BLOCKED]})
    assert requests == []


async def test_optout_allows_known_source_extract_and_preserves_full_native_body():
    requests = []
    raw = {"results": [{"url": BLOCKED, "full_content": "synthetic marker"}], "future": {"opaque": "retained"}}

    def serve(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=raw)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve), base_url="https://api.parallel.ai") as client:
        engine = ParallelClient(client=client, filter_benchmarks=False)
        actual = await engine.request("extract", {"objective": "Evidence", "urls": [BLOCKED]})
    assert requests == [{"objective": "Evidence", "urls": [BLOCKED], "client_model": ""}]
    assert actual == raw
