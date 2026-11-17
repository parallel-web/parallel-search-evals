"""Service response defaults preserve provider evidence through real action boundaries."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from parallel_search_evals.models import ResponsesClient
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import (
    EXTRACT,
    SEARCH,
    ExtractInput,
    SearchInput,
)


@pytest.mark.parametrize("kind", ["search", "extract"])
@pytest.mark.parametrize("mode", ["turbo", "fast", "basic", "advanced"])
async def test_request_uses_service_response_defaults(kind, mode):
    arguments = {
        "objective": "Evidence",
        **({"search_queries": ["evidence"]} if kind == "search" else {"urls": ["https://example.org"]}),
    }
    sent = []

    def serve(request):
        sent.append(request)
        return httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        research_client = ParallelClient(mode=mode, model="fixture", client=client)
        assert await research_client.request(kind, arguments) == {"results": []}
    assert len(sent) == 1
    assert sent[0].url.path == f"/v1/{kind}"
    assert json.loads(sent[0].content) == {
        **arguments,
        "client_model": "fixture",
        **({"mode": mode} if kind == "search" else {}),
    }


@pytest.mark.parametrize("kind", ["search", "extract"])
async def test_http_response_preserves_long_unicode_and_native_fields(kind):
    original = {
        "results": [{"excerpts": ["😀e\u0301界" * 12001, None, {"future": True}]} for _ in range(12)],
        "warnings": [{"type": "provider_warning"}],
        "errors": [{"url": "https://example.org/failed"}],
        "content_budget": {"provider": True},
        "session_id": "native",
        "unknown": [None, True],
    }
    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=original)),
    ) as client:
        result = await ParallelClient(client=client).request(
            kind,
            {
                "objective": "Evidence",
                **({"search_queries": ["evidence"]} if kind == "search" else {"urls": ["https://example.org"]}),
            },
        )
    assert result == original


@pytest.mark.parametrize("definition", [SEARCH, EXTRACT])
def test_tool_schema_leaves_upper_argument_limits_to_service(definition):
    serialized = json.dumps(definition.parameters)
    assert "maxLength" not in serialized
    assert "maxItems" not in serialized


@pytest.mark.parametrize("kind", ["search", "extract"])
async def test_large_valid_arguments_reach_service_unchanged(kind):
    arguments = {"objective": "界" * 5001, "search_queries": ["😀" * 501]}
    if kind == "extract":
        arguments["urls"] = [f"https://example.org/{i}" for i in range(21)]
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"results": [], "warnings": [{"type": "service_limit"}]})

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        research_client = ParallelClient(client=client)
        action = SEARCH if kind == "search" else EXTRACT
        schema = SearchInput if kind == "search" else ExtractInput
        validated = schema.model_validate(arguments)
        result = await research_client.request(action.endpoint, validated.model_dump(exclude_none=True))
    assert sent == [{**arguments, "client_model": "", **({"mode": "basic"} if kind == "search" else {})}]
    assert result == {"results": [], "warnings": [{"type": "service_limit"}]}


@pytest.mark.parametrize("kind", ["search", "extract"])
async def test_tool_output_continues_through_actual_sdk_with_provider_json(kind):
    import json

    original = {
        "results": [{"url": "https://example.org", "excerpts": ["😀e\u0301" * 5000, {"future": None}, "界" * 14000]}],
        "unknown": [None, True],
        "warnings": [{"type": "provider_warning", "detail": "unaltered"}],
        "errors": [{"url": "https://example.org/failed", "detail": "provider error"}],
        "content_budget": {"provider": "native field"},
    }
    expected = original
    seen = []

    def serve(request):
        if request.url.host == "api.parallel.ai":
            return httpx.Response(200, json=original)
        body = json.loads(request.content)
        seen.append(body)
        if len(seen) == 1:
            arguments = (
                {"objective": "Synthetic", "search_queries": ["synthetic"]}
                if kind == "search"
                else {"objective": "Synthetic", "urls": ["https://example.org"]}
            )
            output = [
                {
                    "type": "function_call",
                    "call_id": "native-call",
                    "name": "search_web" if kind == "search" else "web_fetch",
                    "arguments": json.dumps(arguments),
                }
            ]
        else:
            output = [{"type": "message", "content": [{"type": "output_text", "text": "Done"}]}]
        return httpx.Response(200, json={"status": "completed", "output": output})

    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)
    ) as parallel_client:
        research_client = ParallelClient(client=parallel_client)
        action = SEARCH if kind == "search" else EXTRACT
        tools = ToolDispatcher([action], client=research_client)
        async with AsyncOpenAI(
            api_key="fixture", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve))
        ) as client:
            model_client = ResponsesClient("fixture", client=client, max_tokens=64, effort=None)
            history = model_client.new_history().append_question("Q")
            first = await model_client.generate(history, tools)
            history.accept_reply(first)
            results = await tools.execute(first.tool_calls)
            assert not results[0].is_error
            assert json.loads(results[0].output) == expected
            history.append_tool_outputs(results)
            assert (await model_client.generate(history, ToolDispatcher([]))).text == "Done"
    assert len(seen) == 2
    assert seen[1]["input"][-1]["call_id"] == "native-call"
    assert json.loads(seen[1]["input"][-1]["output"]) == expected
