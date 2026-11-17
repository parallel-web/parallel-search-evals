"""Local action boundaries preserve raw JSON objects and validate before reserving budget."""

import json
from importlib.metadata import version

import httpx
import pytest

from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import (
    EXTRACT,
    SEARCH,
    ExtractInput,
    SearchInput,
    ToolInvocation,
)

PAYLOAD = {
    "results": [{"url": "https://example.org", "excerpts": ["Evidence"], "future": None}],
    "unknown": {"nested": [None, {"flag": True, "number": 3.25}]},
    "model_dump": {"opaque": None},
    "output_schema": [],
}


@pytest.mark.parametrize("endpoint", ["search", "extract"])
async def test_owned_parallel_requests_identify_the_installed_package(endpoint, monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "fixture")

    async def serve(transport, request):
        assert f"parallel-search-evals/{version('parallel-search-evals')}" in request.headers["user-agent"]
        assert request.headers["user-agent"].startswith("AsyncParallel/Python ")
        assert request.headers["x-api-key"] == "fixture"
        return httpx.Response(200, json=PAYLOAD)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", serve)
    research_client = ParallelClient(model="fixture")
    try:
        arguments = {
            "objective": "Synthetic evidence",
            **({"search_queries": ["evidence"]} if endpoint == "search" else {"urls": ["https://example.org"]}),
        }
        assert await research_client.request(endpoint, arguments) == PAYLOAD
    finally:
        await research_client.sdk.close()


@pytest.mark.parametrize("endpoint", ["search", "extract"])
async def test_parallel_tool_returns_the_actual_json_object(endpoint):
    def serve(request):
        return httpx.Response(200, json=PAYLOAD)

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        research_client = ParallelClient(model="fixture", client=client)
        if endpoint == "search":
            action = SEARCH
            arguments = SearchInput(objective="Find evidence", search_queries=["evidence"])
        else:
            action = EXTRACT
            arguments = ExtractInput(objective="Read evidence", urls=["https://example.org"])
        result = await research_client.request(action.endpoint, arguments.model_dump(exclude_none=True))
    assert result == PAYLOAD


@pytest.mark.parametrize("endpoint", ["search", "extract"])
async def test_invalid_input_keeps_budget_then_one_raw_json_result_is_serialized(endpoint):
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=PAYLOAD)

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        research_client = ParallelClient(model="fixture", client=client)
        action = SEARCH if endpoint == "search" else EXTRACT
        tools = ToolDispatcher([action], client=research_client, max_calls=1)
        arguments = (
            {"objective": "Evidence", "search_queries": ["q"]}
            if endpoint == "search"
            else {
                "objective": "Evidence",
                "urls": ["https://example.org"],
            }
        )
        invalid = ToolInvocation(id="invalid", name=action.name, input={**arguments, "unknown_argument": True})
        invalid_result = (await tools.execute([invalid]))[0]
        assert invalid_result.is_error and sent == [] and tools.remaining_calls == 1
        valid = ToolInvocation(id="valid", name=action.name, input=arguments)
        result = (await tools.execute([valid]))[0]
        assert not result.is_error and json.loads(result.output) == PAYLOAD
        assert len(sent) == 1 and tools.remaining_calls == 0
        exhausted = (await tools.execute([valid]))[0]
        assert exhausted.is_error and exhausted.output == "Tool budget exhausted" and len(sent) == 1


@pytest.mark.parametrize("endpoint", ["search", "extract"])
@pytest.mark.parametrize("body", [[], None, 42, "not an object"])
async def test_non_object_tool_response_remains_a_visible_error(endpoint, body):
    sent = []

    def serve(request):
        sent.append(request)
        return httpx.Response(200, content=json.dumps(body).encode())

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        research_client = ParallelClient(model="fixture", client=client)
        action = SEARCH if endpoint == "search" else EXTRACT
        tools = ToolDispatcher([action], client=research_client, max_calls=1)
        arguments = (
            {"objective": "Evidence", "search_queries": ["q"]}
            if endpoint == "search"
            else {
                "objective": "Evidence",
                "urls": ["https://example.org"],
            }
        )
        result = (await tools.execute([ToolInvocation(id="invalid-response", name=action.name, input=arguments)]))[0]
        assert result.is_error and result.output == "Error: Parallel response must be a JSON object"
        assert len(sent) == 1 and tools.remaining_calls == 0


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_explicit_manual_parameters_are_used_even_when_empty(provider):
    from pydantic import BaseModel

    from parallel_search_evals.models.anthropic_messages import messages_tool_schemas
    from parallel_search_evals.models.openai_responses import responses_tool_schemas
    from parallel_search_evals.tools import ToolSpec

    class Input(BaseModel):
        required_input: str

    tool = ToolSpec("explicit_empty", "Explicit manual schema", {}, Input, "search")
    render = {"openai": responses_tool_schemas, "anthropic": messages_tool_schemas}[provider]
    definition = render((tool,))[0]
    if provider == "openai":
        assert definition == {
            "type": "function",
            "name": "explicit_empty",
            "description": "Explicit manual schema",
            "parameters": {},
        }
    else:
        assert definition == {
            "name": "explicit_empty",
            "description": "Explicit manual schema",
            "input_schema": {"additionalProperties": False},
        }


@pytest.mark.parametrize("endpoint", ["search", "extract"])
async def test_default_budget_reserves_twenty_before_await_and_keeps_batch_order(endpoint):
    import asyncio

    sent = []

    async def serve(request):
        await asyncio.sleep(0)
        body = json.loads(request.content)
        sent.append(body["objective"])
        return httpx.Response(200, json={"objective": body["objective"]})

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        research_client = ParallelClient(model="fixture", client=client)
        action = SEARCH if endpoint == "search" else EXTRACT
        calls = [
            ToolInvocation(
                id=f"call-{index}",
                name=action.name,
                input={
                    "objective": str(index),
                    **({"search_queries": ["q"]} if endpoint == "search" else {"urls": ["https://example.org"]}),
                },
            )
            for index in range(21)
        ]
        tools = ToolDispatcher([action], client=research_client)
        results = await tools.execute(calls)
        assert len(sent) == 20 and tools.remaining_calls == 0
        assert [result.tool_call_id for result in results] == [call.id for call in calls]
        assert [json.loads(result.output)["objective"] for result in results[:20]] == [str(i) for i in range(20)]
        assert all(not result.is_error for result in results[:20])
        assert results[20].is_error and results[20].output == "Tool budget exhausted"
        fresh_question = ToolDispatcher([action], client=research_client)
        fresh_result = (await fresh_question.execute([calls[0]]))[0]
        assert not fresh_result.is_error and len(sent) == 21
        assert fresh_question.remaining_calls == 19 and tools.remaining_calls == 0


async def test_zero_budget_returns_local_error_without_dispatch():
    sent = []
    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai", transport=httpx.MockTransport(lambda request: sent.append(request))
    ) as client:
        action = SEARCH
        tools = ToolDispatcher([action], client=ParallelClient(client=client), max_calls=0)
        result = (
            await tools.execute(
                [ToolInvocation(id="zero", name=action.name, input={"objective": "Evidence", "search_queries": ["q"]})]
            )
        )[0]
    assert not sent and tools.remaining_calls == 0
    assert result.tool_call_id == "zero" and result.is_error and result.output == "Tool budget exhausted"


@pytest.mark.parametrize(
    "name,arguments,detail", [("unknown", {}, "'unknown'"), ("search_web", {}, "validation error")]
)
async def test_exhausted_budget_still_reports_invalid_call(name, arguments, detail):
    tools = ToolDispatcher([SEARCH], max_calls=0)
    result = (await tools.execute([ToolInvocation(id="invalid", name=name, input=arguments)]))[0]
    assert result.is_error and result.output.startswith("Error:") and detail in result.output
    assert tools.remaining_calls == 0


async def test_tool_result_validation_failure_consumes_reserved_call():
    from pydantic import TypeAdapter

    class InvalidClient:
        async def request(self, endpoint, arguments):
            TypeAdapter(int).validate_python("invalid external result")

    tools = ToolDispatcher([SEARCH], client=InvalidClient(), max_calls=1)
    result = (
        await tools.execute(
            [ToolInvocation(id="response", name="search_web", input={"objective": "Evidence", "search_queries": ["q"]})]
        )
    )[0]
    assert result.is_error and result.output.startswith("Error:") and "validation error" in result.output
    assert tools.remaining_calls == 0


async def test_argument_validator_programming_error_propagates_without_reserving():
    from pydantic import BaseModel, model_validator

    from parallel_search_evals.tools import ToolSpec

    class Input(BaseModel):
        @model_validator(mode="before")
        @classmethod
        def validate_arguments(cls, value):
            raise KeyError("argument validator bug")

    tool = ToolSpec("validator", "Validate fixture", {}, Input, "search")
    tools = ToolDispatcher([tool], max_calls=1)
    with pytest.raises(KeyError, match="argument validator bug"):
        await tools.execute([ToolInvocation(id="validator-bug", name="validator", input={})])
    assert tools.remaining_calls == 1
