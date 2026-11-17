"""Real Parallel SDK requests retain the tool contract and never retry failures."""

import json

import httpx
import pytest

from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import EXTRACT, SEARCH, ToolInvocation


@pytest.mark.parametrize("action", [SEARCH, EXTRACT])
@pytest.mark.parametrize("status", [200, 401, 429, 503])
async def test_parallel_sdk_dispatches_once_and_preserves_tool_errors(action, status):
    requests = []
    arguments = {
        "objective": "Public SDK smoke",
        **({"search_queries": ["python creator"]} if action is SEARCH else {"urls": ["https://python.org"]}),
    }
    response = {"results": [], "future_field": [None, True], "warnings": [{"type": "future_warning"}]}

    def serve(request):
        requests.append(request)
        return httpx.Response(status, json=response if status == 200 else {"message": "synthetic failure"})

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as http:
        client = ParallelClient(mode="fast", model="fixture", client=http)
        tools = ToolDispatcher([action], client=client)
        result = (await tools.execute([ToolInvocation(id="sdk-call", name=action.name, input=arguments)]))[0]
    assert len(requests) == 1
    assert requests[0].headers["x-stainless-package-version"] == "1.3.7"
    assert requests[0].url.path == f"/v1/{action.endpoint}"
    assert result.is_error is (status != 200)
    if status == 200:
        assert json.loads(result.output) == response
    else:
        assert "synthetic failure" in result.output


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
async def test_parallel_sdk_network_failure_is_one_visible_tool_error(failure):
    requests = []

    def serve(request):
        requests.append(request)
        raise failure("synthetic network failure", request=request)

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as http:
        tools = ToolDispatcher([SEARCH], client=ParallelClient(client=http))
        result = (
            await tools.execute(
                [
                    ToolInvocation(
                        id="network", name=SEARCH.name, input={"objective": "Evidence", "search_queries": ["q"]}
                    )
                ]
            )
        )[0]
    assert len(requests) == 1
    assert result.is_error and result.output.startswith("Error:")
    assert tools.remaining_calls == 19
