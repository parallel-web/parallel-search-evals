"""Tool implementation failures propagate after all owned sibling work is joined."""

import asyncio
import json

import httpx
import pytest

from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import SEARCH, ToolInvocation


@pytest.mark.parametrize("error", [KeyError, ValueError, TypeError, RuntimeError, AttributeError, AssertionError])
async def test_programming_error_from_tool_dispatch_is_not_converted_to_model_evidence(error):
    async def serve(request):
        raise error("action implementation bug")

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        action = SEARCH
        tools = ToolDispatcher([action], client=ParallelClient(client=client))
        call = ToolInvocation(id="bug", name=action.name, input={"objective": "Evidence", "search_queries": ["q"]})
        with pytest.raises(error, match="action implementation bug"):
            await tools.execute([call])
        assert tools.remaining_calls == 19


async def test_programming_error_joins_tool_sibling_before_owned_client_closes():
    started = asyncio.Event()
    release = asyncio.Event()
    closed = []
    children = []

    async def serve(request):
        children.append(asyncio.current_task())
        if json.loads(request.content)["objective"] == "bug":
            await started.wait()
            raise RuntimeError("action implementation bug")
        started.set()
        try:
            await release.wait()
            return httpx.Response(200, json={"results": []})
        finally:
            closed.append(not client.is_closed)

    try:
        async with httpx.AsyncClient(
            base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)
        ) as client:
            action = SEARCH
            tools = ToolDispatcher([action], client=ParallelClient(client=client))
            calls = [
                ToolInvocation(id=name, name=action.name, input={"objective": name, "search_queries": ["q"]})
                for name in ["bug", "peer"]
            ]
            with pytest.raises(RuntimeError, match="action implementation bug"):
                await tools.execute(calls)
            assert closed == [True]
            assert all(task.done() for task in children)
    finally:
        for task in children:
            task.cancel()
        await asyncio.gather(*children, return_exceptions=True)


@pytest.mark.parametrize("body", [b"{", b"\xff"])
async def test_malformed_external_tool_json_is_an_error_and_next_valid_call_succeeds(body):
    sent = []

    def serve(request):
        sent.append(request)
        return httpx.Response(200, content=body) if len(sent) == 1 else httpx.Response(200, json={"results": []})

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        action = SEARCH
        tools = ToolDispatcher([action], client=ParallelClient(client=client))
        call = ToolInvocation(id="external", name=action.name, input={"objective": "Evidence", "search_queries": ["q"]})
        invalid = (await tools.execute([call]))[0]
        assert invalid.is_error and invalid.output.startswith("Error:")
        valid = (await tools.execute([call]))[0]
        assert not valid.is_error and json.loads(valid.output) == {"results": []}
        assert len(sent) == 2 and tools.remaining_calls == 18


async def test_unknown_tool_is_a_local_error_without_reserving_or_dispatching():
    sent = []
    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai", transport=httpx.MockTransport(lambda request: sent.append(request))
    ) as client:
        tools = ToolDispatcher([SEARCH], client=ParallelClient(client=client))
        result = (await tools.execute([ToolInvocation(id="unknown", name="unknown", input={})]))[0]
        assert result.is_error and result.output == "Error: 'unknown'"
        assert tools.remaining_calls == 20 and not sent


async def test_external_cancellation_joins_all_tool_cleanup_before_client_exit():
    started = [asyncio.Event(), asyncio.Event()]
    children = []
    closed = []

    async def serve(request):
        children.append(asyncio.current_task())
        index = int(json.loads(request.content)["objective"])
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.01 if index == 0 else 0.05)
            closed.append(not client.is_closed)

    try:
        async with httpx.AsyncClient(
            base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)
        ) as client:
            action = SEARCH
            calls = [
                ToolInvocation(id=str(i), name=action.name, input={"objective": str(i), "search_queries": ["q"]})
                for i in range(2)
            ]
            parent = asyncio.create_task(ToolDispatcher([action], client=ParallelClient(client=client)).execute(calls))
            await asyncio.gather(*(event.wait() for event in started))
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await parent
            assert closed == [True, True]
            assert all(child.done() for child in children)
    finally:
        for child in children:
            child.cancel()
        await asyncio.gather(*children, return_exceptions=True)


async def test_evaluation_cancellation_joins_async_child_cleanup_before_return(tmp_path):
    from parallel_search_evals.benchmarks import Question
    from tests.run.helpers import FixtureRunner

    started = [asyncio.Event(), asyncio.Event()]
    children = []
    closed = []

    class CleanupRunner(FixtureRunner):
        async def _evaluate_question(self, question):
            children.append(asyncio.current_task())
            index = int(question.id)
            started[index].set()
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01 if index == 0 else 0.05)
                closed.append(question.id)

    try:
        runner = CleanupRunner(tmp_path, 2)
        runner.questions = [Question(id=str(i), problem="Question", answer="Answer") for i in range(2)]
        parent = asyncio.create_task(runner.run())
        await asyncio.gather(*(event.wait() for event in started))
        parent.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent
        assert sorted(closed) == ["0", "1"]
        assert all(child.done() for child in children)
        assert not list(tmp_path.glob("*.json"))
    finally:
        for child in children:
            child.cancel()
        await asyncio.gather(*children, return_exceptions=True)
