"""Actual SDK progress cannot extend absolute generation/judge deadlines."""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import parallel_search_evals.agent as research
import parallel_search_evals.benchmarks.judge as judging
from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models import MessagesClient, ModelSettings, ResponsesClient
from parallel_search_evals.models.chat_completions import ChatClient
from parallel_search_evals.results import FailedAnswer, Grade, ScoredAnswer, read_answer
from tests.run.helpers import FixtureRunner

JUDGMENT = json.loads((Path(__file__).parent / "fixtures/reference-hle.json").read_text())["judge_response"]


class ProgressBody(httpx.AsyncByteStream):
    def __init__(self, progress, final):
        self.progress = progress
        self.final = final
        self.chunks = 0
        self.cancelled = False
        self.closed = False
        self.finished = False

    async def __aiter__(self):
        try:
            for _ in range(20):
                await asyncio.sleep(0.1)
                self.chunks += 1
                yield self.progress
            self.finished = True
            yield self.final
        except asyncio.CancelledError:
            self.cancelled = True
            raise

    async def aclose(self):
        self.closed = True


def response_payload(kind):
    if kind == "anthropic":
        return {
            "id": "m1",
            "type": "message",
            "role": "assistant",
            "model": "fixture",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    if kind == "chat":
        return {
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}],
        }
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
    }


@pytest.mark.parametrize("kind", ["responses", "anthropic", "chat"])
async def test_generation_absolute_call_deadline_cancels_progress_and_reuses_failure(kind, monkeypatch, tmp_path):
    monkeypatch.setattr(research, "CALL_SECONDS", 0.5, raising=False)
    monkeypatch.setattr(research, "GENERATION_SECONDS", 5)
    payload = response_payload(kind)
    slow = ProgressBody(b" ", json.dumps(payload).encode())
    requests = []

    def serve(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, headers={"content-type": "application/json"}, stream=slow)
        return httpx.Response(200, json=payload)

    client_type = AsyncAnthropic if kind == "anthropic" else AsyncOpenAI
    async with client_type(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        # Load lazy SDK resources before applying the tiny network-operation test deadline.
        if kind == "anthropic":
            _ = client.messages
        elif kind == "chat":
            _ = client.chat.completions
        else:
            _ = client.responses
        model_type = {"responses": ResponsesClient, "anthropic": MessagesClient, "chat": ChatClient}[kind]
        model_client = model_type(
            "fixture",
            client=client,
            max_tokens=64,
            effort=None,
            **({"thinking_budget": None} if kind == "anthropic" else {}),
        )
        monkeypatch.setattr(research, "open_model", lambda settings: model_client)
        agent = research.AnswerGenerator("baseline", ModelSettings.from_selection({"model": "gpt-fixture"}))

        class Batch(FixtureRunner):
            async def _evaluate_question(self, datum):
                convo = await agent.answer(datum.problem)
                return ScoredAnswer(
                    datum=datum, convo=convo, grader_result=Grade(score=1.0, correct=True, grade_text="fixture")
                )

        suite = Batch(tmp_path / "run", 1)
        suite.questions = [Question(id=str(i), problem="Q", answer="A") for i in range(2)]
        result = await suite.run()
        assert result.failed_samples == 1 and result.score == 0.5
        failure = read_answer(tmp_path / "run/0.json")
        assert isinstance(failure, FailedAnswer) and failure.failure == "TimeoutError"
        assert isinstance(read_answer(tmp_path / "run/1.json"), ScoredAnswer)
        assert slow.cancelled and slow.closed and not slow.finished and 0 < slow.chunks < 20
        assert requests[0].extensions["timeout"] == {"connect": 30, "read": 3600, "write": 3600, "pool": 3600}
        assert await suite.run() == result and len(requests) == 2
    assert client.is_closed()


async def test_overall_generation_deadline_cancels_across_individually_short_calls(monkeypatch):
    monkeypatch.setattr(research, "CALL_SECONDS", 2)
    monkeypatch.setattr(research, "GENERATION_SECONDS", 1)
    monkeypatch.setenv("PARALLEL_API_KEY", "fixture")
    requests = []
    cancelled = []

    async def serve(request):
        requests.append(request)
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": str(len(requests)),
                        "name": "unknown",
                        "arguments": "{}",
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="fixture", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve))
    ) as client:
        _ = client.responses
        model_client = ResponsesClient("fixture", client=client, max_tokens=64, effort=None)
        monkeypatch.setattr(research, "open_model", lambda settings: model_client)
        agent = research.AnswerGenerator("parallel-basic", ModelSettings.from_selection({"model": "gpt-fixture"}))
        try:
            with pytest.raises(TimeoutError):
                await agent.answer("Q")
            assert len(requests) > 1 and cancelled == [True]
        finally:
            await agent.research_client.sdk.close()
    assert client.is_closed()


async def test_judge_absolute_deadline_cancels_nonidle_stream_and_reuses_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(judging, "JUDGE_SECONDS", 0.5, raising=False)
    completed = {
        "type": "response.completed",
        "response": {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": JUDGMENT}]}],
        },
    }
    final = ("data: " + json.dumps(completed) + "\n\n").encode()
    progress = b'data: {"type":"response.output_text.delta","delta":"x"}\n\n'
    slow = ProgressBody(progress, final)
    requests = []

    def serve(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=slow)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=final)

    async with AsyncOpenAI(
        api_key="fixture", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve))
    ) as client:
        _ = client.responses
        grader = judging.BenchmarkGrader("hle", client)

        class Batch(FixtureRunner):
            async def _evaluate_question(self, datum):
                convo = ResponsesClient("fixture", client=client, max_tokens=64, effort=None).new_history()
                convo.append_question(datum.problem).append_answer("answer")
                return ScoredAnswer(datum=datum, convo=convo, grader_result=await grader.grade(datum, convo))

        suite = Batch(tmp_path / "run", 1)
        suite.questions = [Question(id=str(i), problem="Q", answer="A") for i in range(2)]
        result = await suite.run()
        assert result.failed_samples == 1 and result.score == 0.5
        failure = read_answer(tmp_path / "run/0.json")
        assert isinstance(failure, FailedAnswer) and failure.failure == "TimeoutError"
        assert isinstance(read_answer(tmp_path / "run/1.json"), ScoredAnswer)
        assert slow.cancelled and slow.closed and not slow.finished and 0 < slow.chunks < 20
        assert requests[0].extensions["timeout"] == {"connect": 30, "read": 120, "write": 1800, "pool": 1800}
        assert await suite.run() == result and len(requests) == 2
    assert client.is_closed()


async def test_judge_absolute_deadline_also_covers_wait_for_response_headers(monkeypatch):
    monkeypatch.setattr(judging, "JUDGE_SECONDS", 0.5)
    requests = []
    cancelled = []

    async def serve(request):
        requests.append(request)
        try:
            await asyncio.sleep(2)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        pytest.fail("Request opening escaped the judge deadline")

    async with AsyncOpenAI(
        api_key="fixture", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve))
    ) as client:
        _ = client.responses
        with pytest.raises(TimeoutError):
            await judging.BenchmarkGrader("hle", client).judge("Q")
        assert cancelled == [True] and len(requests) == 1
        assert requests[0].extensions["timeout"] == {"connect": 30, "read": 120, "write": 1800, "pool": 1800}
    assert client.is_closed()


async def test_parallel_owned_transport_matches_main_default(monkeypatch):
    from parallel_search_evals.tools.parallel_client import ParallelClient

    monkeypatch.setenv("PARALLEL_API_KEY", "fixture")
    engine = ParallelClient()
    try:

        async def serve(transport, request):
            assert request.extensions["timeout"] == {"connect": 600, "read": 600, "write": 600, "pool": 600}
            return httpx.Response(200, json={"results": []})

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", serve)
        assert await engine.request("search", {"objective": "Evidence", "search_queries": ["q"]}) == {"results": []}
    finally:
        await engine.sdk.close()


@pytest.mark.parametrize("endpoint", ["search", "extract"])
@pytest.mark.parametrize("waiting_headers", [False, True])
async def test_parallel_absolute_deadline_closes_progress_and_header_wait(endpoint, waiting_headers, monkeypatch):
    import parallel_search_evals.tools.parallel_client as research_client
    from parallel_search_evals.tools import ToolDispatcher
    from parallel_search_evals.tools.schemas import EXTRACT, SEARCH, ToolInvocation

    monkeypatch.setattr(research_client, "TOOL_SECONDS", 0.5, raising=False)
    slow = ProgressBody(b" ", b'{"results": []}')
    requests = []
    header_cancelled = []

    async def serve(request):
        requests.append(request)
        if waiting_headers:
            try:
                await asyncio.sleep(2)
            except asyncio.CancelledError:
                header_cancelled.append(True)
                raise
        return httpx.Response(200, stream=slow)

    async with httpx.AsyncClient(base_url="https://api.parallel.ai", transport=httpx.MockTransport(serve)) as client:
        engine = research_client.ParallelClient(client=client)
        tool = SEARCH if endpoint == "search" else EXTRACT
        tools = ToolDispatcher([tool], client=engine)
        arguments = {
            "objective": "Evidence",
            **({"search_queries": ["q"]} if endpoint == "search" else {"urls": ["https://example.org"]}),
        }
        call = ToolInvocation(id="slow", name=tool.name, input=arguments)
        with pytest.raises(TimeoutError):
            await tools.execute([call])
        assert not client.is_closed
        assert len(requests) == 1 and tools.remaining_calls == 19
        if waiting_headers:
            assert header_cancelled == [True] and slow.chunks == 0
        else:
            assert 0 < slow.chunks < 20
            assert slow.cancelled and slow.closed and not slow.finished
