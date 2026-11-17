"""Independent real-SDK regression probes; no network and no product modifications."""

import json

import httpx
import pytest
from google import genai
from google.genai import types

from parallel_search_evals.agent import AnswerGenerator
from parallel_search_evals.models.gemini import GoogleClient
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import ToolChoice
from tests.models.test_gemini_native import SEARCH_TOOL, Credentials, SearchClient


def sdk(vertex, serve):
    return genai.Client(
        vertexai=vertex,
        api_key=None if vertex else "fixture",
        project="fixture-project" if vertex else None,
        location="global" if vertex else None,
        credentials=Credentials() if vertex else None,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            async_client_args={"transport": httpx.MockTransport(serve)},
        ),
    )


@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize("collision", ["same_turn", "future_turn", "future_native"])
async def test_synthetic_ids_do_not_collide_with_native_ids(vertex, collision):
    sent = []

    async def serve(req):
        body = json.loads(req.content)
        sent.append(body)
        step = len(sent)
        call = {"name": "search", "args": {"query": "evidence"}}
        if collision == "same_turn" and step == 1:
            parts = [{"functionCall": call}, {"functionCall": {**call, "id": "1:0"}}]
        elif collision == "future_turn" and step == 1:
            parts = [{"functionCall": {**call, "id": "3:0"}}]
        elif collision == "future_turn" and step == 2 or collision == "future_native" and step == 1:
            parts = [{"functionCall": call}]
        elif collision == "future_native" and step == 2:
            parts = [{"functionCall": {**call, "id": "1:0"}}]
        else:
            parts = [{"text": "answer"}]
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": parts}}]},
            request=req,
        )

    client = sdk(vertex, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    agent = AnswerGenerator.__new__(AnswerGenerator)
    agent.model_client = model_client
    agent.max_turns = 10
    agent.research_mode = "parallel-basic"
    agent.research_client = SearchClient()
    agent.tool_specs = (SEARCH_TOOL,)
    try:
        history = await agent.answer("question")
        assert history.assistant_text == "answer"
        first_results = sent[1]["contents"][2]["parts"]
        if collision == "same_turn":
            assert "id" not in first_results[0]["functionResponse"]
            assert first_results[1]["functionResponse"]["id"] == "1:0"
        elif collision == "future_turn":
            assert first_results[0]["functionResponse"]["id"] == "3:0"
            assert "id" not in sent[2]["contents"][4]["parts"][0]["functionResponse"]
        else:
            assert "id" not in first_results[0]["functionResponse"]
            assert sent[2]["contents"][4]["parts"][0]["functionResponse"]["id"] == "1:0"
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("vertex", [False, True])
async def test_tools_disabled_request_has_no_declarations(vertex):
    sent = []

    async def serve(req):
        sent.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={
                "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "answer"}]}}]
            },
            request=req,
        )

    client = sdk(vertex, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    tools = ToolDispatcher([SEARCH_TOOL], client=SearchClient())
    tools.tool_choice = ToolChoice.NONE
    try:
        await model_client.generate(model_client.new_history().append_question("question"), tools)
        assert sent[0]["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
        assert sent[0]["tools"] == []
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("vertex", [False, True])
async def test_reordered_grouped_results_keep_native_call_identity_and_continuation(vertex):
    from parallel_search_evals.tools.schemas import ToolReply

    parts = [
        {"thought": True, "text": "private", "thoughtSignature": "opaque:not-base64", "unknown": None},
        {"functionCall": {"name": "search", "args": {"query": "first"}}},
        {"text": "visible"},
        {"functionCall": {"name": "search", "args": {"query": "second"}, "id": None}},
        {"functionCall": {"name": "search", "args": {"query": "third"}, "id": "native-third"}},
    ]
    sent = []

    async def serve(req):
        sent.append(json.loads(req.content))
        response_parts = parts if len(sent) == 1 else [{"text": "answer"}]
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": response_parts}}]},
            request=req,
        )

    client = sdk(vertex, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    history = model_client.new_history().append_question("question")
    try:
        first = await model_client.generate(history, ToolDispatcher([SEARCH_TOOL], client=SearchClient()))
        assert [call.id for call in first.tool_calls] == ["1:1", "1:3", "1:4"]
        history.accept_reply(first)
        history.append_tool_outputs(
            [
                ToolReply(tool_call_id="1:4", output="third result"),
                ToolReply(tool_call_id="1:1", output="first error", is_error=True),
                ToolReply(tool_call_id="1:3", output="second result"),
            ],
            continuation="Finish with the available evidence",
        )
        await model_client.generate(history, ToolDispatcher([SEARCH_TOOL], client=SearchClient()))
        assert sent[1]["contents"][1]["parts"] == parts
        assert sent[1]["contents"][2]["parts"] == [
            {"functionResponse": {"name": "search", "id": "native-third", "response": {"output": "third result"}}},
            {"functionResponse": {"name": "search", "response": {"error": "first error"}}},
            {"functionResponse": {"name": "search", "response": {"output": "second result"}}},
        ]
        assert sent[1]["contents"][3] == {"role": "user", "parts": [{"text": "Finish with the available evidence"}]}
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize("concurrent", [False, True])
async def test_answers_each_spend_their_own_tool_budget(vertex, concurrent, monkeypatch):
    import asyncio

    import parallel_search_evals.agent as generation
    from parallel_search_evals.config import FINAL_ANSWER_INSTRUCTIONS
    from parallel_search_evals.models import ModelSettings

    searched = []
    requests = []
    both_started = asyncio.Event()

    class Research(SearchClient):
        async def request(self, endpoint, arguments):
            searched.append(arguments["query"])
            return await super().request(endpoint, arguments)

    async def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        question = body["contents"][0]["parts"][0]["text"]
        first = len(body["contents"]) == 1
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == ("AUTO" if first else "NONE")
        if first:
            if concurrent:
                if len(requests) == 2:
                    both_started.set()
                await both_started.wait()
            parts = [{"functionCall": {"name": "search", "args": {"query": question}, "id": "same-call"}}]
        else:
            result = body["contents"][2]["parts"][0]["functionResponse"]
            assert result["id"] == "same-call"
            assert json.loads(result["response"]["output"]) == {"result": question}
            assert FINAL_ANSWER_INSTRUCTIONS in json.dumps(body)
            parts = [{"text": f"Answer: {question}"}]
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": parts}}]},
            request=request,
        )

    client = sdk(vertex, serve)
    model = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    monkeypatch.setattr(generation, "open_model", lambda settings: model)
    monkeypatch.setattr(generation, "ParallelClient", lambda **kwargs: Research())
    monkeypatch.setattr(generation, "TOOLS", (SEARCH_TOOL,))
    monkeypatch.setattr(generation, "MAX_TOOL_CALLS", 1)
    agent = AnswerGenerator(
        "parallel-basic",
        ModelSettings.from_selection({"provider": "vertex" if vertex else "gemini", "model": "gemini-4-carbon"}),
    )
    try:
        if concurrent:
            histories = await asyncio.gather(agent.answer("first"), agent.answer("second"))
        else:
            histories = [await agent.answer("first"), await agent.answer("second")]
        assert [history.assistant_text for history in histories] == ["Answer: first", "Answer: second"]
        assert sorted(searched) == ["first", "second"]
        assert len(requests) == 4
    finally:
        await client.aio.aclose()
        client.close()
