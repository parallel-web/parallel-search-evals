import copy
import json

import httpx
import pytest

from parallel_search_evals.models.anthropic_messages import messages_tool_schemas
from parallel_search_evals.models.gemini import GoogleClient
from parallel_search_evals.models.openai_responses import responses_tool_schemas
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import ToolReply
from tests.models.test_gemini_native import SEARCH_TOOL, SearchClient
from tests.models.test_gemini_review import sdk
from tests.sdk_transport import mock_owned_transport


def native(parts, finish="STOP", **candidate):
    return {"candidates": [{"finishReason": finish, "content": {"role": "model", "parts": parts}, **candidate}]}


@pytest.mark.parametrize("vertex", [False, True])
async def test_actual_sdk_continuation_preserves_raw_parts_text_and_native_tools(vertex):
    first = [
        {"thought": True, "text": "private", "thoughtSignature": "opaque-not-base64", "unknown": None},
        {"text": "Hel"},
    ]
    second = [{"text": "lo ", "future": {"unknown": [None, 7]}}]
    third = [
        {"text": "world"},
        {
            "functionCall": {"name": "search", "args": {"query": "q"}, "id": "native-id"},
            "thoughtSignature": "signed-opaque",
            "text": "tool-call text must not enter answer",
            "future": False,
        },
    ]
    replies = [
        native(first, "CONTINUATION", continuationToken="token-one"),
        native(second, "CONTINUATION", continuationToken="token-two"),
        native(third),
        native([{"text": "Done"}]),
    ]
    sent = []

    async def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=replies[len(sent) - 1])

    client = sdk(vertex, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    history = model_client.new_history().append_question("question")
    history.system = "system"
    tools = ToolDispatcher([SEARCH_TOOL], client=SearchClient())
    original_openai = responses_tool_schemas(tools.available)
    original_anthropic = messages_tool_schemas(tools.available)
    original_parameters = copy.deepcopy(SEARCH_TOOL.parameters)
    try:
        output = await model_client.generate(history, tools)
        assert len(sent) == 3
        assert output.text == "Hello world"
        assert output.native == first + second + third
        assert sent[1] == dict(sent[0], continuationToken="token-one")
        assert sent[2] == dict(sent[0], continuationToken="token-two")
        assert output.tool_calls[0].id == "1:4"
        history.accept_reply(output).append_tool_outputs(
            [ToolReply(tool_call_id="1:4", output="evidence", is_error=False)]
        )
        final = await model_client.generate(history, tools)
        assert final.text == "Done"
        assert "continuationToken" not in sent[3]
        assert sent[3]["contents"][1]["parts"] == first + second + third
        assert sent[3]["contents"][2]["parts"][0]["functionResponse"]["id"] == "native-id"
        assert responses_tool_schemas(tools.available) == original_openai
        assert messages_tool_schemas(tools.available) == original_anthropic
        assert SEARCH_TOOL.parameters == original_parameters
        assert sent[0]["tools"] == [
            {
                "functionDeclarations": [
                    {key: value for key, value in schema.items() if key != "type"} for schema in original_openai
                ]
            }
        ]
        assert all(body["generationConfig"]["maxOutputTokens"] == 65536 for body in sent)
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("token", [None, "", " \t\n", 42, [], {}])
async def test_actual_sdk_continuation_requires_nonblank_string_token(token):
    sent = []

    async def serve(request):
        sent.append(request)
        return httpx.Response(200, json=native([{"text": "partial"}], "CONTINUATION", continuationToken=token))

    client = sdk(True, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError):
            await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert len(sent) == 1
    finally:
        await client.aio.aclose()
        client.close()


async def test_actual_sdk_continuation_repeated_token_does_not_dispatch_again():
    sent = []

    async def serve(request):
        sent.append(request)
        return httpx.Response(200, json=native([{"text": "partial"}], "CONTINUATION", continuationToken="same-token"))

    client = sdk(True, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError):
            await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert len(sent) == 2
    finally:
        await client.aio.aclose()
        client.close()


async def test_unique_continuations_stop_at_32_physical_hops_without_partial_output():
    sent = []

    async def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json=native([{"text": "partial"}], "CONTINUATION", continuationToken=f"token-{len(sent)}")
        )

    client = sdk(True, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError, match="hop limit"):
            await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert len(sent) == 32
        for index, body in enumerate(sent[1:], 1):
            assert body == dict(sent[0], continuationToken=f"token-{index}")
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize(
    "finish,parts",
    [("MAX_TOKENS", [{"text": "partial"}]), ("CONTINUATION", [{"text": 42}]), ("UNKNOWN", [{"text": "partial"}])],
)
async def test_bad_later_hop_never_grades_or_returns_partial(finish, parts):
    sent = []

    async def serve(request):
        sent.append(request)
        body = (
            native([{"text": "prefix"}], "CONTINUATION", continuationToken="one")
            if len(sent) == 1
            else native(parts, finish, continuationToken="two")
        )
        return httpx.Response(200, json=body)

    client = sdk(True, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError):
            await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert len(sent) == 2
    finally:
        await client.aio.aclose()
        client.close()


async def test_duplicate_native_tool_ids_across_hops_rejected_before_execution():
    sent = []
    part = {"functionCall": {"id": "same-native", "name": "search", "args": {"query": "q"}}}

    async def serve(request):
        sent.append(request)
        return httpx.Response(
            200, json=native([part], "CONTINUATION" if len(sent) == 1 else "STOP", continuationToken="token")
        )

    client = sdk(True, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError, match="duplicate"):
            await model_client.generate(
                model_client.new_history().append_question("question"),
                ToolDispatcher([SEARCH_TOOL], client=SearchClient()),
            )
        assert len(sent) == 2
    finally:
        await client.aio.aclose()
        client.close()


async def test_malformed_continuation_public_error_and_traceback_hide_native_values():
    import traceback

    private = "synthetic-sensitive-opaque-value"

    async def serve(request):
        return httpx.Response(
            200,
            json=native(
                [{"text": "partial", "thoughtSignature": private}],
                "CONTINUATION",
                continuationToken={"opaque": private},
            ),
        )

    client = sdk(True, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError) as failure:
            await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert private not in str(failure.value)
        assert private not in "".join(traceback.format_exception(failure.value))
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize(
    "tail",
    [
        None,
        {},
        {"candidates": [{"finishReason": "STOP", "content": None}]},
        {"candidates": [{"finishReason": "STOP", "content": {"role": "model"}}]},
        {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": None}}]},
        {
            "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "answer"}]}}],
            "promptFeedback": {"blockReason": "SAFETY"},
        },
    ],
)
async def test_malformed_or_blocked_tail_rejected(tail):
    sent = []

    async def serve(request):
        sent.append(request)
        response = native([{"text": "prefix"}], "CONTINUATION", continuationToken="token") if len(sent) == 1 else tail
        return httpx.Response(200, content=json.dumps(response))

    client = sdk(True, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError):
            await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert len(sent) == 2
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("cancel", [False, True])
async def test_outer_call_deadline_and_cancellation_close_later_hop(cancel, monkeypatch):
    import asyncio

    import parallel_search_evals.agent as generation

    entered = asyncio.Event()
    closed = asyncio.Event()
    sent = []

    class WaitingBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            entered.set()
            await asyncio.Event().wait()
            yield b"unreachable"

        async def aclose(self):
            closed.set()

    async def serve(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(200, json=native([{"text": "prefix"}], "CONTINUATION", continuationToken="token"))
        return httpx.Response(200, stream=WaitingBody())

    client = sdk(True, serve)
    agent = generation.AnswerGenerator.__new__(generation.AnswerGenerator)
    agent.model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    agent.max_turns = 26
    agent.research_mode = "baseline"
    agent.research_client = None
    agent.tool_specs = ()
    monkeypatch.setattr(generation, "CALL_SECONDS", 0.05 if not cancel else 60)
    monkeypatch.setattr(generation, "GENERATION_SECONDS", 1 if not cancel else 60)
    try:
        task = asyncio.create_task(agent.answer("question"))
        await entered.wait()
        if cancel:
            task.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await task
        assert closed.is_set() and len(sent) == 2
        await asyncio.sleep(0)
        assert len(sent) == 2
    finally:
        await client.aio.aclose()
        client.close()


def test_real_cli_continuation_failure_preserves_peer_grade_and_terminal_cache(tmp_path, monkeypatch):
    import hashlib

    import chz
    import google.auth
    from google import genai
    from openai import AsyncOpenAI

    from parallel_search_evals import run_eval
    from parallel_search_evals.benchmarks import judge as benchmark
    from tests.cli_oracles import judgment, write_source
    from tests.models.test_gemini_native import Credentials

    source = tmp_path / "questions.jsonl"
    write_source(source, "hle", "Valid")
    row = json.loads(source.read_text())
    source.write_text(json.dumps(dict(row, question="Malformed")) + "\n" + json.dumps(row) + "\n")
    checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    calls = []
    per_question = {}
    clients = []
    denied = False
    private = "synthetic-sensitive-private-token"

    def serve(request):
        assert not denied, "Cached replay attempted HTTP"
        body = json.loads(request.content)
        calls.append((request.url.host, body))
        if request.url.host == "api.openai.com":
            assert "Valid" in body["input"][0]["content"]
            assert "Malformed" not in body["input"][0]["content"]
            grade = {
                "id": "grade",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": judgment("hle")}],
                    }
                ],
            }
            return httpx.Response(
                200,
                text="data: " + json.dumps({"type": "response.completed", "response": grade}) + "\n\n",
                headers={"content-type": "text/event-stream"},
            )
        question = body["contents"][0]["parts"][0]["text"]
        per_question[question] = per_question.get(question, 0) + 1
        assert request.extensions["timeout"]["connect"] == 30.0
        assert body["tools"] == [] and body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
        if per_question[question] == 1:
            response = native([{"text": "Paris\n"}], "CONTINUATION", continuationToken="first-token")
        elif question == "Malformed":
            response = native(
                [{"text": "partial", "thoughtSignature": private}],
                "CONTINUATION",
                continuationToken={"opaque": private},
            )
        else:
            response = native([{"text": "Rome"}])
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(serve)
    mock_owned_transport(monkeypatch, transport)

    original = genai.Client

    def google_client(**kwargs):
        kwargs["http_options"].async_client_args["transport"] = transport
        client = original(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(genai, "Client", google_client)
    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (Credentials(), "fixture-project"))
    monkeypatch.setattr(
        benchmark,
        "AsyncOpenAI",
        lambda **kwargs: AsyncOpenAI(**kwargs),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-judge")
    output = tmp_path / "run"
    argv = [
        "search_engine=baseline",
        "suite=hle",
        "provider=vertex",
        "model=fixture",
        "vertex_project=fixture-project",
        "vertex_location=global",
        f"source={source}",
        f"source_sha256={checksum}",
        "synthetic=True",
        "max_workers=2",
        "limit=2",
        f"output_dir={output}",
    ]
    chz.entrypoint(run_eval.main, argv=argv)
    bad = json.loads((output / "0.json").read_text())
    good = json.loads((output / "1.json").read_text())
    assert bad["status"] == "failed" and good["status"] == "completed"
    assert private not in json.dumps(bad)
    assert good["convo"]["assistant_text"] == "Paris\nRome"
    assert per_question == {"Malformed": 2, "Valid": 2}
    assert sum(host == "api.openai.com" for host, _ in calls) == 1
    before = {p: p.read_bytes() for p in output.rglob("*.json")}
    denied = True
    chz.entrypoint(run_eval.main, argv=argv)
    assert before == {p: p.read_bytes() for p in output.rglob("*.json")}
    assert all(
        client._api_client._httpx_client.is_closed and client._api_client._async_httpx_client.is_closed
        for client in clients
    )


async def test_long_decode_can_finish_on_the_32nd_physical_hop():
    sent = []

    async def serve(request):
        sent.append(json.loads(request.content))
        reply = (
            native([{"text": "answer"}])
            if len(sent) == 32
            else native([{"thought": True, "text": "thinking"}], "CONTINUATION", continuationToken=f"token-{len(sent)}")
        )
        return httpx.Response(200, json=reply)

    client = sdk(True, serve)
    model_client = GoogleClient("gemini-4-carbon", client=client, max_tokens=1048576, thinking_level="high")
    try:
        output = await model_client.generate(model_client.new_history().append_question("question"), ToolDispatcher([]))
        assert output.text == "answer"
        assert len(sent) == 32
        assert all(body["generationConfig"]["maxOutputTokens"] == 1048576 for body in sent)
        assert sent[-1] == dict(sent[0], continuationToken="token-31")
    finally:
        await client.aio.aclose()
        client.close()
