"""Provider histories preserve complete SDK payloads between tool calls."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from parallel_search_evals.models.anthropic_messages import MessagesClient
from parallel_search_evals.models.openai_responses import ResponsesClient
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import SEARCH, ToolChoice, ToolReply


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
async def test_sdk_native_replay_roundtrip(kind, tmp_path):
    if kind == "openai":
        content = [
            {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": "opaque+/=", "future": None},
            {"type": "future_item", "opaque": [None, {"signed": "x"}]},
            {
                "type": "function_call",
                "id": "fc1",
                "call_id": "call1",
                "name": "search_web",
                "arguments": '{"query":"a"}',
            },
        ]
        first = {
            "id": "resp1",
            "object": "response",
            "created_at": 1,
            "model": "native",
            "status": "completed",
            "output": content,
            "usage": None,
        }
        final = {
            **first,
            "output": [
                {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "first", "annotations": []},
                        {"type": "output_text", "text": "second", "annotations": []},
                    ],
                }
            ],
        }
    else:
        content = [
            {"type": "thinking", "thinking": "reason", "signature": "signed+/="},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "future", "unknown": None},
            {"type": "tool_use", "id": "call1", "name": "search_web", "input": {"query": "a"}},
        ]
        first = {
            "id": "msg1",
            "type": "message",
            "role": "assistant",
            "model": "native",
            "content": content,
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 2,
                "output_tokens": 1,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }
        final = {
            **first,
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        }
    if kind == "openai":
        final["output"].append(
            {"type": "message", "id": "m2", "role": "assistant", "status": "completed", "content": []}
        )
    before = copy.deepcopy(content)
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=first if len(sent) == 1 else final)

    transport = httpx.MockTransport(serve)
    client_type = AsyncOpenAI if kind == "openai" else AsyncAnthropic
    endpoint = "https://api.openai.com/v1" if kind == "openai" else "https://api.anthropic.com"
    async with client_type(
        api_key="fixture-key", base_url=endpoint, max_retries=0, http_client=httpx.AsyncClient(transport=transport)
    ) as client:
        client_type = ResponsesClient if kind == "openai" else MessagesClient
        model_client = client_type(
            "native",
            client=client,
            max_tokens=64,
            effort="high" if kind == "openai" else "max",
            **({"thinking_budget": None} if kind == "anthropic" else {}),
        )
        history = model_client.new_history().append_question("question")
        tools = ToolDispatcher([SEARCH])
        output = await model_client.generate(history, tools)
        assert [call.id for call in output.tool_calls] == ["call1"]
        history.accept_reply(output)
        history.append_tool_outputs(
            [ToolReply(tool_call_id="call1", output='{"error":"bad query"}', is_error=True)], "finalize"
        )
        expected_wire = copy.deepcopy(history.request_messages())
        tools.tool_choice = ToolChoice.NONE
        answer = await model_client.generate(history, tools)
        assert answer.text == "first\nsecond"
        history.accept_reply(answer)
        assert history.assistant_text == "first\nsecond"
        path = tmp_path / "history.json"
        path.write_text(history.model_dump_json())
        restored = type(history).model_validate_json(path.read_text())
        assert restored.assistant_text == "first\nsecond"
        assert restored.request_messages() == history.request_messages()
    assert sent[0]["max_output_tokens" if kind == "openai" else "max_tokens"] == 64
    if kind == "openai":
        assert sent[0]["store"] is False
        assert sent[0]["include"] == ["reasoning.encrypted_content"]
    wire = sent[1]["input" if kind == "openai" else "messages"]
    assert sent[1]["tools"] == []
    assert wire == expected_wire
    if kind == "openai":
        assert wire[1 : 1 + len(content)] == before
        assert wire[1 + len(content)]["call_id"] == "call1"
    else:
        assert wire[1] == {"role": "assistant", "content": before}
        assert wire[2]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": "call1",
                "content": '{"error":"bad query"}',
                "is_error": True,
            },
            {"type": "text", "text": "finalize"},
        ]
    assert content == before
    assert client.is_closed()


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
async def test_native_status_error_has_one_attempt_and_closes(kind):
    requests = []

    def serve(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"type": "rate_limit_error", "message": "fixture"}})

    client_type = AsyncOpenAI if kind == "openai" else AsyncAnthropic
    endpoint = "https://api.openai.com/v1" if kind == "openai" else "https://api.anthropic.com"
    async with client_type(
        api_key="fixture-key",
        base_url=endpoint,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model_client = (ResponsesClient if kind == "openai" else MessagesClient)(
            "native",
            client=client,
            max_tokens=128000,
            effort="high" if kind == "openai" else "max",
            **({"thinking_budget": None} if kind == "anthropic" else {}),
        )
        history = model_client.new_history().append_question("question")
        tools = SimpleNamespace(tool_choice=ToolChoice.AUTO, available=())
        with pytest.raises(Exception) as error:
            await model_client.generate(history, tools)
        assert error.value.__class__.__name__ == "RateLimitError"
    assert len(requests) == 1 and client.is_closed()


async def test_sdk_refusal_beside_text_is_a_failure():
    from parallel_search_evals.runtime import NativeResponseError

    body = {
        "id": "resp-refusal",
        "object": "response",
        "created_at": 1,
        "model": "native",
        "status": "completed",
        "usage": None,
        "output": [
            {
                "type": "message",
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": "partial", "annotations": []},
                    {"type": "refusal", "refusal": "denied"},
                ],
            }
        ],
    }
    async with AsyncOpenAI(
        api_key="fixture-key",
        base_url="https://api.openai.com/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))),
    ) as client:
        model_client = ResponsesClient("native", client=client, max_tokens=128000, effort="high")
        history = model_client.new_history().append_answer("Let me search").append_question("question")
        with pytest.raises(NativeResponseError, match="refused"):
            await model_client.generate(history, SimpleNamespace(tool_choice=ToolChoice.NONE, available=()))


@pytest.mark.parametrize("kind", ["openai", "anthropic"])
async def test_native_baseline_does_not_inject_research_finalization(kind):
    body = (
        {
            "id": "r",
            "object": "response",
            "created_at": 1,
            "model": "native",
            "status": "completed",
            "usage": None,
            "output": [
                {
                    "type": "message",
                    "id": "m",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                }
            ],
        }
        if kind == "openai"
        else {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "native",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "answer"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=body)

    client_type = AsyncOpenAI if kind == "openai" else AsyncAnthropic
    endpoint = "https://api.openai.com/v1" if kind == "openai" else "https://api.anthropic.com"
    async with client_type(
        api_key="key",
        base_url=endpoint,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = (ResponsesClient if kind == "openai" else MessagesClient)(
            "native",
            client=client,
            max_tokens=128000,
            effort="high" if kind == "openai" else "max",
            **({"thinking_budget": None} if kind == "anthropic" else {}),
        )
        history = model.new_history().append_question("question")
        history.system = "baseline instructions"
        output = await model.generate(history, SimpleNamespace(tool_choice=ToolChoice.NONE, available=()))
        assert output.text == "answer"
    wire = sent[0]
    if kind == "openai":
        assert wire["instructions"] == "baseline instructions"
        assert wire["input"] == [{"role": "user", "content": "question"}]
    else:
        assert wire["system"] == "baseline instructions"
        assert wire["messages"] == [
            {
                "role": "user",
                "content": [{"type": "text", "text": "question"}],
            }
        ]


@pytest.mark.parametrize(
    "native",
    [
        [{"type": "thinking", "thinking": "private reasoning", "signature": "signed+/=", "unknown": None}],
        [{"type": "redacted_thinking", "data": "opaque+/=", "unknown": None}],
    ],
)
async def test_thinking_only_native_content_is_preserved_but_is_not_an_answer(native):
    def serve(request):
        body = json.loads(request.content)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        return httpx.Response(
            200,
            json={
                "id": "thinking-only",
                "type": "message",
                "role": "assistant",
                "model": "native",
                "content": native,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    async with AsyncAnthropic(
        api_key="fixture-key",
        base_url="https://api.anthropic.com",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = MessagesClient("native", client=client, max_tokens=128000, effort="max", thinking_budget=1024)
        output = await model.generate(
            model.new_history().append_question("question"), ToolDispatcher([], tool_choice=ToolChoice.NONE)
        )
        assert output.text == ""
        assert output.tool_calls == []
        assert output.native == native
        history = model.new_history().accept_reply(output)
        assert history.assistant_text == ""
        assert history.request_messages() == [{"role": "assistant", "content": native}]
    assert client.is_closed()


@pytest.mark.parametrize("kind", ["openai", "anthropic", "chat"])
@pytest.mark.parametrize("bad_text", [42, True])
async def test_sdk_malformed_text_is_rejected_before_joining(kind, bad_text):
    from pydantic import ValidationError

    from parallel_search_evals.models.chat_completions import ChatClient

    if kind == "anthropic":
        response = {
            "id": "malformed",
            "type": "message",
            "role": "assistant",
            "model": "native",
            "content": [{"type": "text", "text": bad_text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    elif kind == "chat":
        response = {
            "id": "malformed",
            "object": "chat.completion",
            "created": 1,
            "model": "native",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": bad_text}}],
        }
    else:
        response = {
            "id": "malformed",
            "object": "response",
            "created_at": 1,
            "model": "native",
            "status": "completed",
            "output": [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": bad_text}]}
            ],
        }
    client_type = AsyncAnthropic if kind == "anthropic" else AsyncOpenAI
    async with client_type(
        api_key="fixture-key",
        base_url="https://fixture.invalid",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response))
        ),
    ) as client:
        model = (
            MessagesClient("native", client=client, max_tokens=128000, effort="max", thinking_budget=None)
            if kind == "anthropic"
            else (ChatClient if kind == "chat" else ResponsesClient)(
                "native", client=client, max_tokens=128000, effort="high"
            )
        )
        with pytest.raises(ValidationError):
            await model.generate(
                model.new_history().append_question("question"), ToolDispatcher([], tool_choice=ToolChoice.NONE)
            )
    assert client.is_closed()
