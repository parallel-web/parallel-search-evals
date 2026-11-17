"""Direct native optional parameters retain their exact emitted protocol fields."""

import json

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from parallel_search_evals.models import MessagesClient
from parallel_search_evals.models.chat_completions import ChatClient
from parallel_search_evals.models.openai_responses import ResponsesClient
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import SEARCH, ToolChoice


@pytest.mark.parametrize("budget", [None, 1024])
@pytest.mark.parametrize(
    "choice,expected", [(ToolChoice.AUTO, "auto"), (ToolChoice.NONE, "none"), (ToolChoice.REQUIRED, "any")]
)
async def test_anthropic_direct_thinking_and_choice_fields(budget, choice, expected):
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "stop_reason": "end_turn",
                "content": [{"type": "text", "text": "answer"}],
            },
        )

    async with AsyncAnthropic(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = MessagesClient("native", client=client, max_tokens=4096, effort="max", thinking_budget=budget)
        assert (
            await model.generate(model.new_history().append_question("Q"), ToolDispatcher([SEARCH], choice))
        ).text == "answer"
    assert len(sent) == 1 and client.is_closed()
    assert sent[0]["tool_choice"] == {"type": "auto" if budget and choice == ToolChoice.REQUIRED else expected}
    assert ("thinking" in sent[0]) == (budget is not None)
    if budget is not None:
        assert sent[0]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert bool(sent[0]["tools"]) == (choice != ToolChoice.NONE)
    assert sent[0]["output_config"] == {"effort": "max"}


@pytest.mark.parametrize("effort", [None, "xhigh", "max"])
async def test_chat_native_effort_omission_and_levels_beyond_old_sdk_enum(effort):
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "answer"},
                    }
                ]
            },
        )

    async with AsyncOpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = ChatClient("native", client=client, max_tokens=64, effort=effort)
        assert (
            await model.generate(model.new_history().append_question("Q"), ToolDispatcher([], ToolChoice.NONE))
        ).text == "answer"
    assert len(sent) == 1 and client.is_closed()
    assert ("reasoning_effort" in sent[0]) == (effort is not None)
    if effort is not None:
        assert sent[0]["reasoning_effort"] == effort
    assert "tools" not in sent[0]


async def test_responses_native_project_cache_namespace_and_private_state_fields():
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}],
            },
        )

    async with AsyncOpenAI(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = ResponsesClient("native", client=client, max_tokens=64, effort="high")
        assert (
            await model.generate(model.new_history().append_question("Q"), ToolDispatcher([], ToolChoice.NONE))
        ).text == "answer"
    assert len(sent) == 1 and client.is_closed()
    assert sent[0]["prompt_cache_key"] == "parallel-search-evals"
    assert sent[0]["store"] is False
    assert sent[0]["include"] == ["reasoning.encrypted_content"]
    assert sent[0]["model"] == "native"
    assert sent[0]["max_output_tokens"] == 64
    assert sent[0]["reasoning"] == {"effort": "high"}
    assert sent[0]["tools"] == []
    assert sent[0]["tool_choice"] == "none"
