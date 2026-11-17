"""Native wire placement: Responses instructions, Chat system messages."""

import copy
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI

import parallel_search_evals.models as providers
from parallel_search_evals.config import (
    CLOSED_BOOK_INSTRUCTIONS,
    FINAL_ANSWER_INSTRUCTIONS,
    RESEARCH_INSTRUCTIONS,
)
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import ToolChoice, ToolReply
from tests.sdk_transport import mock_owned_transport


@pytest.mark.parametrize(
    "text,digest,length",
    [
        (CLOSED_BOOK_INSTRUCTIONS, "7aa27ada960d3bc183fae55503d008b3e0d69eeb29f989687c11f624f66dacb7", 141),
        (RESEARCH_INSTRUCTIONS, "def8b37b8b669d1adb91e61d52677ab6ac6889fbe90d22e53e8a6e6815527d1c", 255),
        (FINAL_ANSWER_INSTRUCTIONS, "d00d3a3da922844966da99bfa81cb94f9d25cca2536aa1354138784f48bd949a", 150),
    ],
)
def test_generation_prompt_utf8_matches_pinned_parallel_agent_v4(text, digest, length):
    # Independently pinned from e54ebda82130ca725405aa42e04e64fdbcf35101 agent_loop.PROMPTS.
    assert len(text) == length
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == digest


@pytest.mark.parametrize("route", ["openai", "openrouter", "unbiased", "compatible"])
@pytest.mark.parametrize("initial_system", ["", "Native system instruction"])
async def test_actual_sdk_responses_instructions_and_lossless_continuation(route, initial_system, monkeypatch):
    endpoint, credential = {
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "unbiased": ("https://api.unbiased.ai/v1", "UNBIASED_API_KEY"),
        "compatible": ("https://custom.example/v2", "COMPATIBLE_API_KEY"),
    }[route]
    native = [
        {"type": "reasoning", "id": "r", "summary": [], "encrypted_content": "opaque+/=", "future": None},
        {"type": "future_item", "unknown": [None, {"signed": "unchanged"}]},
        {
            "type": "function_call",
            "id": "fc",
            "call_id": "call",
            "name": "search_web",
            "arguments": '{"objective":"public"}',
        },
    ]
    before = copy.deepcopy(native)
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        assert str(request.url) == endpoint + "/responses"
        assert request.headers["authorization"] == "Bearer fixture-key"
        output = (
            native
            if len(sent) == 1
            else [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}]
        )
        return httpx.Response(200, json={"status": "completed", "output": output})

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))

    def sdk(**kwargs):
        return AsyncOpenAI(**kwargs)

    monkeypatch.setenv(credential, "fixture-key")
    monkeypatch.setattr(providers, "AsyncOpenAI", sdk)
    model_settings = ModelSettings.from_selection(
        {
            "model": "gpt-fixture" if route == "openai" else "vendor/exact",
            "provider": route,
            "max_tokens": 64,
            "openrouter_backends": ("backend",) if route == "openrouter" else (),
            "base_url": endpoint if route == "compatible" else None,
        }
    )
    model = providers.open_model(model_settings)
    history = model.new_history().append_question("question")
    history.system = initial_system
    definitions = [
        {
            "type": "function",
            "name": "search_web",
            "description": "Public search",
            "parameters": {
                "type": "object",
                "properties": {"objective": {"type": "string"}},
                "required": ["objective"],
            },
        }
    ]
    from pydantic import BaseModel

    from parallel_search_evals.tools import ToolSpec

    definition = definitions[0]
    tools = SimpleNamespace(
        tool_choice=ToolChoice.AUTO,
        available=(
            ToolSpec(definition["name"], definition["description"], definition["parameters"], BaseModel, "search"),
        ),
    )
    try:
        first = await model.generate(history, tools)
        history.accept_reply(first)
        history.append_tool_outputs([ToolReply(tool_call_id="call", output='{"ok":true}')], "Final continuation")
        history.system = "Updated final system"
        tools.tool_choice = ToolChoice.NONE
        result = await model.generate(history, tools)
        assert result.text == "answer"
    finally:
        await model.aclose()
    if initial_system:
        assert sent[0]["instructions"] == initial_system
    else:
        assert "instructions" not in sent[0]
    assert sent[0]["input"] == [{"role": "user", "content": "question"}]
    assert sent[1]["instructions"] == "Updated final system"
    assert sent[1]["input"] == [
        {"role": "user", "content": "question"},
        *before,
        {"type": "function_call_output", "call_id": "call", "output": '{"ok":true}'},
        {"role": "user", "content": "Final continuation"},
    ]
    assert native == before
    assert all(body["store"] is False for body in sent)
    extensions = route in {"openai", "openrouter"}
    assert all(body.get("include") == (["reasoning.encrypted_content"] if extensions else None) for body in sent)
    assert all(("prompt_cache_key" in body) is extensions for body in sent)


async def test_actual_sdk_chat_keeps_system_message_without_instructions(monkeypatch):
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        assert str(request.url) == "https://custom.example/v2/chat/completions"
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "exact",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}
                ],
            },
        )

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))

    monkeypatch.setenv("COMPATIBLE_API_KEY", "fixture-key")
    monkeypatch.setattr(
        providers,
        "AsyncOpenAI",
        lambda **kw: AsyncOpenAI(**kw),
    )
    model = providers.open_model(
        ModelSettings.from_selection(
            {
                "model": "exact",
                "provider": "compatible",
                "max_tokens": 64,
                "base_url": "https://custom.example/v2",
                "api": "chat",
            }
        )
    )
    history = model.new_history().append_question("question")
    history.system = "Chat system"
    try:
        result = await model.generate(
            history,
            ToolDispatcher([], ToolChoice.NONE),
        )
        assert result.text == "answer"
    finally:
        await model.aclose()
    assert sent[0]["messages"] == [
        {"role": "system", "content": "Chat system"},
        {"role": "user", "content": "question"},
    ]
    assert "instructions" not in sent[0]
