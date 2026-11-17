"""Responses-only extensions stay on endpoints that document their support."""

import json

import httpx
import pytest

import parallel_search_evals.models as providers
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import SEARCH, ToolReply
from tests.sdk_transport import mock_owned_transport


@pytest.mark.parametrize("provider", ["openai", "openrouter", "unbiased", "compatible"])
async def test_responses_extensions_follow_endpoint_capabilities_and_preserve_replay(provider, monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "UNBIASED_API_KEY", "COMPATIBLE_API_KEY"):
        monkeypatch.setenv(key, "fixture")
    supports_extensions = provider in {"openai", "openrouter"}
    native = [
        {"type": "reasoning", "id": "r", "encrypted_content": "opaque+/=", "future": None},
        {
            "type": "function_call",
            "call_id": "call",
            "name": "search_web",
            "arguments": '{"objective":"Evidence","search_queries":["public evidence"]}',
        },
    ]
    sent = []

    def serve(request):
        body = json.loads(request.content)
        sent.append(body)
        if not supports_extensions and ({"include", "prompt_cache_key"} & body.keys()):
            return httpx.Response(400, json={"error": {"message": "Unsupported Responses extensions"}})
        output = (
            native if len(sent) == 1 else [{"type": "message", "content": [{"type": "output_text", "text": "answer"}]}]
        )
        return httpx.Response(200, json={"status": "completed", "output": output})

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))
    selection = {"provider": provider, "model": "fixture", "max_tokens": 64, "effort": "none"}
    if provider == "compatible":
        selection["base_url"] = "https://custom.example/v1"
    model = providers.open_model(providers.ModelSettings.from_selection(selection))
    try:
        history = model.new_history().append_question("question")
        first = await model.generate(history, ToolDispatcher([SEARCH]))
        history.accept_reply(first).append_tool_outputs([ToolReply(tool_call_id="call", output='{"results": []}')])
        assert (await model.generate(history, ToolDispatcher([]))).text == "answer"
        assert sent[1]["input"][1:3] == native
        for body in sent:
            assert body["store"] is False
            if supports_extensions:
                assert body["include"] == ["reasoning.encrypted_content"]
                assert body["prompt_cache_key"] == "parallel-search-evals"
            else:
                assert "include" not in body and "prompt_cache_key" not in body
    finally:
        await model.aclose()
