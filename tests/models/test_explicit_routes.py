"""Explicit routes retain native tool state through the actual SDK and local tools."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

import parallel_search_evals.models as providers
from parallel_search_evals.agent import AnswerGenerator
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.tools.parallel_client import ParallelClient
from tests.sdk_transport import mock_owned_transport


@pytest.mark.parametrize("default_routing", [False, True])
async def test_openrouter_preserves_optional_and_explicit_routing_through_native_tools(monkeypatch, default_routing):
    requests = []
    tools = []
    created = []
    native = [
        {"type": "reasoning", "id": "reason1", "summary": [], "encrypted_content": "opaque+/=", "future": None},
        {
            "type": "function_call",
            "id": "fc1",
            "call_id": "call1",
            "name": "search_web",
            "arguments": '{"objective":"find answer","search_queries":["query"]}',
        },
        {
            "type": "function_call",
            "id": "fc2",
            "call_id": "call2",
            "name": "web_fetch",
            "arguments": '{"objective":"extract answer","urls":["https://example.org"]}',
        },
    ]

    def model_serve(request):
        body = json.loads(request.content)
        requests.append(body)
        assert str(request.url) == "https://openrouter.ai/api/v1/responses"
        assert request.headers["authorization"] == "Bearer selected-router"
        assert body["model"] == ("openrouter/auto" if default_routing else "vendor/exact-model-high")
        if default_routing:
            assert "reasoning" not in body
        else:
            assert body["reasoning"] == {"effort": "high"}
        assert body["max_output_tokens"] == (128000 if default_routing else 64)
        assert body["store"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        expected_provider = {
            "only": ["backend-a", "backend-b"],
            "order": ["backend-a", "backend-b"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if default_routing:
            assert "provider" not in body
        else:
            assert body["provider"] == expected_provider
        if len(requests) == 1:
            output = native
        else:
            output = [
                {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                }
            ]
        return httpx.Response(
            200,
            json={
                "id": "r",
                "object": "response",
                "created_at": 1,
                "model": "native",
                "status": "completed",
                "usage": None,
                "output": output,
            },
        )

    mock_owned_transport(monkeypatch, httpx.MockTransport(model_serve))

    def sdk_client(**kwargs):
        client = AsyncOpenAI(**kwargs)
        created.append(client)
        return client

    def parallel_serve(request):
        tools.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"results": [{"url": "https://example.org", "excerpts": ["answer"]}]})

    monkeypatch.setenv("OPENAI_BASE_URL", "https://poison.invalid")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-openai-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "selected-router")
    monkeypatch.setenv("PARALLEL_API_KEY", "parallel-key")
    monkeypatch.setattr(providers, "AsyncOpenAI", sdk_client, raising=False)
    agent = AnswerGenerator(
        "parallel-basic",
        ModelSettings.from_selection(
            {
                "model": "openrouter/auto" if default_routing else "vendor/exact-model-high",
                "provider": "openrouter",
                "openrouter_backends": () if default_routing else ("backend-a", "backend-b"),
                "effort": "none" if default_routing else "high",
                "max_tokens": 128000 if default_routing else 64,
            }
        ),
    )
    model = agent.model_client
    await agent.research_client.sdk.close()
    agent.research_client = ParallelClient(
        mode="basic",
        model=model.model,
        client=httpx.AsyncClient(
            base_url="https://api.parallel.ai",
            headers={"x-api-key": "parallel-key"},
            transport=httpx.MockTransport(parallel_serve),
        ),
    )
    try:
        history = await agent.answer("question")
        assert history.assistant_text == "answer"
    finally:
        await model.client.close()
        await agent.research_client.sdk.close()
    assert [url for url, _ in tools] == ["https://api.parallel.ai/v1/search", "https://api.parallel.ai/v1/extract"]
    assert all(
        payload["client_model"] == ("openrouter/auto" if default_routing else "vendor/exact-model-high")
        for _, payload in tools
    )
    assert requests[1]["input"][1 : 1 + len(native)] == native
    outputs = [item for item in requests[1]["input"] if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["call1", "call2"]
    assert created[0].max_retries == 0 and created[0].is_closed()


@pytest.mark.parametrize("backends", [("",), ("same", "same")])
def test_openrouter_requires_unambiguous_backend_selection(backends):
    with pytest.raises(ValueError):
        providers.open_model(
            ModelSettings.from_selection(
                {"model": "vendor/model", "provider": "openrouter", "openrouter_backends": backends}
            )
        )


def test_openrouter_requires_its_own_key_without_openai_fallback(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-fallback")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        providers.open_model(
            ModelSettings.from_selection(
                {"model": "vendor/model", "provider": "openrouter", "openrouter_backends": ("backend",)}
            )
        )
