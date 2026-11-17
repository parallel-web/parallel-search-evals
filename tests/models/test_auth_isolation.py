"""Selected generation routes emit only their intended authentication headers."""

import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import parallel_search_evals.benchmarks.judge as benchmark
import parallel_search_evals.models as providers
from parallel_search_evals.tools import ToolDispatcher
from tests.sdk_transport import mock_owned_transport


@pytest.mark.parametrize("route", ["anthropic", "openai", "openrouter", "unbiased", "responses", "chat"])
async def test_selected_route_isolates_ambient_credentials(route, monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "UNBIASED_API_KEY", "COMPATIBLE_API_KEY"):
        monkeypatch.setenv(name, "selected-key")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-bearer")
    monkeypatch.setenv("OPENAI_ORG_ID", "ambient-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-project")
    sent = []

    def serve(request):
        sent.append(request)
        if route == "anthropic":
            payload = {
                "id": "msg1",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        elif route == "chat":
            payload = {
                "id": "chat1",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "answer"}}
                ],
            }
        else:
            payload = {
                "id": "resp1",
                "object": "response",
                "created_at": 1,
                "model": "fixture",
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "answer"}]}
                ],
            }
        return httpx.Response(200, json=payload)

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))

    def openai_client(**kwargs):
        return AsyncOpenAI(**kwargs)

    def anthropic_client(**kwargs):
        return AsyncAnthropic(**kwargs)

    monkeypatch.setattr(providers, "AsyncOpenAI", openai_client)
    monkeypatch.setattr(providers, "AsyncAnthropic", anthropic_client)
    selection = {"provider": route, "model": "fixture", "max_tokens": 128}
    if route in {"responses", "chat"}:
        selection.update(provider="compatible", api=route, base_url="https://custom.example/v1")
    if route == "openrouter":
        selection["openrouter_backends"] = ("selected-backend",)
    model = providers.open_model(providers.ModelSettings.from_selection(selection))
    try:
        output = await model.generate(model.new_history().append_question("question"), ToolDispatcher([]))
        assert output.text == "answer"
        request = sent[0]
        if route == "anthropic":
            assert request.headers["x-api-key"] == "selected-key"
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == "Bearer selected-key"
        if route == "openai":
            assert request.headers["openai-organization"] == "ambient-org"
            assert request.headers["openai-project"] == "ambient-project"
        else:
            assert "openai-organization" not in request.headers
            assert "openai-project" not in request.headers
    finally:
        await model.aclose()
    assert model.client.is_closed()


async def test_official_judge_retains_native_openai_tenant_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "judge-key")
    monkeypatch.setenv("OPENAI_ORG_ID", "judge-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "judge-project")
    sent = []

    def serve(request):
        sent.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"type":"response.completed","response":{"id":"r1","object":"response","created_at":1,"model":"fixture","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"{}"}]}]}}\n\ndata: [DONE]\n\n',
        )

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))

    def client(**kwargs):
        return AsyncOpenAI(**kwargs)

    monkeypatch.setattr(benchmark, "AsyncOpenAI", client)
    grader = benchmark.BenchmarkGrader("dsqa")
    try:
        assert await grader.judge("fixture question") == "{}"
        assert sent[0].headers["authorization"] == "Bearer judge-key"
        assert sent[0].headers["openai-organization"] == "judge-org"
        assert sent[0].headers["openai-project"] == "judge-project"
    finally:
        await grader.client.close()
    assert grader.client.is_closed()
