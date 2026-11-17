from functools import wraps

import httpx
import pytest
from anthropic import APIError as AnthropicAPIError
from google.genai.errors import APIError as GoogleAPIError
from openai import APIError as OpenAIAPIError

import parallel_search_evals.models as providers
from parallel_search_evals.benchmarks.judge import BenchmarkGrader
from parallel_search_evals.models import gemini as google_provider
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher
from tests.models.test_gemini_native import Credentials


@pytest.mark.parametrize("target", ["http://redirect.invalid/leak", "https://other.invalid/leak", "/same-origin"])
@pytest.mark.parametrize(
    "route", ["openai", "anthropic", "openrouter", "unbiased", "responses", "chat", "gemini", "vertex", "judge"]
)
async def test_owned_sdk_clients_do_not_follow_redirects(monkeypatch, route, target):
    requests, responses, clients = [], [], []
    for name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "UNBIASED_API_KEY",
        "COMPATIBLE_API_KEY",
        "GOOGLE_API_KEY",
    ):
        monkeypatch.setenv(name, "fixture-key")

    def serve(request):
        requests.append(request)
        response = httpx.Response(307, headers={"location": target}, json={"error": {"message": "fixture redirect"}})
        responses.append(response)
        return response

    original_init = httpx.AsyncClient.__init__

    @wraps(original_init)
    def init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(serve)
        original_init(self, *args, **kwargs)
        clients.append(self)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", init)
    original_google = google_provider.genai.Client

    def google_client(**kwargs):
        if kwargs.get("vertexai"):
            kwargs["credentials"] = Credentials()
        return original_google(**kwargs)

    monkeypatch.setattr(google_provider.genai, "Client", google_client)
    if route == "judge":
        owner = BenchmarkGrader("dsqa")
        client = owner.client
    else:
        selection = {"provider": route, "model": "fixture", "max_tokens": 64, "effort": "none"}
        if route in {"responses", "chat"}:
            selection.update(provider="compatible", api=route, base_url="https://trusted.invalid/v1")
        if route == "openrouter":
            selection["openrouter_backends"] = ["fixture"]
        if route == "vertex":
            selection.update(vertex_project="fixture-project", vertex_location="global")
        owner = providers.open_model(providers.ModelSettings.from_selection(selection))
        client = owner.client
    try:
        assert clients
        with pytest.raises((OpenAIAPIError, AnthropicAPIError, GoogleAPIError, NativeResponseError, httpx.HTTPError)):
            if route == "judge":
                await owner.judge("synthetic question")
            else:
                await owner.generate(owner.new_history().append_question("synthetic question"), ToolDispatcher([]))
        assert len(requests) == 1
        assert all(not instance.follow_redirects for instance in clients)
        assert all(response.is_closed for response in responses)
        assert requests[0].url.scheme == "https"
        assert requests[0].extensions["timeout"]["connect"] == 30
        assert requests[0].extensions["timeout"]["read"] == (120 if route == "judge" else 3600)
        if route in {"gemini", "vertex"}:
            assert client._api_client._httpx_client.follow_redirects is False
    finally:
        if route == "judge":
            await client.close()
        else:
            await owner.aclose()
    assert all(instance.is_closed for instance in clients)


async def test_injected_judge_client_redirect_policy_remains_callers_choice():
    from openai import AsyncOpenAI

    transport = httpx.MockTransport(lambda request: pytest.fail("No request expected"))
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as http_client:
        client = AsyncOpenAI(api_key="fixture", http_client=http_client)
        grader = BenchmarkGrader("dsqa", client=client)
        assert grader.client is client
        assert http_client.follow_redirects is True
        await client.close()
