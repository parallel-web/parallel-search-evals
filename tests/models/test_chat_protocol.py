import httpx
import pytest
from openai import AsyncOpenAI

from parallel_search_evals.models import open_model
from parallel_search_evals.models.chat_completions import ChatClient
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher


@pytest.mark.parametrize("reason,refusal", [("length", None), ("content_filter", None), ("stop", "denied")])
async def test_chat_native_terminal_failure_has_no_fallback(reason, refusal):
    calls = []

    def serve(request):
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "native",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": reason,
                        "message": {"role": "assistant", "content": "partial", "refusal": refusal},
                    }
                ],
            },
        )

    async with AsyncOpenAI(
        api_key="fixture-key",
        base_url="https://custom.example/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve)),
    ) as client:
        model = ChatClient("exact-model", client=client, max_tokens=128000, effort=None)
        with pytest.raises(NativeResponseError):
            await model.generate(model.new_history().append_question("question"), ToolDispatcher([]))
    assert len(calls) == 1 and client.is_closed()


@pytest.mark.parametrize(
    "endpoint", [None, "ftp://host", "https://user:pass@host", "https://host?secret=value", "https://host#fragment"]
)
def test_compatible_requires_explicit_clean_http_endpoint(endpoint, monkeypatch):
    monkeypatch.setenv("COMPATIBLE_API_KEY", "fixture-key")
    with pytest.raises(ValueError, match="base_url"):
        open_model(
            ModelSettings.from_selection(
                {"model": "selected", "provider": "compatible", "max_tokens": 64, "base_url": endpoint}
            )
        )
