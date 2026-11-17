"""Real Google SDK native protocol drives the production local Search/Extract tools."""

import json

import google.auth.credentials
import httpx
import pytest
from google import genai
from google.genai import types

from parallel_search_evals.models.gemini import GoogleClient
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.parallel_client import ParallelClient
from parallel_search_evals.tools.schemas import EXTRACT, SCHEMAS, SEARCH


class Credentials(google.auth.credentials.Credentials):
    def __init__(self):
        super().__init__()
        self.token = "fixture-token"

    def refresh(self, request):
        raise AssertionError("Fixture credentials must not refresh")


@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize("native_ids", [False, True])
async def test_actual_sdk_search_extract_answer(vertex, native_ids, monkeypatch):
    monkeypatch.setenv("PARALLEL_API_KEY", "parallel-fixture")
    model_requests, parallel_requests = [], []
    preserved = []
    provider_json = {
        "results": [{"url": "https://example.org", "excerpts": ["😀e\u0301界" * 3001]}],
        "warnings": [{"type": "native_warning"}],
        "errors": [{"url": "https://example.org/failed"}],
        "future": {"opaque": None},
    }

    async def model_http(request):
        body = json.loads(request.content)
        model_requests.append(body)
        step = len(model_requests)
        if step > 1:
            assert body["contents"][-2]["parts"] == preserved[-1]
            result = body["contents"][-1]["parts"][0]["functionResponse"]
            if native_ids:
                assert result["id"] == f"call-{step - 1}"
            else:
                assert "id" not in result
            assert json.loads(result["response"]["output"]) == provider_json
        assert body["tools"][0]["functionDeclarations"] == [
            {key: schema[key] for key in ("name", "description", "parameters")} for schema in SCHEMAS
        ]
        if step == 1:
            call = {"name": "search_web", "args": {"objective": "Find evidence", "search_queries": ["evidence"]}}
        elif step == 2:
            call = {"name": "web_fetch", "args": {"objective": "Read evidence", "urls": ["https://example.org"]}}
        else:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "Grounded answer"}]}}
                    ]
                },
                request=request,
            )
        parts = [
            {"thought": True, "text": "reason", "thoughtSignature": "opaque:not-base64", "future": None},
            {"functionCall": {**call, **({"id": f"call-{step}"} if native_ids else {})}, "thoughtSignature": None},
        ]
        preserved.append(parts)
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": parts}}]},
            request=request,
        )

    async def parallel_http(request):
        body = json.loads(request.content)
        parallel_requests.append((request.url.path, body))
        assert request.headers["x-api-key"] == "parallel-fixture"
        assert body["client_model"] == "gemini-fixture"
        assert not {"max_chars_total", "max_results", "advanced_settings", "excerpt_settings"}.intersection(body)
        if request.url.path.endswith("search"):
            assert body["mode"] == "basic"
            assert body["search_queries"] == ["evidence"]
        else:
            assert body["urls"] == ["https://example.org"]
        return httpx.Response(200, json=provider_json, request=request)

    client = genai.Client(
        vertexai=vertex,
        api_key=None if vertex else "google-fixture",
        project="project" if vertex else None,
        location="global" if vertex else None,
        credentials=Credentials() if vertex else None,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
            async_client_args={"transport": httpx.MockTransport(model_http)},
        ),
    )
    async with httpx.AsyncClient(
        base_url="https://api.parallel.ai",
        headers={"x-api-key": "parallel-fixture"},
        transport=httpx.MockTransport(parallel_http),
    ) as web_client:
        engine = ParallelClient(model="gemini-fixture", client=web_client)
        tools = ToolDispatcher([SEARCH, EXTRACT], client=engine)
        model_client = GoogleClient("gemini-fixture", client=client, max_tokens=65536, thinking_level="high")
        history = model_client.new_history().append_question("Research evidence")
        try:
            seen = set()
            for _ in range(2):
                output = await model_client.generate(history, tools)
                assert not seen.intersection(call.id for call in output.tool_calls)
                seen.update(call.id for call in output.tool_calls)
                history.accept_reply(output)
                history.append_tool_outputs(await tools.execute(output.tool_calls))
            final = await model_client.generate(history, tools)
            history.accept_reply(final)
            assert history.assistant_text == "Grounded answer"
            assert len(model_requests) == 3
            assert [path for path, _ in parallel_requests] == ["/v1/search", "/v1/extract"]
        finally:
            await client.aio.aclose()
            client.close()
