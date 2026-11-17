import json

import google.auth.credentials
import httpx
import pytest
from google import genai
from google.genai import types
from pydantic import BaseModel

from parallel_search_evals.models import ModelSettings, open_model
from parallel_search_evals.models.gemini import GoogleClient
from parallel_search_evals.tools import ToolDispatcher, ToolSpec


class Credentials(google.auth.credentials.Credentials):
    def __init__(self):
        super().__init__()
        self.token = "fixture-token"

    def refresh(self, request):
        raise AssertionError("No refresh expected")


class Input(BaseModel):
    query: str


SEARCH_TOOL = ToolSpec("search", "Search local fixture", Input.model_json_schema(), Input, "search")


class SearchClient:
    async def request(self, endpoint, arguments):
        return {"result": arguments["query"]}


@pytest.mark.parametrize("vertex", [False, True])
async def test_actual_sdk_native_tool_cycle(vertex):
    requests = []
    parts = [
        {"text": "private", "thought": True, "thoughtSignature": "opaque:not-base64", "future": None},
        {
            "functionCall": {"name": "search", "args": {"query": "evidence"}, "id": "native-call"},
            "thoughtSignature": "second:opaque",
            "unknown": {"null": None},
        },
    ]

    async def handle(request):
        requests.append(request)
        response_parts = parts if len(requests) == 1 else [{"text": "answer"}, {"text": "complete"}]
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": response_parts}}]},
            request=request,
        )

    client = genai.Client(
        vertexai=vertex,
        api_key=None if vertex else "fixture-key",
        project="fixture-project" if vertex else None,
        location="global" if vertex else None,
        credentials=Credentials() if vertex else None,
        http_options=types.HttpOptions(
            api_version="v1" if vertex else "v1beta",
            retry_options=types.HttpRetryOptions(attempts=1),
            async_client_args={"transport": httpx.MockTransport(handle)},
        ),
    )
    model_client = GoogleClient("gemini-fixture", client=client, thinking_level="high", max_tokens=65536)
    history = model_client.new_history().append_question("Question")
    history.system = "System"
    tools = ToolDispatcher([SEARCH_TOOL], client=SearchClient())
    try:
        first = await model_client.generate(history, tools)
        history.accept_reply(first)
        history.append_tool_outputs(await tools.execute(first.tool_calls))
        second = await model_client.generate(history, tools)
        history.accept_reply(second)
        body = json.loads(requests[1].content)
        assert body["contents"][1]["parts"] == parts
        assert body["contents"][2]["parts"] == [
            {
                "functionResponse": {
                    "name": "search",
                    "id": "native-call",
                    "response": {"output": '{"result":"evidence"}'},
                }
            }
        ]
        assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
        assert history.assistant_text == "answer\ncomplete"
        assert (
            type(history).model_validate_json(history.model_dump_json()).request_messages()
            == history.request_messages()
        )
        assert len(requests) == 2
        if vertex:
            assert requests[0].url.host == "aiplatform.googleapis.com"
            assert "/projects/fixture-project/locations/global/" in requests[0].url.path
            assert "x-goog-api-key" not in requests[0].headers
        else:
            assert requests[0].url.host == "generativelanguage.googleapis.com"
            assert requests[0].headers["x-goog-api-key"] == "fixture-key"
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("route", ["developer", "global", "us-central1"])
async def test_owned_native_constructors_auth_endpoints_and_cleanup(route, monkeypatch):
    requests = []
    actual_client = genai.Client

    async def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "answer"}]}}]
            },
            request=request,
        )

    def intercepted_client(**kwargs):
        kwargs["http_options"].async_client_args["transport"] = httpx.MockTransport(handle)
        # A client default alone is overridden by the SDK's per-call scalar timeout.
        kwargs["http_options"].async_client_args["timeout"] = httpx.Timeout(1800, connect=30)
        return actual_client(**kwargs)

    monkeypatch.setattr(genai, "Client", intercepted_client)
    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (Credentials(), "adc-project"))
    for name in (
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GEMINI_BASE_URL",
        "GOOGLE_VERTEX_BASE_URL",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        monkeypatch.setenv(name, "poisoned-ambient")
    monkeypatch.setenv("GOOGLE_API_KEY", "explicit-key")
    if route == "developer":
        model_client = open_model(ModelSettings.from_selection({"provider": "gemini", "model": "gemini-fixture"}))
    else:
        model_client = open_model(
            ModelSettings.from_selection(
                {
                    "provider": "vertex",
                    "model": "gemini-fixture",
                    "vertex_project": "explicit-project",
                    "vertex_location": route,
                }
            )
        )
    sync_client = model_client.client._api_client._httpx_client
    async_client = model_client.client._api_client._async_httpx_client
    try:
        output = await model_client.generate(model_client.new_history().append_question("Question"), ToolDispatcher([]))
        assert output.text == "answer"
        assert json.loads(requests[0].content)["tools"] == []
        assert async_client.timeout.connect == 30
        assert requests[0].extensions["timeout"] == {"connect": 30.0, "read": 3600.0, "write": 3600.0, "pool": 3600.0}
        if route == "developer":
            assert requests[0].url.host == "generativelanguage.googleapis.com"
            assert requests[0].headers["x-goog-api-key"] == "explicit-key"
            assert "authorization" not in requests[0].headers
        else:
            host = "aiplatform.googleapis.com" if route == "global" else f"{route}-aiplatform.googleapis.com"
            assert requests[0].url.host == host
            assert f"/projects/explicit-project/locations/{route}/" in requests[0].url.path
            assert requests[0].headers["authorization"] == "Bearer fixture-token"
            assert "x-goog-api-key" not in requests[0].headers
    finally:
        await model_client.aclose()
    assert sync_client.is_closed
    assert async_client.is_closed


@pytest.mark.parametrize(
    "body",
    [
        None,
        {},
        {"candidates": []},
        {"candidates": [{}]},
        {"candidates": [{"finishReason": "MAX_TOKENS", "content": {"role": "model", "parts": []}}]},
        {"candidates": [{"finishReason": "STOP", "content": {"role": "user", "parts": []}}]},
        {"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": 1}]}}]},
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"role": "model", "parts": [{"functionCall": {"name": "", "args": {}}}]},
                }
            ]
        },
        {
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"role": "model", "parts": [{"functionCall": {"name": "search", "args": []}}]},
                }
            ]
        },
    ],
)
def test_malformed_native_control_rejected(body):
    from parallel_search_evals.runtime import NativeResponseError

    model_client = GoogleClient("fixture", client=None, max_tokens=65536, thinking_level="high")
    with pytest.raises(NativeResponseError):
        model_client._parse_response(body, set())


@pytest.mark.parametrize("status", [201, 429, 503])
async def test_actual_sdk_errors_dispatch_once_and_owned_cleanup(status, monkeypatch):
    from google.genai import errors

    requests = []
    actual_client = genai.Client

    async def handle(request):
        requests.append(request)
        return httpx.Response(
            status,
            json={
                "candidates": [
                    {"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "must not accept"}]}}
                ]
            },
            request=request,
        )

    def intercepted_client(**kwargs):
        kwargs["http_options"].async_client_args["transport"] = httpx.MockTransport(handle)
        return actual_client(**kwargs)

    monkeypatch.setattr(genai, "Client", intercepted_client)
    monkeypatch.setenv("GOOGLE_API_KEY", "explicit-key")
    model_client = open_model(ModelSettings.from_selection({"provider": "gemini", "model": "gemini-fixture"}))
    try:
        with pytest.raises(errors.APIError):
            await model_client.generate(model_client.new_history().append_question("Question"), ToolDispatcher([]))
        assert len(requests) == 1
    finally:
        await model_client.aclose()
    assert model_client.client._api_client._httpx_client.is_closed
    assert model_client.client._api_client._async_httpx_client.is_closed


async def test_cancellation_during_actual_sdk_adc_prevents_late_model_dispatch(monkeypatch):
    import asyncio
    import threading

    started, released, finished = threading.Event(), threading.Event(), threading.Event()
    requests = []
    actual_client = genai.Client

    def adc(**kwargs):
        started.set()
        assert released.wait(5)
        finished.set()
        return Credentials(), "adc-project"

    async def handle(request):
        requests.append(request)
        raise AssertionError("Cancelled ADC must not dispatch model request")

    def intercepted_client(**kwargs):
        kwargs["http_options"].async_client_args["transport"] = httpx.MockTransport(handle)
        return actual_client(**kwargs)

    monkeypatch.setattr(genai, "Client", intercepted_client)
    monkeypatch.setattr(google.auth, "default", adc)
    model_client = open_model(
        ModelSettings.from_selection(
            {"provider": "vertex", "model": "fixture", "vertex_project": "project", "vertex_location": "global"}
        )
    )
    task = asyncio.create_task(
        model_client.generate(model_client.new_history().append_question("Question"), ToolDispatcher([]))
    )
    try:
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert requests == []
    finally:
        released.set()
        assert await asyncio.to_thread(finished.wait, 2)
        await model_client.aclose()
    assert requests == []


def test_duplicate_native_call_ids_rejected():
    from parallel_search_evals.runtime import NativeResponseError

    model_client = GoogleClient("fixture", client=None, max_tokens=65536, thinking_level="high")
    body = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"functionCall": {"name": "search", "args": {}, "id": "duplicate"}},
                        {"functionCall": {"name": "search", "args": {}, "id": "duplicate"}},
                    ],
                },
            }
        ]
    }
    with pytest.raises(NativeResponseError, match="duplicate"):
        model_client._parse_response(body, set())


@pytest.mark.parametrize("vertex", [False, True])
async def test_sdk_finalization_has_separate_text_turn_after_grouped_results(vertex):
    from parallel_search_evals.tools.schemas import ToolChoice

    requests = []
    parts = [
        {"thought": True, "text": "private", "thoughtSignature": "opaque:not-base64", "future": None},
        {"functionCall": {"id": "first-native", "name": "search", "args": {"query": "one"}}, "unknown": None},
        {"functionCall": {"name": "search", "args": {"query": "two"}}, "thoughtSignature": "signed+/="},
        {"functionCall": {"id": "last-native", "name": "search", "args": {"query": "three"}}},
    ]

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": parts if len(requests) == 1 else [{"text": "answer"}]},
                    }
                ]
            },
        )

    client = genai.Client(
        vertexai=vertex,
        api_key=None if vertex else "fixture-key",
        project="fixture-project" if vertex else None,
        location="global" if vertex else None,
        credentials=Credentials() if vertex else None,
        http_options=types.HttpOptions(
            api_version="v1" if vertex else "v1beta",
            retry_options=types.HttpRetryOptions(attempts=1),
            async_client_args={"transport": httpx.MockTransport(handle)},
        ),
    )
    model_client = GoogleClient("gemini-fixture", client=client, max_tokens=65536, thinking_level="high")
    history = model_client.new_history().append_question("Question")
    tools = ToolDispatcher([SEARCH_TOOL], client=SearchClient(), max_calls=2)
    try:
        first = await model_client.generate(history, tools)
        history.accept_reply(first)
        results = await tools.execute(first.tool_calls)
        assert [item.is_error for item in results] == [False, False, True]
        history.append_tool_outputs(results, "Finalize now")
        tools.tool_choice = ToolChoice.NONE
        second = await model_client.generate(history, tools)
        body = json.loads(requests[1].content)
        assert body["contents"] == [
            {"role": "user", "parts": [{"text": "Question"}]},
            {"role": "model", "parts": parts},
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "id": "first-native",
                            "name": "search",
                            "response": {"output": '{"result":"one"}'},
                        }
                    },
                    {"functionResponse": {"name": "search", "response": {"output": '{"result":"two"}'}}},
                    {
                        "functionResponse": {
                            "id": "last-native",
                            "name": "search",
                            "response": {"error": "Tool budget exhausted"},
                        }
                    },
                ],
            },
            {"role": "user", "parts": [{"text": "Finalize now"}]},
        ]
        assert body["tools"] == []
        assert body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
        assert second.text == "answer" and len(requests) == 2
        assert (requests[0].url.host == "aiplatform.googleapis.com") is vertex
    finally:
        await client.aio.aclose()
        client.close()


async def test_native_factories_own_fresh_transport_options_and_close_both_clients(monkeypatch):
    captured = []
    actual_client = genai.Client

    def capture_client(**kwargs):
        captured.append(kwargs["http_options"])
        return actual_client(**kwargs)

    monkeypatch.setattr(genai, "Client", capture_client)
    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (Credentials(), "adc-project"))
    monkeypatch.setenv("GOOGLE_API_KEY", "explicit-key")
    developer = open_model(ModelSettings.from_selection({"provider": "gemini", "model": "fixture"}))
    vertex = open_model(
        ModelSettings.from_selection(
            {
                "provider": "vertex",
                "model": "fixture",
                "vertex_project": "explicit-project",
                "vertex_location": "global",
            }
        )
    )
    models = [developer, vertex]
    clients = [
        (model.client._api_client._httpx_client, model.client._api_client._async_httpx_client) for model in models
    ]
    try:
        assert captured[0] is not captured[1]
        assert captured[0].async_client_args is not captured[1].async_client_args
        transports = [options.async_client_args["transport"] for options in captured]
        assert transports[0] is not transports[1]
        assert all(isinstance(transport, httpx.AsyncHTTPTransport) for transport in transports)
        assert [(options.base_url, options.api_version) for options in captured] == [
            ("https://generativelanguage.googleapis.com", "v1beta"),
            ("https://aiplatform.googleapis.com", "v1beta1"),
        ]
        assert all(options.retry_options.attempts == 1 for options in captured)
    finally:
        for model in models:
            await model.aclose()
    assert all(sync.is_closed and asynchronous.is_closed for sync, asynchronous in clients)
