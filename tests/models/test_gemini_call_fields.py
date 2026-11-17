"""Native Google empty IDs are absent wire identities, while malformed fields still reject."""

import json

import httpx
import pytest

from parallel_search_evals.models.gemini import GoogleClient
from parallel_search_evals.runtime import NativeResponseError
from parallel_search_evals.tools import ToolDispatcher
from tests.models.test_gemini_native import SEARCH_TOOL, SearchClient
from tests.models.test_gemini_review import sdk


@pytest.mark.parametrize("vertex", [False, True])
async def test_google_empty_missing_null_ids_replay_raw_and_omit_result_ids(vertex):
    parts = [
        {"thought": True, "text": "private", "thoughtSignature": "opaque:not-base64", "future": None},
        {"functionCall": {"name": "search", "args": {"query": "empty-first"}, "id": ""}},
        {"functionCall": {"name": "search", "args": {"query": "missing"}}},
        {"functionCall": {"name": "search", "args": {"query": "null"}, "id": None}},
        {"functionCall": {"name": "search", "args": {"query": "empty-second"}, "id": ""}},
        {
            "functionCall": {"name": "search", "args": {"query": "native-collision"}, "id": "1:1"},
            "unknown": {"null": None},
        },
        {"functionCall": {"name": "search", "args": {"query": "native-spaces"}, "id": "  native  "}},
    ]
    sent = []

    async def serve(request):
        sent.append(json.loads(request.content))
        response_parts = parts if len(sent) == 1 else [{"text": "answer"}]
        return httpx.Response(
            200,
            json={"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": response_parts}}]},
            request=request,
        )

    client = sdk(vertex, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    history = model_client.new_history().append_question("question")
    tools = ToolDispatcher([SEARCH_TOOL], client=SearchClient())
    try:
        first = await model_client.generate(history, tools)
        assert [call.id for call in first.tool_calls] == ["1:1", "1:2", "1:3", "1:4", "1:5", "1:6"]
        history.accept_reply(first)
        history.append_tool_outputs(await tools.execute(first.tool_calls))
        restored = type(history).model_validate_json(history.model_dump_json())
        assert restored.request_messages() == history.request_messages()
        final = await model_client.generate(restored, tools)
        assert final.text == "answer"
        assert sent[1]["contents"][1]["parts"] == parts
        responses = sent[1]["contents"][2]["parts"]
        assert responses == [
            {"functionResponse": {"name": "search", "response": {"output": '{"result":"empty-first"}'}}},
            {"functionResponse": {"name": "search", "response": {"output": '{"result":"missing"}'}}},
            {"functionResponse": {"name": "search", "response": {"output": '{"result":"null"}'}}},
            {"functionResponse": {"name": "search", "response": {"output": '{"result":"empty-second"}'}}},
            {
                "functionResponse": {
                    "name": "search",
                    "response": {"output": '{"result":"native-collision"}'},
                    "id": "1:1",
                }
            },
            {
                "functionResponse": {
                    "name": "search",
                    "response": {"output": '{"result":"native-spaces"}'},
                    "id": "  native  ",
                }
            },
        ]
        assert len(sent) == 2
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("vertex", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "\n"),
        ("id", " \t"),
        ("id", "\x1c"),
        ("id", "\x1d"),
        ("id", "\x1e"),
        ("id", "\x1f"),
        ("id", False),
        ("id", 42),
        ("id", {}),
        ("name", ""),
        ("name", " \t\n"),
        ("name", "\x1c\x1d\x1e\x1f"),
        ("name", 42),
    ],
)
async def test_actual_google_sdk_rejects_malformed_call_fields(vertex, field, value):
    requests = []

    async def serve(request):
        requests.append(request)
        call = {"name": "search", "args": {"query": "q"}, "id": "native", field: value}
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"finishReason": "STOP", "content": {"role": "model", "parts": [{"functionCall": call}]}}
                ]
            },
            request=request,
        )

    client = sdk(vertex, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError, match="Malformed or incomplete"):
            await model_client.generate(
                model_client.new_history().append_question("question"),
                ToolDispatcher([SEARCH_TOOL], client=SearchClient()),
            )
        assert len(requests) == 1
    finally:
        await client.aio.aclose()
        client.close()


@pytest.mark.parametrize("vertex", [False, True])
async def test_google_still_rejects_duplicate_nonempty_native_ids(vertex):
    async def serve(request):
        call = {"name": "search", "args": {"query": "q"}, "id": "duplicate"}
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"role": "model", "parts": [{"functionCall": call}, {"functionCall": call}]},
                    }
                ]
            },
            request=request,
        )

    client = sdk(vertex, serve)
    model_client = GoogleClient("fixture", client=client, max_tokens=65536, thinking_level="high")
    try:
        with pytest.raises(NativeResponseError, match="duplicate"):
            await model_client.generate(
                model_client.new_history().append_question("question"),
                ToolDispatcher([SEARCH_TOOL], client=SearchClient()),
            )
    finally:
        await client.aio.aclose()
        client.close()
