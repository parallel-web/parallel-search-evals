"""Explicit model routes execute the real CLI, local tools, official judge and reuse."""

import json
from types import SimpleNamespace

import chz
import httpx
import pytest
from openai import AsyncOpenAI

from parallel_search_evals import models, run_eval
from parallel_search_evals.benchmarks import judge as judging
from parallel_search_evals.config import RESEARCH_INSTRUCTIONS
from parallel_search_evals.tools import parallel_client as research_client
from parallel_search_evals.tools.schemas import SCHEMAS
from tests.cli_oracles import (
    assert_judge_request,
    assert_native_tool_ids,
    assert_terminal,
    expected_score,
    write_source,
)
from tests.cli_oracles import (
    judgment as make_judgment,
)
from tests.sdk_transport import mock_owned_transport


@pytest.mark.parametrize("suite", ["dsqa", "hle"])
@pytest.mark.parametrize("empty_final", [False, True])
@pytest.mark.parametrize("effort_selection", ["default", "low", "none"])
def test_chat_actual_cli_tools_judge_and_terminal_reuse(tmp_path, monkeypatch, empty_final, suite, effort_selection):
    provider = "compatible"
    endpoint = "https://custom.example/v2"
    credential = "CUSTOM_TEST_KEY"
    source = tmp_path / f"synthetic.{suite}"
    checksum = write_source(source, suite, "Requested cities?")
    requests, clients = [], []
    judgment = make_judgment(suite)

    def serve(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        if request.url.host == "api.parallel.ai":
            assert body["client_model"] == "vendor/exact-high"
            return httpx.Response(200, json={"results": [{"url": "https://example.org", "excerpts": ["Paris"]}]})
        if body.get("stream"):
            assert_judge_request(request, body, suite, "Requested cities?", "Paris Rome")
            assert request.url.host == "api.openai.com"
            assert request.headers["authorization"] == "Bearer judge-key"
            assert body["model"] == "gpt-6-sol"
            assert body["max_output_tokens"] == 16384 and body["reasoning"] == {"effort": "medium"}
            assert body["text"]["format"]["strict"] is (suite == "hle")
            assert "Paris Rome" in body["input"][0]["content"]
            response = {
                "id": "judge",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "m",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": judgment}],
                    }
                ],
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps({'type': 'response.completed', 'response': response})}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        assert str(request.url) == endpoint + "/chat/completions"
        assert request.headers["authorization"] == "Bearer selected-key"
        assert body["model"] == "vendor/exact-high"
        assert body["max_tokens"] == 64
        assert body.get("reasoning_effort") == ("low" if effort_selection == "low" else None)
        assert request.extensions["timeout"]["read"] == 3600.0
        if provider == "openrouter":
            assert body["provider"] == {
                "only": ["backend-a", "backend-b"],
                "order": ["backend-a", "backend-b"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        else:
            assert "provider" not in body
        step = sum(url == endpoint + "/chat/completions" for url, _ in requests)
        history = body["messages"]
        assert "instructions" not in body
        assert history[0] == {"role": "system", "content": RESEARCH_INSTRUCTIONS}
        if step == 1:
            assert body["tools"] == [
                {
                    "type": "function",
                    "function": {key: value for key, value in schema.items() if key != "type"},
                }
                for schema in SCHEMAS
            ]
            message = {
                "role": "assistant",
                "content": None,
                "refusal": None,
                "reasoning_details": [{"type": "encrypted", "data": "opaque+/="}],
                "future": None,
                "tool_calls": [
                    {
                        "id": "s",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": '{"objective":"cities","search_queries":["cities"]}',
                        },
                    }
                ],
            }
        elif step == 2:
            assert history[2]["reasoning_details"] == [{"type": "encrypted", "data": "opaque+/="}]
            assert history[2]["future"] is None and history[2]["content"] is None
            assert history[3]["role"] == "tool" and history[3]["tool_call_id"] == "s"
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "e",
                        "type": "function",
                        "function": {
                            "name": "web_fetch",
                            "arguments": '{"objective":"verify","urls":["https://example.org"]}',
                        },
                    }
                ],
            }
        else:
            assert [item["tool_call_id"] for item in history if item.get("role") == "tool"] == ["s", "e"]
            message = {"role": "assistant", "content": None if empty_final else "Paris Rome", "tool_calls": None}
        return httpx.Response(
            200,
            json={
                "id": "chat",
                "object": "chat.completion",
                "created": 1,
                "model": "native",
                "choices": [{"index": 0, "finish_reason": "stop" if step == 3 else "tool_calls", "message": message}],
                "usage": None,
            },
        )

    transport = httpx.MockTransport(serve)

    mock_owned_transport(monkeypatch, transport)

    def client(**kwargs):
        result = AsyncOpenAI(**kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(models, "AsyncOpenAI", client)
    monkeypatch.setattr(judging, "AsyncOpenAI", client)
    monkeypatch.setattr(
        research_client,
        "httpx",
        SimpleNamespace(AsyncClient=lambda **kw: httpx.AsyncClient(transport=transport, **kw), Timeout=httpx.Timeout),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "judge-key")
    monkeypatch.setenv(credential, "selected-key")
    monkeypatch.setenv("PARALLEL_API_KEY", "parallel-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://poison.invalid")
    output = tmp_path / "run"
    argv = [
        "search_engine=parallel-basic",
        f"suite={suite}",
        "model=vendor/exact-high",
        f"provider={provider}",
        "api=chat",
        f"source={source}",
        f"source_sha256={checksum}",
        f"output_dir={output}",
        "limit=1",
        "synthetic=True",
        "max_tokens=64",
        f"effort={effort_selection}",
    ]
    if provider == "openrouter":
        argv += ["openrouter_backends=backend-a,backend-b"]
    if provider == "compatible":
        argv += [f"base_url={endpoint}", f"api_key_env={credential}"]
    chz.entrypoint(run_eval.main, argv=argv)
    terminal = assert_terminal(output, suite, empty_final, "Paris Rome")
    if not empty_final:
        assert_native_tool_ids(terminal.convo, ["s", "e"])
    assert len(requests) == (5 if empty_final else 6)
    summary = json.loads((output.parent / "results" / "run.json").read_text())
    assert summary["score"] == (0 if empty_final else expected_score(suite))
    from parallel_search_evals.results import FailedAnswer, read_answer

    restored = read_answer(output / "0.json")
    if empty_final:
        assert isinstance(restored, FailedAnswer) and "no current final answer" in restored.failure
    else:
        assert restored.convo.api == "chat" and restored.convo.assistant_text == "Paris Rome"
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == (5 if empty_final else 6)
    assert all(client.max_retries == 0 and client.is_closed() for client in clients)
