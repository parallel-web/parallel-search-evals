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


@pytest.mark.parametrize("empty_final", [False, True])
@pytest.mark.parametrize("suite", ["dsqa", "hle"])
@pytest.mark.parametrize("provider", ["openrouter", "unbiased", "compatible"])
@pytest.mark.parametrize("irrelevant_options", [False, True])
@pytest.mark.parametrize("settings_case", ["default", "override", "none"])
def test_explicit_responses_cli_tools_judge_and_terminal_reuse(
    tmp_path, monkeypatch, provider, irrelevant_options, suite, settings_case, empty_final
):
    endpoint = {
        "openrouter": "https://openrouter.ai/api/v1",
        "unbiased": "https://api.unbiased.ai/v1",
        "compatible": "https://custom.example/v2",
    }[provider]
    credential = {"openrouter": "OPENROUTER_API_KEY", "unbiased": "UNBIASED_API_KEY", "compatible": "CUSTOM_TEST_KEY"}[
        provider
    ]
    source = tmp_path / f"synthetic.{suite}"
    checksum = write_source(source, suite, "Requested cities?")
    requests, clients = [], []
    native = {"type": "reasoning", "id": "reason", "encrypted_content": "opaque+/=", "future": None}
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
        assert str(request.url) == endpoint + "/responses"
        assert request.headers["authorization"] == "Bearer selected-key"
        assert body["model"] == "vendor/exact-high"
        assert body["instructions"] == RESEARCH_INSTRUCTIONS
        assert not any(item.get("role") == "system" for item in body["input"])
        assert body["max_output_tokens"] == (128000 if provider == "openrouter" and settings_case == "default" else 64)
        expected_effort = (
            "high"
            if settings_case == "default" and provider == "openrouter"
            else "low"
            if settings_case == "override"
            else None
        )
        assert body.get("reasoning") == ({"effort": expected_effort} if expected_effort is not None else None)
        if provider == "openrouter":
            assert body["provider"] == {
                "only": ["backend-a", "backend-b"],
                "order": ["backend-a", "backend-b"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        else:
            assert "provider" not in body
        step = sum(url == endpoint + "/responses" for url, _ in requests)
        if step == 1:
            output = [
                native,
                {
                    "type": "function_call",
                    "call_id": "s",
                    "name": "search_web",
                    "arguments": '{"objective":"cities","search_queries":["cities"]}',
                },
            ]
        elif step == 2:
            assert native in body["input"]
            assert [i["call_id"] for i in body["input"] if i.get("type") == "function_call_output"] == ["s"]
            output = [
                {
                    "type": "function_call",
                    "call_id": "e",
                    "name": "web_fetch",
                    "arguments": '{"objective":"verify","urls":["https://example.org"]}',
                }
            ]
        else:
            assert [i["call_id"] for i in body["input"] if i.get("type") == "function_call_output"] == ["s", "e"]
            output = [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [] if empty_final else [{"type": "output_text", "text": "Paris Rome"}],
                }
            ]
        return httpx.Response(200, json={"status": "completed", "output": output})

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
        f"source={source}",
        f"source_sha256={checksum}",
        f"output_dir={output}",
        "limit=1",
        "synthetic=True",
    ]
    if provider != "openrouter" or settings_case != "default":
        argv += ["max_tokens=64"]
    if settings_case != "default":
        argv += [f"effort={'none' if settings_case == 'none' else 'low'}"]
    if provider == "openrouter":
        argv += ["openrouter_backends=backend-a,backend-b"]
    if provider == "compatible":
        argv += [f"base_url={endpoint}", f"api_key_env={credential}"]
    if irrelevant_options:
        argv += ["vertex_project=unused-project", "vertex_location=unused-region"]
        if provider != "compatible":
            argv += ["base_url=https://poison.invalid/v1", "api_key_env=UNUSED_POISON_KEY", "api=chat"]
        if provider != "openrouter":
            argv += ["openrouter_backends=unused,unused"]
    chz.entrypoint(run_eval.main, argv=argv)
    terminal = assert_terminal(output, suite, empty_final, "Paris Rome")
    if not empty_final:
        assert_native_tool_ids(terminal.convo, ["s", "e"])
    assert len(requests) == (5 if empty_final else 6)
    assert [url.rsplit("/", 1)[-1] for url, _ in requests] == [
        "responses",
        "search",
        "responses",
        "extract",
        "responses",
        "responses",
    ][: 5 if empty_final else 6]
    summary = json.loads((output.parent / "results" / "run.json").read_text())
    assert summary["score"] == (0 if empty_final else expected_score(suite))
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == (5 if empty_final else 6)
    assert all(client.max_retries == 0 and client.is_closed() for client in clients)
