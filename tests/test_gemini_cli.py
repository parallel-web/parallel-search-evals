"""Native Google SDKs execute the real CLI, local tools, official judge and terminal reuse."""

import json
from types import SimpleNamespace

import chz
import google.auth
import httpx
import pytest
from google import genai
from openai import AsyncOpenAI

from parallel_search_evals import run_eval
from parallel_search_evals.benchmarks import judge as judging
from parallel_search_evals.config import FINAL_ANSWER_INSTRUCTIONS
from parallel_search_evals.results import ScoredAnswer
from parallel_search_evals.tools import parallel_client as research_client
from tests.cli_oracles import (
    assert_judge_request,
    assert_terminal,
    expected_score,
    write_source,
)
from tests.cli_oracles import (
    judgment as make_judgment,
)
from tests.models.test_gemini_native import Credentials
from tests.sdk_transport import mock_owned_transport


@pytest.mark.parametrize("suite", ["dsqa", "hle"])
@pytest.mark.parametrize("provider", ["gemini", "vertex"])
@pytest.mark.parametrize("settings_case", ["default", "override", "none"])
@pytest.mark.parametrize("variant", ["research", "baseline", "budget", "http_error", "empty"])
def test_native_google_cli_tools_grade_terminal_reuse(tmp_path, monkeypatch, provider, variant, suite, settings_case):
    source = tmp_path / f"synthetic.{suite}"
    checksum = write_source(source, suite, "Which cities?")
    requests, native_clients, judge_clients = [], [], []
    native_parts = []
    model_calls = []
    actual_client = genai.Client

    provider_json = {
        "results": [
            {
                "url": "https://example.org",
                "excerpts": ["Paris Rome", "😀e\u0301界" * (3001 if variant == "research" else 1)],
            }
        ],
        "warnings": [{"type": "provider_warning", "detail": "unaltered"}],
        "errors": [{"url": "https://example.org/failed", "detail": "provider error"}],
        "session_id": "native-session",
        "unknown": [None, True],
    }

    def serve(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body))
        if request.url.host == "api.parallel.ai":
            assert body["client_model"] == "gemini-fixture"
            assert not {"max_chars_total", "max_results", "advanced_settings", "excerpt_settings"}.intersection(body)
            return httpx.Response(200, json=provider_json)
        if request.url.host == "api.openai.com":
            assert_judge_request(request, body, suite, "Which cities?", "Paris\nRome")
            assert body["model"] == "gpt-6-sol"
            assert body["tools"] == [] and body["tool_choice"] == "none"
            assert "Paris Rome" in body["input"][0]["content"].replace("\n", " ")
            judgment = make_judgment(suite)
            result = {
                "id": "judge",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "msg",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": judgment}],
                    }
                ],
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps({'type': 'response.completed', 'response': result})}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        assert body["generationConfig"] == {
            "maxOutputTokens": 65536 if settings_case == "default" else 64,
            "candidateCount": 1,
            **(
                {"thinkingConfig": {"thinkingLevel": "high" if settings_case == "default" else "low"}}
                if settings_case != "none"
                else {}
            ),
        }
        assert request.extensions["timeout"]["read"] == 3600.0
        model_calls.append(body)
        step = len(model_calls)
        if variant == "http_error":
            return httpx.Response(503, json={"error": {"code": 503, "message": "fixture failure"}})
        if step > 1:
            if variant == "budget":
                assert body["contents"][-3]["parts"] == native_parts[-1]
                results = body["contents"][-2]["parts"]
                assert len(results) == 21
                assert "error" in results[20]["functionResponse"]["response"]
                assert [item["functionResponse"]["id"] for item in results] == [f"call-1-{i}" for i in range(21)]
                assert body["contents"][-1] == {"role": "user", "parts": [{"text": FINAL_ANSWER_INSTRUCTIONS}]}
            else:
                assert body["contents"][-2]["parts"] == native_parts[-1]
                results = body["contents"][-1]["parts"]
        if variant == "research" and step > 1:
            assert len(results) == 1
            assert json.loads(results[0]["functionResponse"]["response"]["output"]) == provider_json
        final = variant in {"baseline", "empty"} or step == (2 if variant == "budget" else 3)
        if final:
            if variant in {"baseline", "budget"}:
                assert body["tools"] == []
                assert body["toolConfig"]["functionCallingConfig"]["mode"] == "NONE"
            if variant == "baseline":
                assert body["tools"] == []
            parts = [] if variant == "empty" else [{"text": "Paris"}, {"text": "Rome"}]
        else:
            if step == 1:
                name, args = "search_web", {"objective": "cities", "search_queries": ["cities"]}
            else:
                name, args = "web_fetch", {"objective": "verify", "urls": ["https://example.org"]}
            parts = [{"thought": True, "text": "private", "thoughtSignature": "opaque:not-base64", "future": None}]
            parts.extend(
                {
                    "functionCall": {
                        "name": name,
                        "args": args,
                        **({"id": f"call-{step}-{i}"} if variant != "research" else {}),
                    },
                    "thoughtSignature": None,
                }
                for i in range(21 if variant == "budget" else 1)
            )
            native_parts.append(parts)
        return httpx.Response(
            200, json={"candidates": [{"finishReason": "STOP", "content": {"role": "model", "parts": parts}}]}
        )

    transport = httpx.MockTransport(serve)

    def google_client(**kwargs):
        kwargs["http_options"].async_client_args["transport"] = transport
        client = actual_client(**kwargs)
        native_clients.append(client)
        return client

    mock_owned_transport(monkeypatch, transport)

    def judge_client(**kwargs):
        client = AsyncOpenAI(**kwargs)
        judge_clients.append(client)
        return client

    monkeypatch.setattr(genai, "Client", google_client)
    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (Credentials(), "fixture-project"))
    monkeypatch.setattr(judging, "AsyncOpenAI", judge_client)
    monkeypatch.setattr(
        research_client,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs), Timeout=httpx.Timeout
        ),
    )
    monkeypatch.setenv("GOOGLE_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-judge")
    monkeypatch.setenv("PARALLEL_API_KEY", "fixture-web")
    output = tmp_path / "run"
    argv = [
        f"search_engine={'baseline' if variant == 'baseline' else 'parallel-basic'}",
        f"suite={suite}",
        "model=gemini-fixture",
        f"provider={provider}",
        f"source={source}",
        f"output_dir={output}",
        "limit=1",
        "synthetic=True",
        f"source_sha256={checksum}",
    ]
    if settings_case != "default":
        argv += ["max_tokens=64", f"effort={'none' if settings_case == 'none' else 'low'}"]
    if provider == "vertex":
        argv += ["vertex_project=fixture-project", "vertex_location=global"]
    argv += [
        "base_url=https://poison.invalid/v1",
        "api_key_env=UNUSED_POISON_KEY",
        "api=chat",
        "openrouter_backends=unused,unused",
    ]
    if provider == "gemini":
        argv += ["vertex_project=unused-project", "vertex_location=unused-region"]
    chz.entrypoint(run_eval.main, argv=argv)
    terminal = assert_terminal(output, suite, variant in {"http_error", "empty"}, "Paris\nRome")
    if isinstance(terminal, ScoredAnswer):
        assert [message.parts for message in terminal.convo.messages if message.role == "model"][:-1] == native_parts
    saved = json.loads((output / "0.json").read_text())
    failed = variant in {"http_error", "empty"}
    assert saved["status"] == ("failed" if failed else "completed")
    assert (
        "grader_result" not in saved
        if failed
        else saved["grader_result"]["score"] == pytest.approx(expected_score(suite))
    )
    assert not any(host == "api.openai.com" for host, _ in requests) if failed else True
    total = len(requests)
    assert len(model_calls) == {"research": 3, "baseline": 1, "budget": 2, "http_error": 1, "empty": 1}[variant]
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == total
    assert all(client._api_client._httpx_client.is_closed for client in native_clients)
    assert all(client._api_client._async_httpx_client.is_closed for client in native_clients)
    assert all(client.is_closed() for client in judge_clients)
