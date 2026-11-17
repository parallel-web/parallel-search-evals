"""Actual chz CLI and native SDKs execute local tools, official grading and terminal reuse."""

import asyncio
import json
from types import SimpleNamespace

import chz
import httpx
import pytest
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

import parallel_search_evals.agent as generation
from parallel_search_evals import models, run_eval
from parallel_search_evals.benchmarks import judge as judging
from parallel_search_evals.config import (
    CLOSED_BOOK_INSTRUCTIONS,
    FINAL_ANSWER_INSTRUCTIONS,
    RESEARCH_INSTRUCTIONS,
)
from parallel_search_evals.results import read_answer
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


@pytest.mark.parametrize("suite", ["dsqa", "hle"])
@pytest.mark.parametrize("model", ["gpt-native", "gpt-native-medium", "claude-native", "claude-native-thinking"])
@pytest.mark.parametrize("empty_final", [False, True])
@pytest.mark.parametrize("settings_case", ["default", "override", "none"])
@pytest.mark.parametrize("variant", ["research", "budget", "baseline", "iteration"])
def test_cli_native_answer_tools_official_f1_and_terminal_resume(
    tmp_path, monkeypatch, model, empty_final, variant, suite, settings_case
):
    source = tmp_path / f"synthetic.{suite}"
    checksum = write_source(source, suite, "What are the requested cities?")
    expected_cap = 128000 if settings_case == "default" else 64
    expected_effort = (
        None
        if settings_case == "none"
        else "low"
        if settings_case == "override"
        else ("max" if model.startswith("claude") else "medium" if model.endswith("-medium") else "high")
    )
    timeouts = []

    def generation_timeout(seconds):
        timeouts.append(seconds)
        return asyncio.timeout(seconds)

    monkeypatch.setattr(generation, "asyncio", SimpleNamespace(timeout=generation_timeout))
    requests = []
    clients = []
    native = {"type": "reasoning", "id": "reason", "encrypted_content": "opaque+/=", "future": None}
    thinking = {"type": "thinking", "thinking": "native", "signature": "signed+/=", "future": None}
    judgment = make_judgment(suite)

    provider_json = {
        "results": [
            {
                "url": "https://example.org",
                "excerpts": ["Paris", "😀e\u0301界" * (3001 if variant == "research" else 1)],
            }
        ],
        "warnings": [{"type": "provider_warning", "detail": "unaltered"}],
        "errors": [{"url": "https://example.org/failed", "detail": "provider error"}],
        "session_id": "native-session",
        "unknown": [None, True],
    }

    def serve(request):
        body = json.loads(request.content)
        requests.append((request.url.host, request.url.path, body))
        if request.url.host == "api.parallel.ai":
            assert body["client_model"] == model.removesuffix("-medium").removesuffix("-thinking")
            assert not {"max_chars_total", "max_results", "advanced_settings", "excerpt_settings"}.intersection(body)
            if request.url.path.endswith("search"):
                assert body["mode"] == "basic"
            return httpx.Response(200, json=provider_json)
        if body.get("stream"):
            assert_judge_request(
                request,
                body,
                suite,
                "What are the requested cities?",
                "Paris\nRome" if model.startswith("claude") else "Paris Rome",
            )
            assert body["model"] == "gpt-6-sol"
            assert "Paris Rome" in body["input"][0]["content"].replace("\n", " ")
            assert body["max_output_tokens"] == 16384
            response = {
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
                text=f"data: {json.dumps({'type': 'response.completed', 'response': response})}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        if request.url.host == "api.openai.com":
            assert body["max_output_tokens"] == expected_cap
            assert body.get("reasoning") == ({"effort": expected_effort} if expected_effort is not None else None)
            assert body["store"] is False
            assert body["include"] == ["reasoning.encrypted_content"]
        else:
            assert body.get("thinking") == (
                {"type": "enabled", "budget_tokens": 1024} if model.endswith("-thinking") else None
            )
            assert body["max_tokens"] == expected_cap
            assert body["output_config"] == ({"effort": expected_effort} if expected_effort is not None else {})
        assert request.extensions["timeout"]["read"] == 3600.0
        assert "Paris;Rome" not in json.dumps(body)
        calls = [value for host, _, value in requests if host != "api.parallel.ai" and not value.get("stream")]
        step = len(calls) % (
            1 if variant == "baseline" else 2 if variant == "budget" else 26 if variant == "iteration" else 3
        )
        is_anthropic = request.url.host == "api.anthropic.com"
        if not is_anthropic:
            expected_system = (
                CLOSED_BOOK_INSTRUCTIONS
                if variant == "baseline"
                else RESEARCH_INSTRUCTIONS
                + ("\n\n" + FINAL_ANSWER_INSTRUCTIONS if variant in {"budget", "iteration"} and step == 0 else "")
            )
            assert body["instructions"] == expected_system
            assert not any(item.get("role") == "system" for item in body["input"])
        history = body["messages"] if is_anthropic else body["input"]
        if variant == "research" and step != 1:
            outputs = (
                [
                    part["content"]
                    for message in history
                    for part in message["content"]
                    if part.get("type") == "tool_result"
                ]
                if is_anthropic
                else [item["output"] for item in history if item.get("type") == "function_call_output"]
            )
            assert len(outputs) == (1 if step == 2 else 2)
            assert all(json.loads(output) == provider_json for output in outputs)
        if step == 0:
            if variant != "baseline":
                assert (thinking if is_anthropic else native) in (history[1]["content"] if is_anthropic else history)
            if variant in {"budget", "baseline", "iteration"}:
                assert body["tools"] == []
                assert body["tool_choice"] == ({"type": "none"} if is_anthropic else "none")
                system = body["system"] if is_anthropic else body["instructions"]
                assert system == (
                    CLOSED_BOOK_INSTRUCTIONS
                    if variant == "baseline"
                    else RESEARCH_INSTRUCTIONS + "\n\n" + FINAL_ANSWER_INSTRUCTIONS
                )
                if variant in {"budget", "iteration"}:
                    if is_anthropic:
                        results = history[-1]["content"]
                        expected_ids = [f"s{i}" for i in range(21)] if variant == "budget" else ["invalid-25"]
                        assert [item["tool_use_id"] for item in results[:-1]] == expected_ids
                        assert results[-2]["is_error"] is True
                        assert results[-1] == {"type": "text", "text": FINAL_ANSWER_INSTRUCTIONS}
                        assert len(history) == (3 if variant == "budget" else 51)
                    else:
                        assert history[-1] == {"role": "user", "content": FINAL_ANSWER_INSTRUCTIONS}
                        expected_ids = (
                            [f"s{i}" for i in range(21)]
                            if variant == "budget"
                            else [f"invalid-{i}" for i in range(1, 26)]
                        )
                        assert [
                            item["call_id"] for item in history if item.get("type") == "function_call_output"
                        ] == expected_ids
                else:
                    assert len(history) == 1
            if is_anthropic:
                content = [] if empty_final else [{"type": "text", "text": "Paris"}, {"type": "text", "text": "Rome"}]
                return httpx.Response(
                    200,
                    json={
                        "id": "final",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-native",
                        "stop_reason": "end_turn",
                        "content": content,
                        "usage": {"input_tokens": 2, "output_tokens": 2},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [] if empty_final else [{"type": "output_text", "text": "Paris Rome"}],
                        }
                    ],
                },
            )
        name, identity, arguments = (
            ("search_web", "s", {"objective": "cities", "search_queries": ["cities"]})
            if step == 1
            else ("web_fetch", "e", {"objective": "verify", "urls": ["https://example.org"]})
        )
        if variant == "reused_ids":
            identity = "same-native-id"
        if variant == "iteration":
            name, identity, arguments = "unknown_tool", f"invalid-{step}", {}
        specifications = (
            [(f"s{i}", "search_web", {"objective": "cities", "search_queries": ["cities"]}) for i in range(21)]
            if variant == "budget"
            else [(identity, name, arguments)]
        )
        if is_anthropic:
            content = [thinking, {"type": "text", "text": "Previous research, not a final answer"}]
            content.extend(
                {"type": "tool_use", "id": identity, "name": name, "input": arguments}
                for identity, name, arguments in specifications
            )
            return httpx.Response(
                200,
                json={
                    "id": "tool",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-native",
                    "stop_reason": "tool_use",
                    "content": content,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        output = [
            native,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Previous research, not a final answer"}],
            },
        ]
        output.extend(
            {"type": "function_call", "call_id": identity, "name": name, "arguments": json.dumps(arguments)}
            for identity, name, arguments in specifications
        )
        return httpx.Response(200, json={"status": "completed", "output": output})

    transport = httpx.MockTransport(serve)

    mock_owned_transport(monkeypatch, transport)

    def sdk(cls, **kwargs):
        client = cls(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(models, "AsyncOpenAI", lambda **kwargs: sdk(AsyncOpenAI, **kwargs))
    monkeypatch.setattr(models, "AsyncAnthropic", lambda **kwargs: sdk(AsyncAnthropic, **kwargs))
    monkeypatch.setattr(judging, "AsyncOpenAI", lambda **kwargs: sdk(AsyncOpenAI, **kwargs))
    monkeypatch.setattr(
        research_client,
        "httpx",
        SimpleNamespace(
            AsyncClient=lambda **kwargs: httpx.AsyncClient(transport=transport, **kwargs), Timeout=httpx.Timeout
        ),
    )
    monkeypatch.setenv("PARALLEL_API_KEY", "synthetic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic")
    output = tmp_path / "run"
    argv = [
        f"search_engine={'baseline' if variant == 'baseline' else 'parallel-basic'}",
        f"suite={suite}",
        f"model={model}",
        f"source={source}",
        f"output_dir={output}",
        "limit=1",
        "synthetic=True",
        f"source_sha256={checksum}",
    ]
    if settings_case != "default":
        argv += ["max_tokens=64", f"effort={'none' if settings_case == 'none' else 'low'}"]
    argv += [
        "base_url=https://poison.invalid/v1",
        "api_key_env=UNUSED_POISON_KEY",
        "api=chat",
        "openrouter_backends=unused,unused",
    ]
    argv += ["vertex_project=unused-project", "vertex_location=unused-region"]
    chz.entrypoint(run_eval.main, argv=argv)
    terminal = assert_terminal(
        output, suite, empty_final, "Paris\nRome" if model.startswith("claude") else "Paris Rome"
    )
    if not empty_final:
        assert_native_tool_ids(
            terminal.convo,
            (
                []
                if variant == "baseline"
                else [f"s{i}" for i in range(21)]
                if variant == "budget"
                else [f"invalid-{i}" for i in range(1, 26)]
                if variant == "iteration"
                else ["same-native-id", "same-native-id"]
                if variant == "reused_ids"
                else ["s", "e"]
            ),
        )
    metadata = json.loads((output / "config.json").read_text())
    assert "prompt_version" not in metadata
    assert "native_protocol_version" not in metadata
    assert "excerpt_budget_version" not in metadata
    assert metadata["model"]["max_tokens"] == expected_cap
    assert metadata["model"]["effort"] == expected_effort
    generation_calls = sum(host != "api.parallel.ai" and not body.get("stream") for host, _, body in requests)
    assert timeouts == [5400] + [3600] * generation_calls
    saved = json.loads((output / "0.json").read_text())
    result_path = output.parent / "results" / "run.json"
    if empty_final:
        assert "no current final answer" in saved["failure"]
        assert "grader_result" not in saved
        assert json.loads(result_path.read_text())["score"] == 0
        assert not any(body.get("stream") for _, _, body in requests)
    else:
        assert read_answer(output / "0.json").convo.assistant_text.replace("\n", " ") == "Paris Rome"
        assert saved["grader_result"]["score"] == pytest.approx(expected_score(suite))
        assert json.loads(result_path.read_text())["score"] == pytest.approx(expected_score(suite))
    expected = {"baseline": 2, "budget": 23, "research": 6, "reused_ids": 6, "iteration": 27}[variant] - int(
        empty_final
    )
    assert len(requests) == expected
    assert all(client.is_closed() for client in clients)
    cached_bytes = {file: file.read_bytes() for file in output.rglob("*") if file.is_file()}
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == expected
    assert {file: file.read_bytes() for file in output.rglob("*") if file.is_file()} == cached_bytes
    if model.startswith("claude"):
        original = {file: file.read_bytes() for file in output.rglob("*") if file.is_file()}
        other_model = model.removesuffix("-thinking") if model.endswith("-thinking") else model + "-thinking"
        changed_argv = [f"model={other_model}" if arg.startswith("model=") else arg for arg in argv]
        with pytest.raises(ValueError, match="different model/mode/suite"):
            chz.entrypoint(run_eval.main, argv=changed_argv)
        assert len(requests) == expected
        assert {file: file.read_bytes() for file in output.rglob("*") if file.is_file()} == original
        assert metadata["model"]["thinking_budget"] == (1024 if model.endswith("-thinking") else None)
        assert all(client.is_closed() for client in clients)
    (output / "0.json").write_text('{"datum":')
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == expected * 2


@pytest.mark.parametrize("model", ["gpt-native", "claude-native"])
@pytest.mark.parametrize("suite", ["dsqa", "hle"])
def test_native_call_id_can_recur_in_later_turns(tmp_path, monkeypatch, model, suite):
    test_cli_native_answer_tools_official_f1_and_terminal_resume(
        tmp_path, monkeypatch, model, False, "reused_ids", suite, "override"
    )
