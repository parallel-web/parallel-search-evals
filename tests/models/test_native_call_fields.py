"""Malformed native identity stops at the decoded-call boundary before local tool execution."""

import json

import chz
import httpx
import pytest
from pydantic import ValidationError

from parallel_search_evals import run_eval
from parallel_search_evals.results import FailedAnswer, read_answer
from parallel_search_evals.tools.schemas import ToolInvocation
from tests.cli_oracles import judgment, write_source


@pytest.mark.parametrize("provider", ["openai", "anthropic", "chat"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("id", ""),
        ("id", " \t\n"),
        ("id", "\x1c"),
        ("id", "\x1d"),
        ("id", "\x1e"),
        ("id", "\x1f"),
        ("name", ""),
        ("name", " \t\n"),
        ("name", "\x1c\x1d\x1e\x1f"),
        ("id", 42),
        ("name", False),
    ],
)
def test_actual_sdk_cli_bad_call_identity_is_one_terminal_model_failure(provider, field, value, tmp_path, monkeypatch):
    source = tmp_path / "synthetic.hle"
    checksum = write_source(source, "hle", "Which cities?")
    requests, clients = [], []
    call = {"id": "native-call", "name": "search_web", field: value}
    arguments = {"objective": "Find cities", "search_queries": ["cities"]}
    model_steps = 0

    def serve(request):
        nonlocal model_steps
        body = json.loads(request.content)
        requests.append((request.url.host, body))
        if request.url.host == "api.parallel.ai":
            return httpx.Response(200, json={"results": []})
        if body.get("stream"):
            response = {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": judgment("hle")}],
                    }
                ],
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps({'type': 'response.completed', 'response': response})}\n\n",
                headers={"content-type": "text/event-stream"},
            )
        model_steps += 1
        if provider == "chat":
            message = (
                {"role": "assistant", "content": "Paris Rome"}
                if model_steps > 1
                else {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": json.dumps(arguments)},
                        }
                    ],
                }
            )
            return httpx.Response(
                200,
                json={
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "fixture",
                    "choices": [
                        {"index": 0, "finish_reason": "stop" if model_steps > 1 else "tool_calls", "message": message}
                    ],
                },
            )
        if provider == "anthropic":
            content = (
                [{"type": "text", "text": "Paris Rome"}]
                if model_steps > 1
                else [{"type": "tool_use", "id": call["id"], "name": call["name"], "input": arguments}]
            )
            return httpx.Response(
                200,
                json={
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fixture",
                    "stop_reason": "end_turn" if model_steps > 1 else "tool_use",
                    "content": content,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        output = (
            [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris Rome"}]}]
            if model_steps > 1
            else [
                {
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": call["name"],
                    "arguments": json.dumps(arguments),
                }
            ]
        )
        return httpx.Response(200, json={"status": "completed", "output": output})

    original_init = httpx.AsyncClient.__init__

    def client_init(client, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(serve)
        original_init(client, *args, **kwargs)
        clients.append(client)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", client_init)
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "PARALLEL_API_KEY", "CUSTOM_TEST_KEY"):
        monkeypatch.setenv(name, "fixture-key")
    output = tmp_path / "run"
    argv = [
        "search_engine=parallel-basic",
        "suite=hle",
        f"model={'claude-fixture' if provider == 'anthropic' else 'gpt-fixture'}",
        f"source={source}",
        f"source_sha256={checksum}",
        f"output_dir={output}",
        "limit=1",
        "max_workers=1",
        "synthetic=True",
        "max_tokens=64",
    ]
    if provider == "chat":
        argv += ["provider=compatible", "api=chat", "base_url=https://custom.example/v1", "api_key_env=CUSTOM_TEST_KEY"]
    else:
        argv += [f"provider={provider}"]
    chz.entrypoint(run_eval.main, argv=argv)
    terminal = read_answer(output / "0.json")
    assert isinstance(terminal, FailedAnswer) and terminal.failure == "ValidationError"
    assert len(requests) == model_steps == 1
    assert json.loads((tmp_path / "results/run.json").read_text()) == {
        "score": 0.0,
        "total_samples": 1,
        "total_correct": 0,
        "failed_samples": 1,
    }
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == 1 and before == {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert all(client.is_closed for client in clients)


@pytest.mark.parametrize("field", ["id", "name"])
@pytest.mark.parametrize("value", [b"native", 0, False, None])
def test_decoded_control_identity_requires_strings_without_coercion(field, value):
    with pytest.raises(ValidationError):
        ToolInvocation(**{"id": "native", "name": "search", "input": {}, field: value})


def test_meaningful_identity_is_validated_without_stripping_provider_values():
    call = ToolInvocation(id="  native  ", name="  search  ", input={})
    assert call.id == "  native  " and call.name == "  search  "
