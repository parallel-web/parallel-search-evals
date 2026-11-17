"""The actual CLI carries local source policy through native tools and run identity."""

import json

import chz
import httpx
import pytest

from parallel_search_evals import run as evaluation_run
from parallel_search_evals import run_eval
from tests.cli_oracles import assert_native_tool_ids, assert_terminal, judgment, write_source
from tests.sdk_transport import mock_owned_transport

BLOCKED = "https://huggingface.co/datasets/cais/hle/resolve/main/test.jsonl"
ALLOWED = "https://example.org/synthetic"
BLOCK_ERROR = "Error: Benchmark source URLs are blocked by the retrieval filter"
RAW = {
    "results": [
        {"url": BLOCKED, "excerpts": ["Restricted synthetic fixture"], "future": None},
        {"url": ALLOWED, "excerpts": ["Paris"], "future": {"opaque": [1, None]}},
    ],
    "errors": [{"url": BLOCKED, "message": "synthetic error"}, {"url": ALLOWED, "message": "allowed error"}],
    "warnings": ["Synthetic warning"],
    "future": {"opaque": [False, None, {"value": "retained"}]},
}
FILTERED = {
    "results": [{"url": ALLOWED, "excerpts": ["Paris"], "future": {"opaque": [1, None]}}],
    "errors": [{"url": ALLOWED, "message": "allowed error"}],
    "warnings": ["Synthetic warning"],
    "future": {"opaque": [False, None, {"value": "retained"}]},
}


def native_response(api, call=None):
    if api == "responses":
        output = (
            [{"type": "function_call", "call_id": call[0], "name": call[1], "arguments": json.dumps(call[2])}]
            if call
            else [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris Rome"}]}]
        )
        return {"status": "completed", "output": output}
    message = {"role": "assistant", "content": "Paris Rome"}
    if call:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call[0], "type": "function", "function": {"name": call[1], "arguments": json.dumps(call[2])}}
            ],
        }
    return {
        "id": "chat",
        "object": "chat.completion",
        "created": 1,
        "model": "native",
        "choices": [{"index": 0, "finish_reason": "tool_calls" if call else "stop", "message": message}],
    }


@pytest.fixture
def native_run(tmp_path, monkeypatch):
    """Replace HTTP only; keep CLI, SDKs, dispatch, grading and storage real."""
    source = tmp_path / "synthetic.hle"
    checksum = write_source(source, "hle", "Requested cities?")
    requests = []
    model_bodies = []
    state = {}
    transport_errors = []

    def serve(request):
        body = json.loads(request.content)
        requests.append((str(request.url), body))
        if request.url.host == "api.parallel.ai":
            assert body == {
                **(
                    {"mode": "basic", "search_queries": ["synthetic cities"]}
                    if request.url.path.endswith("search")
                    else {"urls": state["extract_urls"]}
                ),
                "objective": "synthetic cities",
                "client_model": "synthetic-model",
            }
            return httpx.Response(200, json=RAW)
        if body.get("stream"):
            assert request.url.host == "api.openai.com"
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
                text=f"data: {json.dumps({'type': 'response.completed', 'response': response})}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        model_bodies.append(body)
        step = len(model_bodies)
        calls = [
            ("search", "search_web", {"objective": "synthetic cities", "search_queries": ["synthetic cities"]}),
            ("blocked", "web_fetch", {"objective": "synthetic cities", "urls": [ALLOWED, BLOCKED]}),
            ("allowed", "web_fetch", {"objective": "synthetic cities", "urls": [ALLOWED]}),
        ]
        if step > 1:
            history = body["input"] if state["api"] == "responses" else body["messages"]
            replies = (
                [item["output"] for item in history if item.get("type") == "function_call_output"]
                if state["api"] == "responses"
                else [item["content"] for item in history if item.get("role") == "tool"]
            )
            assert json.loads(replies[0]) == (FILTERED if state["filtered"] else RAW)
            if step > 2:
                if state["filtered"]:
                    assert replies[1] == BLOCK_ERROR
                else:
                    assert json.loads(replies[1]) == RAW
            if step > 3:
                assert json.loads(replies[2]) == (FILTERED if state["filtered"] else RAW)
        if step < 4:
            state["extract_urls"] = calls[step - 1][2].get("urls")
        return httpx.Response(200, json=native_response(state["api"], calls[step - 1] if step < 4 else None))

    def checked_serve(request):
        try:
            return serve(request)
        except AssertionError as error:
            transport_errors.append(error)
            raise

    mock_owned_transport(monkeypatch, httpx.MockTransport(checked_serve))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-judge-key")
    monkeypatch.setenv("CUSTOM_TEST_KEY", "synthetic-model-key")
    monkeypatch.setenv("PARALLEL_API_KEY", "synthetic-parallel-key")

    def run(api="responses", filtered=True):
        state.update(api=api, filtered=filtered)
        output = tmp_path / "run"
        argv = [
            "search_engine=parallel-basic",
            "suite=hle",
            "model=synthetic-model",
            "provider=compatible",
            f"api={api}",
            "base_url=https://synthetic.example/v1",
            "api_key_env=CUSTOM_TEST_KEY",
            "max_tokens=64",
            f"source={source}",
            f"source_sha256={checksum}",
            f"output_dir={output}",
            "limit=1",
            "synthetic=True",
        ]
        if not filtered:
            argv.append("filter_benchmarks=False")
        chz.entrypoint(run_eval.main, argv=argv)
        if transport_errors:
            raise transport_errors[0]
        return argv, output, requests, model_bodies

    return run


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("filtered", [True, False])
def test_cli_filter_defaults_optout_recovery_and_terminal_reuse(native_run, api, filtered):
    argv, output, requests, model_bodies = native_run(api, filtered)
    terminal = assert_terminal(output, "hle", False, "Paris Rome")
    assert_native_tool_ids(terminal.convo, ["search", "blocked", "allowed"])
    parallel = [body for url, body in requests if "api.parallel.ai" in url]
    assert len(parallel) == (2 if filtered else 3)
    assert parallel[-1]["urls"] == [ALLOWED]
    if filtered:
        assert not any(BLOCKED in body.get("urls", []) for body in parallel)
    else:
        assert parallel[1]["urls"] == [ALLOWED, BLOCKED]
    assert len(model_bodies) == 4
    config = json.loads((output / "config.json").read_text())
    from parallel_search_evals.tools.source_filter import BENCHMARK_SOURCES

    assert config["filter_benchmarks"] is filtered
    assert config["benchmark_sources"] == list(BENCHMARK_SOURCES)
    count = len(requests)
    terminal_bytes = (output / "0.json").read_bytes()
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == count
    assert (output / "0.json").read_bytes() == terminal_bytes


@pytest.mark.parametrize("previous", ["enabled", "disabled", "legacy_enabled", "legacy_disabled", "changed_sources"])
def test_cli_rejects_incompatible_filter_identity_before_setup(native_run, monkeypatch, previous):
    filtered = previous not in {"enabled", "legacy_disabled"}
    argv, output, requests, _ = native_run(filtered=filtered)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    if previous.startswith("legacy"):
        config.pop("filter_benchmarks", None)
        config.pop("benchmark_sources", None)
    elif previous == "changed_sources":
        config["benchmark_sources"] = ["https://obsolete.example/synthetic/"]
    else:
        config["filter_benchmarks"] = not filtered
    config_path.write_text(json.dumps(config))
    before = config_path.read_bytes(), (output / "0.json").read_bytes(), len(requests)

    def forbidden(*args, **kwargs):
        pytest.fail("Incompatible policy reached provider setup")

    monkeypatch.setattr(evaluation_run, "build_evaluation", forbidden)
    with pytest.raises(ValueError, match="different"):
        chz.entrypoint(run_eval.main, argv=argv)
    assert (config_path.read_bytes(), (output / "0.json").read_bytes(), len(requests)) == before
