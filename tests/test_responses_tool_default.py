"""Provider-default automatic tools work without weakening explicit controls."""

import json

import chz
import httpx
import pytest
from openai import AsyncOpenAI

from parallel_search_evals import run_eval
from parallel_search_evals.models import ResponsesClient
from parallel_search_evals.results import read_answer
from parallel_search_evals.tools import ToolDispatcher
from parallel_search_evals.tools.schemas import ToolChoice
from tests.cli_oracles import judgment, write_source
from tests.sdk_transport import mock_owned_transport


def test_step_cli_search_extract_and_official_judge_without_explicit_auto(tmp_path, monkeypatch):
    source = tmp_path / "public.csv"
    checksum = write_source(source, "dsqa", "What are the requested cities?")
    counts = {"model": 0, "search": 0, "extract": 0, "judge": 0}

    def serve(request):
        body = json.loads(request.content)
        if request.url.host == "api.parallel.ai":
            counts["search" if request.url.path.endswith("search") else "extract"] += 1
            return httpx.Response(200, json={"results": [{"url": "https://example.org", "excerpts": ["Paris Rome"]}]})
        if body.get("stream"):
            counts["judge"] += 1
            assert request.url.host == "api.openai.com" and body["tool_choice"] == "none"
            response = {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": judgment("dsqa")}],
                    }
                ],
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps({'type': 'response.completed', 'response': response})}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        assert request.url.host == "openrouter.ai" and body["model"] == "stepfun/step-5-preview"
        assert body["max_output_tokens"] == 64000
        assert body["provider"] == {
            "only": ["stepfun/fp8"],
            "order": ["stepfun/fp8"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        if "tool_choice" in body:
            return httpx.Response(
                404, json={"error": {"message": "No endpoints found that support the provided tool_choice value"}}
            )
        counts["model"] += 1
        step = counts["model"]
        if step < 3:
            name, args = (
                ("search_web", {"objective": "cities", "search_queries": ["cities"]})
                if step == 1
                else ("web_fetch", {"objective": "verify", "urls": ["https://example.org"]})
            )
            output = [{"type": "function_call", "call_id": str(step), "name": name, "arguments": json.dumps(args)}]
        else:
            output = [
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris Rome"}]}
            ]
        return httpx.Response(200, json={"status": "completed", "output": output})

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "PARALLEL_API_KEY"):
        monkeypatch.setenv(key, "fixture")
    output = tmp_path / "run"
    chz.entrypoint(
        run_eval.main,
        argv=[
            "search_engine=parallel-fast",
            "suite=dsqa",
            "model=stepfun/step-5-preview",
            "provider=openrouter",
            "openrouter_backends=stepfun/fp8",
            "max_tokens=64000",
            f"source={source}",
            f"source_sha256={checksum}",
            "synthetic=True",
            f"output_dir={output}",
            "limit=1",
            "max_workers=1",
        ],
    )
    task = read_answer(output / "0.json")
    assert task.status == "completed" and task.grader_result is not None
    assert counts == {"model": 3, "search": 1, "extract": 1, "judge": 1}


@pytest.mark.parametrize("choice", list(ToolChoice))
async def test_responses_preserves_explicit_none_and_required(choice):
    sent = []

    def serve(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "completed", "output": []})

    async with AsyncOpenAI(
        api_key="fixture", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(serve))
    ) as client:
        model = ResponsesClient("native", client=client, max_tokens=64, effort=None)
        await model.generate(model.new_history().append_question("Q"), ToolDispatcher([], choice))
    if choice == ToolChoice.AUTO:
        assert "tool_choice" not in sent[0]
    else:
        assert sent[0]["tool_choice"] == choice.value
