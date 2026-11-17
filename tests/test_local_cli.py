"""The normal CLI runs local exports with external judging assets and safe resume."""

import hashlib
import json
from pathlib import Path

import chz
import httpx
import pytest

from parallel_search_evals import run_eval
from parallel_search_evals.benchmarks.judge import ASSETS
from tests.sdk_transport import mock_owned_transport


def local_inputs(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "original",
                "problem": "Who created Python?",
                "answer": "Guido van Rossum",
                "answer_type": "Single Answer",
            }
        )
        + "\n"
    )
    assets = tmp_path / "private-judge"
    assets.mkdir()
    (assets / "judge.system.txt").write_text("PRIVATE_SYSTEM_SENTINEL\n")
    (assets / "judge.user.txt").write_text(
        "Question: {question_json}\nReference: {reference_json}\nAnswer: {answer_json}\nType: {answer_type_json}\n"
    )
    schema = json.loads(ASSETS.joinpath("hle-schema.json").read_text())
    schema["properties"].pop("strict", None)
    schema["required"] = [key for key in schema["required"] if key != "strict"]
    (assets / "judge-schema.json").write_text(json.dumps(schema))
    (assets / "LICENSE").write_text("PRIVATE_LICENSE_SENTINEL\n")
    output = tmp_path / "run"
    argv = [
        "suite=local",
        "model=gpt-fixture",
        "provider=openai",
        f"source={source}",
        f"source_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}",
        f"output_dir={output}",
        f"judge_assets={assets}",
        "limit=1",
        "max_workers=1",
    ]
    return assets, output, argv


@pytest.mark.parametrize("mode", ["baseline", "parallel-fast"])
@pytest.mark.parametrize("changed_asset", ["judge.system.txt", "judge.user.txt", "judge-schema.json", "LICENSE"])
def test_local_cli_external_assets_search_grading_and_resume(tmp_path, monkeypatch, mode, changed_asset):
    monkeypatch.setenv("PARALLEL_API_KEY", "fixture")
    assets, output, argv = local_inputs(tmp_path)
    argv.append(f"search_engine={mode}")
    sent = []

    def serve(request):
        body = json.loads(request.content)
        sent.append((request.url.host, request.url.path, body))
        assert "PRIVATE_LICENSE_SENTINEL" not in json.dumps(body)
        if request.url.host == "api.parallel.ai":
            assert request.headers["x-stainless-package-version"] == "1.3.7"
            return httpx.Response(
                200, json={"results": [{"url": "https://example.org", "excerpts": ["Public evidence"]}]}
            )
        if body.get("stream"):
            assert body["model"] == "gpt-6-sol"
            assert body["input"] == [
                {"role": "system", "content": "PRIVATE_SYSTEM_SENTINEL"},
                {
                    "role": "user",
                    "content": 'Question: "Who created Python?"\nReference: "Guido van Rossum"\nAnswer: "Guido van Rossum"\nType: "Single Answer"',
                },
            ]
            assert body["text"]["format"]["strict"] is True
            judgment = {
                "extracted_final_answer": "Guido van Rossum",
                "reasoning": "Matches the reference",
                "correct": "yes",
                "confidence": 100,
            }
            response = {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(judgment)}]}],
            }
            return httpx.Response(
                200,
                text=f"data: {json.dumps({'type': 'response.completed', 'response': response})}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        assert "PRIVATE_SYSTEM_SENTINEL" not in json.dumps(body)
        assert "PRIVATE_LICENSE_SENTINEL" not in json.dumps(body)
        generations = [b for host, _, b in sent if host == "api.openai.com" and not b.get("stream")]
        if mode != "baseline" and len(generations) == 1:
            calls = [
                {"type": "function_call", "call_id": name, "name": name, "arguments": json.dumps(arguments)}
                for name, arguments in [
                    ("search_web", {"objective": "Public evidence", "search_queries": ["python creator"]}),
                    ("web_fetch", {"objective": "Public evidence", "urls": ["https://example.org"]}),
                ]
            ]
            return httpx.Response(200, json={"status": "completed", "output": calls})
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Guido van Rossum"}]}],
            },
        )

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))
    chz.entrypoint(run_eval.main, argv=argv)
    record = json.loads((output / "0.json").read_text())
    assert record["status"] == "completed" and record["grader_result"]["score"] == 1
    assert {path for host, path, _ in sent if host == "api.parallel.ai"} == (
        {"/v1/search", "/v1/extract"} if mode != "baseline" else set()
    )
    count = len(sent)
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(sent) == count
    changed = assets / changed_asset
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        chz.entrypoint(run_eval.main, argv=argv)
    assert len(sent) == count


def test_local_cli_requires_external_assets_before_opening_clients(tmp_path, monkeypatch):
    _, _, argv = local_inputs(tmp_path)
    argv = [arg for arg in argv if not arg.startswith("judge_assets=")] + ["search_engine=baseline"]
    from parallel_search_evals import models

    def forbidden(*args, **kwargs):
        pytest.fail("Clients opened before local asset admission")

    monkeypatch.setattr(models, "AsyncOpenAI", forbidden)
    with pytest.raises(ValueError):
        chz.entrypoint(run_eval.main, argv=argv)


@pytest.mark.parametrize("suite", ["dsqa", "hle"])
def test_official_suite_rejects_local_judge_override(tmp_path, suite):
    _, _, argv = local_inputs(tmp_path)
    argv = [f"suite={suite}" if arg == "suite=local" else arg for arg in argv] + ["search_engine=baseline"]
    with pytest.raises(ValueError):
        chz.entrypoint(run_eval.main, argv=argv)
