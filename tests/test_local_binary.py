"""The local binary contract remains strict even when external JSON Schema is weak."""

import json

import chz
import httpx
import pytest

from parallel_search_evals import run_eval
from tests.sdk_transport import mock_owned_transport
from tests.test_local_cli import local_inputs


@pytest.mark.parametrize(
    "field,value",
    [
        ("correct", "maybe"),
        ("confidence", 1000),
        ("reasoning", 1),
        ("reasoning", " \t"),
        ("extracted_final_answer", " \t"),
        ("confidence", 1.0),
        ("confidence", True),
        ("unexpected", "extra field"),
    ],
)
def test_weak_local_schema_cannot_admit_invalid_binary_judgments(tmp_path, monkeypatch, field, value):
    assets, output, argv = local_inputs(tmp_path)
    (assets / "judge-schema.json").write_text(
        json.dumps({"type": "object", "required": ["extracted_final_answer", "reasoning", "correct", "confidence"]})
    )
    requests = []
    judgment = {
        "extracted_final_answer": "Guido van Rossum",
        "reasoning": "Matches the reference",
        "correct": "yes",
        "confidence": 100,
        field: value,
    }

    def serve(request):
        body = json.loads(request.content)
        requests.append(body)
        answer = json.dumps(judgment) if body.get("stream") else "Guido van Rossum"
        response = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": answer}]}],
        }
        if body.get("stream"):
            event = {"type": "response.completed", "response": response}
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(event)}\n\n"
            )
        return httpx.Response(200, json=response)

    mock_owned_transport(monkeypatch, httpx.MockTransport(serve))
    chz.entrypoint(run_eval.main, argv=[*argv, "search_engine=baseline"])
    record = json.loads((output / "0.json").read_text())
    assert record["status"] == "failed"
    assert record["failure"].startswith(("ValidationError", "GradingError"))
    assert "grader_result" not in record
    summary = json.loads((output.parent / "results" / f"{output.name}.json").read_text())
    assert summary == {"score": 0, "total_samples": 1, "total_correct": 0, "failed_samples": 1}
    assert len(requests) == 2
