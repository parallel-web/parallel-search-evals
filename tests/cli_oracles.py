"""Shared source and frozen official-wire checks for the actual CLI matrix."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from parallel_search_evals.results import FailedAnswer, ScoredAnswer, read_answer


def write_source(source, suite, question):
    if suite == "hle":
        source.write_text(
            json.dumps({"id": "fixture-text", "question": question, "answer": "Paris;Rome", "image": ""}) + "\n"
        )
    else:
        with source.open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["problem", "problem_category", "answer", "answer_type"])
            writer.writerow([question, "synthetic", "Paris;Rome", "Set Answer"])
    return hashlib.sha256(source.read_bytes()).hexdigest()


def expected_score(suite):
    return 1.0 if suite != "dsqa" else 2 / 3


def judgment(suite):
    if suite != "dsqa":
        return json.dumps(
            {
                "extracted_final_answer": "Paris Rome",
                "reasoning": "The requested cities match.",
                "correct": "yes",
                "confidence": 88,
                "strict": True,
            }
        )
    return json.dumps(
        {
            "Answer Correctness": {
                "Explanation": "Fixture partial credit.",
                "Correctness Details": {"Paris": True, "Rome": False},
                "Excessive Answers": [],
            }
        }
    )


def assert_judge_request(request, body, suite, question, answer):
    oracle = json.loads((Path(__file__).parent / "fixtures" / f"reference-{suite}.json").read_text())
    expected = copy.deepcopy(oracle["wire_request"])
    prompt = expected["input"][0]["content"].replace(oracle["answer"], "__NATIVE_FINAL_ANSWER__")
    prompt = prompt.replace(oracle["task"]["question"], question)
    prompt = prompt.replace(oracle["task"]["reference"], "Paris;Rome")
    expected["input"][0]["content"] = prompt.replace("__NATIVE_FINAL_ANSWER__", answer)
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert body == expected


def assert_terminal(output, suite, failed, answer):
    result = read_answer(output / "0.json")
    assert result is not None
    summary = json.loads((output.parent / "results" / f"{output.name}.json").read_text())
    assert summary == {
        "score": 0 if failed else pytest.approx(expected_score(suite)),
        "total_samples": 1,
        "total_correct": int(not failed and suite != "dsqa"),
        "failed_samples": int(failed),
    }
    if failed:
        assert isinstance(result, FailedAnswer) and result.failure
    else:
        assert isinstance(result, ScoredAnswer) and result.convo.assistant_text == answer
        assert result.grader_result.score == pytest.approx(expected_score(suite))
        assert result.grader_result.correct is (suite != "dsqa")
        assert result.grader_result.grade_text == judgment(suite)
        roundtrip = type(result.convo).model_validate_json(result.convo.model_dump_json())
        assert roundtrip.request_messages() == result.convo.request_messages()
    return result


def assert_native_tool_ids(conversation, expected):
    calls, results = [], []
    for message in conversation.request_messages():
        if message.get("type") == "function_call":
            calls.append(message["call_id"])
        if message.get("type") == "function_call_output":
            results.append(message["call_id"])
        calls.extend(call["id"] for call in message.get("tool_calls") or [])
        if message.get("role") == "tool":
            results.append(message["tool_call_id"])
        content = message.get("content")
        if isinstance(content, list):
            calls.extend(part["id"] for part in content if part["type"] == "tool_use")
            results.extend(part["tool_use_id"] for part in content if part["type"] == "tool_result")
    assert calls == results == expected
