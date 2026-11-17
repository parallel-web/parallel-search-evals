"""Remote error bodies cannot enter terminal failure files through SDK exceptions."""

import json

import chz
import google.auth
import httpx
import pytest
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import APIError as GoogleAPIError
from pydantic import TypeAdapter, ValidationError

from parallel_search_evals import run_eval
from parallel_search_evals.benchmarks import Question
from parallel_search_evals.results import FailedAnswer, read_answer
from parallel_search_evals.runtime import GradingError, NativeResponseError
from tests.cli_oracles import write_source
from tests.models.test_gemini_native import Credentials
from tests.run.helpers import FixtureRunner

SECRET = "offline-dummy-credential-must-not-persist"


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "vertex"])
@pytest.mark.parametrize("stage", ["generation", "judge"])
@pytest.mark.parametrize("status", [400, 503])
def test_native_sdk_cli_error_echo_keeps_only_type_status_and_reuses_terminal(
    provider, stage, status, tmp_path, monkeypatch, capsys
):
    source = tmp_path / "synthetic.hle"
    checksum = write_source(source, "hle", "Which cities?")
    requests, clients = [], []

    def serve(request):
        body = json.loads(request.content)
        requests.append((request.url.host, body))
        if stage == "generation" or body.get("stream"):
            return httpx.Response(
                status,
                json={"error": {"code": status, "type": "fixture_remote_error", "message": SECRET}},
            )
        if request.url.host == "api.anthropic.com":
            return httpx.Response(
                200,
                json={
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-fixture",
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "Paris Rome"}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        if request.url.host != "api.openai.com":
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {"finishReason": "STOP", "content": {"role": "model", "parts": [{"text": "Paris Rome"}]}}
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Paris Rome"}]}
                ],
            },
        )

    original_client_init = httpx.AsyncClient.__init__

    def client_init(client, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(serve)
        original_client_init(client, *args, **kwargs)
        clients.append(client)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", client_init)
    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (Credentials(), "fixture-project"))
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, SECRET)
    output = tmp_path / "run"
    model = "claude-fixture" if provider == "anthropic" else "gpt-fixture" if provider == "openai" else "gemini-fixture"
    argv = [
        "search_engine=baseline",
        "suite=hle",
        f"model={model}",
        f"provider={provider}",
        f"source={source}",
        f"source_sha256={checksum}",
        f"output_dir={output}",
        "limit=1",
        "max_workers=1",
        "synthetic=True",
        "max_tokens=64",
    ]
    if provider == "vertex":
        argv += ["vertex_project=fixture-project", "vertex_location=global"]
    chz.entrypoint(run_eval.main, argv=argv)
    terminal = read_answer(output / "0.json")
    expected_type = "BadRequestError" if status == 400 else "InternalServerError"
    if stage == "generation" and provider in {"gemini", "vertex"}:
        expected_type = "ClientError" if status == 400 else "ServerError"
    assert terminal.failure == f"{expected_type}: HTTP {status}"
    assert isinstance(terminal, FailedAnswer)
    assert len(requests) == (1 if stage == "generation" else 2)
    assert json.loads((tmp_path / "results/run.json").read_text()) == {
        "score": 0.0,
        "total_samples": 1,
        "total_correct": 0,
        "failed_samples": 1,
    }
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert all(SECRET.encode() not in value for value in before.values())
    chz.entrypoint(run_eval.main, argv=argv)
    assert len(requests) == (1 if stage == "generation" else 2)
    assert before == {str(path): path.read_bytes() for path in tmp_path.rglob("*.json")}
    assert all(client.is_closed for client in clients)
    console = capsys.readouterr()
    assert SECRET not in console.out + console.err


def validation_error():
    try:
        TypeAdapter(int).validate_python(SECRET)
    except ValidationError as error:
        return error
    raise AssertionError("Expected the external validation value to fail")


@pytest.mark.parametrize(
    "error,expected",
    [
        (httpx.ConnectError(SECRET), "ConnectError"),
        (TimeoutError(SECRET), "TimeoutError"),
        (GoogleAuthError(SECRET), "GoogleAuthError"),
        (GoogleAPIError(0, {"error": {"code": 0, "message": SECRET}}), "APIError"),
        (GoogleAPIError(0, {"error": {"code": SECRET, "message": SECRET}}), "APIError"),
        (GoogleAPIError(True, {"error": {"message": SECRET}}), "APIError"),
        *[
            (
                GoogleAPIError(code, {"error": {"message": SECRET}}),
                f"APIError: HTTP {code}" if 100 <= code <= 599 else "APIError",
            )
            for code in (-1, 99, 100, 599, 600)
        ],
        (validation_error(), "ValidationError"),
        (json.JSONDecodeError(SECRET, SECRET, 0), "JSONDecodeError"),
        (
            httpx.HTTPStatusError(
                SECRET, request=httpx.Request("GET", "https://example.org/" + SECRET), response=httpx.Response(418)
            ),
            "HTTPStatusError: HTTP 418",
        ),
        (
            NativeResponseError("Model returned no current final answer"),
            "NativeResponseError: Model returned no current final answer",
        ),
        (
            GradingError("Judge stream ended without a completed response"),
            "GradingError: Judge stream ended without a completed response",
        ),
    ],
)
async def test_terminal_failure_policy_preserves_trusted_messages_but_not_external_values(error, expected, tmp_path):
    calls = []

    class Suite(FixtureRunner):
        async def _evaluate_question(self, datum):
            calls.append(datum.id)
            raise error

    suite = Suite(tmp_path / "run", 1)
    suite.questions = [Question(id="failed", problem="Q", answer="A")]
    first = await suite.run()
    before = (tmp_path / "run/failed.json").read_bytes()
    second = await suite.run()
    assert calls == ["failed"] and before == (tmp_path / "run/failed.json").read_bytes()
    assert first == second and first.total_samples == first.failed_samples == 1 and first.score == 0
    assert read_answer(tmp_path / "run/failed.json").failure == expected
    assert SECRET.encode() not in before


@pytest.mark.parametrize("error", [RuntimeError("programmer error"), TypeError("programmer error")])
async def test_programmer_errors_propagate_without_becoming_cached_failures(error, tmp_path):
    class Suite(FixtureRunner):
        async def _evaluate_question(self, datum):
            raise error

    suite = Suite(tmp_path / "run", 1)
    suite.questions = [Question(id="failed", problem="Q", answer="A")]
    with pytest.raises(type(error), match="programmer error"):
        await suite.run()
    assert not (tmp_path / "run/failed.json").exists()
