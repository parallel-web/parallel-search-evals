"""Shared judge transport, JSON schema admission, and benchmark dispatch."""

import asyncio
import json
from importlib.resources import files
from typing import Any

import httpx
from jsonschema import Draft202012Validator, ValidationError
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from openai.types.responses import ResponseInputItemParam
from pydantic import BaseModel, InstanceOf, TypeAdapter

from parallel_search_evals import config
from parallel_search_evals.benchmarks.base import Question, benchmark_catalog
from parallel_search_evals.benchmarks.local import JudgeAssets
from parallel_search_evals.config import JUDGE_MODEL as JUDGE_MODEL
from parallel_search_evals.config import JUDGE_SECONDS
from parallel_search_evals.models import ModelHistory
from parallel_search_evals.models.openai_responses import parse_responses_reply
from parallel_search_evals.results import Grade
from parallel_search_evals.runtime import GradingError

ASSETS = files("parallel_search_evals.benchmarks").joinpath("assets")
SDK_EVENT = TypeAdapter(InstanceOf[BaseModel])


class BenchmarkGrader:
    """Use benchmark-owned semantics with one shared judge SDK transport.

    Each instance reads a fresh template and schema so callers may customize them
    without changing peer graders. Client construction precedes resource reads,
    preserving credential and asset error ordering.
    """

    def __init__(
        self, benchmark: str, client: AsyncOpenAI | None = None, *, judge_assets: JudgeAssets | None = None
    ) -> None:
        self.benchmark = benchmark
        self.client = client or AsyncOpenAI(
            base_url=config.ENDPOINTS["openai"],
            max_retries=config.SDK_MAX_RETRIES,
            http_client=DefaultAsyncHttpxClient(follow_redirects=False),
        )
        definition = benchmark_catalog()[benchmark]
        assets = judge_assets or JudgeAssets(
            "",
            ASSETS.joinpath(definition.prompt_asset).read_text(),
            json.loads(ASSETS.joinpath(definition.schema_asset).read_text()),
            {},
        )
        self.template = assets.template.rstrip("\n")
        self.schema = assets.schema
        self.system = assets.system.rstrip("\n")

    def render(self, question: Question, answer: str) -> str:
        if not answer.strip():
            raise GradingError("Cannot grade an empty answer")
        return benchmark_catalog()[self.benchmark].render(self.template, question, answer)

    async def judge(self, prompt: str) -> str:
        """Accept only a completed response through the native envelope parser.

        Deltas and incomplete events never become grades. The native parser owns
        output admission, including concatenation and refusal/error handling.
        """
        messages: list[ResponseInputItemParam] = [{"role": "user", "content": prompt}]
        if self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        async with asyncio.timeout(JUDGE_SECONDS):
            stream = await self.client.responses.create(
                model=JUDGE_MODEL,
                timeout=httpx.Timeout(
                    config.JUDGE_SECONDS, connect=config.CONNECT_SECONDS, read=config.JUDGE_IDLE_SECONDS
                ),
                input=messages,
                max_output_tokens=config.JUDGE_MAX_OUTPUT_TOKENS,
                reasoning={"effort": config.JUDGE_REASONING_EFFORT},
                store=False,
                include=["reasoning.encrypted_content"],
                tools=[],
                tool_choice="none",
                stream=True,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "judgement",
                        "schema": self.schema,
                        "strict": benchmark_catalog()[self.benchmark].strict_schema,
                    }
                },
            )
            async with stream:
                async for event in stream:
                    native = SDK_EVENT.validate_python(event).model_dump(warnings=False)
                    if native.get("type") == "response.completed":
                        return "".join(parse_responses_reply(native.get("response")).texts)
            raise GradingError("Judge stream ended without a completed response")

    async def grade(self, question: Question, history: ModelHistory) -> Grade:
        answer = history.assistant_text
        response = await self.judge(self.render(question, answer))
        return parse_grade(response, self.schema, self.benchmark)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate object keys at every JSON nesting level."""
    value = dict(pairs)
    if len(value) != len(pairs):
        raise GradingError("Judge response repeats a JSON key")
    return value


def parse_grade(response: str, schema: dict[str, Any], benchmark: str) -> Grade:
    """Require one complete schema-valid JSON value before benchmark scoring.

    Preserve the original response as grade_text. No markdown extraction, partial
    JSON repair, or schema-error detail is substituted into the terminal grade.
    """
    try:
        value = json.loads(response, object_pairs_hook=unique_object)
        Draft202012Validator(schema).validate(value)
    except (json.JSONDecodeError, ValidationError) as error:
        raise GradingError("Judge response does not match the official schema") from error
    score, correct = benchmark_catalog()[benchmark].score(value)
    return Grade(score=score, correct=correct, grade_text=response)
