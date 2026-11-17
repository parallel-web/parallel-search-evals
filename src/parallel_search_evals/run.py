"""Construct, own and execute one evaluation run.

Source admission precedes client creation. Saved settings are checked under the
exclusive run lock before terminal reuse; changed model, source or retrieval
policy settings cannot share a run directory. The runner bounds active question
transactions and persists failures in the same denominator as successful grades.
All sibling tasks are joined before their shared clients close in reverse order.
The public CLI delegates here without owning provider or persistence lifetimes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import AsyncExitStack, suppress
from pathlib import Path

import anthropic
import httpx
import openai
import orjson
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import APIError as GoogleAPIError
from pydantic import ValidationError

from parallel_search_evals.agent import AnswerGenerator
from parallel_search_evals.benchmarks import Question, QuestionList
from parallel_search_evals.benchmarks.base import BenchmarkQuestions, benchmark_catalog
from parallel_search_evals.benchmarks.judge import JUDGE_MODEL, BenchmarkGrader
from parallel_search_evals.benchmarks.local import JudgeAssets, load_judge_assets
from parallel_search_evals.config import DEFAULT_MAX_WORKERS, DEFAULT_QUESTION_LIMIT, FILTER_BENCHMARKS
from parallel_search_evals.models.settings import ModelSettings
from parallel_search_evals.results import (
    AnswerRecord,
    EvaluationSummary,
    FailedAnswer,
    ScoredAnswer,
    read_answer,
    write_answer,
)
from parallel_search_evals.runtime import NativeResponseError, join_tasks
from parallel_search_evals.storage import atomic_write, durable_mkdir, regular_file, run_lock
from parallel_search_evals.tools.source_filter import BENCHMARK_SOURCES


def build_evaluation(
    benchmark: str,
    research_mode: str,
    model_settings: ModelSettings,
    output_dir: Path,
    source: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
    limit: int | None = DEFAULT_QUESTION_LIMIT,
    source_sha256: str | None = None,
    synthetic: bool = False,
    filter_benchmarks: bool = FILTER_BENCHMARKS,
    judge_assets: JudgeAssets | None = None,
) -> EvaluationRunner:
    catalog = benchmark_catalog()
    if benchmark not in catalog:
        raise ValueError("Suite must be " + " or ".join(catalog))
    questions = BenchmarkQuestions(source, benchmark, limit, source_sha256, synthetic)
    generator = AnswerGenerator(research_mode, model_settings, filter_benchmarks=filter_benchmarks)
    grader = BenchmarkGrader(benchmark, judge_assets=judge_assets)
    return EvaluationRunner(questions, generator, grader, output_dir, max_workers)


logger = logging.getLogger(__name__)


def failure_detail(error: Exception) -> str:
    from parallel_search_evals.runtime import GradingError

    kind = type(error).__name__
    if isinstance(error, (NativeResponseError, GradingError)):
        return f"{kind}: {error}"
    status = getattr(error, "status_code", None)
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
    if isinstance(error, GoogleAPIError):
        status = error.code
    if type(status) is int and 100 <= status <= 599:
        return f"{kind}: HTTP {status}"
    return kind


class EvaluationRunner:
    def __init__(
        self,
        questions: QuestionList,
        generator: AnswerGenerator,
        grader: BenchmarkGrader,
        output_dir: Path,
        max_workers: int,
    ) -> None:
        self.questions = questions
        self.generator = generator
        self.grader = grader
        self.output_dir = output_dir
        self.max_workers = max_workers

    async def _evaluate_question(self, question: Question) -> ScoredAnswer:
        history = await self.generator.answer(question.problem)
        grade = await self.grader.grade(question, history)
        return ScoredAnswer(datum=question, convo=history, grader_result=grade)

    async def _terminal_record(self, question: Question) -> AnswerRecord:
        from parallel_search_evals.runtime import GradingError

        task_file = self.output_dir / f"{question.id}.json"
        cached = read_answer(task_file)
        if cached is not None:
            logger.info("Question %s %s", question.id, cached.status)
            return cached
        result: AnswerRecord
        try:
            result = await self._evaluate_question(question)
        except (
            TimeoutError,
            httpx.HTTPError,
            openai.APIError,
            anthropic.APIError,
            GoogleAPIError,
            GoogleAuthError,
            ValidationError,
            json.JSONDecodeError,
            UnicodeDecodeError,
            NativeResponseError,
            GradingError,
        ) as error:
            result = FailedAnswer(datum=question, failure=failure_detail(error))
            logger.warning("Question %s failed: %s", question.id, result.failure)
        write_answer(result, task_file)
        logger.info("Question %s %s", question.id, result.status)
        return result

    async def run(self) -> EvaluationSummary:
        """Reuse terminal results and bound active work by max_workers.

        Failed samples remain in the full cohort denominator when averaging scores.
        """
        durable_mkdir(self.output_dir)
        semaphore = asyncio.Semaphore(self.max_workers)

        async def evaluate(question: Question) -> AnswerRecord:
            async with semaphore:
                return await self._terminal_record(question)

        results = await join_tasks([asyncio.create_task(evaluate(question)) for question in self.questions])
        summary = EvaluationSummary.from_records(results)
        summary.save(self.output_dir)
        return summary


async def run_evaluation(
    research_mode: str,
    model_settings: ModelSettings,
    benchmark: str,
    output_dir: Path,
    max_workers: int,
    source: Path,
    limit: int | None = DEFAULT_QUESTION_LIMIT,
    source_sha256: str | None = None,
    synthetic: bool = False,
    filter_benchmarks: bool = FILTER_BENCHMARKS,
    judge_assets: Path | None = None,
) -> None:
    """Check saved settings before opening clients or reusing terminal answers.

    Record the known-source list even when filtering is disabled, so the saved
    configuration describes the policy available to this run without a version counter.
    """
    assets = load_judge_assets(benchmark, judge_assets)
    config = {
        "model": model_settings.model_dump(mode="json"),
        "judge_model": JUDGE_MODEL,
        "search_engine": research_mode,
        "suite": benchmark,
        "synthetic": synthetic,
        "source": str(source.resolve()),
        "source_sha256": source_sha256,
        "filter_benchmarks": filter_benchmarks,
        "benchmark_sources": list(BENCHMARK_SOURCES),
        **({"judge_assets": assets.sha256} if assets is not None else {}),
    }
    with run_lock(output_dir) as output_dir:
        config_path = output_dir / "config.json"
        with suppress(FileNotFoundError), regular_file(config_path) as stream:
            if orjson.loads(stream.read()) != config:
                raise ValueError("Output directory belongs to a different model/mode/suite")
        evaluation = build_evaluation(
            benchmark,
            research_mode,
            model_settings,
            output_dir=output_dir,
            max_workers=max_workers,
            source=source,
            limit=limit,
            source_sha256=source_sha256,
            synthetic=synthetic,
            filter_benchmarks=filter_benchmarks,
            judge_assets=assets,
        )
        async with AsyncExitStack() as stack:
            stack.push_async_callback(evaluation.generator.model_client.aclose)
            stack.push_async_callback(evaluation.grader.client.close)
            if evaluation.generator.research_client is not None:
                stack.push_async_callback(evaluation.generator.research_client.sdk.close)
            atomic_write(config_path, orjson.dumps(config))
            result = await evaluation.run()
            logger.info("Evaluation score: %s", result.score)
