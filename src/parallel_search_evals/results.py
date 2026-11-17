"""Persist terminal answers and compute summaries over the complete cohort.

Records retain the original question/reference and complete native model history,
so they are private run data. Both successes and failures are reusable terminals;
missing or malformed records restart the whole question rather than a partial turn.
The serialized fields are the storage contract even when Python modules move.
Filesystem admission and atomic replacement are delegated to storage.py.
"""

from pathlib import Path
from typing import Annotated, Literal, Self, cast

import orjson
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from parallel_search_evals.benchmarks import Question
from parallel_search_evals.models import ModelHistory
from parallel_search_evals.storage import atomic_write, durable_mkdir, regular_file


class EvaluationSummary(BaseModel):
    score: float
    total_samples: int
    total_correct: int
    failed_samples: int = 0

    @classmethod
    def from_records(cls, results: list["AnswerRecord"]) -> Self:
        """Average all selected records, counting failures as zero-scoring samples."""
        completed = [result.grader_result for result in results if isinstance(result, ScoredAnswer)]
        return cls(
            score=sum(grade.score for grade in completed) / len(results),
            total_samples=len(results),
            total_correct=sum(grade.correct for grade in completed),
            failed_samples=len(results) - len(completed),
        )

    def save(self, output_dir: Path) -> None:
        results_dir = output_dir.parent / "results"
        durable_mkdir(results_dir)
        atomic_write(
            results_dir / f"{output_dir.name}.json", orjson.dumps(self.model_dump(), option=orjson.OPT_INDENT_2)
        )


class Grade(BaseModel):
    score: float = Field(strict=True, ge=0, le=1, allow_inf_nan=False)
    correct: bool = Field(strict=True)
    grade_text: str = Field(strict=True, min_length=1, pattern=r"[^\s\x1c-\x1f]")


class ScoredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["completed"] = "completed"
    datum: Question
    convo: ModelHistory
    grader_result: Grade

    @model_validator(mode="after")
    def require_current_answer(self) -> Self:
        if not self.convo.assistant_text.strip() or not self.convo.messages:
            raise ValueError("Completed conversation requires current answer text and native history")
        return self


class FailedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["failed"] = "failed"
    datum: Question
    failure: str = Field(strict=True, min_length=1, pattern=r"[^\s\x1c-\x1f]")


AnswerRecord = Annotated[ScoredAnswer | FailedAnswer, Field(discriminator="status")]
ANSWER_RECORD_ADAPTER: TypeAdapter[AnswerRecord] = TypeAdapter(cast(type[AnswerRecord], AnswerRecord))


def write_answer(result: AnswerRecord, task_file: Path) -> None:
    atomic_write(task_file, orjson.dumps(result.model_dump(), option=orjson.OPT_INDENT_2))


def read_answer(task_file: Path) -> AnswerRecord | None:
    try:
        with regular_file(task_file) as stream:
            return ANSWER_RECORD_ADAPTER.validate_json(stream.read())
    except (FileNotFoundError, ValueError):
        return None
