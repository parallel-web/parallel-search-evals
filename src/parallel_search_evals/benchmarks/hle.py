"""HLE text-source selection, official prompt rendering, and strict grading."""

from typing import Any

from pydantic import BaseModel, ConfigDict

from parallel_search_evals.benchmarks.base import Question
from parallel_search_evals.benchmarks.jsonl import jsonl_rows
from parallel_search_evals.runtime import GradingError

PROMPT_ASSET = "hle-judge.user.txt"
SCHEMA_ASSET = "hle-schema.json"


class HLEQuestionRow(BaseModel):
    model_config = ConfigDict(hide_input_in_errors=True)

    image: str


def read_hle_questions(text: str) -> list[Question]:
    """Select text-only rows while retaining their original JSONL indices.

    Validate image as a strict string before selecting the row. Question/answer
    fields on excluded image rows are deliberately not validated.
    """
    questions = []
    for index, row in jsonl_rows(text):
        if HLEQuestionRow.model_validate(row, strict=True).image == "":
            questions.append(
                Question(
                    id=str(index),
                    problem=row.get("question"),
                    answer=row.get("answer"),
                    metadata={"source_id": row.get("id")},
                )
            )
    return questions


def render(template: str, question: Question, answer: str) -> str:
    """Substitute the official template without altering source whitespace."""
    return template.format(question=question.problem, correct_answer=question.answer, response=answer)


def hle_score(value: dict[str, Any]) -> tuple[float, bool]:
    """Enforce meaningful judge text and strict confidence beyond JSON Schema.

    JSON Schema admits integer-valued floats, whereas this contract requires a
    Python int and excludes bool. Correctness follows the exact yes/no judgment.
    """
    if type(value["confidence"]) is not int:
        raise GradingError("HLE confidence must be a strict integer")
    if not all(value[key].strip() for key in ("extracted_final_answer", "reasoning")):
        raise GradingError("HLE judge text fields must be meaningful")
    correct = value["correct"] == "yes"
    return float(correct), correct
