"""DSQA source parsing, official prompt rendering, and F1 grading."""

import csv
import io
from typing import Any

from parallel_search_evals.benchmarks.base import Question
from parallel_search_evals.runtime import GradingError

PROMPT_ASSET = "dsqa-judge.user.txt"
SCHEMA_ASSET = "dsqa-schema.json"
DSQA_SHA256 = "25d48dcf7efa872e5467032e8b8eedf38d301f59a252d0da95cda584baa78396"


def read_dsqa_questions(text: str) -> list[Question]:
    """Validate the export's exact columns and preserve source-row numbering.

    CSV values remain verbatim in Question and metadata; whitespace normalization
    belongs to the official judge renderer rather than source admission.
    """
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if set(reader.fieldnames or ()) != {"problem", "problem_category", "answer", "answer_type"}:
        raise ValueError("Unexpected DSQA source columns")
    return [
        Question(
            id=str(index),
            problem=row["problem"],
            answer=row["answer"],
            metadata={"answer_type": row["answer_type"], "problem_category": row["problem_category"]},
        )
        for index, row in enumerate(reader)
    ]


def render(template: str, question: Question, answer: str) -> str:
    """Apply upstream stripping and its two supported answer categories.

    Older records may carry prompt_type instead of answer_type. An existing but
    invalid answer_type must fail rather than fall back to that older field.
    """
    metadata = question.metadata
    prompt_type = metadata.get("answer_type", metadata.get("prompt_type"))
    if prompt_type not in {"Single Answer", "Set Answer"}:
        raise GradingError("DSQA requires Single Answer or Set Answer metadata")
    return template.format(
        prompt=question.problem.strip(),
        answer=question.answer.strip(),
        response=answer.strip(),
        prompt_type=prompt_type,
    )


def unique_parts(parts: list[str]) -> None:
    """Reject blank or repeated labels after whitespace and case normalization.

    Check reference and excessive-answer labels separately: matching labels across
    those two collections are permitted by the upstream grading contract.
    """
    normalized = [part.strip().casefold() for part in parts]
    if len(set(normalized) - {""}) != len(normalized):
        raise GradingError("Answer parts must be meaningful and unique")


def dsqa_score(value: dict[str, Any]) -> tuple[float, bool]:
    """Score schema-admitted answer parts with the official F1 arithmetic.

    Each true reference part contributes a match; every excessive answer lowers
    precision. Perfect correctness requires every reference and no extra answer.
    The zero-match case returns before the precision denominator can be zero.
    """
    judgment = value["Answer Correctness"]
    if not judgment["Explanation"].strip():
        raise GradingError("Judge explanation must be meaningful")
    parts = judgment["Correctness Details"]
    excessive = judgment["Excessive Answers"]
    unique_parts(list(parts))
    unique_parts(excessive)
    found, expected, extra = sum(parts.values()), len(parts), len(excessive)
    if found == 0:
        return 0.0, False
    precision = found / (found + extra)
    recall = found / expected
    f1 = 2 * precision * recall / (precision + recall)
    return f1, found == expected and extra == 0
