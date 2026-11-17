"""Shared JSONL decoding and normalized question/answer exports."""

import io
import json
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any

from pydantic import ConfigDict, TypeAdapter

from parallel_search_evals.benchmarks.base import Question, QuestionList

LOCAL_QUESTION_ROW = TypeAdapter(dict[str, str], config=ConfigDict(hide_input_in_errors=True))


def jsonl_rows(text: str, decoder: Callable[[str], Any] = json.loads) -> Iterator[tuple[int, Any]]:
    """Decode rows in source order, retaining universal-newline line numbers."""
    for index, line in enumerate(io.StringIO(text, newline=None)):
        yield index, decoder(line)


def read_local_questions(text: str) -> QuestionList:
    """Read normalized questions while validating every metadata value as a string."""
    questions = []
    for index, row in jsonl_rows(text, partial(LOCAL_QUESTION_ROW.validate_json, strict=True)):
        problem, answer = row.pop("problem", None), row.pop("answer", None)
        row["source_id"] = row.pop("id", None)
        questions.append(Question(id=str(index), problem=problem, answer=answer, metadata=row))
    return questions
