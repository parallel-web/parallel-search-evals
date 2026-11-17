"""Question records, verified source admission, and the explicit benchmark catalog.

Catalog imports are lazy: loading benchmark assets must not initialize models or
terminal result records. Each definition links a benchmark's local source reader,
prompt renderer and scorer to its official prompt and schema files.
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, Field


class Question(BaseModel):
    id: str
    problem: str = Field(min_length=1, pattern=r"[^\s\x1c-\x1f]")
    answer: str = Field(min_length=1, pattern=r"[^\s\x1c-\x1f]")
    metadata: dict[str, str] = Field(default_factory=dict)


QuestionList = list[Question]


class BenchmarkDefinition(NamedTuple):
    loader: Callable[[str], QuestionList]
    render: Callable[[str, Question, str], str]
    score: Callable[[dict[str, Any]], tuple[float, bool]]
    prompt_asset: str
    schema_asset: str
    strict_schema: bool


def benchmark_catalog() -> dict[str, BenchmarkDefinition]:
    """Resolve concrete benchmark ownership after the question types exist."""
    from parallel_search_evals.benchmarks import binary, dsqa, hle, jsonl, local

    return {
        "dsqa": BenchmarkDefinition(
            dsqa.read_dsqa_questions, dsqa.render, dsqa.dsqa_score, dsqa.PROMPT_ASSET, dsqa.SCHEMA_ASSET, False
        ),
        "hle": BenchmarkDefinition(
            hle.read_hle_questions, hle.render, hle.hle_score, hle.PROMPT_ASSET, hle.SCHEMA_ASSET, True
        ),
        "local": BenchmarkDefinition(
            jsonl.read_local_questions, local.render, binary.score, "judge.user.txt", "judge-schema.json", True
        ),
    }


class BenchmarkQuestions(QuestionList):
    def __init__(
        self, path: Path, benchmark: str, limit: int | None, source_sha256: str | None = None, synthetic: bool = False
    ) -> None:
        if limit is not None and limit < 1:
            raise ValueError("Limit must be positive")
        from parallel_search_evals.benchmarks.dsqa import DSQA_SHA256

        expected = DSQA_SHA256 if benchmark == "dsqa" and not synthetic else source_sha256
        text = read_verified_export(path, expected)
        super().__init__(benchmark_catalog()[benchmark].loader(text)[:limit])
        if not self:
            raise ValueError("Source has no selected text questions")


def read_verified_export(path: Path, expected: str | None) -> str:
    """Admit a local export by file type and exact bytes before decoding questions."""
    if not path.is_file():
        raise ValueError("Local source must be a regular file")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Local source checksum does not match the pinned export")
    return content.decode("utf-8")
